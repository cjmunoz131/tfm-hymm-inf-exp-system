"""
==============================================================================
HYMM-REC Explainability: Fine-Tuning Llama 3.1 8B con QLoRA
==============================================================================
SageMaker Training Job que ejecuta:
  1. Carga train.jsonl + val.jsonl desde S3
  2. Aplica template de prompt LLaMA 3 (Instruction + Input + Response)
  3. Configura QLoRA 4-bit (BitsAndBytes + LoRA adapters)
  4. Entrena con SFTTrainer (masked loss sobre respuesta unicamente)
  5. Guarda adapter weights + tokenizer en /opt/ml/model/
  6. Copia code/inference.py + code/requirements.txt al model dir
     (para deploy directo con PyTorchModel sin re-empaquetado)

Input channels:
  /opt/ml/input/data/train/ -> train.jsonl, val.jsonl

Output (model.tar.gz):
  /opt/ml/model/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json, tokenizer_config.json, special_tokens_map.json
    └── code/
        ├── inference.py       (carga base BF16 + adapter LoRA)
        └── requirements.txt   (transformers, peft, accelerate)

Deploy:
  El model.tar.gz se despliega con PyTorchModel (framework_version="2.1").
  El container detecta code/inference.py, instala code/requirements.txt,
  y ejecuta model_fn() que carga base BF16 + adapter LoRA sin merge.

Estimator: HuggingFace (ml.g5.12xlarge para training QLoRA 4-bit)
Framework: transformers + peft + trl (SFTTrainer)
==============================================================================
"""

import subprocess
import sys

# Instalar dependencias antes de importar (garantiza versiones correctas)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.44.2",
    "trl==0.10.1",
    "peft==0.12.0",
    "accelerate==0.33.0",
    "bitsandbytes==0.43.1",
    "datasets==2.20.0",
])

import os
import json
import argparse
import random
import shutil
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ==============================================================================
# REPRODUCIBILIDAD
# ==============================================================================

def seed_everything(seed=42):
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Reproducibilidad] Semilla fijada en: {seed}")


# ==============================================================================
# PROMPT TEMPLATE (LLaMA 3 SFT Format)
# ==============================================================================

LLAMA_3_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}<|eot_id|>"""


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_format_datasets(data_dir):
    """
    Carga train.jsonl y val.jsonl, aplica el template de LLaMA 3.
    Returns: datasets.DatasetDict con splits 'train' y 'validation'.
    """
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No se encontro train.jsonl en {data_dir}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"No se encontro val.jsonl en {data_dir}")

    dataset_dict = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )

    def format_row(fila):
        return {"text": LLAMA_3_PROMPT_TEMPLATE.format(**fila)}

    dataset_dict = dataset_dict.map(
        format_row, remove_columns=dataset_dict["train"].column_names
    )

    print(f"Datasets cargados y formateados:")
    print(f"  Train: {len(dataset_dict['train'])} ejemplos")
    print(f"  Validation: {len(dataset_dict['validation'])} ejemplos")

    return dataset_dict


# ==============================================================================
# MODELO Y TOKENIZADOR (QLoRA 4-bit)
# ==============================================================================

def load_model_and_tokenizer(model_id, hf_token=None):
    """Carga modelo base con cuantizacion 4-bit y tokenizador."""
    print(f"Cargando modelo: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )

    model = prepare_model_for_kbit_training(model)
    print(f"Modelo cargado exitosamente en 4-bit (NF4)")

    return model, tokenizer


# ==============================================================================
# CUSTOM DATA COLLATOR (Masked Loss)
# ==============================================================================

def get_data_collator(tokenizer):
    """
    Collator que enmascara el prompt (Loss = -100)
    para que el modelo solo aprenda a predecir las 3 keywords.
    """
    response_template = "### Response:\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    class ExplainabilityCollator(DataCollatorForLanguageModeling):
        def __init__(self, response_ids, tok):
            super().__init__(tokenizer=tok, mlm=False)
            self.response_template_ids = response_ids

        def torch_call(self, examples):
            batch = super().torch_call(examples)
            for i in range(len(batch["labels"])):
                etiquetas = batch["labels"][i].tolist()
                patron = self.response_template_ids
                longitud_patron = len(patron)

                idx_inicio = -1
                for j in range(len(etiquetas) - longitud_patron):
                    if etiquetas[j : j + longitud_patron] == patron:
                        idx_inicio = j + longitud_patron
                        break

                if idx_inicio != -1:
                    batch["labels"][i][:idx_inicio] = -100
                else:
                    batch["labels"][i][:] = -100

            return batch

    return ExplainabilityCollator(response_template_ids, tokenizer)


# ==============================================================================
# INFERENCE ARTIFACTS — Incluir code/ en el model.tar.gz
# ==============================================================================

def include_inference_code(model_dir):
    """
    Copia inference.py y requirements.txt al directorio code/ dentro de
    SM_MODEL_DIR. SageMaker empaqueta todo SM_MODEL_DIR como model.tar.gz.

    Cuando se despliega con PyTorchModel, el container:
      1. Detecta code/requirements.txt -> pip install
      2. Detecta code/inference.py -> usa model_fn, predict_fn, etc.
    """
    code_dir = os.path.join(model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    # Intentar copiar desde source_dir del training job
    source_dir = os.path.dirname(os.path.abspath(__file__))
    inference_src = os.path.join(source_dir, "..", "inference", "inference.py")
    requirements_src = os.path.join(source_dir, "..", "inference", "requirements.txt")

    # Si los archivos estan en el source_dir del job (empaquetados por SDK)
    if not os.path.exists(inference_src):
        inference_src = os.path.join(source_dir, "inference.py")
    if not os.path.exists(requirements_src):
        requirements_src = os.path.join(source_dir, "requirements.txt")

    if os.path.exists(inference_src):
        shutil.copy(inference_src, os.path.join(code_dir, "inference.py"))
        print(f"  Copiado: inference.py -> {code_dir}")
    else:
        # Fallback: generar inference.py inline
        print("  WARN: inference.py no encontrado en source_dir, generando inline...")
        _write_inference_py_inline(code_dir)

    if os.path.exists(requirements_src):
        shutil.copy(requirements_src, os.path.join(code_dir, "requirements.txt"))
        print(f"  Copiado: requirements.txt -> {code_dir}")
    else:
        # Fallback: generar requirements.txt
        with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
            f.write("transformers==4.44.2\npeft==0.12.0\naccelerate==0.33.0\n")
        print(f"  Generado: requirements.txt -> {code_dir}")


def _write_inference_py_inline(code_dir):
    """Genera inference.py como fallback si no se encuentra en source_dir."""
    inference_code = '''import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def model_fn(model_dir):
    config_path = os.path.join(model_dir, "adapter_config.json")
    with open(config_path, "r") as f:
        adapter_config = json.load(f)
    base_model_id = adapter_config.get("base_model_name_or_path", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    hf_token = os.environ.get("HF_TOKEN", "")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token if hf_token else None)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto",
        token=hf_token if hf_token else None,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    prompt = input_data.get("inputs", "")
    parameters = input_data.get("parameters", {})
    max_new_tokens = parameters.get("max_new_tokens", 20)
    temperature = parameters.get("temperature", 0.1)
    repetition_penalty = parameters.get("repetition_penalty", 1.1)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_token_id is not None and eot_token_id != tokenizer.eos_token_id:
        eos_ids.append(eot_token_id)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            repetition_penalty=repetition_penalty, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids, do_sample=True if temperature > 0 else False,
        )
    resultado_crudo = tokenizer.decode(outputs[0], skip_special_tokens=True)
    etiqueta_separadora = "### Response:\\n"
    if etiqueta_separadora in resultado_crudo:
        prediccion = resultado_crudo.split(etiqueta_separadora)[-1].strip()
    else:
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediccion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    prediccion = prediccion.split("://")[0].strip().split("http")[0].strip()
    prediccion = prediccion.split(".")[0].strip()
    prediccion = prediccion.split("\\n")[0].strip()
    prediccion = prediccion.rstrip(".")
    etiquetas = [e.strip() for e in prediccion.split(",")]
    prediccion_final = ", ".join(etiquetas[:3])
    del inputs, outputs
    torch.cuda.empty_cache()
    return [{"generated_text": prediccion_final}]

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported response type: {response_content_type}")
'''
    with open(os.path.join(code_dir, "inference.py"), "w") as f:
        f.write(inference_code)


# ==============================================================================
# TRAINING — ORCHESTRATOR
# ==============================================================================

def train(args):
    """Orquesta el fine-tuning completo con QLoRA. Guarda solo adapter."""

    seed_everything(args.seed)

    # 1. Cargar datos
    print("\n[1/5] Cargando datasets...")
    dataset_dict = load_and_format_datasets(args.data_dir)

    # 2. Cargar modelo
    print("\n[2/5] Cargando modelo y tokenizador...")
    hf_token = os.environ.get("HF_TOKEN", args.hf_token)
    model, tokenizer = load_model_and_tokenizer(args.model_id, hf_token)

    # 3. Configurar LoRA
    print("\n[3/5] Configurando adaptadores LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Configurar entrenamiento
    print("\n[4/5] Configurando SFTTrainer...")

    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="steps",
        save_steps=0.2,
        save_total_limit=2,
        logging_steps=10,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        seed=args.seed,
        data_seed=args.seed,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )

    collator = get_data_collator(tokenizer)

    # 5. Entrenar
    print("\n[5/5] Iniciando entrenamiento SFT...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        args=sft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        data_collator=collator,
    )

    trainer.train()

    # Guardar adapter + tokenizer en SM_MODEL_DIR
    print(f"\nGuardando adapter LoRA + tokenizer en: {model_dir}")
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Incluir code/inference.py + code/requirements.txt para deploy
    print("\nIncluyendo inference artifacts para deploy con PyTorchModel:")
    include_inference_code(model_dir)

    # Guardar metricas de entrenamiento
    train_metrics = trainer.state.log_history
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Metricas guardadas en: {metrics_path}")

    print("\nFine-Tuning completado exitosamente.")
    print(f"Adapter guardado en: {model_dir}")
    print(f"Deploy: PyTorchModel(model_data=<s3_model_tar_gz>, framework_version='2.1', py_version='py310')")


# ==============================================================================
# MAIN — ENTRY POINT PARA SAGEMAKER
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Llama 3.1 8B")

    # Model
    parser.add_argument("--model-id", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--hf-token", type=str, default="")

    # Data
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))

    # LoRA Hyperparameters
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
