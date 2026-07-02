"""
==============================================================================
HYMM-REC Explainability: Fine-Tuning Llama 3.1 8B con QLoRA
==============================================================================
SageMaker Training Job que ejecuta:
  1. Carga train.jsonl + val.jsonl desde S3
  2. Aplica template de prompt LLaMA 3 (Instruction + Input + Response)
  3. Configura QLoRA 4-bit (BitsAndBytes + LoRA adapters)
  4. Entrena con SFTTrainer (masked loss sobre respuesta únicamente)
  5. Guarda adapter weights + tokenizer en /opt/ml/model/

Input channels:
  /opt/ml/input/data/train/ → train.jsonl, val.jsonl
  
Output:
  /opt/ml/model/ → adapter weights (LoRA) + tokenizer + config

Estimator: HuggingFace (ml.g5.2xlarge — GPU A10G 24GB para QLoRA 4-bit)
Framework: transformers + peft + trl (SFTTrainer)
==============================================================================
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
from pathlib import Path


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
    from datasets import load_dataset

    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    # Verificar existencia
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No se encontró train.jsonl en {data_dir}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"No se encontró val.jsonl en {data_dir}")

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
    """
    Carga modelo base con cuantización 4-bit y tokenizador.
    QLoRA reduce el modelo de ~16GB a ~6GB VRAM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    print(f"Cargando modelo: {model_id}")

    # Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configuración BitsAndBytes (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Cargar modelo cuantizado
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
    Collator personalizado que enmascara el prompt (Loss = -100)
    para que el modelo solo aprenda a predecir las 3 keywords.
    """
    from transformers import DataCollatorForLanguageModeling

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
# TRAINING — ORCHESTRATOR
# ==============================================================================

def train(args):
    """Orquesta el fine-tuning completo con QLoRA."""
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

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
        processing_class=tokenizer,
        args=sft_config,
        data_collator=collator,
    )

    trainer.train()

    # Guardar adapter final
    print(f"\nGuardando adapter LoRA en: {model_dir}")
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Guardar métricas de entrenamiento
    train_metrics = trainer.state.log_history
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)

    print(f"Métricas guardadas en: {metrics_path}")
    print("\nFine-Tuning completado exitosamente.")


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
