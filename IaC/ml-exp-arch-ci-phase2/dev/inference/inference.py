"""
==============================================================================
HYMM-REC Explainability: Custom Inference Script (PyTorchModel Endpoint)
==============================================================================
Script de inferencia para SageMaker PyTorchModel endpoint.
Carga Llama 3.1 8B en BF16 + aplica adapter LoRA (sin merge, sin cuantizacion).

Reproduce exactamente el flujo de Colab (saveAndEvaluation.py):
  base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=bfloat16)
  model = PeftModel.from_pretrained(base_model, adapter_path)

El model.tar.gz debe contener:
  - adapter_config.json
  - adapter_model.safetensors (o .bin)
  - tokenizer.json, tokenizer_config.json, special_tokens_map.json
  - code/inference.py (este archivo)
  - code/requirements.txt

Deploy con PyTorchModel:
  from sagemaker.pytorch import PyTorchModel
  model = PyTorchModel(
      model_data="s3://.../model.tar.gz",
      role=ROLE,
      framework_version="2.1",
      py_version="py310",
      env={"HF_TOKEN": "...", "BASE_MODEL_ID": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
  )
  predictor = model.deploy(instance_type="ml.g5.2xlarge", ...)
==============================================================================
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def model_fn(model_dir):
    """
    Carga modelo base en BF16 + aplica adapter LoRA.
    NO usa cuantizacion, NO hace merge. Calidad completa del fine-tuning.
    """
    print(f"[inference.py] Cargando modelo desde: {model_dir}")

    # Leer base_model_id desde adapter_config.json
    config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get("base_model_name_or_path")
    else:
        base_model_id = os.environ.get("BASE_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")

    print(f"[inference.py] Base model: {base_model_id}")

    hf_token = os.environ.get("HF_TOKEN", "")

    # Tokenizador (configurado igual que en entrenamiento)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token if hf_token else None)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Modelo base en BF16 completo (sin cuantizacion)
    print("[inference.py] Cargando modelo base en BF16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token if hf_token else None,
    )

    # Aplicar adapter LoRA
    print("[inference.py] Aplicando adapter LoRA...")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    print("[inference.py] Modelo listo (Base BF16 + Adapter LoRA)")
    return {"model": model, "tokenizer": tokenizer}


def input_fn(request_body, request_content_type):
    """Parsea el input JSON."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """
    Genera 3 keywords de explicacion dado un prompt.

    Input esperado:
      {"inputs": "<prompt completo>", "parameters": {"max_new_tokens": 20, ...}}

    Output:
      [{"generated_text": "keyword1, keyword2, keyword3"}]
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    prompt = input_data.get("inputs", "")
    parameters = input_data.get("parameters", {})

    max_new_tokens = parameters.get("max_new_tokens", 20)
    temperature = parameters.get("temperature", 0.1)
    repetition_penalty = parameters.get("repetition_penalty", 1.1)

    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Tokens de parada
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_token_id is not None and eot_token_id != tokenizer.eos_token_id:
        eos_ids.append(eot_token_id)

    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            do_sample=True if temperature > 0 else False,
        )

    # Decodificar resultado completo
    resultado_crudo = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la parte generada (despues de "### Response:\n")
    etiqueta_separadora = "### Response:\n"
    if etiqueta_separadora in resultado_crudo:
        prediccion = resultado_crudo.split(etiqueta_separadora)[-1].strip()
    else:
        # Si no encuentra el separador, extraer solo tokens nuevos
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediccion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Post-procesamiento (mismo que Colab — limpieza robusta)
    prediccion = prediccion.split("://")[0].strip()
    prediccion = prediccion.split("http")[0].strip()
    prediccion = prediccion.split(".")[0].strip()
    prediccion = prediccion.split("\n")[0].strip()
    prediccion = prediccion.rstrip(".")

    # Forzar exactamente 3 keywords
    etiquetas = [e.strip() for e in prediccion.split(",")]
    prediccion_final = ", ".join(etiquetas[:3])

    # Liberar memoria
    del inputs, outputs
    torch.cuda.empty_cache()

    return [{"generated_text": prediccion_final}]


def output_fn(prediction, response_content_type):
    """Serializa el output."""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported response type: {response_content_type}")
