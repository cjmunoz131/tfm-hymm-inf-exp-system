"""
==============================================================================
HYMM-REC Explainability: Evaluation Job (Adapter LoRA sobre Base BF16)
==============================================================================
SageMaker Processing Job (GPU) que ejecuta:
  1. Carga modelo base Llama 3.1 8B en BF16 (sin cuantización)
  2. Aplica adapter LoRA encima (PeftModel.from_pretrained)
  3. Genera predicciones para cada ejemplo del test set
  4. Calcula métricas: ROUGE-L, Exact Match, Keyword Overlap
  5. Guarda reporte de evaluación como JSON

IMPORTANTE: Este script NO hace merge. Carga base + adapter dinámicamente,
igual que en Google Colab (saveAndEvaluation.py). Esto preserva la calidad
del fine-tuning sin degradación numérica del merge.

Input channels:
  /opt/ml/processing/input/model/ → adapter weights (adapter_config.json +
                                     adapter_model.safetensors + tokenizer)
  /opt/ml/processing/input/test/  → test.jsonl

Output:
  /opt/ml/processing/output/metrics/ → evaluation_report.json, evaluation_details.json

Processor: PyTorchProcessor (ml.g5.2xlarge — GPU A10G 24GB)
==============================================================================
"""

import subprocess
import sys

# Instalar dependencias necesarias (transformers con soporte Llama 3.1 + PEFT)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.44.2",
    "peft==0.12.0",
    "accelerate==0.33.0",
    "datasets==2.20.0",
])

import os
import json
import re
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==============================================================================
# PROMPT TEMPLATE (debe coincidir con el training)
# ==============================================================================

LLAMA_3_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}<|eot_id|>"""


# ==============================================================================
# METRICAS DE EVALUACION
# ==============================================================================

def compute_exact_match(predicted_keywords, gold_keywords):
    """1 si las 3 keywords coinciden exactamente (case-insensitive, order-insensitive)."""
    pred_set = set(k.strip().lower() for k in predicted_keywords)
    gold_set = set(k.strip().lower() for k in gold_keywords)
    return 1.0 if pred_set == gold_set else 0.0


def compute_keyword_overlap(predicted_keywords, gold_keywords):
    """Proporcion de keywords gold presentes en la prediccion."""
    pred_set = set(k.strip().lower() for k in predicted_keywords)
    gold_set = set(k.strip().lower() for k in gold_keywords)
    if len(gold_set) == 0:
        return 0.0
    return len(pred_set.intersection(gold_set)) / len(gold_set)


def compute_rouge_l(prediction, reference):
    """ROUGE-L F1 basado en Longest Common Subsequence."""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        table = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[m][n]

    pred_tokens = prediction.lower().replace(",", "").split()
    ref_tokens = reference.lower().replace(",", "").split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ==============================================================================
# CARGA DEL MODELO: Base BF16 + Adapter LoRA (mismo enfoque que Colab)
# ==============================================================================

def load_model_with_adapter(adapter_dir, base_model_id=None, hf_token=None):
    """
    Carga el modelo base en BF16 nativo + aplica adapter LoRA encima.
    NO usa cuantizacion, NO hace merge. Preserva calidad del fine-tuning.

    Si el adapter_dir contiene un model.tar.gz (Processing Job input),
    lo descomprime primero.

    Mismo enfoque que saveAndEvaluation.py de Colab:
      base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
      model = PeftModel.from_pretrained(base_model, adapter_path)
    """
    import tarfile
    import glob

    # Descomprimir model.tar.gz si existe (Processing Jobs no lo hacen automaticamente)
    tar_files = glob.glob(os.path.join(adapter_dir, "*.tar.gz"))
    if tar_files:
        print(f"Descomprimiendo {tar_files[0]}...")
        with tarfile.open(tar_files[0], "r:gz") as tar:
            tar.extractall(path=adapter_dir)
        print(f"Contenido descomprimido: {os.listdir(adapter_dir)}")

    # Buscar adapter_config.json (puede estar en subdirectorio "model/")
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        # Buscar en subdirectorios
        for root, dirs, files in os.walk(adapter_dir):
            if "adapter_config.json" in files:
                adapter_dir = root
                config_path = os.path.join(root, "adapter_config.json")
                print(f"adapter_config.json encontrado en: {adapter_dir}")
                break

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontro adapter_config.json en {adapter_dir}")

    # Leer base_model_id desde adapter_config.json si no se proporciona
    if base_model_id is None:
        with open(config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get("base_model_name_or_path")
        if base_model_id is None:
            base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    print(f"Cargando modelo base en BF16: {base_model_id}")

    # Tokenizador (reconstruido como en entrenamiento)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Modelo base en BF16 completo (sin cuantizacion)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # Aplicar adapter LoRA
    print(f"Conectando adapter LoRA desde: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    print("Modelo cargado exitosamente (Base BF16 + Adapter LoRA)")
    return model, tokenizer


# ==============================================================================
# GENERACION DE PREDICCIONES
# ==============================================================================

def generate_prediction(model, tokenizer, muestra):
    """
    Genera prediccion (3 keywords) para una muestra del test set.
    Reproduce exactamente el flujo de inferencia de Colab.
    """
    # Construir texto completo y separar prompt de respuesta
    texto_completo = LLAMA_3_PROMPT_TEMPLATE.format(**muestra)
    etiqueta_separadora = "### Response:\n"

    if etiqueta_separadora in texto_completo:
        prompt_entrada = texto_completo.split(etiqueta_separadora)[0] + etiqueta_separadora
    else:
        prompt_entrada = texto_completo

    # Tokenizar
    inputs = tokenizer(prompt_entrada, return_tensors="pt").to(model.device)

    # IDs de tokens de parada
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_token_id is not None and eot_token_id != tokenizer.eos_token_id:
        eos_ids.append(eot_token_id)

    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            do_sample=True,
        )

    # Decodificar
    resultado_crudo = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la parte generada (despues de "### Response:\n")
    if etiqueta_separadora in resultado_crudo:
        prediccion = resultado_crudo.split(etiqueta_separadora)[-1].strip()
    else:
        prediccion = resultado_crudo.strip()

    # Post-procesamiento (mismo que Colab)
    prediccion = prediccion.split("://")[0].strip()
    prediccion = prediccion.split("http")[0].strip()
    prediccion = prediccion.rstrip(".")

    # Forzar exactamente 3 keywords
    etiquetas = [e.strip() for e in prediccion.split(",")]
    prediccion_final = ", ".join(etiquetas[:3])

    # Liberar memoria
    del inputs, outputs
    torch.cuda.empty_cache()

    return prediccion_final


# ==============================================================================
# EVALUACION COMPLETA
# ==============================================================================

def evaluate(args):
    """Ejecuta la evaluacion completa sobre test set."""

    hf_token = os.environ.get("HF_TOKEN", "")

    # Cargar modelo (base BF16 + adapter LoRA)
    model, tokenizer = load_model_with_adapter(
        adapter_dir=args.model_dir,
        hf_token=hf_token if hf_token else None,
    )

    # Cargar test set
    test_path = None
    for f in sorted(os.listdir(args.test_dir)):
        if "test" in f and f.endswith(".jsonl"):
            test_path = os.path.join(args.test_dir, f)
            break
    if test_path is None:
        for f in sorted(os.listdir(args.test_dir)):
            if f.endswith(".jsonl"):
                test_path = os.path.join(args.test_dir, f)
                break

    if test_path is None:
        raise FileNotFoundError(f"No se encontro archivo test .jsonl en {args.test_dir}")

    print(f"Cargando test set: {test_path}")
    test_data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for linea in f:
            test_data.append(json.loads(linea))

    print(f"Total ejemplos de test: {len(test_data)}")

    # Evaluar cada ejemplo
    exact_matches = []
    keyword_overlaps = []
    rouge_l_scores = []
    results = []

    for i, muestra in enumerate(test_data):
        prediccion = generate_prediction(model, tokenizer, muestra)
        gold = muestra["output"]

        # Calcular metricas
        pred_keywords = [k.strip() for k in prediccion.split(",")]
        gold_keywords = [k.strip() for k in gold.split(",")]

        em = compute_exact_match(pred_keywords, gold_keywords)
        ko = compute_keyword_overlap(pred_keywords, gold_keywords)
        rl = compute_rouge_l(prediccion, gold)

        exact_matches.append(em)
        keyword_overlaps.append(ko)
        rouge_l_scores.append(rl)

        results.append({
            "idx": i,
            "gold": gold,
            "predicted": prediccion,
            "exact_match": em,
            "keyword_overlap": ko,
            "rouge_l": rl,
        })

        if (i + 1) % 25 == 0:
            print(f"  Evaluados: {i+1}/{len(test_data)} | "
                  f"EM={np.mean(exact_matches):.3f} | "
                  f"KO={np.mean(keyword_overlaps):.3f} | "
                  f"ROUGE-L={np.mean(rouge_l_scores):.3f}")

    # Metricas agregadas
    metrics = {
        "total_examples": len(test_data),
        "exact_match": round(float(np.mean(exact_matches)), 4),
        "keyword_overlap": round(float(np.mean(keyword_overlaps)), 4),
        "rouge_l_f1": round(float(np.mean(rouge_l_scores)), 4),
        "exact_match_std": round(float(np.std(exact_matches)), 4),
        "keyword_overlap_std": round(float(np.std(keyword_overlaps)), 4),
        "rouge_l_std": round(float(np.std(rouge_l_scores)), 4),
    }

    print(f"\n{'='*60}")
    print("RESULTADOS DE EVALUACION")
    print(f"{'='*60}")
    print(f"  Exact Match:     {metrics['exact_match']:.4f} (+/- {metrics['exact_match_std']:.4f})")
    print(f"  Keyword Overlap: {metrics['keyword_overlap']:.4f} (+/- {metrics['keyword_overlap_std']:.4f})")
    print(f"  ROUGE-L F1:      {metrics['rouge_l_f1']:.4f} (+/- {metrics['rouge_l_std']:.4f})")

    # Guardar resultados
    os.makedirs(args.output_dir, exist_ok=True)

    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReporte: {report_path}")

    detail_path = os.path.join(args.output_dir, "evaluation_details.json")
    with open(detail_path, "w") as f:
        json.dump(results[:50], f, indent=2, ensure_ascii=False)

    print("Evaluacion completada.")
    return metrics


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Explainability Model (Adapter)")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/input/model")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/input/test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()
    evaluate(args)
