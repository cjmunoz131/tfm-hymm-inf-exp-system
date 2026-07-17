"""
==============================================================================
HYMM-REC Explainability: Evaluation Job (Merged Model — GPU Processing Job)
==============================================================================
SageMaker Processing Job que ejecuta:
  1. Carga modelo merged completo (FP16) desde /opt/ml/processing/input/model/
  2. Genera predicciones para cada ejemplo del test set
  3. Calcula métricas: ROUGE-L, Exact Match, Keyword Overlap
  4. Guarda reporte de evaluación como JSON

Input channels:
  /opt/ml/processing/input/model/ → modelo merged completo (safetensors + config + tokenizer)
  /opt/ml/processing/input/test/  → test.jsonl

Output:
  /opt/ml/processing/output/metrics/ → evaluation_report.json, evaluation_details.json

Processor: PyTorchProcessor (ml.g5.2xlarge — GPU A10G para inferencia del modelo merged)

NOTA: El modelo se carga directamente con AutoModelForCausalLM (no necesita PEFT ni BitsAndBytes).
      El modelo merged ya tiene los pesos del fine-tuning integrados.
==============================================================================
"""

import subprocess
import sys

# Instalar dependencias necesarias
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.44.2",
    "accelerate==0.33.0",
    "datasets==2.20.0",
])

import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# PROMPT TEMPLATE (debe coincidir con el training)
# ==============================================================================

LLAMA_3_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


# ==============================================================================
# MÉTRICAS DE EVALUACIÓN
# ==============================================================================

def compute_exact_match(predicted_keywords, gold_keywords):
    """1 si las 3 keywords coinciden exactamente (case-insensitive, order-insensitive)."""
    pred_set = set(k.strip().lower() for k in predicted_keywords)
    gold_set = set(k.strip().lower() for k in gold_keywords)
    return 1.0 if pred_set == gold_set else 0.0


def compute_keyword_overlap(predicted_keywords, gold_keywords):
    """Proporción de keywords gold presentes en la predicción."""
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

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ==============================================================================
# CARGA DEL MODELO MERGED (sin PEFT, sin BitsAndBytes)
# ==============================================================================

def load_merged_model(model_dir):
    """
    Carga el modelo merged completo en FP16/BF16.
    No necesita PEFT ni cuantización — es un modelo estándar de HuggingFace.
    """
    print(f"Cargando modelo merged desde: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Modelo cargado en {next(model.parameters()).device}")
    return model, tokenizer


# ==============================================================================
# GENERACIÓN DE PREDICCIONES
# ==============================================================================

def generate_prediction(model, tokenizer, prompt_text, max_new_tokens=20):
    """Genera predicción (3 keywords) dado un prompt formateado."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_token_id is not None and eot_token_id != tokenizer.eos_token_id:
        eos_ids.append(eot_token_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            do_sample=True,
        )

    # Decode only generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Post-procesamiento: limpiar y forzar 3 keywords
    generated_text = generated_text.split("\n")[0].strip()
    generated_text = generated_text.rstrip(".")
    import re
    generated_text = re.sub(r"<\|eot_id\|>", "", generated_text).strip()
    etiquetas = [e.strip() for e in generated_text.split(",")]
    return ", ".join(etiquetas[:3])


# ==============================================================================
# EVALUACIÓN COMPLETA
# ==============================================================================

def evaluate(args):
    """Ejecuta la evaluación completa sobre test set."""

    # Cargar modelo merged
    model, tokenizer = load_merged_model(args.model_dir)

    # Cargar test set
    test_path = None
    for f in os.listdir(args.test_dir):
        if "test" in f and f.endswith(".jsonl"):
            test_path = os.path.join(args.test_dir, f)
            break
    if test_path is None:
        # Fallback: cualquier jsonl
        for f in os.listdir(args.test_dir):
            if f.endswith(".jsonl"):
                test_path = os.path.join(args.test_dir, f)
                break

    if test_path is None:
        raise FileNotFoundError(f"No se encontró archivo test .jsonl en {args.test_dir}")

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
        # Construir prompt
        prompt = LLAMA_3_PROMPT_TEMPLATE.format(
            instruction=muestra["instruction"],
            input=muestra["input"],
        )

        # Generar predicción
        prediccion = generate_prediction(model, tokenizer, prompt)
        gold = muestra["output"]

        # Calcular métricas
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

    # Métricas agregadas
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
    print("RESULTADOS DE EVALUACIÓN")
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

    print("Evaluación completada.")
    return metrics


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Explainability Model (Merged)")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/input/model")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/input/test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()
    evaluate(args)
