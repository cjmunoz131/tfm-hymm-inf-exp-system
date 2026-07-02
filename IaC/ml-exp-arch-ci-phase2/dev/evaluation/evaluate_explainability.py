"""
==============================================================================
HYMM-REC Explainability: Evaluation Job
==============================================================================
SageMaker Processing Job que ejecuta:
  1. Carga test.jsonl + modelo fine-tuned (adapter LoRA)
  2. Genera predicciones para cada ejemplo del test set
  3. Calcula métricas: ROUGE-L, Exact Match, Keyword Overlap
  4. Guarda reporte de evaluación como JSON

Input channels:
  /opt/ml/processing/input/model/ → adapter weights + tokenizer
  /opt/ml/processing/input/test/  → test.jsonl

Output:
  /opt/ml/processing/output/metrics/ → evaluation_report.json

Processor: HuggingFace PyTorchProcessor (ml.g5.xlarge — GPU A10G para inferencia)
==============================================================================
"""

import os
import json
import argparse
import random
import numpy as np
import torch
from pathlib import Path


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
# MÉTRICAS DE EVALUACIÓN
# ==============================================================================

def compute_exact_match(predicted_keywords, gold_keywords):
    """
    Exact Match: 1 si las 3 keywords predichas coinciden exactamente
    con las 3 keywords gold (case-insensitive, order-insensitive).
    """
    pred_set = set(k.strip().lower() for k in predicted_keywords)
    gold_set = set(k.strip().lower() for k in gold_keywords)
    return 1.0 if pred_set == gold_set else 0.0


def compute_keyword_overlap(predicted_keywords, gold_keywords):
    """
    Keyword Overlap: Proporción de keywords gold que aparecen en la predicción.
    Score: |intersection| / |gold| (precision-like metric)
    """
    pred_set = set(k.strip().lower() for k in predicted_keywords)
    gold_set = set(k.strip().lower() for k in gold_keywords)
    if len(gold_set) == 0:
        return 0.0
    overlap = len(pred_set.intersection(gold_set))
    return overlap / len(gold_set)


def compute_rouge_l(prediction, reference):
    """
    ROUGE-L F1 basado en Longest Common Subsequence.
    Compara el texto completo de predicción vs referencia.
    """
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
    precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ==============================================================================
# INFERENCIA CON MODELO FINE-TUNED
# ==============================================================================

def load_model_for_inference(model_dir):
    """Carga el modelo fine-tuned (base + adapter LoRA) para inferencia."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    print(f"Cargando modelo desde: {model_dir}")

    # Leer configuración del adapter para obtener el base model
    peft_config = PeftConfig.from_pretrained(model_dir)
    base_model_id = peft_config.base_model_name_or_path

    print(f"Base model: {base_model_id}")

    # Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Cargar base model en 4-bit (igual que training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    hf_token = os.environ.get("HF_TOKEN", "")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token if hf_token else None,
    )

    # Cargar adapter LoRA
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    print("Modelo cargado correctamente para inferencia.")
    return model, tokenizer


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
        )

    resultado = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la parte después de "### Response:\n"
    response_marker = "### Response:\n"
    if response_marker in resultado:
        prediccion = resultado.split(response_marker)[-1].strip()
    else:
        prediccion = resultado.strip()

    # Post-procesamiento: limpiar y forzar 3 keywords
    prediccion = prediccion.split("://")[0].strip()
    prediccion = prediccion.split("http")[0].strip()
    prediccion = prediccion.rstrip(".")

    etiquetas = [e.strip() for e in prediccion.split(",")]
    prediccion_final = ", ".join(etiquetas[:3])

    return prediccion_final


# ==============================================================================
# EVALUACIÓN COMPLETA
# ==============================================================================

def evaluate(args):
    """Ejecuta la evaluación completa sobre test set."""

    # Cargar modelo
    model, tokenizer = load_model_for_inference(args.model_dir)

    # Cargar test set
    test_path = os.path.join(args.test_dir, "test.jsonl")
    if not os.path.exists(test_path):
        # Buscar cualquier .jsonl
        for f in os.listdir(args.test_dir):
            if f.endswith(".jsonl"):
                test_path = os.path.join(args.test_dir, f)
                break

    print(f"Cargando test set: {test_path}")
    test_data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for linea in f:
            test_data.append(json.loads(linea))

    print(f"Total ejemplos de test: {len(test_data)}")

    # Evaluar cada ejemplo
    results = []
    exact_matches = []
    keyword_overlaps = []
    rouge_l_scores = []

    for i, muestra in enumerate(test_data):
        # Construir prompt (sin la respuesta)
        prompt_text = LLAMA_3_PROMPT_TEMPLATE.format(
            instruction=muestra["instruction"],
            input=muestra["input"],
            output="",  # vacío para que el modelo genere
        ).split("### Response:\n")[0] + "### Response:\n"

        # Generar predicción
        prediccion = generate_prediction(model, tokenizer, prompt_text)
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

        if (i + 1) % 50 == 0:
            print(f"  Evaluados: {i+1}/{len(test_data)} | "
                  f"EM={np.mean(exact_matches):.3f} | "
                  f"KO={np.mean(keyword_overlaps):.3f} | "
                  f"ROUGE-L={np.mean(rouge_l_scores):.3f}")

    # Calcular métricas agregadas
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
    print(f"  Total ejemplos:  {metrics['total_examples']}")

    # Guardar resultados
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Reporte resumido
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReporte guardado: {report_path}")

    # Resultados detallados (primeros 50 ejemplos como muestra)
    detail_path = os.path.join(output_dir, "evaluation_details.json")
    with open(detail_path, "w") as f:
        json.dump(results[:50], f, indent=2, ensure_ascii=False)
    print(f"Detalles guardados: {detail_path}")

    print("\nEvaluación completada exitosamente.")
    return metrics


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Explainability Model")

    parser.add_argument("--model-dir", type=str,
                        default="/opt/ml/processing/input/model")
    parser.add_argument("--test-dir", type=str,
                        default="/opt/ml/processing/input/test")
    parser.add_argument("--output-dir", type=str,
                        default="/opt/ml/processing/output/metrics")

    args = parser.parse_args()
    evaluate(args)
