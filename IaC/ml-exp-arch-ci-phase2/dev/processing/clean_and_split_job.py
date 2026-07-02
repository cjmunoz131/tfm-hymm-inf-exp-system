"""
==============================================================================
HYMM-REC Explainability: Clean Gold Set + Split Job
==============================================================================
SageMaker Processing Job que ejecuta:
  1. Carga gold_dataset_clean.jsonl (post human-in-the-loop review)
  2. Valida y limpia respuestas (exactamente 3 keywords por comma)
  3. Split estratificado: train 80% / val 10% / test 10%
  4. Guarda train.jsonl, val.jsonl, test.jsonl en /opt/ml/processing/output/

Input:
  /opt/ml/processing/input/gold/ → gold_dataset_clean.jsonl

Output:
  /opt/ml/processing/output/splits/ → train.jsonl, val.jsonl, test.jsonl
  /opt/ml/processing/output/metrics/ → split_metrics.json

Processor: SKLearnProcessor (ml.m5.large)
==============================================================================
"""

import os
import json
import re
import argparse
import random
import numpy as np
from pathlib import Path


# ==============================================================================
# CONFIGURACIÓN Y SEMILLAS
# ==============================================================================

def seed_everything(seed=42):
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Reproducibilidad] Semilla fijada en: {seed}")


SEED = 42
seed_everything(SEED)


# ==============================================================================
# FUNCIONES DE VALIDACIÓN Y LIMPIEZA
# ==============================================================================

def clean_text_results(texto):
    """
    Elimina introducciones conversacionales del LLM.
    """
    patrones_basura = [
        r"(?i)^here are.*?:",
        r"(?i)^sure.*?:",
        r"(?i)^based on.*?:",
        r"(?i)^the 3 keywords.*?:",
        r"(?i)^the three keywords.*?:",
    ]
    texto_limpio = texto
    for patron in patrones_basura:
        texto_limpio = re.sub(patron, "", texto_limpio).strip()

    # Quitar puntos finales
    if texto_limpio.endswith("."):
        texto_limpio = texto_limpio[:-1]

    return texto_limpio


def es_respuesta_valida(output_text):
    """
    Valida estrictamente que el output sea exactamente 3 keywords separadas por coma.
    Returns: (es_valido: bool, texto_limpio_o_motivo: str)
    """
    if not output_text or output_text == "ERROR_API_TIMEOUT":
        return False, "Error de API o vacío"

    texto_limpio = clean_text_results(output_text)

    # Validar separación por comas
    palabras = [p.strip() for p in texto_limpio.split(",")]

    # Validar exactamente 3 conceptos
    if len(palabras) != 3:
        return False, f"Cantidad incorrecta de keywords ({len(palabras)})"

    # Validar longitud (evitar alucinaciones largas)
    for palabra in palabras:
        if len(palabra.split(" ")) > 4:
            return False, "Concepto demasiado largo (posible alucinación)"

    return True, ", ".join(palabras)


# ==============================================================================
# LIMPIEZA SILVER → GOLD (Validación programática)
# ==============================================================================

def limpiar_y_validar(registros_raw):
    """
    Aplica validación programática sobre los registros del gold set.
    Filtra registros inválidos y limpia los válidos.
    """
    registros_gold = []
    metricas = {"leidos": 0, "aprobados": 0, "rechazados": 0}

    for fila in registros_raw:
        metricas["leidos"] += 1
        es_valido, resultado = es_respuesta_valida(fila.get("output", ""))

        if es_valido:
            fila["output"] = resultado
            registros_gold.append(fila)
            metricas["aprobados"] += 1
        else:
            metricas["rechazados"] += 1

    print(f"Resultados de Validación:")
    print(f"  - Total leídos: {metricas['leidos']}")
    print(f"  - Aprobados (Gold): {metricas['aprobados']}")
    print(f"  - Rechazados: {metricas['rechazados']}")

    if metricas["aprobados"] == 0:
        raise ValueError("Ningún registro pasó la validación. Revisa el gold set.")

    return registros_gold, metricas


# ==============================================================================
# SPLIT TRAIN / VAL / TEST
# ==============================================================================

def split_dataset(registros, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Divide el dataset en train/val/test.
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    """
    random.seed(seed)
    np.random.seed(seed)

    indices = list(range(len(registros)))
    random.shuffle(indices)

    n = len(registros)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    train_data = [registros[i] for i in train_indices]
    val_data = [registros[i] for i in val_indices]
    test_data = [registros[i] for i in test_indices]

    print(f"Split completado:")
    print(f"  - Train: {len(train_data)} ({len(train_data)/n*100:.1f}%)")
    print(f"  - Val:   {len(val_data)} ({len(val_data)/n*100:.1f}%)")
    print(f"  - Test:  {len(test_data)} ({len(test_data)/n*100:.1f}%)")

    return train_data, val_data, test_data


# ==============================================================================
# GUARDAR EN FORMATO JSONL
# ==============================================================================

def guardar_jsonl(registros, filepath):
    """Guarda lista de dicts como JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    print(f"  Guardado: {filepath} ({len(registros)} registros)")


# ==============================================================================
# MAIN — ORQUESTADOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean Gold Set + Split")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Paths SageMaker Processing
    input_dir = "/opt/ml/processing/input/gold"
    output_splits_dir = "/opt/ml/processing/output/splits"
    output_metrics_dir = "/opt/ml/processing/output/metrics"

    os.makedirs(output_splits_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    # 1. Cargar gold dataset
    gold_file = None
    for f in os.listdir(input_dir):
        if f.endswith(".jsonl"):
            gold_file = os.path.join(input_dir, f)
            break

    if gold_file is None:
        raise FileNotFoundError(f"No se encontró archivo .jsonl en {input_dir}")

    print(f"Cargando gold dataset: {gold_file}")
    registros_raw = []
    with open(gold_file, "r", encoding="utf-8") as f:
        for linea in f:
            registros_raw.append(json.loads(linea))

    print(f"Total registros cargados: {len(registros_raw)}")

    # 2. Validar y limpiar
    registros_gold, metricas_limpieza = limpiar_y_validar(registros_raw)

    # 3. Split
    train_data, val_data, test_data = split_dataset(
        registros_gold,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 4. Guardar splits
    print("\nGuardando splits en formato JSONL:")
    guardar_jsonl(train_data, os.path.join(output_splits_dir, "train.jsonl"))
    guardar_jsonl(val_data, os.path.join(output_splits_dir, "val.jsonl"))
    guardar_jsonl(test_data, os.path.join(output_splits_dir, "test.jsonl"))

    # 5. Guardar métricas del proceso
    split_metrics = {
        "input_file": os.path.basename(gold_file),
        "total_raw_records": metricas_limpieza["leidos"],
        "approved_records": metricas_limpieza["aprobados"],
        "rejected_records": metricas_limpieza["rechazados"],
        "approval_rate": round(
            metricas_limpieza["aprobados"] / max(metricas_limpieza["leidos"], 1), 4
        ),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 2),
        "seed": args.seed,
    }

    metrics_path = os.path.join(output_metrics_dir, "split_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(split_metrics, f, indent=2)
    print(f"\nMétricas guardadas en: {metrics_path}")
    print(json.dumps(split_metrics, indent=2))

    print("\n Pipeline de Clean + Split completado exitosamente.")


if __name__ == "__main__":
    main()
