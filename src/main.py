"""
main.py
=======
Оркестратор аналітичного конвеєру виявлення аномалій вебтрафіку.

Запуск:
    python src/main.py              # повний pipeline
    python src/main.py --from eda   # починаючи з EDA (CLEAN Parquet вже є)

Кроки:
    1. ingestion        → raw_logs.parquet
    2. preprocessing    → clean_logs.parquet
    3. eda              → вивід статистики (опціонально)
    4. feature_engineering → features.parquet
    5. modelling        → scored.parquet + *.joblib
    6. reporting        → anomaly_report.csv + вивід топ-аномалій
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Додаємо src/ до sys.path щоб імпорти працювали при запуску з кореня
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    RAW_DIR,
    RAW_PARQUET,
    CLEAN_PARQUET,
    FEATURES_PARQUET,
    SCORED_PARQUET,
    REPORT_PATH,
)
from src.ingestion import load_raw
from src.preprocessing import clean
from eda import run_eda
from src.feature_engineering import build_features
from src.modelling import fit_and_score
from src.reporting import generate_report

# ─── Логування ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─── Утиліта для вимірювання часу ─────────────────────────────────────────────
class _Timer:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        logger.info("  ✓ %s  —  %.2f с", self.name, elapsed)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(start_from: str = "ingestion") -> None:
    steps = [
        "ingestion",
        "preprocessing",
        "eda",
        "feature_engineering",
        "modelling",
        "reporting",
    ]

    if start_from not in steps:
        raise ValueError(
            f"Невідомий крок '{start_from}'. Доступні: {steps}"
        )

    active = steps[steps.index(start_from):]
    logger.info("Pipeline: кроки %s", active)

    t_total = time.perf_counter()

    # ── Крок 1: Інгестія ──────────────────────────────────────────────────────
    if "ingestion" in active:
        logger.info("━━ Крок 1/6: INGESTION ━━")
        with _Timer("ingestion"):
            load_raw(raw_dir=RAW_DIR, out_path=RAW_PARQUET)

    # ── Крок 2: Очищення ──────────────────────────────────────────────────────
    if "preprocessing" in active:
        logger.info("━━ Крок 2/6: PREPROCESSING ━━")
        with _Timer("preprocessing"):
            clean(raw_path=RAW_PARQUET, out_path=CLEAN_PARQUET)

    # ── Крок 3: EDA ───────────────────────────────────────────────────────────
    if "eda" in active:
        logger.info("━━ Крок 3/6: EDA ━━")
        with _Timer("eda"):
            run_eda(clean_path=CLEAN_PARQUET)

    # ── Крок 4: Feature Engineering ───────────────────────────────────────────
    if "feature_engineering" in active:
        logger.info("━━ Крок 4/6: FEATURE ENGINEERING ━━")
        with _Timer("feature_engineering"):
            build_features(
                clean_path=CLEAN_PARQUET,
                out_path=FEATURES_PARQUET,
            )

    # ── Крок 5: Моделювання ───────────────────────────────────────────────────
    if "modelling" in active:
        logger.info("━━ Крок 5/6: MODELLING ━━")
        with _Timer("modelling"):
            fit_and_score(
                features_path=FEATURES_PARQUET,
                out_path=SCORED_PARQUET,
            )

    # ── Крок 6: Звіт ──────────────────────────────────────────────────────────
    if "reporting" in active:
        logger.info("━━ Крок 6/6: REPORTING ━━")
        with _Timer("reporting"):
            generate_report(
                scored_path=SCORED_PARQUET,
                report_path=REPORT_PATH,
            )

    elapsed_total = time.perf_counter() - t_total
    logger.info("Pipeline завершено за %.2f с", elapsed_total)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Детектор аномалій вебтрафіку (Polars + Scikit-learn)"
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        default="ingestion",
        choices=[
            "ingestion", "preprocessing", "eda",
            "feature_engineering", "modelling", "reporting",
        ],
        help="З якого кроку починати (якщо проміжні Parquet вже є)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(start_from=args.start_from)