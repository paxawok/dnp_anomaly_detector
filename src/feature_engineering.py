"""
feature_engineering.py
=======================
Крок 3: Sessionization → обчислення поведінкових ознак → збереження Parquet.

Відповідальність:
  - Агрегація записів у часові вікна (60 с за замовчуванням) для кожного host
  - Обчислення 9 поведінкових ознак сесії
  - Збереження матриці ознак у FEATURES_PARQUET
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import (
    CLEAN_PARQUET,
    FEATURES_PARQUET,
    SESSION_WINDOW_SEC,
    FEATURE_COLS,
)

logger = logging.getLogger(__name__)


def build_features(
    clean_path: Path = CLEAN_PARQUET,
    out_path: Path   = FEATURES_PARQUET,
    window_sec: int  = SESSION_WINDOW_SEC,
) -> pl.DataFrame:
    """
    Будує матрицю поведінкових ознак методом sessionization.

    Алгоритм:
      1. Зчитує CLEAN Parquet (вже відсортований за ts)
      2. group_by_dynamic("ts", every="{window_sec}s", group_by="host")
         → агрегує кожне 60-секундне вікно для кожного IP
      3. Обчислює 9 ознак (див. FEATURE_COLS у config.py)
      4. Зберігає результат у FEATURES_PARQUET

    Returns
    -------
    DataFrame із колонками: host, window_start + FEATURE_COLS
    """
    logger.info("Зчитуємо CLEAN Parquet: %s", clean_path)
    df = pl.read_parquet(clean_path)
    logger.info("  Записів: %d  |  Унікальних host: %d", len(df), df["host"].n_unique())

    every = f"{window_sec}s"
    logger.info("Sessionization: вікно = %s", every)

    agg = (
        df.sort("ts")                          # group_by_dynamic вимагає сортування
          .group_by_dynamic(
              "ts",
              every=every,
              group_by="host",
              closed="left",
              label="left",                    # мітка вікна = початок інтервалу
          )
          .agg([
              # ── Інтенсивність ─────────────────────────────────────────────
              pl.len().alias("req_count"),

              # ── Різноманітність запитів ───────────────────────────────────
              pl.col("resource").n_unique().alias("unique_urls"),
              pl.col("user_agent").n_unique().alias("unique_ua"),

              # ── Профіль помилок ───────────────────────────────────────────
              (
                  pl.col("status").is_between(400, 499).sum() / pl.len()
              ).alias("error_4xx_rate"),
              (
                  pl.col("status").is_between(500, 599).sum() / pl.len()
              ).alias("error_5xx_rate"),

              # ── Обсяг даних ───────────────────────────────────────────────
              pl.col("bytes").mean().fill_null(0).alias("avg_bytes"),
              pl.col("bytes").sum().alias("total_bytes"),
              pl.col("bytes").std().fill_null(0).alias("bytes_std"),

              # ── Нелюдський User-Agent ─────────────────────────────────────
              (
                  pl.col("is_bot_ua").sum() / pl.len()
              ).alias("bot_ua_rate"),
          ])
          .rename({"ts": "window_start"})
          .sort(["host", "window_start"])
    )

    # Замінюємо null у числових колонках на 0
    agg = agg.with_columns(
        [pl.col(c).fill_null(0) for c in FEATURE_COLS]
    )

    logger.info(
        "Сформовано %d сесійних векторів  "
        "(з %d записів, коефіцієнт стиснення %.2f:1)",
        len(agg), len(df), len(df) / max(len(agg), 1),
    )

    agg.write_parquet(out_path, compression="zstd")
    logger.info(
        "FEATURES Parquet збережено: %s  (%.1f KB)",
        out_path, out_path.stat().st_size / 1024,
    )
    return agg