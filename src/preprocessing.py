"""
preprocessing.py
================
Крок 2: Парсинг типів → очищення даних → збереження clean Parquet.

Відповідальність:
  - Конвертація datetime-рядка у тип Polars Datetime
  - Приведення status та bytes до числових типів
  - Видалення рядків із некоректними/відсутніми ключовими полями
  - Видалення дублікатів
  - Збереження результату у CLEAN_PARQUET
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import (
    RAW_PARQUET,
    CLEAN_PARQUET,
    DATETIME_FORMAT,
    BOT_UA_PATTERN,
)

logger = logging.getLogger(__name__)


def _parse_and_cast(df: pl.DataFrame) -> pl.DataFrame:
    """
    Типізує колонки:
      - datetime → Polars Datetime (мікросекунди, UTC-naive)
      - status   → Int16
      - bytes    → Int64  ("-" та пусті → 0)
      - host, resource, user_agent → рядки без зайвих пробілів
    """
    return df.with_columns([
        # Часова мітка
        pl.col("datetime")
          .str.strip_chars()
          .str.to_datetime(format=DATETIME_FORMAT, strict=False)
          .alias("ts"),

        # HTTP-статус
        pl.col("status")
          .str.strip_chars()
          .cast(pl.Int16, strict=False),

        # Розмір відповіді: прочерк або порожній → 0
        pl.col("bytes")
          .str.strip_chars()
          .str.replace(r"^-$", "0")
          .cast(pl.Int64, strict=False)
          .fill_null(0),

        # Очищення рядкових полів
        pl.col("host").str.strip_chars(),
        pl.col("resource").str.strip_chars(),
        pl.col("method").str.strip_chars().str.to_uppercase(),
        pl.col("user_agent").str.strip_chars(),

        # Прапорець нелюдського User-Agent (обчислюється один раз тут)
        pl.col("user_agent")
          .str.contains(BOT_UA_PATTERN)
          .cast(pl.Int8)
          .fill_null(0)
          .alias("is_bot_ua"),
    ])


def _filter_invalid(df: pl.DataFrame) -> pl.DataFrame:
    """
    Відкидає рядки, що не придатні для аналізу:
      - відсутня або непарсована часова мітка
      - порожній host
      - некоректний HTTP-статус (поза [100, 599])
    """
    before = len(df)
    df = df.filter(
        pl.col("ts").is_not_null()
        & pl.col("host").is_not_null()
        & pl.col("host").str.len_chars().gt(0)
        & pl.col("status").is_not_null()
        & pl.col("status").is_between(100, 599)
    )
    after = len(df)
    logger.info("Фільтрація некоректних рядків: відкинуто %d  →  залишилось %d",
                before - after, after)
    return df


def clean(
    raw_path: Path = RAW_PARQUET,
    out_path: Path = CLEAN_PARQUET,
) -> pl.DataFrame:
    """
    Виконує повне очищення та типізацію даних.

    Parameters
    ----------
    raw_path : шлях до RAW Parquet (вихід ingestion.py)
    out_path : шлях збереження CLEAN Parquet

    Returns
    -------
    Очищений типізований DataFrame
    """
    logger.info("Зчитуємо RAW Parquet: %s", raw_path)
    df = pl.read_parquet(raw_path)
    logger.info("  Записів до очищення: %d", len(df))

    df = _parse_and_cast(df)
    df = _filter_invalid(df)

    # Видаляємо дублікати за ключовими полями
    before_dedup = len(df)
    df = df.unique(
        subset=["host", "ts", "method", "resource", "status"],
        keep="first",
    )
    logger.info("Дедублікація: видалено %d дублікатів", before_dedup - len(df))

    # Сортуємо за часом — потрібно для group_by_dynamic у feature_engineering
    df = df.sort("ts")

    df.write_parquet(out_path, compression="zstd")
    logger.info(
        "CLEAN Parquet збережено: %s  (%d рядків, %.1f KB)",
        out_path, len(df), out_path.stat().st_size / 1024,
    )
    return df