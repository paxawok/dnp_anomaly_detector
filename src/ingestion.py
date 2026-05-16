"""
ingestion.py
============
Крок 1: Завантаження CSV-файлів → нормалізація заголовків → збереження raw Parquet.

Відповідальність цього модуля:
  - Знайти всі *.csv у RAW_DIR
  - Зчитати кожен файл із правильним роздільником і кодуванням
  - Перейменувати колонки з оригінальних (з дужками) на snake_case
  - Об'єднати всі файли в єдиний DataFrame
  - Зберегти результат у RAW_PARQUET (Parquet, стиснення zstd)
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import (
    RAW_DIR,
    RAW_PARQUET,
    CSV_SEPARATOR,
    CSV_HAS_HEADER,
    CSV_RAW_COLUMNS,
    CSV_COLUMNS,
)

logger = logging.getLogger(__name__)


_SEPARATORS = ["\t", ",", ";", "|"]
_ENCODINGS  = ["utf-8", "utf-8-sig", "windows-1250", "latin-1", "cp1252"]
 
 
def _sniff_csv(path: Path) -> tuple[str, str] | None:
    """
    Повертає (separator, encoding) першої комбінації, що дає рівно 9 колонок.
    Читає лише перших 5 рядків — швидко і без зайвого RAM.
    """
    for enc in _ENCODINGS:
        try:
            raw_bytes = path.read_bytes()
            text = raw_bytes.decode(enc, errors="replace")
        except Exception:
            continue
 
        lines = [l for l in text.splitlines() if l.strip()][:5]
        if not lines:
            continue
 
        for sep in _SEPARATORS:
            if len(lines[0].split(sep)) == len(CSV_RAW_COLUMNS):
                logger.debug(
                    "  %s → sep=%r  enc=%s  cols=%d",
                    path.name, sep, enc, len(CSV_RAW_COLUMNS),
                )
                return sep, enc
 
    return None
 
 
def _read_single_csv(path: Path) -> pl.DataFrame | None:
    """
    Зчитує один CSV-файл з автодетекцією роздільника та кодування.
    Повертає None якщо файл не підходить за структурою.
    """
    sniff = _sniff_csv(path)
    if sniff is None:
        logger.warning(
            "Файл '%s': не вдалося визначити роздільник/кодування "
            "(жодна комбінація не дала %d колонок) — пропуск",
            path.name, len(CSV_RAW_COLUMNS),
        )
        return None
 
    sep, enc = sniff
 
    try:
        df = pl.read_csv(
            path,
            separator=sep,
            has_header=CSV_HAS_HEADER,
            encoding=enc,
            infer_schema_length=0,       
            ignore_errors=True,
            truncate_ragged_lines=True,
            eol_char="\n",                
        )
    except Exception as exc:
        logger.warning("Не вдалося зчитати '%s': %s", path.name, exc)
        return None
 
    if df.is_empty():
        logger.warning("Файл '%s' порожній — пропуск", path.name)
        return None
 
    # Очищуємо назви колонок від зайвих пробілів і \r
    df = df.rename({col: col.strip() for col in df.columns})
 
    actual = list(df.columns)
    if actual == CSV_RAW_COLUMNS:
        df = df.rename(dict(zip(CSV_RAW_COLUMNS, CSV_COLUMNS)))
    elif len(actual) == len(CSV_COLUMNS):
        df = df.rename(dict(zip(actual, CSV_COLUMNS)))
    else:
        logger.warning(
            "Файл '%s': несподівана кількість колонок %d — пропуск\n"
            "  Знайдені колонки: %s",
            path.name, len(actual), actual,
        )
        return None
 
    df = df.with_columns(pl.lit(path.name).alias("_source_file"))
 
    logger.info(
        "  [OK] %-35s  %8d рядків  sep=%-3r  enc=%s",
        path.name, len(df), sep, enc,
    )
    return df


def load_raw(raw_dir: Path = RAW_DIR, out_path: Path = RAW_PARQUET) -> pl.DataFrame:
    """
    Завантажує всі CSV із raw_dir, об'єднує та зберігає у Parquet.

    Parameters
    ----------
    raw_dir  : каталог з вхідними CSV-файлами
    out_path : шлях збереження RAW Parquet

    Returns
    -------
    Об'єднаний DataFrame (усі колонки — рядкові типи, без очищення)
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Жодного CSV-файлу не знайдено у '{raw_dir}'. "
            "Перевірте значення RAW_DIR у config.py."
        )

    logger.info("Знайдено %d CSV-файлів у '%s'", len(csv_files), raw_dir)

    frames: list[pl.DataFrame] = []
    for path in csv_files:
        frame = _read_single_csv(path)
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise RuntimeError("Жоден файл не вдалося зчитати. Перевірте формат даних.")

    raw = pl.concat(frames, how="diagonal_relaxed")

    logger.info(
        "Завантажено разом: %d рядків із %d файлів",
        len(raw), len(frames),
    )

    # Зберігаємо у Parquet (zstd дає ~3–4× стиснення порівняно з CSV)
    raw.write_parquet(out_path, compression="zstd")
    logger.info("RAW Parquet збережено: %s  (%.1f KB)", out_path,
                out_path.stat().st_size / 1024)

    return raw