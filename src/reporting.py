"""
reporting.py
============
Крок 5: Формування звіту про аномалії.

Відповідальність:
  - Зчитує SCORED Parquet
  - Фільтрує аномальні сесії
  - Формує підсумкову статистику
  - Зберігає CSV-звіт для SIEM/Firewall (REPORT_PATH)
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import SCORED_PARQUET, REPORT_PATH

logger = logging.getLogger(__name__)

# Колонки, що потрапляють у фінальний звіт
REPORT_COLS = [
    "host",
    "window_start",
    "req_count",
    "unique_urls",
    "unique_ua",
    "error_4xx_rate",
    "error_5xx_rate",
    "avg_bytes",
    "total_bytes",
    "bot_ua_rate",
    "anomaly_score_if",
    "label_if",
    "label_dbscan",
    "anomaly_type",
]


def generate_report(
    scored_path: Path = SCORED_PARQUET,
    report_path: Path = REPORT_PATH,
) -> pl.DataFrame:
    """
    Генерує CSV-звіт аномалій та виводить підсумкову статистику.

    Returns
    -------
    DataFrame аномальних сесій (відсортований від найпідозріліших до менш підозрілих)
    """
    logger.info("Зчитуємо SCORED Parquet: %s", scored_path)
    df = pl.read_parquet(scored_path)

    total      = len(df)
    n_anomaly  = int((df["label_combined"] == -1).sum())
    n_normal   = total - n_anomaly

    sep = "=" * 62

    # ── Загальна статистика ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ")
    print(sep)
    print(f"  Сесійних векторів усього : {total:>8,}")
    print(f"  Нормальних               : {n_normal:>8,}  ({n_normal/total*100:.1f} %)")
    print(f"  Аномальних (combined)    : {n_anomaly:>8,}  ({n_anomaly/total*100:.1f} %)")

    # Score-розподіл
    anom_df   = df.filter(pl.col("label_combined") == -1)
    normal_df = df.filter(pl.col("label_combined") == 1)

    print(f"\n  Anomaly Score IF (аномальні) :")
    print(f"    min={anom_df['anomaly_score_if'].min():.4f}  "
          f"max={anom_df['anomaly_score_if'].max():.4f}  "
          f"mean={anom_df['anomaly_score_if'].mean():.4f}")
    print(f"  Anomaly Score IF (нормальні)  :")
    print(f"    min={normal_df['anomaly_score_if'].min():.4f}  "
          f"max={normal_df['anomaly_score_if'].max():.4f}  "
          f"mean={normal_df['anomaly_score_if'].mean():.4f}")

    # ── Розподіл типів аномалій ───────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Розподіл аномальних сесій за типом загрози")
    print(f"{'─'*50}")
    type_dist = (
        anom_df.group_by("anomaly_type")
               .agg(pl.len().alias("count"))
               .sort("count", descending=True)
    )
    for row in type_dist.to_dicts():
        pct = row["count"] / n_anomaly * 100 if n_anomaly else 0
        print(f"  {row['anomaly_type']:<40}  {row['count']:>4}  ({pct:.1f} %)")

    # ── Топ-15 найпідозріліших IP ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Топ-15 найпідозріліших IP-адрес (за Anomaly Score IF)")
    print(f"{'─'*72}")
    top_ips = (
        anom_df.group_by("host")
               .agg([
                   pl.len().alias("anom_windows"),
                   pl.col("anomaly_score_if").mean().alias("avg_score"),
                   pl.col("req_count").sum().alias("total_reqs"),
                   pl.col("anomaly_type").first().alias("type"),
               ])
               .sort("avg_score")     # від'ємніше = підозріліше
               .head(15)
    )
    print(f"  {'IP-адреса':<22} {'Вікон':>7} {'Score':>8} "
          f"{'Запитів':>8}  Тип")
    print(f"  {'─'*22} {'─'*7} {'─'*8} {'─'*8}  {'─'*30}")
    for row in top_ips.to_dicts():
        print(
            f"  {row['host']:<22} {row['anom_windows']:>7} "
            f"{row['avg_score']:>8.4f} {row['total_reqs']:>8}  {row['type']}"
        )

    # ── Середні ознаки по типах ───────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Середні ознаки за типом аномалії")
    print(f"{'─'*62}")
    feat_summary = (
        anom_df.group_by("anomaly_type")
               .agg([
                   pl.col("req_count").mean().round(1).alias("avg_req"),
                   pl.col("unique_urls").mean().round(1).alias("avg_urls"),
                   pl.col("error_4xx_rate").mean().round(3).alias("avg_4xx"),
                   pl.col("bot_ua_rate").mean().round(3).alias("avg_bot"),
               ])
               .sort("avg_req", descending=True)
    )
    print(f"  {'Тип':<40} {'req':>7} {'urls':>7} {'4xx':>7} {'bot':>7}")
    print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for row in feat_summary.to_dicts():
        print(
            f"  {row['anomaly_type']:<40} "
            f"{row['avg_req']:>7.1f} {row['avg_urls']:>7.1f} "
            f"{row['avg_4xx']:>7.3f} {row['avg_bot']:>7.3f}"
        )

    print(f"\n{sep}\n")

    # ── Збереження CSV-звіту ──────────────────────────────────────────────────
    report = (
        anom_df.select([c for c in REPORT_COLS if c in anom_df.columns])
               .sort("anomaly_score_if")   # найпідозріліші зверху
    )
    report.write_csv(report_path)
    logger.info(
        "CSV-звіт збережено: %s  (%d аномальних сесій)",
        report_path, len(report),
    )
    return report