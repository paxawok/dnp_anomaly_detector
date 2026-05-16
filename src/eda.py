"""
eda.py
======
Розвідковий аналіз даних (EDA).
Зчитує CLEAN Parquet і виводить ключову статистику.

Може запускатися як окремий скрипт:
    python src/eda.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import CLEAN_PARQUET

logger = logging.getLogger(__name__)


def run_eda(clean_path: Path = CLEAN_PARQUET) -> None:
    """
    Виводить розвідкову статистику по очищеному датасету:
      - Загальні параметри
      - Топ-10 IP за кількістю запитів
      - Розподіл HTTP-статусів
      - Топ-10 User-Agent
      - Топ-10 найзапитуваніших ресурсів
      - Погодинний розподіл навантаження
    """
    logger.info("EDA: зчитуємо %s", clean_path)
    df = pl.read_parquet(clean_path)

    sep = "=" * 62

    print(f"\n{sep}")
    print("  РОЗВІДКОВИЙ АНАЛІЗ ДАНИХ (EDA)")
    print(sep)

    # ── Загальні параметри ────────────────────────────────────────────────────
    print(f"\n{'Записів усього':<30}: {len(df):>10,}")
    print(f"{'Унікальних IP (host)':<30}: {df['host'].n_unique():>10,}")
    print(f"{'Унікальних User-Agent':<30}: {df['user_agent'].n_unique():>10,}")
    print(f"{'Часовий діапазон':<30}: {df['ts'].min()}  →  {df['ts'].max()}")

    total = len(df)
    n_bot = int(df["is_bot_ua"].sum())
    print(f"{'Запитів з bot UA':<30}: {n_bot:>10,}  ({n_bot/total*100:.1f} %)")

    # ── Статуси ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  Розподіл HTTP-статусів")
    print(f"{'─'*40}")
    status_dist = (
        df.group_by("status")
          .agg(pl.len().alias("count"))
          .with_columns(
              (pl.col("count") / total * 100).round(1).alias("pct")
          )
          .sort("count", descending=True)
    )
    for row in status_dist.to_dicts():
        print(f"  {row['status']:>5}   {row['count']:>7,}   {row['pct']:>5.1f} %")

    # ── Топ-10 IP ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Топ-10 найактивніших IP-адрес")
    print(f"{'─'*50}")
    top_ips = (
        df.group_by("host")
          .agg(pl.len().alias("n"))
          .with_columns(
              (pl.col("n") / total * 100).round(1).alias("pct")
          )
          .sort("n", descending=True)
          .head(10)
    )
    for row in top_ips.to_dicts():
        print(f"  {row['host']:<22} {row['n']:>7,}   {row['pct']:>5.1f} %")

    # ── Топ-10 User-Agent ─────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Топ-10 User-Agent рядків")
    print(f"{'─'*62}")
    ua_dist = (
        df.group_by("user_agent")
          .agg(pl.len().alias("n"))
          .with_columns(
              (pl.col("n") / total * 100).round(1).alias("pct")
          )
          .sort("n", descending=True)
          .head(10)
    )
    for row in ua_dist.to_dicts():
        ua = (row["user_agent"] or "")[:55]
        print(f"  {ua:<55}  {row['n']:>6,}  {row['pct']:>5.1f} %")

    # ── Топ-10 ресурсів ───────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Топ-10 запитуваних ресурсів")
    print(f"{'─'*62}")
    res_dist = (
        df.group_by("resource")
          .agg(pl.len().alias("n"))
          .sort("n", descending=True)
          .head(10)
    )
    for row in res_dist.to_dicts():
        r = (row["resource"] or "")[:52]
        print(f"  {r:<52}  {row['n']:>6,}")

    # ── Погодинний розподіл ───────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  Розподіл запитів по годинах доби")
    print(f"{'─'*40}")
    hourly = (
        df.with_columns(pl.col("ts").dt.hour().alias("hour"))
          .group_by("hour")
          .agg(pl.len().alias("count"))
          .sort("hour")
    )
    median_count = hourly["count"].median()
    for row in hourly.to_dicts():
        bar_len = int(row["count"] / hourly["count"].max() * 30)
        flag = "  ⚠" if row["count"] > median_count * 4 else ""
        print(
            f"  {row['hour']:>2}:00  {'█'*bar_len:<30}  "
            f"{row['count']:>5,}{flag}"
        )

    print(f"\n{sep}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_eda()