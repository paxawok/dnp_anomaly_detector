"""
modelling.py
============
Крок 4: Масштабування → Isolation Forest → LOF → класифікація типу аномалії.

Стратегія вибору другого алгоритму за розміром датасету:
  - n ≤ 20 000  →  DBSCAN (точний, але O(n²) RAM)
  - n >  20 000  →  Local Outlier Factor на вибірці (масштабований)

Чому не DBSCAN на великих даних:
  DBSCAN будує матрицю відстаней O(n²). При n=120k це ~55 GB RAM → MemoryError.
  LOF із novelty=True дозволяє навчатись на підвибірці та класифікувати решту.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl # type: ignore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLS,
    FEATURES_PARQUET,
    IF_CONTAMINATION,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    DBSCAN_EPS,
    DBSCAN_MIN_PTS,
    PROCESSED_DIR,
    SCORED_PARQUET,
)

logger = logging.getLogger(__name__)

# Поріг: якщо більше — використовуємо LOF замість DBSCAN
_DBSCAN_MAX_ROWS = 20_000

# Розмір вибірки для навчання LOF (не більше цього числа рядків)
_LOF_SAMPLE_SIZE = 15_000

# Шляхи серіалізованих моделей
_SCALER_PATH = PROCESSED_DIR / "scaler.joblib"
_IF_PATH     = PROCESSED_DIR / "isolation_forest.joblib"
_SEC_PATH    = PROCESSED_DIR / "secondary_model.joblib"   # DBSCAN або LOF


# ─── Евристична класифікація типу аномалії ────────────────────────────────────

def _classify_type(
    label: int,
    req_count: float,
    unique_urls: float,
    error_4xx_rate: float,
    bot_ua_rate: float,
) -> str:
    if label != -1:
        return "normal"
        
    # Коефіцієнт: скільки унікальних сторінок відкрито на 1 запит
    url_ratio = unique_urls / max(req_count, 1)

    if bot_ua_rate > 0.5:
        return "bot / crawler"
        
    if error_4xx_rate > 0.15: # Знизив поріг, сканери часто ховаються
        return "vulnerability scan / dir-brute"
        
    if req_count > 100 and url_ratio < 0.05:
        return "DDoS / flood (single endpoint)"
        
    if unique_urls > 30 or url_ratio > 0.6:
        return "web scraper (content theft)"
        
    if req_count > 50:
        return "high-volume anomaly"
        
    return "low-volume anomaly (needs manual review)"


# ─── Другорядний алгоритм (DBSCAN або LOF) ────────────────────────────────────

def _run_secondary(X: np.ndarray) -> np.ndarray:
    """
    Повертає мітки (-1 = аномалія, 1 = норма) від другорядного алгоритму.
    Автоматично обирає між DBSCAN (малий датасет) та LOF (великий).
    """
    n = len(X)

    if n <= _DBSCAN_MAX_ROWS:
        # ── DBSCAN — точний, підходить для малих наборів ──────────────────────
        logger.info(
            "Вторинна модель: DBSCAN  (n=%d ≤ %d)  eps=%.2f  min_samples=%d",
            n, _DBSCAN_MAX_ROWS, DBSCAN_EPS, DBSCAN_MIN_PTS,
        )
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS, n_jobs=-1)
        raw_labels: np.ndarray = db.fit_predict(X)
        # DBSCAN повертає: -1 = шум, 0,1,2... = кластер
        labels = np.where(raw_labels == -1, -1, 1).astype(np.int8)
        n_noise = int((labels == -1).sum())
        logger.info("  Шумових точок: %d  |  Кластерів: %d",
                    n_noise, len(set(raw_labels.tolist()) - {-1}))
        joblib.dump(db, _SEC_PATH)

    else:
        # ── SGD One-Class SVM — блискавично для великих датасетів ────────────
        logger.info(
            "Вторинна модель: SGDOneClassSVM  (n=%d > %d)",
            n, _DBSCAN_MAX_ROWS
        )
        # Навчаємо лінійний SVM на всьому датасеті (він лінійно залежить від N)
        svm = SGDOneClassSVM(
            nu=IF_CONTAMINATION,   # аналог contamination
            random_state=IF_RANDOM_STATE,
        )
        svm.fit(X)
        labels = svm.predict(X).astype(np.int8)   # -1 = аномалія, 1 = норма

        n_anom = int((labels == -1).sum())
        logger.info(
            "  SVM аномалій: %d  (%.1f %%)", n_anom, n_anom / n * 100
        )
        joblib.dump(svm, _SEC_PATH)

    logger.info("Вторинна модель збережена: %s", _SEC_PATH)
    return labels


# ─── Основна функція ──────────────────────────────────────────────────────────

def fit_and_score(
    features_path: Path = FEATURES_PARQUET,
    out_path: Path      = SCORED_PARQUET,
) -> pl.DataFrame:
    """
    Навчає моделі та виставляє мітки аномальності кожній сесії.

    Returns
    -------
    DataFrame = features + anomaly_score_if + label_if +
                label_secondary + label_combined + anomaly_type
    """
    logger.info("Зчитуємо FEATURES Parquet: %s", features_path)
    df = pl.read_parquet(features_path)
    n  = len(df)
    logger.info("  Сесійних векторів: %d", n)

    # ── 1. Матриця ознак → NumPy ──────────────────────────────────────────────
    X_raw: np.ndarray = df.select(FEATURE_COLS).to_numpy().astype(np.float64)

    # ── 2. StandardScaler ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X: np.ndarray = scaler.fit_transform(X_raw)
    joblib.dump(scaler, _SCALER_PATH)
    logger.info("StandardScaler збережено: %s", _SCALER_PATH)

    # ── 3. Isolation Forest ───────────────────────────────────────────────────
    logger.info(
        "Isolation Forest: n_estimators=%d  contamination=%.2f",
        IF_N_ESTIMATORS, IF_CONTAMINATION,
    )
    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X)
    if_scores: np.ndarray = iso.score_samples(X)    # нижче → аномальніше
    if_labels: np.ndarray = iso.predict(X)           # -1 = аномалія

    n_if = int((if_labels == -1).sum())
    logger.info("  Виявлено аномалій: %d  (%.1f %%)", n_if, n_if / n * 100)
    joblib.dump(iso, _IF_PATH)
    logger.info("IsolationForest збережено: %s", _IF_PATH)

    # ── 4. Вторинна модель (DBSCAN або LOF) ───────────────────────────────────
    sec_labels: np.ndarray = _run_secondary(X)

    # ── 5. Об'єднана мітка ────────────────────────────────────────────────────
    combined: np.ndarray = np.where(
        (if_labels == -1) | (sec_labels == -1), -1, 1
    ).astype(np.int8)

    n_combined = int((combined == -1).sum())
    logger.info(
        "  Об'єднана аномалія (IF ∪ secondary): %d  (%.1f %%)",
        n_combined, n_combined / n * 100,
    )

    # ── 6. Приєднуємо мітки до DataFrame ─────────────────────────────────────
    # IF видає скор де від'ємне = аномалія. Зробимо Risk Score (0-100)
    # Наприклад, -0.8 -> 80, -0.4 -> 40. Чим більше, тим гірше.
    risk_score = np.clip(-if_scores * 100, 0, 100)

    scored = df.with_columns([
        pl.Series("anomaly_score_if",  if_scores.astype(np.float32)),
        pl.Series("risk_score_100",    risk_score.astype(np.float32)), # НОВА КОЛОНКА
        pl.Series("label_if",          if_labels.astype(np.int8)),
        pl.Series("label_secondary",   sec_labels),
        pl.Series("label_combined",    combined),
    ])

    # ── 7. Евристична класифікація типу загрози ───────────────────────────────
    anomaly_types = [
        _classify_type(
            row["label_combined"],
            row["req_count"],
            row["unique_urls"],
            row["error_4xx_rate"],
            row["bot_ua_rate"],
        )
        for row in scored.select([
            "label_combined", "req_count", "unique_urls",
            "error_4xx_rate", "bot_ua_rate",
        ]).to_dicts()
    ]
    scored = scored.with_columns(
        pl.Series("anomaly_type", anomaly_types)
    )

    # ── 8. Збереження ─────────────────────────────────────────────────────────
    scored.write_parquet(out_path, compression="zstd")
    logger.info(
        "SCORED Parquet збережено: %s  (%.1f KB)",
        out_path, out_path.stat().st_size / 1024,
    )
    return scored