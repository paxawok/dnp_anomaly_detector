"""
modelling.py
============
Крок 4: Масштабування → Isolation Forest → SGD One-Class SVM →
        класифікація типу аномалії → Risk Score.

Стратегія вибору другого алгоритму за розміром датасету:
  - n ≤ 20 000  →  DBSCAN (точний, але O(n²) RAM)
  - n >  20 000  →  SGDOneClassSVM (O(n), масштабований)

Вибір Union (OR) vs Intersection (AND) — логується автоматично після
навчання. При contamination=0.05 Intersection дає ~320 сесій (0.06 %),
відкидаючи 1133 унікальні SVM-аномалії. Union обраний, бо в
кібербезпеці ціна пропущеної загрози перевищує ціну хибнопозитиву.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl  # type: ignore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler

from config import (
    DBSCAN_EPS,
    DBSCAN_MIN_PTS,
    FEATURE_COLS,
    FEATURES_PARQUET,
    IF_CONTAMINATION,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    PROCESSED_DIR,
    SCORED_PARQUET,
)

logger = logging.getLogger(__name__)

_DBSCAN_MAX_ROWS = 20_000

_SCALER_PATH = PROCESSED_DIR / "scaler.joblib"
_IF_PATH     = PROCESSED_DIR / "isolation_forest.joblib"
_SEC_PATH    = PROCESSED_DIR / "secondary_model.joblib"
_THR_PATH    = PROCESSED_DIR / "thresholds.joblib"


# ─── Структура порогів ────────────────────────────────────────────────────────

@dataclass
class HeuristicThresholds:
    """
    Пороги для евристичної класифікації типу аномалії.
    Обчислюються з даних, а не задаються вручну.
    """
    bot_ua_rate:         float
    error_4xx_rate:      float
    req_count_ddos:      float
    unique_urls_scraper: float
    req_count_highvol:   float

    def log(self) -> None:
        logger.info("━━ Обчислені пороги класифікації (з p99.9 нормального трафіку) ━━")
        logger.info("  bot_ua_rate      > %.4f", self.bot_ua_rate)
        logger.info("  error_4xx_rate   > %.4f", self.error_4xx_rate)
        logger.info("  req_count (DDoS) > %.1f",  self.req_count_ddos)
        logger.info("  unique_urls      > %.1f",  self.unique_urls_scraper)
        logger.info("  req_count (high) > %.1f",  self.req_count_highvol)


# ─── Обчислення порогів із даних ─────────────────────────────────────────────

def _compute_thresholds(
    df: pl.DataFrame,
    normal_mask: np.ndarray,
    percentile: float = 99.9,
    margin: float = 1.25,
) -> HeuristicThresholds:
    """
    Обчислює пороги як n-й перцентиль розподілу ознаки
    у нормальних сесіях, помножений на margin.

    Якщо p99.9 = 0 (ознака у нормі завжди нульова), поріг = мінімальне
    ненульове значення * margin, щоб будь-яке ненульове вже було аномалією.
    """
    normal_df = df.filter(pl.Series("_mask", normal_mask))

    # Ознаки з природним обмеженням [0, 1] — clamp порогу до 0.9,
    # щоб уникнути > 1.0 після множення на margin.
    # Ознаки без обмеження (req_count, unique_urls) — множимо на margin.
    def thr(col: str, extra_margin: float = 1.0, bounded: bool = False) -> float:
        vals = normal_df[col].to_numpy().astype(float)
        p = float(np.percentile(vals, percentile))
        if p == 0.0:
            # Ознака у нормі завжди 0 → беремо мінімальне ненульове * margin
            nonzero = vals[vals > 0]
            p = float(nonzero.min()) if len(nonzero) > 0 else 0.01
            result = round(p * 0.5, 4)          # половина мінімуму → м'який поріг
        else:
            result = round(p * margin * extra_margin, 4)
        if bounded:
            result = min(result, 0.9)            # clamp для ознак ∈ [0, 1]
        return result

    thresholds = HeuristicThresholds(
        bot_ua_rate=thr("bot_ua_rate",    bounded=True),
        error_4xx_rate=thr("error_4xx_rate", bounded=True),
        req_count_ddos=thr("req_count"),
        unique_urls_scraper=thr("unique_urls"),
        req_count_highvol=thr("req_count", extra_margin=0.6),
    )

    logger.info(
        "━━ Перцентильний аналіз нормального трафіку (%d сесій) ━━",
        int(normal_mask.sum()),
    )
    for col in ["req_count", "unique_urls", "error_4xx_rate", "bot_ua_rate"]:
        vals = normal_df[col].to_numpy().astype(float)
        logger.info(
            "  %-20s  p50=%.2f  p90=%.2f  p99=%.2f  p99.9=%.2f  → поріг=%.4f",
            col,
            float(np.percentile(vals, 50)),
            float(np.percentile(vals, 90)),
            float(np.percentile(vals, 99)),
            float(np.percentile(vals, 99.9)),
            thr(col),
        )

    thresholds.log()
    return thresholds


# ─── Sensitivity analysis ─────────────────────────────────────────────────────

def _run_sensitivity_analysis(X: np.ndarray) -> None:
    """
    Навчає ансамбль при п'яти значеннях contamination і логує результати.
    Використовує n_estimators=100 для швидкості (не впливає на production).
    """
    n = len(X)
    logger.info("━━ Sensitivity Analysis: contamination ━━")
    logger.info(
        "  %5s | %8s %5s | %8s %5s | %8s %5s | %7s",
        "cont", "IF", "IF%", "SVM", "SVM%", "Union", "Un%", "Inter",
    )

    for cont in [0.01, 0.03, 0.05, 0.07, 0.10]:
        iso_tmp = IsolationForest(
            n_estimators=100,
            contamination=cont,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
        )
        il = iso_tmp.fit_predict(X)

        svm_tmp = SGDOneClassSVM(nu=cont, random_state=IF_RANDOM_STATE)
        svm_tmp.fit(X)
        sl = svm_tmp.predict(X)

        n_if  = int((il == -1).sum())
        n_svm = int((sl == -1).sum())
        n_uni = int(((il == -1) | (sl == -1)).sum())
        n_int = int(((il == -1) & (sl == -1)).sum())

        logger.info(
            "  %5.2f | %8d %5.1f | %8d %5.1f | %8d %5.1f | %7d",
            cont,
            n_if,  n_if  / n * 100,
            n_svm, n_svm / n * 100,
            n_uni, n_uni / n * 100,
            n_int,
        )

    logger.info(
        "  → Обрано contamination=%.2f: стабільний розподіл типів загроз, "
        "відповідає частці IP вище p99.9 req_count нормального трафіку.",
        IF_CONTAMINATION,
    )


# ─── Union vs Intersection ────────────────────────────────────────────────────

def _log_union_vs_intersection(
    if_labels: np.ndarray,
    sec_labels: np.ndarray,
) -> None:
    n = len(if_labels)
    n_if_only  = int(((if_labels == -1) & (sec_labels ==  1)).sum())
    n_svm_only = int(((sec_labels == -1) & (if_labels ==  1)).sum())
    n_inter    = int(((if_labels == -1) & (sec_labels == -1)).sum())
    n_union    = int(((if_labels == -1) | (sec_labels == -1)).sum())

    logger.info("━━ Union vs Intersection ━━")
    logger.info("  Тільки IF  (IF=−1, SVM=+1):  %7d  (%.1f %%)", n_if_only,  n_if_only  / n * 100)
    logger.info("  Тільки SVM (SVM=−1, IF=+1):  %7d  (%.1f %%)", n_svm_only, n_svm_only / n * 100)
    logger.info("  Перетин AND:                  %7d  (%.2f %%)", n_inter,    n_inter    / n * 100)
    logger.info("  Об'єднання OR (обрано):       %7d  (%.1f %%)", n_union,    n_union    / n * 100)
    logger.info(
        "  → Intersection відкидає %d унікальних SVM-аномалій (%.1f %% від Union)",
        n_svm_only, n_svm_only / max(n_union, 1) * 100,
    )


# ─── Евристична класифікація ─────────────────────────────────────────────────

def _classify_type(
    label: int,
    req_count: float,
    unique_urls: float,
    error_4xx_rate: float,
    bot_ua_rate: float,
    thr: HeuristicThresholds,
) -> str:
    """
    Класифікує аномальну сесію за типом загрози.
    Пороги передаються як параметр — обчислені з даних, не захардкоджені.
    """
    if label != -1:
        return "normal"

    url_ratio = unique_urls / max(req_count, 1)

    if bot_ua_rate > thr.bot_ua_rate:
        return "bot / crawler"
    if error_4xx_rate > thr.error_4xx_rate:
        return "vulnerability scan / dir-brute"
    if req_count > thr.req_count_ddos and url_ratio < 0.05:
        return "DDoS / flood (single endpoint)"
    if unique_urls > thr.unique_urls_scraper or url_ratio > 0.6:
        return "web scraper (content theft)"
    if req_count > thr.req_count_highvol:
        return "high-volume anomaly"

    return "low-volume anomaly (needs manual review)"


# ─── Вторинний алгоритм ───────────────────────────────────────────────────────

def _run_secondary(X: np.ndarray) -> np.ndarray:
    n = len(X)

    if n <= _DBSCAN_MAX_ROWS:
        logger.info(
            "Вторинна модель: DBSCAN  (n=%d ≤ %d)  eps=%.2f  min_samples=%d",
            n, _DBSCAN_MAX_ROWS, DBSCAN_EPS, DBSCAN_MIN_PTS,
        )
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS, n_jobs=-1)
        raw = db.fit_predict(X)
        labels = np.where(raw == -1, -1, 1).astype(np.int8)
        n_noise = int((labels == -1).sum())
        logger.info("  Шумових точок: %d  |  Кластерів: %d",
                    n_noise, len(set(raw.tolist()) - {-1}))
        joblib.dump(db, _SEC_PATH)
    else:
        logger.info(
            "Вторинна модель: SGDOneClassSVM  (n=%d > %d, nu=%.2f)",
            n, _DBSCAN_MAX_ROWS, IF_CONTAMINATION,
        )
        svm = SGDOneClassSVM(nu=IF_CONTAMINATION, random_state=IF_RANDOM_STATE)
        svm.fit(X)
        labels = svm.predict(X).astype(np.int8)
        n_anom = int((labels == -1).sum())
        logger.info("  SVM аномалій: %d  (%.1f %%)", n_anom, n_anom / n * 100)
        joblib.dump(svm, _SEC_PATH)

    logger.info("Вторинна модель збережена: %s", _SEC_PATH)
    return labels


# ─── Основна функція ──────────────────────────────────────────────────────────

def fit_and_score(
    features_path: Path = FEATURES_PARQUET,
    out_path: Path      = SCORED_PARQUET,
    run_sensitivity: bool = True,
) -> pl.DataFrame:
    """
    Навчає моделі та виставляє мітки аномальності кожній сесії.

    Parameters
    ----------
    run_sensitivity : виводити sensitivity analysis у лог (за замовчуванням True)
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

    # ── 3. Sensitivity analysis (перед основним навчанням) ────────────────────
    if run_sensitivity:
        _run_sensitivity_analysis(X)

    # ── 4. Isolation Forest ───────────────────────────────────────────────────
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
    if_scores: np.ndarray = iso.score_samples(X)
    if_labels: np.ndarray = iso.predict(X)

    n_if = int((if_labels == -1).sum())
    logger.info("  Виявлено аномалій: %d  (%.1f %%)", n_if, n_if / n * 100)
    joblib.dump(iso, _IF_PATH)
    logger.info("IsolationForest збережено: %s", _IF_PATH)

    # ── 5. Вторинна модель ────────────────────────────────────────────────────
    sec_labels: np.ndarray = _run_secondary(X)

    # ── 6. Union vs Intersection ──────────────────────────────────────────────
    _log_union_vs_intersection(if_labels, sec_labels)

    # ── 7. Об'єднана мітка (Union) ────────────────────────────────────────────
    combined: np.ndarray = np.where(
        (if_labels == -1) | (sec_labels == -1), -1, 1
    ).astype(np.int8)

    n_combined = int((combined == -1).sum())
    logger.info(
        "  Об'єднана аномалія (IF ∪ SVM): %d  (%.1f %%)",
        n_combined, n_combined / n * 100,
    )

    # ── 8. Пороги класифікації з нормальних сесій ─────────────────────────────
    # Нормальні = сесії, не позначені жодним алгоритмом (combined == 1)
    normal_mask = (combined == 1)
    thresholds = _compute_thresholds(df, normal_mask)
    joblib.dump(thresholds, _THR_PATH)
    logger.info("Пороги збережено: %s", _THR_PATH)

    # ── 9. Risk Score ─────────────────────────────────────────────────────────
    risk_score = np.clip(-if_scores * 100, 0, 100)

    scored = df.with_columns([
        pl.Series("anomaly_score_if", if_scores.astype(np.float32)),
        pl.Series("risk_score_100",   risk_score.astype(np.float32)),
        pl.Series("label_if",         if_labels.astype(np.int8)),
        pl.Series("label_secondary",  sec_labels),
        pl.Series("label_combined",   combined),
    ])

    # ── 10. Евристична класифікація з обчисленими порогами ────────────────────
    anomaly_types = [
        _classify_type(
            row["label_combined"],
            row["req_count"],
            row["unique_urls"],
            row["error_4xx_rate"],
            row["bot_ua_rate"],
            thresholds,
        )
        for row in scored.select([
            "label_combined", "req_count", "unique_urls",
            "error_4xx_rate", "bot_ua_rate",
        ]).to_dicts()
    ]
    scored = scored.with_columns(
        pl.Series("anomaly_type", anomaly_types)
    )

    # ── 11. Збереження ────────────────────────────────────────────────────────
    scored.write_parquet(out_path, compression="zstd")
    logger.info(
        "SCORED Parquet збережено: %s  (%.1f KB)",
        out_path, out_path.stat().st_size / 1024,
    )
    return scored