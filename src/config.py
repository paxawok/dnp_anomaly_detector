"""
config.py
=========
Централізована конфігурація проєкту.
Усі параметри — в одному місці; імпортується усіма модулями.
"""

from pathlib import Path

# ─── Шляхи ────────────────────────────────────────────────────────────────────

ROOT_DIR       = Path(__file__).resolve().parent.parent   # корінь проєкту
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "polish_data"                 # вхідні CSV
PROCESSED_DIR  = DATA_DIR / "processed"                   # Parquet + звіти
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Parquet-файли
RAW_PARQUET    = PROCESSED_DIR / "raw_logs.parquet"       # об'єднані сирі логи
CLEAN_PARQUET  = PROCESSED_DIR / "clean_logs.parquet"     # очищені логи
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"     # матриця ознак
SCORED_PARQUET   = PROCESSED_DIR / "scored.parquet"       # + мітки моделей

# CSV-звіт для SIEM/Firewall
REPORT_PATH    = PROCESSED_DIR / "anomaly_report.csv"

# ─── Формат вхідних CSV ───────────────────────────────────────────────────────

CSV_HAS_HEADER = True

# Оригінальні назви заголовків у файлі (з дужками та опечаткою оригіналу)
CSV_RAW_COLUMNS = [
    "[Host]",
    "[Date time]",
    "[Request method]",
    "[Reguested resourse]",   
    "[Protocol version]",
    "[Response code]",
    "[Bytes]",
    "[Referrer]",
    "[User Agent]",
]

# Нормалізовані назви (без дужок, snake_case) — використовуються в коді
CSV_COLUMNS = [
    "host",
    "datetime",
    "method",
    "resource",
    "protocol",
    "status",
    "bytes",
    "referrer",
    "user_agent",
]

# Формат дати у логах: 05/Feb/2018 06:25:05
DATETIME_FORMAT = "%d/%b/%Y %H:%M:%S"
# DATETIME_FORMAT_IRAN = "%d/%b/%Y:%H:%M:%S %z"

# ─── Feature Engineering ──────────────────────────────────────────────────────

SESSION_WINDOW_SEC = 60   # розмір часового вікна сесії (секунди)

# Regex-шаблон нелюдських User-Agent рядків
BOT_UA_PATTERN = (
    r"(?i)(bot|crawler|spider|scraper|python|curl|wget|java/"
    r"|go-http|httpclient|libwww|ahrefsbot|semrushbot|mj12bot|"
    r"baiduspider|yandexbot|googlebot|bingbot|scrapy|"
    r"facebookexternalhit|archive\.org_bot)"
)

# ─── Isolation Forest ─────────────────────────────────────────────────────────

IF_N_ESTIMATORS  = 200
IF_CONTAMINATION = 0.05   # очікувана частка аномалій
IF_RANDOM_STATE  = 42

# ─── DBSCAN ───────────────────────────────────────────────────────────────────

DBSCAN_EPS     = 0.8
DBSCAN_MIN_PTS = 5

# ─── Вектор ознак ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "req_count",
    "unique_urls",
    "unique_ua",
    "error_4xx_rate",
    "error_5xx_rate",
    "avg_bytes",
    "total_bytes",
    "bytes_std",
    "bot_ua_rate",
]