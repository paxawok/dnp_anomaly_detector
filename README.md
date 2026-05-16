# 🕵️‍♂️ Web Log Anomaly Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-Blazing%20Fast-orange.svg?logo=polars)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg?logo=scikit-learn)
![uv](https://img.shields.io/badge/uv-Package%20Manager-purple.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Повний Machine Learning пайплайн для аналізу HTTP-логів та виявлення аномальної активності без використання розмічених даних (Unsupervised Learning). Проєкт ефективно обробляє великі обсяги даних, виявляючи ботів, скрейпери, сканери вразливостей та brute-force атаки.

---

## ✨ Ключові особливості

- 🚀 **Блискавична швидкість:** Обробка ~3.6 млн записів та навчання моделей займає **менше 15 секунд** завдяки `Polars` та лінійній масштабованості алгоритмів.
- 🧠 **Unsupervised Ensemble ML:** Використання комбінації **Isolation Forest** та **SGD One-Class SVM** для максимальної точності та уникнення MemoryError на великих даних.
- ⏱️ **Advanced Sessionization:** Динамічне групування запитів у сесії з вікном у 60 секунд для аналізу поведінкових патернів.
- 🎯 **Risk Scoring:** Автоматичний розрахунок нормалізованого **Risk Score (0-100)** для кожної сесії, що ідеально підходить для інтеграції з SIEM-системами та Firewall.

---

## 🏗 Архітектура Пайплайну

Проєкт побудований за класичною схемою ETL + ML. 

```mermaid
graph LR
    A[(Raw CSV Logs)] -->|Ingestion| B(Polars DataFrame)
    B -->|Preprocessing| C[Clean Dataset]
    C -->|Sessionization| D[Feature Vectors]
    D --> E{ML Ensemble}
    E -->|Isolation Forest| F[Anomaly Scores]
    E -->|SGD One-Class SVM| F
    F --> G[Reporting & Classification]
    G --> H[(Parquet / CSV Reports)]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#fdb,stroke:#333,stroke-width:2px
```

---

## 💾 Дані

Для навчання та тестування використовується відкритий датасет веб-логів:
- 📎 **Джерело:** [Zenodo - HTTP access logs](https://zenodo.org/records/3332970)
- **Період:** Січень–Лютий 2018
- **Обсяг:** ~3.6 млн записів після очищення

---

## 🚀 Getting Started (Швидкий старт)

### Передумови
Проєкт використовує пакетний менеджер [uv](https://github.com/astral-sh/uv) для управління залежностями.

### Встановлення та запуск

1. **Клонуйте репозиторій:**
   ```bash
   git clone https://github.com/paxawok/dnp_anomaly_detector.git
   cd dnp_anomaly_detector
   ```

2. **Встановіть залежності (за допомогою `uv`):**
   ```bash
   uv sync
   ```

3. **Запустіть пайплайн:**
   ```bash
   uv run python -m src.main
   ```

*(Можливий також частковий запуск пайплайну, наприклад `uv run python -m src.main --from modelling`, якщо проміжні дані вже згенеровані).*

---

## 🧠 ML Підхід та Класифікація

Оскільки реальні логи рідко мають розмітку аномалій, ми використовуємо **Unsupervised Learning**. 
Моделі аналізують поведінкові фічі (інтенсивність запитів, коефіцієнт унікальних URL, частка 4xx/5xx статусів, наявність бот-маркерів у User-Agent).

**Класифікатор загроз розподіляє аномалії за такими типами:**
* 🕷️ `web scraper (content theft)` — агресивне викачування контенту з великою кількістю унікальних сторінок.
* 🤖 `bot / crawler` — автоматизовані скрипти та боти, що ігнорують правила сайту.
* 🔓 `vulnerability scan / dir-brute` — сканери вразливостей, що генерують аномальну кількість 4xx помилок.
* ❓ `low-volume anomaly` — нестандартна поведінка низької інтенсивності (потребує ручного рев'ю).

---

## 📈 Результати роботи

Після прогону пайплайну на повному датасеті були отримані наступні метрики:

> **Всього проаналізовано сесій:** `550,736`
> **Виявлено аномалій (Ensemble):** `~5.2%`

**Розподіл типів аномалій:**
| Тип загрози | Відсоток | Опис |
| :--- | :--- | :--- |
| **Web Scraper** | ~61% | Цілеспрямований збір даних з сайту (широке охоплення URL) |
| **Bot / Crawler** | ~37% | Автоматизовані скрипти, пошукові боти без обмежень |
| **Low-volume anomaly**| ~1.1% | Атиповий трафік, який не підпадає під жорсткі патерни |
| **Vulnerability scan**| ~0.1% | Спроби пошуку прихованих директорій / сканування вразливостей |

---

## 📁 Структура проєкту та Артефакти

Після успішного запуску, пайплайн генерує артефакти у папку `data/processed/`:

```text
📦 data/processed/
 ┣ 📜 raw_logs.parquet         # Конвертовані сирі дані
 ┣ 📜 clean_logs.parquet       # Очищені та відфільтровані дані
 ┣ 📜 features.parquet         # Згенеровані фічі (вектори сесій)
 ┣ 📜 scored.parquet           # Дані з оцінками аномальності
 ┣ 📊 anomaly_report.csv       # Бізнес-звіт із розрахованим Risk Score (0-100)
 ┣ 🧠 scaler.joblib            # Збережений об'єкт стандартизації
 ┣ 🧠 isolation_forest.joblib  # Навчена модель Isolation Forest
 ┗ 🧠 secondary_model.joblib   # Навчена модель SGD One-Class SVM
```