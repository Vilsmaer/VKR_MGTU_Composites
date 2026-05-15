# CompositePredictor 🚀
**Система предиктивной аналитики для композитных материалов | Predictive Analytics System for Composite Materials**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🇷🇺 Описание (RU)
**CompositePredictor** — это интерактивное веб-приложение для загрузки, очистки, разведочного анализа (EDA) и подготовки данных для машинного обучения в области исследования композитных материалов.

### 🔑 Ключевые возможности:
1.  **Умная загрузка данных**: Поддержка CSV и XLSX с авто-определением кодировок (UTF-8, CP1251) и разделителей. Корректная обработка кириллицы.
2.  **Разведочный анализ (EDA)**:
    *   Автоматический расчет метрик качества данных (пропуски, дубликаты, плотность).
    *   **Тест на нормальность (Shapiro-Wilk)** с адаптивным сэмплированием для больших датасетов.
    *   Визуализация: матрица пропусков, Q-Q графики, распределение типов.
3.  **Препроцессинг**:
    *   Импуляция пропусков (Mean/Median/Mode).
    *   Удаление строк/колонок.
    *   Масштабирование (MinMax, StandardScaler).
    *   One-Hot Encoding для категориальных признаков.
4.  **Безопасность данных**: Все операции выполняются локально, данные не покидают браузер пользователя до момента явного экспорта.

### 🛠️ Технологический стек
*   **Frontend**: Streamlit
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly, Seaborn
*   **ML Prep**: Scikit-Learn
*   **Stats**: SciPy (для тестов нормальности)

### 🚀 Быстрый старт
1.  **Клонирование**:
    ```bash
    git clone https://github.com/Vilsmaer/VKR_MGTU_Composites.git
    cd VKR_MGTU_Composites
    ```
2.  **Установка зависимостей**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Запуск**:
    ```bash
    streamlit run app.py
    ```

---

## 🇬🇧 Description (EN)
**CompositePredictor** is an interactive web application designed for loading, cleaning, Exploratory Data Analysis (EDA), and preprocessing data for machine learning in the field of composite materials research.

### 🔑 Key Features:
1.  **Smart Data Loading**: Supports CSV and XLSX with auto-detection of encodings (UTF-8, CP1251) and delimiters. Full Cyrillic support.
2.  **Exploratory Data Analysis (EDA)**:
    *   Automatic calculation of data quality metrics (missing values, duplicates, density).
    *   **Normality Test (Shapiro-Wilk)** with adaptive sampling for large datasets.
    *   Visualizations: Missing value matrix, Q-Q plots, data type distribution.
3.  **Preprocessing**:
    *   Imputation of missing values (Mean/Median/Mode).
    *   Row/Column deletion.
    *   Scaling (MinMax, StandardScaler).
    *   One-Hot Encoding for categorical features.
4.  **Data Privacy**: All operations are performed locally; data does not leave the user's browser until explicit export.

### 🛠️ Tech Stack
*   **Frontend**: Streamlit
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly, Seaborn
*   **ML Prep**: Scikit-Learn
*   **Stats**: SciPy (for normality tests)

### 🚀 Quick Start
1.  **Clone**:
    ```bash
    git clone https://github.com/Vilsmaer/VKR_MGTU_Composites.git
    cd VKR_MGTU_Composites
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run**:
    ```bash
    streamlit run app.py
    ```

---

## 📝 История изменений (Changelog)

### v0.1.0 (Initial Release) - 2026
*   **📂 Загрузка данных**: Реализована загрузка множественных файлов, объединение (concat/merge), редактирование имен колонок. Исправлена проблема с кодировкой кириллицы (UTF-8-SIG/CP1251).
*   **🎯 EDA**: Добавлен расширенный модуль анализа. Внедрен тест Шапиро-Уилка для проверки нормальности распределения. Построение Q-Q графиков. Корректный подсчет NaN (пустые строки конвертируются в NaN).
*   **🧹 Препроцессинг**: Полноценный пайплайн очистки: импутация, удаление, скалеры, OHE. Сохранение объектов скалеров/энкодеров в сессию для будущего использования в моделях.
*   **🐛 Исправления**: Устранены ошибки синтаксиса, артефакты копирования, проблемы с ключами `st.session_state`. Оптимизирована работа с большими данными (сэмплинг в тестах).
*   **🌐 GitHub**: Проект опубликован в репозитории `VKR_MGTU_Composites`.

---

## 📁 Структура проекта
```text
CompositePredictor/
├── app.py              # Главный файл UI (Streamlit)
├── core/               # Бизнес-логика (планируется)
├── utils/
│   ├── load.py         # Загрузка, слияние, экспорт (NaN-логика)
│   ├── eda.py          # Метрики, статистика, графики (Shapiro, Q-Q)
│   └── clean.py        # Пайплайны очистки и трансформации
├── data/               # Папка для временных файлов
├── models/             # Папка для сохраненных моделей (.pkl)
├── requirements.txt    # Зависимости Python
└── README.md           # Документация
```

## 📄 Лицензия
MIT License. См. файл [LICENSE](LICENSE) для деталей.

## 👥 Авторы
*   **Vilsmaer** - [GitHub Profile](https://github.com/Vilsmaer)
*   Проект разработан в рамках ВКР МГТУ им. Н.Э. Баумана.
