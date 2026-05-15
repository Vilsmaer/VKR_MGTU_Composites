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
    *   Автоматический расчет метрик качества данных (пропуски, дубликаты, плотность), базовые статметрики.
    *   Визуализация: матрица пропусков, матрица уникальных значений, распределение типов.
3.  **Препроцессинг**:
    *   Импуляция пропусков (Mean/Median/Mode).
    *   Удаление строк/колонок.
    *   Масштабирование (MinMax, StandardScaler).
    *   One-Hot Encoding для категориальных признаков.

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

## 📄 Лицензия
MIT License. См. файл [LICENSE](LICENSE) для деталей.

## 👥 Авторы
*   **Vilsmaer** - [GitHub Profile](https://github.com/Vilsmaer)
*   Проект разработан в рамках ВКР МГТУ им. Н.Э. Баумана.
