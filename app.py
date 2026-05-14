"""
Главный файл приложения CompositePredictor.
UI для загрузки, объединения и подготовки данных (Вкладка 1).
"""
import streamlit as st
import pandas as pd
import os
from typing import List, Dict, Any, Optional

from utils.load import (
    load_file,
    preview_random_rows,
    concatenate_dataframes,
    merge_dataframes,
    merge_by_inner_index,
    rename_columns,
    export_to_csv,
    get_duplicate_columns
)
from utils.eda import run_eda
from utils.clean import run_cleaning_pipeline, impute_numeric_nan

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ============================================================================
def init_session_state():
    """Инициализирует ключи st.session_state при первом запуске."""
    if 'df_raw' not in st.session_state:
        st.session_state['df_raw'] = None
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []
    if 'file_dataframes' not in st.session_state:
        st.session_state['file_dataframes'] = {}  # {filename: df}
    if 'merge_method' not in st.session_state:
        st.session_state['merge_method'] = 'concat_0'
    if 'merge_key_left' not in st.session_state:
        st.session_state['merge_key_left'] = None
    if 'merge_key_right' not in st.session_state:
        st.session_state['merge_key_right'] = None
    
    # Ключи для вкладки очистки данных
    if 'df_clean' not in st.session_state:
        st.session_state['df_clean'] = None
    if 'scalers' not in st.session_state:
        st.session_state['scalers'] = {}
    if 'encoders' not in st.session_state:
        st.session_state['encoders'] = {}


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ UI
# ============================================================================
def display_file_preview(df: pd.DataFrame, filename: str):
    """Отображает превью файла (10 случайных строк)."""
    try:
        preview_df = preview_random_rows(df, n=10)
        st.write(f"**Превью файла:** `{filename}` ({len(df)} строк, {len(df.columns)} колонок)")
        st.dataframe(preview_df, use_container_width=True, height=250)
    except Exception as e:
        st.error(f"Ошибка при отображении превью файла {filename}: {str(e)}")


def handle_file_upload(uploaded_files) -> Dict[str, pd.DataFrame]:
    """
    Обрабатывает загруженные файлы и возвращает словарь {имя: DataFrame}.
    # TODO: Добавить прогресс-бар для больших файлов
    """
    file_dataframes = {}
    
    for uploaded_file in uploaded_files:
        try:
            # Сохраняем во временную директорию
            temp_path = os.path.join('data', uploaded_file.name)
            os.makedirs('data', exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Загружаем через утилиту
            df = load_file(temp_path)
            file_dataframes[uploaded_file.name] = df
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки файла `{uploaded_file.name}`: {str(e)}")
    
    return file_dataframes


def perform_merge(
    file_dataframes: Dict[str, pd.DataFrame],
    merge_method: str,
    merge_key: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Выполняет объединение файлов согласно выбранному методу.
    # TODO: [описание] Добавить поддержку множественных ключей для merge
    """
    if not file_dataframes:
        return None
    
    dfs = list(file_dataframes.values())
    
    if len(dfs) == 1:
        return dfs[0].copy()
    
    try:
        if merge_method == 'concat_0':
            # Вертикальное объединение (строки)
            result = concatenate_dataframes(dfs, axis=0, ignore_index=True)
            
        elif merge_method == 'concat_1':
            # Горизонтальное объединение (колонки)
            result = concatenate_dataframes(dfs, axis=1, ignore_index=False)
            
        elif merge_method == 'merge_by_col':
            # Объединение по колонке (только для 2 файлов)
            if len(dfs) != 2:
                st.warning("Метод 'merge_by_col' поддерживает только 2 файла. Используются первые два.")
                dfs = dfs[:2]
            
            if not merge_key:
                st.warning("Ключ для объединения не выбран. Используется первая общая колонка.")
                common_cols = set(dfs[0].columns) & set(dfs[1].columns)
                merge_key = list(common_cols)[0] if common_cols else None
            
            if merge_key:
                result = merge_dataframes(
                    dfs[0], dfs[1],
                    on=merge_key,
                    how='inner'
                )
            else:
                st.error("Не найдено общих колонок для объединения.")
                return None
                
        elif merge_method == 'inner_index':
            # Объединение по индексам (только для 2 файлов)
            if len(dfs) != 2:
                st.warning("Метод 'inner_index' поддерживает только 2 файла. Используются первые два.")
                dfs = dfs[:2]
            
            result = merge_by_inner_index(dfs[0], dfs[1], how='inner')
            
        else:
            st.error(f"Неизвестный метод объединения: {merge_method}")
            return None
        
        # Проверка на дубликаты колонок после merge
        duplicates = get_duplicate_columns(result)
        if duplicates:
            st.warning(f"⚠️ Обнаружены дублирующиеся имена колонок: {duplicates}")
        
        return result
        
    except Exception as e:
        st.error(f"❌ Ошибка при объединении данных: {str(e)}")
        return None


def display_column_editor(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Отображает редактор для переименования колонок через st.data_editor.
    Возвращает DataFrame с обновленными именами или None если изменений нет.
    """
    st.subheader("📝 Переименование колонок")
    
    # Создаем DataFrame с текущими именами
    editor_df = pd.DataFrame({
        'Текущее имя': list(df.columns),
        'Новое имя': list(df.columns)
    })
    
    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Текущее имя": st.column_config.TextColumn(disabled=True),
            "Новое имя": st.column_config.TextColumn(help="Измените имя колонки")
        },
        num_rows="fixed",
        key="column_rename_editor"
    )
    
    # Проверяем изменения
    changes_made = False
    new_names = {}
    
    for _, row in edited_df.iterrows():
        old_name = row['Текущее имя']
        new_name = row['Новое имя']
        
        if old_name != new_name and new_name.strip():
            new_names[old_name] = new_name.strip()
            changes_made = True
    
    if changes_made:
        if st.button("💾 Применить переименование", key="apply_rename_btn"):
            try:
                result_df = rename_columns(df, new_names)
                st.success(f"✅ Переименовано колонок: {len(new_names)}")
                return result_df
            except ValueError as ve:
                st.error(f"Ошибка валидации: {str(ve)}")
            except Exception as e:
                st.error(f"Ошибка при переименовании: {str(e)}")
    
    return None


def display_export_section(df: pd.DataFrame):
    """Отображает секцию экспорта результата в CSV с корректной кодировкой."""
    st.subheader("💾 Экспорт данных")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        filename = st.text_input(
            "Имя файла:",
            value="prepared_data.csv",
            help="Введите имя файла для экспорта"
        )
    
    with col2:
        export_btn = st.button("📥 Скачать CSV", type="primary", use_container_width=True)
    
    if export_btn and df is not None:
        try:
            output_path = os.path.join('data', filename)
            export_to_csv(df, output_path, index=False)
            
            # Читаем файл в бинарном режиме для скачивания (сохраняет UTF-8-SIG BOM)
            with open(output_path, 'rb') as f:
                csv_data = f.read()
            
            st.download_button(
                label="⬇️ Подтвердить скачивание",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="download_csv_confirm"
            )
            st.success(f"Файл сохранен: `{output_path}`")
            
        except Exception as e:
            st.error(f"Ошибка при экспорте: {str(e)}")


# ============================================================================
# ОСНОВНОЙ ИНТЕРФЕЙС (ВКЛАДКА 1)
# ============================================================================
def render_data_upload_tab():
    """Рендерит вкладку загрузки и подготовки данных."""
    st.header("📂 Загрузка и подготовка данных")
    
    # 1. Загрузка файлов
    st.subheader("1️⃣ Загрузка файлов (CSV / XLSX)")
    
    uploaded_files = st.file_uploader(
        "Выберите файлы для загрузки",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Можно выбрать несколько файлов одновременно"
    )
    
    # Обработка новых загруженных файлов
    if uploaded_files:
        current_filenames = [f.name for f in uploaded_files]
        previous_filenames = [f for f in st.session_state.get('uploaded_files', [])]
        
        # Если список файлов изменился, перезагружаем
        if set(current_filenames) != set(previous_filenames):
            with st.spinner("Загрузка файлов..."):
                file_dataframes = handle_file_upload(uploaded_files)
                st.session_state['uploaded_files'] = current_filenames
                st.session_state['file_dataframes'] = file_dataframes
                st.session_state['df_raw'] = None  # Сбрасываем результат при новой загрузке
                st.rerun()
    
    # Отображение превью загруженных файлов
    file_dataframes = st.session_state.get('file_dataframes', {})
    
    if file_dataframes:
        st.success(f"✅ Загружено файлов: {len(file_dataframes)}")
        
        # Показываем превью каждого файла в expanders
        for filename, df in file_dataframes.items():
            with st.expander(f"📄 {filename} ({len(df)} строк)", expanded=False):
                display_file_preview(df, filename)
        
        # 2. Выбор метода объединения
        st.subheader("2️⃣ Метод объединения")
        
        merge_method = st.selectbox(
            "Выберите способ объединения:",
            options=[
                ('concat_0', 'Вертикально (добавить строки)'),
                ('concat_1', 'Горизонтально (добавить колонки)'),
                ('merge_by_col', 'Объединение по ключевой колонке (2 файла)'),
                ('inner_index', 'Объединение по индексам (2 файла)')
            ],
            format_func=lambda x: x[1],
            help="concat_0: добавляет строки снизу; concat_1: добавляет колонки справа; merge_by_col: SQL-like join; inner_index: join по индексу"
        )[0]
        
        st.session_state['merge_method'] = merge_method
        
        # Дополнительные настройки для merge_by_col
        merge_key = None
        if merge_method == 'merge_by_col' and len(file_dataframes) >= 2:
            dfs_list = list(file_dataframes.values())
            common_cols = set(dfs_list[0].columns) & set(dfs_list[1].columns)
            
            if common_cols:
                merge_key = st.selectbox(
                    "Выберите ключевую колонку для объединения:",
                    options=list(common_cols),
                    help="Колонка должна присутствовать в обоих файлах"
                )
            else:
                st.warning("Нет общих колонок между первыми двумя файлами!")
        
        # Кнопка выполнения объединения
        if st.button("🔗 Объединить данные", type="primary"):
            with st.spinner("Объединение данных..."):
                merged_df = perform_merge(file_dataframes, merge_method, merge_key)
                
                if merged_df is not None:
                    st.session_state['df_raw'] = merged_df
                    st.success(f"✅ Данные успешно объединены! Строк: {len(merged_df)}, Колонок: {len(merged_df.columns)}")
                    st.rerun()
    
    # 3. Работа с объединенным DataFrame
    df_raw = st.session_state.get('df_raw')
    
    if df_raw is not None:
        st.divider()
        st.subheader("3️⃣ Результат объединения")
        
        # Превью итога
        st.write(f"**Итоговый датасет:** {len(df_raw)} строк × {len(df_raw.columns)} колонок")
        st.dataframe(df_raw.head(10), use_container_width=True, height=300)
        
        # Статистика
        with st.expander("📊 Базовая статистика"):
            st.write(df_raw.describe(include='all'))
        
        # 4. Переименование колонок
        renamed_df = display_column_editor(df_raw)
        
        if renamed_df is not None:
            st.session_state['df_raw'] = renamed_df
            df_raw = renamed_df
            st.rerun()
        
        # 5. Экспорт
        st.divider()
        display_export_section(df_raw)
        
    elif file_dataframes:
        st.info("👆 Выберите метод объединения и нажмите кнопку выше, чтобы создать итоговый датасет.")
    else:
        st.info("📁 Загрузите файлы (CSV/XLSX) для начала работы.")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================
def main():
    """Основная функция приложения."""
    st.set_page_config(
        page_title="CompositePredictor",
        page_icon="🔮",
        layout="wide"
    )
    
    init_session_state()
    
    # Боковая панель
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/crystal-ball.png", width=80)
        st.title("CompositePredictor")
        st.markdown("---")
        st.markdown("**Меню:**")
        
        tabs = st.radio(
            "Навигация",
            ["📂 Загрузка данных", "🧹 Препроцессинг", "🎯 EDA", "🤖 Обучение модели", "🔮 Прогноз"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ О проекте")
        st.markdown("""
        **CompositePredictor** - инструмент для:
        - Загрузки и объединения данных
        - Предобработки и очистки
        - Разведочного анализа (EDA)
        - Обучения ML-моделей
        - Прогнозирования
        """)
    
    # Рендеринг активной вкладки
    if tabs == "📂 Загрузка данных":
        render_data_upload_tab()
    elif tabs == "🧹 Препроцессинг":
        render_cleaning_tab()
    elif tabs == "🎯 EDA":
        render_eda_tab()
    elif tabs == "🤖 Обучение модели":
        st.header("🤖 Обучение модели")
        st.info("Этот раздел находится в разработке. Вернитесь позже!")
    elif tabs == "🔮 Прогноз":
        st.header("🔮 Прогноз")
        st.info("Этот раздел находится в разработке. Вернитесь позже!")


def render_eda_tab():
    """Рендерит вкладку первичного анализа данных (EDA)."""
    st.header("🎯 Первичный анализ данных (EDA)")
    
    df_raw = st.session_state.get('df_raw')
    
    # Блокировка вкладки, если данных нет
    if df_raw is None or df_raw.empty:
        st.warning("⚠️ Данные не загружены. Перейдите на вкладку **📂 Загрузка данных**, чтобы загрузить файл.")
        st.stop()
        return
    
    st.success(f"✅ Анализ данных: {len(df_raw)} строк × {len(df_raw.columns)} колонок")
    
    with st.spinner("Выполнение анализа..."):
        try:
            eda_result = run_eda(df_raw)
        except Exception as e:
            st.error(f"❌ Ошибка при выполнении EDA: {str(e)}")
            return
    
    metrics = eda_result.get('metrics', {})
    stats_df = eda_result.get('stats_df', pd.DataFrame())
    col_uniqueness = eda_result.get('col_uniqueness', pd.Series())
    row_uniqueness_msg = eda_result.get('row_uniqueness_msg', 'N/A')
    figures = eda_result.get('figures', {})
    
    # 1. Основные метрики
    st.subheader("1️⃣ Основные метрики качества")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Строки / Столбцы",
            value=f"{metrics.get('rows', 0)} / {metrics.get('cols', 0)}",
            help="Общее количество строк и столбцов в датасете"
        )
    
    with col2:
        nan_pct = metrics.get('nan_pct', 0.0)
        delta_nan = f"{nan_pct:.1f}%" if nan_pct > 0 else "0%"
        st.metric(
            label="🕳️ Пропуски (NaN)",
            value=f"{metrics.get('nan_count', 0)} ({delta_nan})",
            delta=f"-{nan_pct:.1f}%" if nan_pct > 0 else "0%",
            delta_color="inverse" if nan_pct < 10 else "normal"
        )
    
    with col3:
        density = metrics.get('density_pct', 0.0)
        st.metric(
            label="💎 Плотность данных",
            value=f"{density:.1f}%",
            delta=f"{density:.1f}%",
            delta_color="normal" if density > 90 else "inverse"
        )
    
    with col4:
        dup_pct = metrics.get('duplicates_pct', 0.0)
        st.metric(
            label="🔄 Дубликаты строк",
            value=f"{metrics.get('duplicates_count', 0)} ({dup_pct:.1f}%)",
            delta=f"-{dup_pct:.1f}%" if dup_pct > 0 else "0%",
            delta_color="inverse" if dup_pct < 5 else "normal"
        )
    
    # Дополнительная информация
    with st.expander("📝 Детальная информация"):
        st.write(f"**Уникальность строк:** {row_uniqueness_msg}")
        st.write(f"**Потребление памяти:** ~{metrics.get('memory_mb', 0):.2f} MB")
        
        if not col_uniqueness.empty:
            st.write("**% Уникальных значений по столбцам (топ-10):**")
            top_unique = col_uniqueness.sort_values(ascending=False).head(10)
            st.bar_chart(top_unique)

    # 2. Графики
    st.subheader("2️⃣ Визуализация")
    
    fig_missing = figures.get('missing_matrix')
    fig_types = figures.get('types_dist')
    
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        if fig_types:
            st.plotly_chart(fig_types, use_container_width=True)
        else:
            st.info("Нет данных для отображения графика типов.")
    
    with col_graph2:
        if fig_missing:
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.info("Нет данных для отображения матрицы пропусков.")
    
    # 3. Детальная статистика по столбцам
    st.subheader("3️⃣ Статистика по столбцам")
    
    if not stats_df.empty:
        # Фильтры для таблицы
        filter_col = st.selectbox(
            "Фильтр по типу данных:",
            options=["Все"] + list(stats_df['Type'].unique()),
            key="eda_type_filter"
        )
        
        display_df = stats_df
        if filter_col != "Все":
            display_df = stats_df[stats_df['Type'] == filter_col]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Экспорт статистики
        csv_stats = display_df.to_csv(index=False, encoding='utf-8-sig', sep=';').encode('utf-8')
        st.download_button(
            label="📥 Скачать статистику (CSV)",
            data=csv_stats,
            file_name="eda_statistics.csv",
            mime="text/csv",
            key="download_eda_stats"
        )
    else:
        st.info("Статистика недоступна.")


def render_cleaning_tab():
    """
    Рендерит вкладку препроцессинга данных.
    Реализует: импутацию NaN, удаление строк/колонок, скалеры, OHE.
    """
    st.header("🧹 Препроцессинг")
    
    df_raw = st.session_state.get('df_raw')
    
    # Блокировка вкладки, если данных нет
    if df_raw is None or df_raw.empty:
        st.warning("⚠️ Данные не загружены. Перейдите на вкладку **📂 Загрузка данных**, чтобы загрузить файл.")
        st.stop()
        return
    
    st.success(f"✅ Исходные данные: {len(df_raw)} строк × {len(df_raw.columns)} колонок")
    
    # Инициализация переменных сессии для формы
    if 'clean_nan_strategy' not in st.session_state:
        st.session_state['clean_nan_strategy'] = 'mean'
    if 'clean_rows_to_drop' not in st.session_state:
        st.session_state['clean_rows_to_drop'] = ''
    if 'clean_cols_to_drop' not in st.session_state:
        st.session_state['clean_cols_to_drop'] = []
    if 'clean_scaler_type' not in st.session_state:
        st.session_state['clean_scaler_type'] = 'none'
    if 'clean_ohe_columns' not in st.session_state:
        st.session_state['clean_ohe_columns'] = []
    
    # ========================================================================
    # ШАГ 1: Импуляция NaN
    # ========================================================================
    st.subheader("1️⃣ Заполнение пропусков (NaN)")
    st.markdown("Применяется только к **числовым** колонкам.")
    
    nan_strategy = st.selectbox(
        "Стратегия заполнения:",
        options=['mean', 'median', 'mode'],
        format_func=lambda x: {'mean': 'Среднее значение (Mean)', 'median': 'Медиана (Median)', 'mode': 'Мода (Mode)'}[x],
        help="Mean - среднее арифметическое; Median - медиана; Mode - наиболее частое значение",
        key='clean_nan_strategy'
    )
    
    # ========================================================================
    # ШАГ 2: Удаление строк
    # ========================================================================
    st.subheader("2️⃣ Удаление строк по индексам")
    st.markdown("Введите индексы через запятую или диапазоны (например: `0, 5, 10-15`).")
    
    rows_input = st.text_input(
        "Индексы для удаления:",
        value=st.session_state.get('clean_rows_to_drop', ''),
        placeholder="0, 5, 10-15",
        help="Можно указывать отдельные индексы и диапазоны через дефис",
        key='clean_rows_input'
    )
    
    def parse_indices(input_str: str) -> List[int]:
        """Парсит строку индексов в список целых чисел."""
        indices = []
        if not input_str.strip():
            return indices
        
        parts = input_str.replace(' ', '').split(',')
        for part in parts:
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start, end + 1))
                except ValueError:
                    st.warning(f"Некорректный диапазон: {part}")
            else:
                try:
                    indices.append(int(part))
                except ValueError:
                    st.warning(f"Некорректный индекс: {part}")
        
        return sorted(list(set(indices)))
    
    rows_to_drop = parse_indices(rows_input)
    if rows_to_drop:
        st.info(f"Будет удалено строк: {len(rows_to_drop)} (индексы: {rows_to_drop[:10]}{'...' if len(rows_to_drop) > 10 else ''})")
    
    # ========================================================================
    # ШАГ 3: Удаление колонок
    # ========================================================================
    st.subheader("3️⃣ Удаление колонок")
    
    all_columns = list(df_raw.columns)
    cols_to_drop = st.multiselect(
        "Выберите колонки для удаления:",
        options=all_columns,
        default=st.session_state.get('clean_cols_to_drop', []),
        help="Удалите ненужные колонки перед трансформацией",
        key='clean_cols_to_drop'
    )
    
    # ========================================================================
    # ШАГ 4: Трансформация данных
    # ========================================================================
    st.subheader("4️⃣ Трансформация данных")
    
    col_transf_1, col_transf_2 = st.columns(2)
    
    with col_transf_1:
        st.markdown("**Числовые данные:**")
        scaler_type = st.selectbox(
            "Метод масштабирования:",
            options=['none', 'minmax', 'standard'],
            format_func=lambda x: {
                'none': 'Без масштабирования',
                'minmax': 'MinMaxScaler (диапазон [0, 1])',
                'standard': 'StandardScaler (Z-score normalization)'
            }[x],
            key='clean_scaler_type'
        )
    
    with col_transf_2:
        st.markdown("**Категориальные данные:**")
        categorical_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
        
        ohe_columns = st.multiselect(
            "One-Hot Encoding для колонок:",
            options=categorical_cols,
            default=st.session_state.get('clean_ohe_columns', []),
            help="Преобразует категории в бинарные столбцы 0/1",
            key='clean_ohe_columns'
        )
    
    # ========================================================================
    # ПРЕДПРОСМОТР И ПРИМЕНЕНИЕ
    # ========================================================================
    st.divider()
    
    preview_col, apply_col = st.columns([1, 1])
    
    with preview_col:
        if st.button("👁️ Предпросмотр изменений", use_container_width=True):
            with st.spinner("Обработка предпросмотра..."):
                try:
                    # Парсим индексы заново для предпросмотра
                    rows_preview = parse_indices(rows_input)
                    
                    preview_df = run_cleaning_pipeline(
                        df=df_raw,
                        nan_strategy=nan_strategy,
                        rows_to_drop=rows_preview if rows_preview else None,
                        cols_to_drop=cols_to_drop if cols_to_drop else None,
                        scaler_type=scaler_type if scaler_type != 'none' else None,
                        ohe_columns=ohe_columns if ohe_columns else None
                    )[0]
                    
                    st.success(f"✅ После обработки: {len(preview_df)} строк × {len(preview_df.columns)} колонок")
                    st.dataframe(preview_df.head(10), use_container_width=True, height=300)
                    
                    # Показываем изменения
                    st.write("**Изменения:**")
                    changes_info = []
                    if len(preview_df) != len(df_raw):
                        changes_info.append(f"• Строк: {len(df_raw)} → {len(preview_df)}")
                    if len(preview_df.columns) != len(df_raw.columns):
                        changes_info.append(f"• Колонок: {len(df_raw.columns)} → {len(preview_df.columns)}")
                    if changes_info:
                        st.markdown("\n".join(changes_info))
                    else:
                        st.info("Значимых изменений в размерах не обнаружено.")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при предпросмотре: {str(e)}")
    
    with apply_col:
        if st.button("💾 Применить изменения", type="primary", use_container_width=True):
            with st.spinner("Применение изменений..."):
                try:
                    df_clean, scalers_dict, encoders_dict = run_cleaning_pipeline(
                        df=df_raw,
                        nan_strategy=nan_strategy,
                        rows_to_drop=rows_to_drop if rows_to_drop else None,
                        cols_to_drop=cols_to_drop if cols_to_drop else None,
                        scaler_type=scaler_type if scaler_type != 'none' else None,
                        ohe_columns=ohe_columns if ohe_columns else None
                    )
                    
                    # Сохраняем в session_state
                    st.session_state['df_clean'] = df_clean
                    st.session_state['scalers'] = scalers_dict
                    st.session_state['encoders'] = encoders_dict
                    
                    st.success(f"✅ Данные очищены и сохранены!")
                    st.write(f"**Результат:** {len(df_clean)} строк × {len(df_clean.columns)} колонок")
                    st.write(f"**Скалеров сохранено:** {len(scalers_dict)}")
                    st.write(f"**Энкодеров сохранено:** {len(encoders_dict)}")
                    
                    # Превью результата
                    st.dataframe(df_clean.head(10), use_container_width=True, height=300)
                    
                    # Экспорт очищенных данных
                    csv_clean = df_clean.to_csv(index=False, encoding='utf-8-sig', sep=';').encode('utf-8')
                    st.download_button(
                        label="📥 Скачать очищенные данные (CSV)",
                        data=csv_clean,
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                        key="download_cleaned_csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при применении изменений: {str(e)}")
    
    # Кнопка сброса
    st.divider()
    if st.button("🔄 Сбросить очистку (вернуться к исходным данным)", key="reset_cleaning"):
        st.session_state['df_clean'] = None
        st.session_state['scalers'] = {}
        st.session_state['encoders'] = {}
        st.session_state['clean_rows_to_drop'] = ''
        st.session_state['clean_cols_to_drop'] = []
        st.session_state['clean_ohe_columns'] = []
        st.success("Очистка сброшена. Данные возвращены к исходному состоянию.")
        st.rerun()
    
    # Инфо о сохраненных артефактах
    if st.session_state.get('df_clean') is not None:
        st.info("""
        **ℹ️ Артефакты сохранены в памяти:**
        - Скалеры (`st.session_state['scalers']`) — для обратной трансформации числовых данных.
        - Энкодеры (`st.session_state['encoders']`) — для обратной трансформации категориальных данных.
        Эти объекты будут использованы на этапе прогнозирования (Этап 8).
        """)


if __name__ == "__main__":
    main()
