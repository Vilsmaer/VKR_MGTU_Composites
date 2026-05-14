"""
Модуль первичного анализа данных (EDA).
Расчет статистик, проверка качества данных и визуализация.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
import warnings

# Игнорируем предупреждения о смешанных типах для чистоты логов
warnings.filterwarnings('ignore', category=UserWarning, module='plotly')


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Расчет основных метрик качества данных.
    
    Args:
        df: Исходный DataFrame.
        
    Returns:
        Словарь с метриками (строки, столбцы, NaN, дубликаты, плотность и т.д.).
    """
    if df.empty:
        return {
            "rows": 0, "cols": 0, "nan_count": 0, "nan_pct": 0.0,
            "density_pct": 0.0, "duplicates_pct": 0.0, "memory_mb": 0.0
        }

    total_cells = df.size
    nan_count = df.isna().sum().sum()
    
    # Защита от деления на ноль
    nan_pct = (nan_count / total_cells * 100) if total_cells > 0 else 0.0
    density_pct = 100.0 - nan_pct
    
    dup_count = df.duplicated().sum()
    # % дубликатов считаем от количества строк
    duplicates_pct = (dup_count / len(df) * 100) if len(df) > 0 else 0.0
    
    # Потребление памяти (примерное)
    try:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    except Exception:
        memory_mb = 0.0

    return {
        "rows": len(df),
        "cols": len(df.columns),
        "nan_count": int(nan_count),
        "nan_pct": round(nan_pct, 2),
        "density_pct": round(density_pct, 2),
        "duplicates_count": int(dup_count),
        "duplicates_pct": round(duplicates_pct, 2),
        "memory_mb": round(memory_mb, 2)
    }


def calculate_uniqueness(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Расчет процента уникальных значений для каждого столбца и строки.
    
    Returns:
        (col_uniqueness, row_uniqueness) - Series с процентами.
    """
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Уникальность по столбцам
    col_unique_counts = df.nunique(dropna=False)
    col_uniqueness = (col_unique_counts / len(df) * 100).round(2)
    
    # Уникальность по строкам (вычислительно дорого для больших данных, делаем сэмпл если > 10000)
    if len(df) > 10000:
        # Для оптимизации считаем только на сэмпле или возвращаем заглушку
        # В полной версии можно реализовать хеширование строк
        row_uniqueness = pd.Series([0.0] * len(df), index=df.index, name="Row Uniqueness (%)")
        # Быстрый расчет для небольших датасетов или сэмпла
        sample_df = df.sample(n=1000, random_state=42) if len(df) > 1000 else df
        row_unique_counts = sample_df.drop_duplicates().shape[0]
        # Это упрощение: глобальный % уникальных строк
        global_row_unique_pct = (row_unique_counts / len(sample_df) * 100)
        row_uniqueness_msg = f"~{global_row_unique_pct:.2f}% (оценено по сэмплу)"
    else:
        row_unique_counts = df.drop_duplicates().shape[0]
        global_row_unique_pct = (row_unique_counts / len(df) * 100)
        row_uniqueness_msg = f"{global_row_unique_pct:.2f}%"

    return col_uniqueness, row_uniqueness_msg


def get_column_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Детальная статистика по каждому столбцу.
    """
    if df.empty:
        return pd.DataFrame()

    stats = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        non_null = s.count()
        nulls = s.isna().sum()
        unique = s.nunique()
        unique_pct = round((unique / len(df) * 100), 2) if len(df) > 0 else 0
        
        # Попытка получить числовые статистики
        mean_val, std_val, min_val, max_val = "-", "-", "-", "-"
        if pd.api.types.is_numeric_dtype(s):
            mean_val = round(s.mean(), 2)
            std_val = round(s.std(), 2)
            min_val = s.min()
            max_val = s.max()
        
        stats.append({
            "Column": col,
            "Type": dtype,
            "Non-Null": non_null,
            "Nulls": nulls,
            "Unique": unique,
            "Unique %": unique_pct,
            "Mean": mean_val,
            "Std": std_val,
            "Min": min_val,
            "Max": max_val
        })
    
    return pd.DataFrame(stats)


def plot_missing_matrix(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Тепловая карта пропусков (Missing Matrix).
    """
    if df.empty or df.size == 0:
        return None
    
    # Создаем маску пропусков
    missing_df = df.isna()
    # Оптимизация: если колонок очень много, берем первые 50
    if len(missing_df.columns) > 50:
        missing_df = missing_df.iloc[:, :50]
    
    fig = px.imshow(
        missing_df.T, 
        aspect="auto",
        color_continuous_scale=[[0, '#636EFA'], [1, '#EF553B']], # Синий (нет пропуска), Красный (пропуск)
        labels={'x': 'Index', 'y': 'Column', 'color': 'Missing'},
        title='Матрица пропусков (Красный = NaN)',
        origin='lower'
    )
    fig.update_layout(height=max(300, len(missing_df.columns) * 20))
    return fig


def plot_distribution_overview(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Гистограмма распределения типов данных и количества уникальных значений.
    """
    if df.empty:
        return None

    col_types = df.dtypes.astype(str).value_counts().reset_index()
    col_types.columns = ['Type', 'Count']
    
    fig = px.bar(
        col_types, 
        x='Type', 
        y='Count', 
        text='Count',
        title='Распределение типов данных по столбцам',
        color='Type'
    )
    fig.update_traces(textposition='outside')
    return fig


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Главная функция EDA. Агрегирует все метрики и графики.
    
    Args:
        df: DataFrame для анализа.
        
    Returns:
        Словарь с результатами: metrics, stats_df, uniqueness_msg, figures.
    """
    # TODO: Добавить анализ выбросов (IQR/Z-score) в будущих версиях
    # TODO: Добавить корреляционную матрицу для числовых признаков
    
    metrics = calculate_metrics(df)
    col_uniq, row_uniq_msg = calculate_uniqueness(df)
    stats_df = get_column_stats(df)
    
    fig_missing = plot_missing_matrix(df)
    fig_types = plot_distribution_overview(df)
    
    return {
        "metrics": metrics,
        "stats_df": stats_df,
        "col_uniqueness": col_uniq,
        "row_uniqueness_msg": row_uniq_msg,
        "figures": {
            "missing_matrix": fig_missing,
            "types_dist": fig_types
        }
    }
