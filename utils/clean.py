"""
utils/clean.py
Модуль очистки и трансформации данных.
Содержит функции для импутации, удаления строк/колонок, масштабирования и кодирования.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import warnings

# Отключаем предупреждения о FutureWarning для чистоты логов
warnings.filterwarnings('ignore')


def impute_numeric_nan(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Заполняет пропуски (NaN) в числовых колонках.
    
    Параметры:
        df: Исходный DataFrame.
        strategy: Стратегия заполнения ('mean', 'median', 'mode').
    
    Возвращает:
        DataFrame с заполненными пропусками в числовых колонках.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    for col in numeric_cols:
        if df_copy[col].isnull().sum() > 0:
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                # mode() возвращает Series, берем первое значение
                fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy


def drop_rows_by_indices(df: pd.DataFrame, indices: List[Union[int, str]]) -> pd.DataFrame:
    """
    Удаляет строки по списку индексов.
    
    Параметры:
        df: Исходный DataFrame.
        indices: Список индексов для удаления (int или str).
    
    Возвращает:
        DataFrame без указанных строк.
    """
    df_copy = df.copy()
    
    # Фильтруем только те индексы, которые реально есть в DF
    valid_indices = [idx for idx in indices if idx in df_copy.index]
    invalid_indices = [idx for idx in indices if idx not in df_copy.index]
    
    if invalid_indices:
        print(f"Предупреждение: Индексы {invalid_indices} не найдены и будут проигнорированы.")
    
    if valid_indices:
        df_copy = df_copy.drop(index=valid_indices)
    
    return df_copy


def drop_columns_by_names(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Удаляет колонки по списку имен.
    
    Параметры:
        df: Исходный DataFrame.
        columns: Список имен колонок для удаления.
    
    Возвращает:
        DataFrame без указанных колонок.
    """
    df_copy = df.copy()
    
    # Фильтруем только существующие колонки
    existing_cols = [col for col in columns if col in df_copy.columns]
    missing_cols = [col for col in columns if col not in df_copy.columns]
    
    if missing_cols:
        print(f"Предупреждение: Колонки {missing_cols} не найдены и будут проигнорированы.")
    
    if existing_cols:
        df_copy = df_copy.drop(columns=existing_cols)
    
    return df_copy


def apply_scaler(df: pd.DataFrame, scaler_type: str = 'minmax') -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Применяет масштабирование к числовым колонкам.
    
    Параметры:
        df: Исходный DataFrame.
        scaler_type: Тип скалера ('minmax', 'standard', 'none').
    
    Возвращает:
        Tuple[DataFrame с масштабированными данными, Словарь объектов скалеров {col_name: scaler}].
    """
    df_copy = df.copy()
    scalers = {}
    
    if scaler_type == 'none':
        return df_copy, scalers
    
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy, scalers
    
    # Выбираем класс скалера
    if scaler_type == 'minmax':
        scaler_class = MinMaxScaler
    elif scaler_type == 'standard':
        scaler_class = StandardScaler
    else:
        return df_copy, scalers
    
    for col in numeric_cols:
        # Пропускаем колонки, где все значения NaN (уже обработано импутацией, но на всякий случай)
        if df_copy[col].isnull().all():
            continue
            
        scaler = scaler_class()
        # Reshape для sklearn (n_samples, n_features)
        df_copy[col] = scaler.fit_transform(df_copy[[col]])
        scalers[col] = scaler
    
    return df_copy, scalers


def apply_one_hot_encoding(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
    """
    Применяет One-Hot Encoding к указанным категориальным колонкам.
    
    Параметры:
        df: Исходный DataFrame.
        columns: Список колонок для кодирования. Если None, выбираются все object/category колонки.
    
    Возвращает:
        Tuple[DataFrame с OHE, Словарь энкодеров {col_name: encoder}, Список новых имен колонок].
    """
    df_copy = df.copy()
    encoders = {}
    new_column_names = []
    
    if columns is None:
        # Авто-выбор категориальных колонок
        columns = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not columns:
        return df_copy, encoders, new_column_names
    
    # Фильтруем только существующие колонки
    valid_cols = [col for col in columns if col in df_copy.columns]
    
    for col in valid_cols:
        # Пропускаем, если колонка пустая или все NaN
        if df_copy[col].isnull().all():
            continue
            
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Преобразуем колонку в 2D массив
        encoded_data = encoder.fit_transform(df_copy[[col]].astype(str))
        
        # Получаем имена новых колонок
        feature_names = [f"{col}_{cat}" for cat in encoder.get_feature_names_out([col])]
        new_column_names.extend(feature_names)
        
        # Создаем DataFrame с закодированными данными
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_copy.index)
        
        # Удаляем старую колонку и добавляем новые
        df_copy = df_copy.drop(columns=[col])
        df_copy = pd.concat([df_copy, encoded_df], axis=1)
        
        encoders[col] = encoder
    
    return df_copy, encoders, new_column_names


def run_cleaning_pipeline(
    df: pd.DataFrame,
    nan_strategy: str = 'mean',
    rows_to_drop: Optional[List[Union[int, str]]] = None,
    cols_to_drop: Optional[List[str]] = None,
    scaler_type: str = 'none',
    ohe_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    """
    Главный пайплайн очистки и трансформации данных.
    
    Параметры:
        df: Исходный DataFrame.
        nan_strategy: Стратегия импутации ('mean', 'median', 'mode').
        rows_to_drop: Список индексов строк для удаления.
        cols_to_drop: Список имен колонок для удаления.
        scaler_type: Тип скалера ('minmax', 'standard', 'none').
        ohe_columns: Список колонок для OHE.
    
    Возвращает:
        Tuple[Очищенный DataFrame, Словарь скалеров, Словарь энкодеров].
    """
    # TODO: Добавить обработку выбросов (IQR/Z-score) перед импутацией
    # TODO: Добавить KNN импутацию для более сложных случаев
    # TODO: Добавить Rare Encoding для категориальных переменных
    
    current_df = df.copy()
    scalers = {}
    encoders = {}
    
    try:
        # Шаг 1: Импуляция NaN
        current_df = impute_numeric_nan(current_df, strategy=nan_strategy)
        
        # Шаг 2: Удаление строк
        if rows_to_drop:
            current_df = drop_rows_by_indices(current_df, rows_to_drop)
        
        # Шаг 3: Удаление колонок
        if cols_to_drop:
            current_df = drop_columns_by_names(current_df, cols_to_drop)
        
        # Шаг 4: Масштабирование
        current_df, scalers = apply_scaler(current_df, scaler_type=scaler_type)
        
        # Шаг 5: One-Hot Encoding
        current_df, encoders, _ = apply_one_hot_encoding(current_df, columns=ohe_columns)
        
    except Exception as e:
        raise Exception(f"Ошибка в пайплайне очистки: {str(e)}")
    
    return current_df, scalers, encoders
