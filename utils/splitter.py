import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split


def validate_target_columns(df: pd.DataFrame, y_cols: List[str]) -> None:
    """Проверяет, что целевые колонки существуют и являются числовыми."""
    for col in y_cols:
        if col not in df.columns:
            raise ValueError(f"Целевая колонка '{col}' не найдена в данных.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Попытка конвертации
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                raise ValueError(f"Целевая колонка '{col}' не является числовой и не может быть конвертирована.")


def validate_feature_columns(df: pd.DataFrame, x_cols: List[str]) -> None:
    """Проверяет наличие колонок признаков."""
    missing = [col for col in x_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Колонки признаков не найдены: {missing}")


def split_data(
    df: pd.DataFrame,
    y_cols: List[str],
    x_cols: Optional[List[str]] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, any]:
    """
    Разделяет данные на train/val/test.
    
    Логика:
    1. Валидация пропорций (сумма должна быть ~1.0).
    2. Если X не задан, выбираются все числовые колонки кроме Y.
    3. Двухэтапный split: сначала отделяем test, потом от остатка val.
    
    Returns:
        Словарь с X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # 1. Валидация пропорций
    total = train_size + val_size + test_size
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Сумма пропорций ({total}) должна быть равна 1.0. Текущие: {train_size}, {val_size}, {test_size}")
    
    if len(df) < 10:
        raise ValueError("Недостаточно данных для разделения (минимум 10 строк).")

    # 2. Определение X
    if x_cols is None or not x_cols:
        # Авто-выбор: все числовые, кроме target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_cols = [col for col in numeric_cols if col not in y_cols]
        
        if not x_cols:
            raise ValueError("Не удалось автоматически выбрать признаки. Укажите колонки вручную.")

    # 3. Проверка колонок
    validate_target_columns(df, y_cols)
    validate_feature_columns(df, x_cols)

    # Подготовка данных (удаление NaN)
    subset_cols = x_cols + y_cols
    df_clean = df[subset_cols].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("После удаления пропусков в выбранных колонках не осталось данных.")

    X = df_clean[x_cols].values
    y = df_clean[y_cols].values

    # 4. Двухэтапное разделение
    # Сначала отделяем TEST
    temp_X, X_test, temp_y, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Затем от оставшегося отделяем VAL (относительно нового размера)
    # val_ratio = val_size / (train_size + val_size)
    remaining_total = train_size + val_size
    val_ratio = val_size / remaining_total
    
    X_train, X_val, y_train, y_val = train_test_split(
        temp_X, temp_y,
        test_size=val_ratio,
        random_state=random_state
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": x_cols,
        "target_names": y_cols
    }
