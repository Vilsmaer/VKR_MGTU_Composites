"""
Модуль прогнозирования (Inference).
Загрузка артефактов, предобработка новых данных и инференс.
"""
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

def load_artifacts(file_path: str) -> Dict[str, Any]:
    """Загружает сохранённые артефакты модели (.pkl)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    return joblib.load(file_path)

def preprocess_new_data(
    df_raw: pd.DataFrame,
    scalers: Dict[str, Any],
    encoders: Dict[str, Any],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Применяет сохранённые трансформации к новым данным и выравнивает колонки 
    в точном порядке, ожидаемом моделью.
    """
    df = df_raw.copy()

    # 1. Масштабирование числовых колонок
    for col, scaler in scalers.items():
        if col in df.columns:
            # flatten() нужен для избежания Shape mismatch в pandas
            df[col] = scaler.transform(df[[col]]).flatten()

    # 2. One-Hot Encoding категориальных колонок
    for orig_col, encoder in encoders.items():
        if orig_col in df.columns:
            encoded = encoder.transform(df[[orig_col]].astype(str))
            new_cols = [f"{orig_col}_{cat}" for cat in encoder.get_feature_names_out([orig_col])]
            enc_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
            df = df.drop(columns=[orig_col])
            df = pd.concat([df, enc_df], axis=1)

    # 3. Выравнивание с feature_names (порядок + безопасное заполнение отсутствующих)
    aligned_df = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            aligned_df[col] = df[col]
        else:
            # Если модель ожидает колонку, которой нет в новых данных, заполняем 0
            aligned_df[col] = 0.0

    # Возвращаем строго в порядке обучения
    return aligned_df[feature_names]

def predict_batch(model, X_processed: pd.DataFrame) -> np.ndarray:
    """Пакетное предсказание"""
    return model.predict(X_processed)