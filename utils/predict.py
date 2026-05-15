"""
Модуль прогнозирования (Inference).
Загружает последнюю модель, предобрабатывает новые данные и возвращает предсказания.
"""
import pandas as pd
import numpy as np
import joblib
import os
import glob
from typing import Dict, Any, Tuple, Optional
import warnings

def get_latest_model_path(models_dir: str = "models") -> Optional[str]:
    """Находит самый последний .pkl файл в директории."""
    if not os.path.exists(models_dir):
        return None
    files = glob.glob(os.path.join(models_dir, "*.pkl"))
    return max(files, key=os.path.getmtime) if files else None

def load_latest_artifact(models_dir: str = "models") -> Tuple[Dict[str, Any], str]:
    """Загружает артефакт последней модели."""
    path = get_latest_model_path(models_dir)
    if path is None:
        raise FileNotFoundError("В папке 'models' не найдено обученных моделей. Сначала обучите модель.")
    return joblib.load(path), path

def preprocess_for_prediction(
    df_input: pd.DataFrame,
    feature_names: list,
    scalers: Optional[Dict[str, Any]] = None,
    encoders: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Применяет скалеры/энкодеры и выравнивает колонки под модель."""
    df = df_input.copy()
    if scalers:
        for col, scaler in scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]]).flatten()
    if encoders:
        for col, encoder in encoders.items():
            if col in df.columns:
                encoded = encoder.transform(df[[col]].astype(str))
                new_cols = [f"{col}_{cat}" for cat in encoder.get_feature_names_out([col])]
                df = df.drop(columns=[col])
                df = pd.concat([df, pd.DataFrame(encoded, columns=new_cols, index=df.index)], axis=1)

    aligned_df = pd.DataFrame(index=df.index)
    for col in feature_names:
        aligned_df[col] = df[col] if col in df.columns else 0.0
    return aligned_df[feature_names]

def predict_from_latest_model(
    input_data: pd.DataFrame,
    models_dir: str = "models",
    session_scalers: Optional[Dict] = None,
    session_encoders: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Основной пайплайн: загрузка модели -> предобработка -> прогноз."""
    artifact, model_path = load_latest_artifact(models_dir)
    model = artifact["model"]
    feature_names = artifact.get("feature_names", [])
    target_names = artifact.get("target_names", ["Target"])
    config = artifact.get("config", {})
    
    scalers = artifact.get("scalers", session_scalers)
    encoders = artifact.get("encoders", session_encoders)
    
    X_processed = preprocess_for_prediction(input_data, feature_names, scalers, encoders)
    predictions = model.predict(X_processed)
    pred_array = predictions if predictions.ndim > 1 else predictions.reshape(-1, 1)
    
    result_df = input_data.copy()
    for i, t_name in enumerate(target_names):
        result_df[f"Прогноз_{t_name}"] = pred_array[:, i]
        
    return result_df, {
        "model_path": os.path.basename(model_path),
        "model_type": config.get("type", "unknown"),
        "targets_predicted": target_names
    }