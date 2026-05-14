import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
from datetime import datetime

from utils.model_config import create_model_pipeline, get_grid_search_params


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Расчет метрик качества: MAE, MSE, R²."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    use_auto_tuning: bool = False
) -> Tuple[Any, Dict[str, float], List[float]]:
    """
    Обучение модели с опциональным GridSearch.
    
    Args:
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные (для early stopping или оценки)
        config: Конфигурация модели
        use_auto_tuning: Если True, используется GridSearchCV
        
    Returns:
        model: Обученная модель
        metrics: Словарь метрик на валидации
        loss_history: История потерь (для MLP, иначе пустой список)
    """
    if use_auto_tuning:
        base_model = create_model_pipeline(config)
        param_grid = get_grid_search_params(config.get("type", ""))
        
        if not param_grid:
            # Если сетка пуста, обучаем без tuning
            model = base_model
            model.fit(X_train, y_train)
        else:
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
    else:
        model = create_model_pipeline(config)
        model.fit(X_train, y_train)
    
    # Предсказание на валидации для метрик
    y_pred_val = model.predict(X_val)
    metrics = calculate_metrics(y_val, y_pred_val)
    
    # История loss (эмуляция для MLP через partial_fit или просто пустой список)
    # В sklearn MLPRegressor не возвращает историю явно при fit(), 
    # но мы можем извлечь её, если обучать вручную или использовать callback-подобную логику.
    # Для упрощения вернем пустой список или заглушку.
    loss_history = []
    if hasattr(model, 'loss_curve_') and model.loss_curve_:
        loss_history = list(model.loss_curve_)
        
    return model, metrics, loss_history


def evaluate_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, float]:
    """Финальная оценка на тестовой выборке."""
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred)


def save_model_artifacts(
    model: Any, 
    metrics: Dict[str, float], 
    config: Dict[str, Any],
    feature_names: List[str],
    target_names: List[str],
    output_dir: str = "models"
) -> str:
    """
    Сохраняет модель, метрики и конфиг в файлы.
    
    Имя файла: model_{type}_{timestamp}.pkl
    Возвращает путь к сохраненному файлу.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config.get("type", "unknown").lower().replace("regressor", "").replace(" ", "_")
    filename = f"model_{model_type}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Упаковка артефактов
    artifacts = {
        "model": model,
        "metrics": metrics,
        "config": config,
        "feature_names": feature_names,
        "target_names": target_names,
        "timestamp": timestamp
    }
    
    joblib.dump(artifacts, filepath)
    return filepath
