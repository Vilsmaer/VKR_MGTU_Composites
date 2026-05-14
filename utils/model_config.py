from typing import Dict, Any, List, Optional
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def get_default_mlp_config() -> Dict[str, Any]:
    """Конфигурация по умолчанию для MLP."""
    return {
        "type": "MLPRegressor",
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,  # L2 регуляризация
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "max_iter": 200,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42
    }


def get_default_poly_config() -> Dict[str, Any]:
    """Конфигурация по умолчанию для Polynomial Regression."""
    return {
        "type": "PolynomialRegression",
        "degree": 2,
        "include_bias": False,
        "alpha": 1.0,  # Ridge regularization
        "random_state": 42
    }


def create_model_pipeline(config: Dict[str, Any]) -> Any:
    """
    Создает sklearn-пайплайн на основе конфига.
    
    Поддерживает:
    - MLPRegressor
    - PolynomialRegression (через PolynomialFeatures + Ridge)
    """
    model_type = config.get("type")
    
    if model_type == "MLPRegressor":
        return MLPRegressor(
            hidden_layer_sizes=config.get("hidden_layer_sizes", (100,)),
            activation=config.get("activation", "relu"),
            solver=config.get("solver", "adam"),
            alpha=config.get("alpha", 0.0001),
            learning_rate=config.get("learning_rate", "constant"),
            learning_rate_init=config.get("learning_rate_init", 0.001),
            max_iter=config.get("max_iter", 200),
            early_stopping=config.get("early_stopping", True),
            validation_fraction=config.get("validation_fraction", 0.1),
            random_state=config.get("random_state", 42),
            verbose=False
        )
        
    elif model_type == "PolynomialRegression":
        degree = config.get("degree", 2)
        alpha = config.get("alpha", 1.0)
        
        # Полиномиальная регрессия реализуется как пайплайн: PolyFeatures -> Ridge
        pipeline = Pipeline([
            ("poly_features", PolynomialFeatures(
                degree=degree, 
                include_bias=config.get("include_bias", False)
            )),
            ("ridge", Ridge(alpha=alpha, random_state=config.get("random_state", 42)))
        ])
        return pipeline
        
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


def get_grid_search_params(model_type: str) -> Dict[str, List[Any]]:
    """
    Возвращает сетку параметров для GridSearchCV.
    """
    if model_type == "MLPRegressor":
        return {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01]
        }
    elif model_type == "PolynomialRegression":
        return {
            "poly_features__degree": [2, 3, 4],
            "ridge__alpha": [0.1, 1.0, 10.0]
        }
    else:
        return {}
