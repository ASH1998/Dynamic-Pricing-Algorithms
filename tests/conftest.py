"""Pytest fixtures for neuroprice tests."""

from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing.

    Returns:
        DataFrame with product data including required and optional columns.
    """
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame({
        "product_id": [f"PROD_{i:03d}" for i in range(n_rows)],
        "current_price": np.random.uniform(10, 100, n_rows),
        "inventory_level": np.random.randint(0, 500, n_rows),
        "competitor_price": np.random.uniform(10, 100, n_rows),
        "day_of_week": np.random.randint(0, 7, n_rows),
        "promotion_active": np.random.choice([True, False], n_rows),
        "demand": np.random.poisson(50, n_rows),
    })


@pytest.fixture
def minimal_dataframe() -> pd.DataFrame:
    """Create a minimal DataFrame with only required columns.

    Returns:
        DataFrame with just the required columns.
    """
    return pd.DataFrame({
        "product_id": ["A", "B", "C"],
        "current_price": [49.99, 79.99, 29.99],
        "inventory_level": [100, 50, 200],
    })


@pytest.fixture
def historical_dataframe() -> pd.DataFrame:
    """Create historical data suitable for training.

    Returns:
        DataFrame with historical sales data.
    """
    np.random.seed(42)
    n_rows = 1000

    # Simulate price-demand relationship with noise
    prices = np.random.uniform(20, 80, n_rows)
    base_demand = 100
    elasticity = -1.5

    # D = D0 * (P/P_ref)^elasticity + noise
    demands = base_demand * (prices / 50) ** elasticity
    demands = demands + np.random.normal(0, 10, n_rows)
    demands = np.maximum(0, demands).astype(int)

    return pd.DataFrame({
        "product_id": [f"PROD_{i % 10:03d}" for i in range(n_rows)],
        "price": prices,
        "current_price": prices,
        "inventory_level": np.random.randint(50, 200, n_rows),
        "demand": demands,
        "day_of_week": np.random.randint(0, 7, n_rows),
        "promotion_active": np.random.choice([True, False], n_rows),
        "competitor_price": prices + np.random.uniform(-10, 10, n_rows),
    })


@pytest.fixture
def sample_strategy_dict() -> Dict:
    """Create a sample strategy configuration dictionary.

    Returns:
        Dictionary with complete strategy configuration.
    """
    return {
        "version": "1.0",
        "name": "test_strategy",
        "model": {
            "type": "rl",
            "algorithm": "PPO"
        },
        "pricing": {
            "min_price": 5.0,
            "max_price": 100.0,
            "price_step": 0.50,
            "currency": "USD"
        },
        "demand": {
            "elasticity": -1.5,
            "base_demand": 1000,
            "seasonality": {
                "enabled": False,
                "period": 7
            }
        },
        "inventory": {
            "initial_stock": 150,
            "salvage_value": 2.0,
            "deadline_days": 30
        },
        "rl_config": {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "n_steps": 64,
            "batch_size": 32,
            "training_episodes": 100  # Small for fast testing
        },
        "causal_config": {
            "treatment_variable": "price",
            "outcome_variable": "demand",
            "confounders": ["day_of_week", "promotion_active"],
            "estimation_method": "backdoor.linear_regression"
        },
        "features": {
            "required": ["product_id", "current_price", "inventory_level"],
            "optional": ["competitor_price", "day_of_week", "promotion_active"]
        },
        "output": {
            "price_column": "recommended_price",
            "confidence_column": "confidence_score"
        }
    }


@pytest.fixture
def sample_strategy_file(sample_strategy_dict: Dict, tmp_path: Path) -> Path:
    """Create a temporary strategy YAML file.

    Args:
        sample_strategy_dict: Strategy configuration dictionary
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to the created YAML file.
    """
    file_path = tmp_path / "test_strategy.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(sample_strategy_dict, f)
    return file_path


@pytest.fixture
def sample_config(sample_strategy_dict: Dict):
    """Create a StrategyConfig object from sample dictionary.

    Args:
        sample_strategy_dict: Strategy configuration dictionary

    Returns:
        StrategyConfig instance.
    """
    from neuroprice.models import StrategyConfig
    return StrategyConfig(**sample_strategy_dict)


@pytest.fixture
def invalid_yaml_file(tmp_path: Path) -> Path:
    """Create an invalid YAML file for testing error handling.

    Returns:
        Path to invalid YAML file.
    """
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text("invalid: yaml: content: [unclosed")
    return file_path


@pytest.fixture
def incomplete_strategy_file(tmp_path: Path) -> Path:
    """Create a strategy file with missing required fields.

    Returns:
        Path to incomplete YAML file.
    """
    file_path = tmp_path / "incomplete.yaml"
    content = {
        "version": "1.0",
        "name": "incomplete",
        # Missing required fields: pricing, demand, inventory, features
    }
    with open(file_path, 'w') as f:
        yaml.dump(content, f)
    return file_path


@pytest.fixture
def empty_strategy_file(tmp_path: Path) -> Path:
    """Create an empty strategy file.

    Returns:
        Path to empty YAML file.
    """
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    return file_path


# ---------------------------------------------------------------------------
# New fixtures for Wave 2 tests
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Generate synthetic data using the data_utils module.

    Returns:
        DataFrame with synthetic pricing data including confounders.
    """
    from neuroprice.data_utils import generate_synthetic_data

    return generate_synthetic_data(
        n_samples=200,
        n_products=5,
        elasticity=-1.5,
        base_demand=100.0,
        include_confounders=True,
        seed=42,
    )


@pytest.fixture
def causal_config():
    """Create a StrategyConfig suitable for causal inference testing.

    Returns:
        StrategyConfig with causal confounders configured.
    """
    from neuroprice.models import StrategyConfig

    return StrategyConfig(
        version="1.0",
        name="causal_test",
        model={"type": "causal", "algorithm": "PPO"},
        pricing={"min_price": 10.0, "max_price": 100.0, "price_step": 1.0},
        demand={
            "elasticity": -1.5,
            "base_demand": 100.0,
            "seasonality": {"enabled": False, "period": 7},
        },
        inventory={"initial_stock": 100, "salvage_value": 0.0, "deadline_days": 30},
        causal_config={
            "treatment_variable": "price",
            "outcome_variable": "demand",
            "confounders": ["day_of_week", "promotion_active", "competitor_price"],
        },
        features={
            "required": ["product_id", "current_price", "inventory_level", "demand"],
            "optional": ["day_of_week", "promotion_active", "competitor_price"],
        },
    )


@pytest.fixture
def rl_config_rich():
    """Create a StrategyConfig for rich observation RL testing.

    Returns:
        StrategyConfig with RL parameters for rich observation mode.
    """
    from neuroprice.models import StrategyConfig

    return StrategyConfig(
        version="1.0",
        name="rl_rich_test",
        model={"type": "rl", "algorithm": "PPO"},
        pricing={"min_price": 10.0, "max_price": 50.0, "price_step": 1.0},
        demand={
            "elasticity": -1.5,
            "base_demand": 300.0,
            "seasonality": {"enabled": True, "period": 7},
        },
        inventory={"initial_stock": 50, "salvage_value": 2.0, "deadline_days": 20},
        rl_config={
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "n_steps": 64,
            "batch_size": 32,
            "training_episodes": 100,
        },
        features={
            "required": ["product_id", "current_price", "inventory_level", "demand"],
            "optional": ["competitor_price"],
        },
    )


@pytest.fixture
def mock_trained_causal_estimator(causal_config, synthetic_data):
    """Create a mock CausalEstimator that is pre-fitted.

    Avoids the heavy EconML dependency by mocking the model.

    Returns:
        MagicMock configured to behave like a fitted CausalEstimator.
    """
    mock_estimator = MagicMock()
    mock_estimator.is_fitted = True
    mock_estimator._causal_effect = -2.5
    mock_estimator._effect_stderr = 0.3
    mock_estimator._n_observations = len(synthetic_data)
    mock_estimator._model_t_score = 0.6
    mock_estimator._model_y_score = 0.4
    mock_estimator._cate_std = 0.8
    mock_estimator._engineered_features = [
        "day_of_week", "promotion_active", "competitor_price",
        "day_of_week^2", "day_of_week promotion_active",
        "day_of_week competitor_price", "promotion_active^2",
    ]
    mock_estimator.config = causal_config

    # Mock estimate_prices
    n = 5
    mock_estimator.estimate_prices.return_value = (
        np.array([50.0] * n),
        np.array([0.75] * n),
    )

    # Mock effect_inference
    mock_estimator.effect_inference.return_value = {
        "effect": -2.5,
        "stderr": 0.3,
        "confidence_interval": (-3.1, -1.9),
        "pvalue": 0.001,
    }

    return mock_estimator
