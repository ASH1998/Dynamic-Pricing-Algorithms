"""NeuroPrize: Dynamic Pricing Engine using Reinforcement Learning and Causal Inference.

This library provides tools for building intelligent dynamic pricing systems
that adapt to market conditions using machine learning techniques.

Main Components:
    - PricingEngine: The main interface for training and prediction
    - load_strategy: Load pricing strategy from YAML configuration
    - StrategyConfig: Pydantic model for configuration validation

Data Utilities:
    - validate_input_data: Validate and clean input DataFrames
    - detect_outliers: IQR-based outlier detection
    - fill_missing_values: NaN filling with median/mean/zero
    - generate_synthetic_data: Create realistic test data
    - split_train_test: Random or time-aware train/test splitting

Metrics:
    - mae, rmse, mape: Standard regression metrics
    - revenue_comparison: Compare revenue between pricing strategies
    - price_distribution_summary: Summary statistics for price arrays

Quick Start:
    >>> from neuroprice import load_strategy, PricingEngine
    >>>
    >>> # Load strategy configuration
    >>> config = load_strategy("my_strategy.yaml")
    >>>
    >>> # Create and train engine
    >>> with PricingEngine(config) as engine:
    ...     engine.train(historical_data)
    ...     result = engine.predict(products_df)
    ...     metrics = engine.evaluate(test_df)
    ...     engine.save_model("models/my_engine")

For more information, see:
    - README.md: Installation and usage guide
    - templates/strategy_template.yaml: Example strategy configuration
"""

from neuroprice.core import PricingEngine
from neuroprice.data_utils import (
    detect_outliers,
    fill_missing_values,
    generate_synthetic_data,
    split_train_test,
    validate_input_data,
)
from neuroprice.exceptions import (
    ConfigurationError,
    NeuroPriceError,
    PredictionError,
    StateRollbackError,
    StrategyFileNotFoundError,
    StrategyValidationError,
    TrainingError,
)
from neuroprice.io import load_strategy, save_strategy, validate_strategy_dict
from neuroprice.metrics import (
    mae,
    mape,
    price_distribution_summary,
    revenue_comparison,
    rmse,
)
from neuroprice.models import (
    CausalConfig,
    DemandConfig,
    FeaturesConfig,
    InventoryConfig,
    ModelConfig,
    ModelType,
    OutputConfig,
    PreprocessingConfig,
    PricingConfig,
    RLAlgorithm,
    RLConfig,
    SeasonalityConfig,
    StrategyConfig,
)
from neuroprice.state import StateManager

__version__ = "1.1.0"
__author__ = "Dynamic Pricing Team"

__all__ = [
    # Main interface
    "PricingEngine",
    "load_strategy",
    "save_strategy",
    "validate_strategy_dict",
    # Configuration models
    "StrategyConfig",
    "ModelConfig",
    "ModelType",
    "RLAlgorithm",
    "PricingConfig",
    "DemandConfig",
    "SeasonalityConfig",
    "InventoryConfig",
    "RLConfig",
    "CausalConfig",
    "FeaturesConfig",
    "OutputConfig",
    "PreprocessingConfig",
    # Data utilities
    "validate_input_data",
    "detect_outliers",
    "fill_missing_values",
    "generate_synthetic_data",
    "split_train_test",
    # Metrics
    "mae",
    "rmse",
    "mape",
    "revenue_comparison",
    "price_distribution_summary",
    # State management
    "StateManager",
    # Exceptions
    "NeuroPriceError",
    "StrategyFileNotFoundError",
    "StrategyValidationError",
    "StateRollbackError",
    "TrainingError",
    "PredictionError",
    "ConfigurationError",
    # Version
    "__version__",
]
