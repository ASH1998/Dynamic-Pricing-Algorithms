"""NeuroPrize: Dynamic Pricing Engine using Reinforcement Learning and Causal Inference.

This library provides tools for building intelligent dynamic pricing systems
that adapt to market conditions using machine learning techniques.

Main Components:
    - PricingEngine: The main interface for training and prediction
    - load_strategy: Load pricing strategy from YAML configuration
    - StrategyConfig: Pydantic model for configuration validation

Quick Start:
    >>> from neuroprice import load_strategy, PricingEngine
    >>>
    >>> # Load strategy configuration
    >>> config = load_strategy("my_strategy.yaml")
    >>>
    >>> # Create and train engine
    >>> engine = PricingEngine(config)
    >>> engine.train(historical_data)
    >>>
    >>> # Generate price recommendations
    >>> result = engine.predict(products_df)

For more information, see:
    - README.md: Installation and usage guide
    - templates/strategy_template.yaml: Example strategy configuration
"""

from neuroprice.core import PricingEngine
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
from neuroprice.models import (
    CausalConfig,
    DemandConfig,
    FeaturesConfig,
    InventoryConfig,
    ModelConfig,
    ModelType,
    OutputConfig,
    PricingConfig,
    RLAlgorithm,
    RLConfig,
    SeasonalityConfig,
    StrategyConfig,
)
from neuroprice.state import StateManager

__version__ = "0.1.0"
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
