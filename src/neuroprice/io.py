"""IO module for loading and parsing strategy files."""

from pathlib import Path
from typing import Union

import yaml

from neuroprice.exceptions import StrategyFileNotFoundError, StrategyValidationError
from neuroprice.models import StrategyConfig


def load_strategy(file_path: Union[str, Path]) -> StrategyConfig:
    """Load and validate a pricing strategy from a YAML file.

    This function reads a YAML configuration file and validates it
    against the StrategyConfig schema using Pydantic.

    Args:
        file_path: Path to the YAML strategy file

    Returns:
        StrategyConfig: Validated configuration object

    Raises:
        StrategyFileNotFoundError: If the file does not exist.
            The error message includes helpful guidance on how to
            create a strategy file.
        StrategyValidationError: If the file content is invalid.
            This includes YAML syntax errors and schema validation failures.

    Example:
        >>> config = load_strategy("my_strategy.yaml")
        >>> print(config.name)
        'retail_pricing'
        >>> print(config.pricing.min_price)
        10.0
    """
    path = Path(file_path)

    # Check if file exists and provide helpful error message
    if not path.exists():
        raise StrategyFileNotFoundError(file_path=str(path))

    # Parse YAML content
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise StrategyValidationError(
            message=f"Invalid YAML syntax: {e}",
            field=None,
        )

    # Handle empty file
    if raw_config is None:
        raise StrategyValidationError(
            message="Strategy file is empty",
            field=None,
        )

    # Validate against Pydantic schema
    try:
        config = StrategyConfig(**raw_config)
    except Exception as e:
        # Extract field information from Pydantic error if available
        error_msg = str(e)
        raise StrategyValidationError(
            message=error_msg,
            field=None,
        )

    return config


def save_strategy(config: StrategyConfig, file_path: Union[str, Path]) -> None:
    """Save a strategy configuration to a YAML file.

    Args:
        config: StrategyConfig object to save
        file_path: Path where to save the YAML file

    Example:
        >>> save_strategy(config, "my_strategy.yaml")
    """
    path = Path(file_path)

    # Convert to dictionary (Pydantic v2)
    config_dict = config.model_dump(mode="json")

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def validate_strategy_dict(config_dict: dict) -> StrategyConfig:
    """Validate a strategy configuration dictionary.

    This is useful when you have configuration data from a source
    other than a YAML file (e.g., from an API or database).

    Args:
        config_dict: Dictionary containing strategy configuration

    Returns:
        StrategyConfig: Validated configuration object

    Raises:
        StrategyValidationError: If the configuration is invalid
    """
    try:
        return StrategyConfig(**config_dict)
    except Exception as e:
        raise StrategyValidationError(
            message=str(e),
            field=None,
        )
