"""Custom exceptions for the neuroprice library."""

from typing import List, Optional


class NeuroPriceError(Exception):
    """Base exception for neuroprice library."""

    pass


class StrategyFileNotFoundError(NeuroPriceError):
    """Raised when strategy file cannot be found.

    Provides helpful guidance on how to create a strategy file.
    """

    def __init__(self, file_path: str, message: Optional[str] = None):
        self.file_path = file_path
        if message is None:
            self.message = (
                f"Strategy file not found: {file_path}\n\n"
                "Please ensure the file exists at the specified path.\n"
                "You can create a new strategy file using the template:\n\n"
                "  1. Copy the template: cp templates/strategy_template.yaml your_strategy.yaml\n"
                "  2. Edit the YAML file with your pricing parameters\n"
                "  3. Load it with: config = neuroprice.load_strategy('your_strategy.yaml')\n\n"
                "See README.md for more details on strategy file format."
            )
        else:
            self.message = message
        super().__init__(self.message)


class StrategyValidationError(NeuroPriceError):
    """Raised when strategy file content is invalid.

    This can occur due to:
    - Invalid YAML syntax
    - Missing required fields
    - Invalid field values
    - Type mismatches
    """

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        if field:
            self.message = f"Validation error in field '{field}': {message}"
        else:
            self.message = f"Strategy validation failed: {message}"
        super().__init__(self.message)


class StateRollbackError(NeuroPriceError):
    """Raised when state rollback fails.

    This occurs when attempting to restore a state that doesn't exist.
    """

    def __init__(self, state_name: str, available_states: Optional[List[str]] = None):
        self.state_name = state_name
        self.available_states = available_states or []
        if self.available_states:
            states_str = ", ".join(f"'{s}'" for s in self.available_states)
            self.message = (
                f"State '{state_name}' not found. "
                f"Available states: {states_str}"
            )
        else:
            self.message = f"State '{state_name}' not found. No states have been saved."
        super().__init__(self.message)


class TrainingError(NeuroPriceError):
    """Raised when model training fails."""

    def __init__(self, message: str, model_type: Optional[str] = None):
        self.model_type = model_type
        if model_type:
            self.message = f"Training failed for {model_type} model: {message}"
        else:
            self.message = f"Training failed: {message}"
        super().__init__(self.message)


class PredictionError(NeuroPriceError):
    """Raised when prediction fails."""

    def __init__(self, message: str):
        self.message = f"Prediction failed: {message}"
        super().__init__(self.message)


class ConfigurationError(NeuroPriceError):
    """Raised when there's a configuration issue."""

    def __init__(self, message: str):
        self.message = f"Configuration error: {message}"
        super().__init__(self.message)
