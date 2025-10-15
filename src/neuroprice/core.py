"""Core module containing the main PricingEngine class."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from neuroprice.exceptions import ConfigurationError, PredictionError, TrainingError
from neuroprice.models import ModelType, StrategyConfig
from neuroprice.state import StateManager


class PricingEngine:
    """Main dynamic pricing engine combining RL and Causal Inference.

    This is the primary interface for the neuroprice library. It provides
    methods for training models and generating price recommendations.

    Key Features:
    - Train on historical data using RL, Causal Inference, or both
    - Generate price predictions that preserve input DataFrame structure
    - Save and restore configuration states for rollback

    Attributes:
        config: Strategy configuration
        is_trained: Whether the model has been trained

    Example:
        >>> from neuroprice import load_strategy, PricingEngine
        >>> config = load_strategy("strategy.yaml")
        >>> engine = PricingEngine(config)
        >>> engine.train(historical_data)
        >>> result = engine.predict(products_df)
    """

    def __init__(self, config: StrategyConfig):
        """Initialize the pricing engine with a configuration.

        Args:
            config: StrategyConfig object with all pricing parameters
        """
        self._config = config
        self._state_manager = StateManager()
        self._rl_agent: Optional[object] = None
        self._causal_estimator: Optional[object] = None
        self._is_trained = False

        # Save initial state for rollback
        self._state_manager.save_state("initial", self._config)

    @property
    def config(self) -> StrategyConfig:
        """Get the current configuration."""
        return self._config

    @property
    def is_trained(self) -> bool:
        """Check if the engine has been trained."""
        return self._is_trained

    def train(
        self,
        historical_data: pd.DataFrame,
        verbose: int = 0,
    ) -> "PricingEngine":
        """Train the pricing model on historical data.

        Depending on the model type configured:
        - 'rl': Trains a reinforcement learning agent
        - 'causal': Fits a causal inference model
        - 'hybrid': Trains both and combines predictions

        Args:
            historical_data: DataFrame with historical sales data
            verbose: Verbosity level (0=silent, 1=progress, 2=debug)

        Returns:
            Self for method chaining

        Raises:
            TrainingError: If training fails
            ValueError: If required columns are missing
        """
        # Validate DataFrame has required columns
        self._validate_dataframe(historical_data)

        model_type = self._config.model.type

        try:
            # Train RL agent if needed
            if model_type in (ModelType.RL, ModelType.HYBRID):
                from neuroprice.rl.agents import RLAgent

                self._rl_agent = RLAgent(self._config)
                self._rl_agent.train(historical_data, verbose=verbose)

                if verbose > 0:
                    print("RL agent training complete")

            # Fit causal model if needed
            if model_type in (ModelType.CAUSAL, ModelType.HYBRID):
                from neuroprice.causal.estimator import CausalEstimator

                self._causal_estimator = CausalEstimator(self._config)
                self._causal_estimator.fit(historical_data, verbose=bool(verbose))

                if verbose > 0:
                    print("Causal model fitting complete")

            self._is_trained = True

            # Save trained state
            self._state_manager.save_state("trained", self._config)

            return self

        except ImportError as e:
            raise TrainingError(
                message=f"Missing dependency: {e}",
                model_type=model_type.value,
            )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price recommendations for input DataFrame.

        IMPORTANT: This method preserves all original DataFrame columns.
        Only new columns are appended (configured price and confidence columns).
        The input DataFrame is never modified.

        Args:
            df: Input DataFrame with required feature columns

        Returns:
            New DataFrame with original columns plus:
            - price_column: Recommended price
            - confidence_column: Prediction confidence (0-1)

        Raises:
            PredictionError: If prediction fails
            ValueError: If model is not trained or columns are missing
        """
        if not self._is_trained:
            raise ValueError(
                "Engine must be trained before making predictions. "
                "Call engine.train(historical_data) first."
            )

        # Validate input DataFrame
        self._validate_dataframe(df)

        # Create a copy to avoid modifying input
        result = df.copy()

        try:
            # Get predictions based on model type
            prices, confidences = self._generate_predictions(df)

            # Append new columns (never modify existing)
            result[self._config.output.price_column] = prices
            result[self._config.output.confidence_column] = confidences

            return result

        except Exception as e:
            raise PredictionError(str(e))

    def _generate_predictions(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions based on configured model type.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (prices, confidences) arrays
        """
        model_type = self._config.model.type

        if model_type == ModelType.RL:
            return self._rl_agent.predict(df)

        elif model_type == ModelType.CAUSAL:
            return self._causal_estimator.estimate_prices(df)

        else:  # HYBRID
            # Get predictions from both models
            rl_prices, rl_conf = self._rl_agent.predict(df)
            causal_prices, causal_conf = self._causal_estimator.estimate_prices(df)

            # Weighted combination (equal weights by default)
            # Weight by confidence scores
            total_conf = rl_conf + causal_conf
            rl_weight = rl_conf / np.maximum(total_conf, 1e-6)
            causal_weight = causal_conf / np.maximum(total_conf, 1e-6)

            prices = rl_weight * rl_prices + causal_weight * causal_prices
            confidences = (rl_conf + causal_conf) / 2

            # Clamp to price bounds
            prices = np.clip(
                prices,
                self._config.pricing.min_price,
                self._config.pricing.max_price,
            )

            return prices, confidences

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = set(self._config.features.required)
        actual_cols = set(df.columns)
        missing = required_cols - actual_cols

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Required columns are: {self._config.features.required}"
            )

    def save_state(self, name: str) -> None:
        """Save current configuration state for later rollback.

        Args:
            name: Unique identifier for this state
        """
        self._state_manager.save_state(name, self._config)

    def rollback(self, state_name: str = "initial") -> "PricingEngine":
        """Rollback to a previously saved configuration state.

        Args:
            state_name: Name of the state to restore (default: "initial")

        Returns:
            Self for method chaining

        Raises:
            StateRollbackError: If the state doesn't exist
        """
        self._config = self._state_manager.restore_state(state_name)
        return self

    def list_states(self) -> List[str]:
        """List all saved state names.

        Returns:
            List of state names
        """
        return self._state_manager.list_states()

    def get_model_info(self) -> dict:
        """Get information about the current model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            "name": self._config.name,
            "model_type": self._config.model.type.value,
            "algorithm": self._config.model.algorithm.value,
            "is_trained": self._is_trained,
            "price_range": (
                self._config.pricing.min_price,
                self._config.pricing.max_price,
            ),
            "features_required": self._config.features.required,
            "output_columns": (
                self._config.output.price_column,
                self._config.output.confidence_column,
            ),
        }

    def update_config(self, **kwargs) -> "PricingEngine":
        """Update configuration parameters.

        This creates a new configuration with updated values.
        Previous configuration is preserved in state manager.

        Args:
            **kwargs: Configuration fields to update

        Returns:
            Self for method chaining
        """
        # Save current state before update
        self.save_state("before_update")

        # Create updated config dict
        config_dict = self._config.model_dump()

        # Update nested fields
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "pricing.min_price"
                parts = key.split(".")
                target = config_dict
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
            else:
                config_dict[key] = value

        # Create new config
        self._config = StrategyConfig(**config_dict)

        # Model needs retraining after config change
        self._is_trained = False

        return self

    def __repr__(self) -> str:
        """String representation of the engine."""
        return (
            f"PricingEngine(name={self._config.name!r}, "
            f"model={self._config.model.type.value}, "
            f"trained={self._is_trained})"
        )
