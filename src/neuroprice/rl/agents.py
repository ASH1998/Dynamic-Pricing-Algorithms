"""RL agent wrappers for stable-baselines3 algorithms."""

from typing import Optional, Tuple, Type

import numpy as np
import pandas as pd

from neuroprice.exceptions import TrainingError
from neuroprice.models import RLAlgorithm, StrategyConfig
from neuroprice.rl.environment import DynamicPricingEnv

# Import stable-baselines3 with graceful fallback
try:
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.base_class import BaseAlgorithm

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseAlgorithm = None  # type: ignore


class RLAgent:
    """Wrapper for stable-baselines3 RL agents.

    This class provides a unified interface for training and inference
    with different RL algorithms (PPO, A2C, DQN).

    Attributes:
        config: Strategy configuration
        model: Trained stable-baselines3 model
        env: Training environment

    Example:
        >>> agent = RLAgent(config)
        >>> agent.train(historical_data)
        >>> prices, confidences = agent.predict(products_df)
    """

    ALGORITHM_MAP: dict = {}

    def __init__(self, config: StrategyConfig):
        """Initialize the RL agent.

        Args:
            config: Strategy configuration with RL parameters

        Raises:
            ImportError: If stable-baselines3 is not installed
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for RL models. "
                "Install it with: pip install stable-baselines3"
            )

        # Initialize algorithm map after confirming import
        self.ALGORITHM_MAP = {
            RLAlgorithm.PPO: PPO,
            RLAlgorithm.A2C: A2C,
            RLAlgorithm.DQN: DQN,
        }

        self.config = config
        self.model: Optional[BaseAlgorithm] = None
        self.env: Optional[DynamicPricingEnv] = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if the agent has been trained."""
        return self._is_trained and self.model is not None

    def train(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        verbose: int = 0,
    ) -> "RLAgent":
        """Train the RL agent.

        Args:
            historical_data: Optional historical data for environment
            verbose: Verbosity level (0=none, 1=info, 2=debug)

        Returns:
            Self for method chaining

        Raises:
            TrainingError: If training fails
        """
        try:
            # Create environment
            self.env = DynamicPricingEnv(self.config, historical_data)

            # Get algorithm class
            algorithm_class: Type[BaseAlgorithm] = self.ALGORITHM_MAP[
                self.config.model.algorithm
            ]

            # Configure model parameters
            model_kwargs = {
                "policy": "MlpPolicy",
                "env": self.env,
                "learning_rate": self.config.rl_config.learning_rate,
                "gamma": self.config.rl_config.gamma,
                "verbose": verbose,
            }

            # Add algorithm-specific parameters
            if self.config.model.algorithm != RLAlgorithm.DQN:
                model_kwargs["n_steps"] = self.config.rl_config.n_steps

            # DQN uses buffer_size instead of n_steps
            if self.config.model.algorithm == RLAlgorithm.DQN:
                model_kwargs["buffer_size"] = 10000
                model_kwargs["batch_size"] = self.config.rl_config.batch_size

            # Create and train model
            self.model = algorithm_class(**model_kwargs)
            self.model.learn(total_timesteps=self.config.rl_config.training_episodes)

            self._is_trained = True
            return self

        except Exception as e:
            raise TrainingError(
                message=str(e),
                model_type=self.config.model.algorithm.value,
            )

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate price predictions for input DataFrame.

        Args:
            df: Input DataFrame with required features

        Returns:
            Tuple of (prices array, confidences array)

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")

        prices = []
        confidences = []

        for _, row in df.iterrows():
            obs = self._row_to_observation(row)
            action, _states = self.model.predict(obs, deterministic=True)

            # Convert action to price
            price = self._action_to_price(int(action))
            prices.append(price)

            # Compute confidence based on action probability
            confidence = self._compute_confidence(obs)
            confidences.append(confidence)

        return np.array(prices), np.array(confidences)

    def predict_single(self, observation: np.ndarray) -> Tuple[float, float]:
        """Predict price for a single observation.

        Args:
            observation: Normalized observation array

        Returns:
            Tuple of (price, confidence)
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")

        action, _states = self.model.predict(observation, deterministic=True)
        price = self._action_to_price(int(action))
        confidence = self._compute_confidence(observation)

        return price, confidence

    def _row_to_observation(self, row: pd.Series) -> np.ndarray:
        """Convert DataFrame row to environment observation.

        Args:
            row: Single row from input DataFrame

        Returns:
            Normalized observation array
        """
        # Extract inventory level (default to initial if not present)
        inventory = row.get("inventory_level", self.config.inventory.initial_stock)

        # Normalize inventory
        norm_inventory = (
            inventory / self.config.inventory.initial_stock
            if self.config.inventory.initial_stock > 0
            else 0.5
        )

        # Default time position (middle of period)
        norm_time = 0.5

        return np.array([norm_inventory, norm_time], dtype=np.float32)

    def _action_to_price(self, action: int) -> float:
        """Convert action index to price.

        Args:
            action: Discrete action index

        Returns:
            Actual price value
        """
        return self.config.pricing.min_price + action * self.config.pricing.price_step

    def _compute_confidence(self, observation: np.ndarray) -> float:
        """Compute prediction confidence.

        For now, uses a simple heuristic based on observation values.
        Higher inventory and more time remaining = higher confidence.

        Args:
            observation: Current observation

        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence heuristic
        norm_inventory = observation[0]
        norm_time = observation[1]

        # Higher confidence when we have more inventory and time
        base_confidence = 0.7
        inventory_bonus = 0.15 * norm_inventory
        time_penalty = 0.15 * norm_time  # Less confident as time runs out

        confidence = base_confidence + inventory_bonus - time_penalty
        return max(0.0, min(1.0, confidence))

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str) -> "RLAgent":
        """Load a trained model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Self for method chaining
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required to load models")

        algorithm_class = self.ALGORITHM_MAP[self.config.model.algorithm]
        self.env = DynamicPricingEnv(self.config)
        self.model = algorithm_class.load(path, env=self.env)
        self._is_trained = True

        return self
