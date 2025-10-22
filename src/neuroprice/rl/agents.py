"""RL agent wrappers for stable-baselines3 algorithms.

Provides a unified interface for training, inference, and confidence
estimation using PPO, A2C, or DQN backends from stable-baselines3.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from neuroprice.exceptions import TrainingError
from neuroprice.models import RLAlgorithm, StrategyConfig
from neuroprice.rl.environment import (
    DynamicPricingEnv,
    LEGACY_FEATURE_NAMES,
    RICH_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional stable-baselines3 import
# ---------------------------------------------------------------------------
try:
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.base_class import BaseAlgorithm

    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    SB3_AVAILABLE = False
    BaseAlgorithm = None  # type: ignore[assignment,misc]


class RLAgent:
    """Wrapper for stable-baselines3 RL agents.

    Supports PPO, A2C and DQN with:

    * **Entropy / Q-value based confidence** instead of simple heuristics.
    * **Batch prediction** via vectorised observations.
    * **Native save / load** through stable-baselines3.
    * **Feature importance** stub for model interpretability.

    Attributes:
        config: Strategy configuration.
        model: Trained stable-baselines3 model.
        env: Training environment instance.

    Example:
        >>> agent = RLAgent(config)
        >>> agent.train(historical_data)
        >>> prices, confidences = agent.predict(products_df)
    """

    ALGORITHM_MAP: Dict[RLAlgorithm, Type[Any]] = {}

    def __init__(self, config: StrategyConfig, rich_observation: bool = True):
        """Initialize the RL agent.

        Args:
            config: Strategy configuration with RL parameters.
            rich_observation: Whether to use the 7-feature (True) or
                2-feature (False) observation space.

        Raises:
            ImportError: If stable-baselines3 is not installed.
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for RL models. "
                "Install it with: pip install stable-baselines3"
            )

        # Build algorithm map after confirming the import succeeded
        self.ALGORITHM_MAP = {
            RLAlgorithm.PPO: PPO,
            RLAlgorithm.A2C: A2C,
            RLAlgorithm.DQN: DQN,
        }

        self.config = config
        self.rich_observation = rich_observation
        self.model: Optional[BaseAlgorithm] = None
        self.env: Optional[DynamicPricingEnv] = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether the agent has been trained and a model is available."""
        return self._is_trained and self.model is not None

    @property
    def feature_names(self) -> List[str]:
        """Names of observation features in the current mode."""
        return RICH_FEATURE_NAMES if self.rich_observation else LEGACY_FEATURE_NAMES

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        verbose: int = 0,
    ) -> "RLAgent":
        """Train the RL agent.

        Args:
            historical_data: Optional historical data for environment.
            verbose: Verbosity level (0=none, 1=info, 2=debug).

        Returns:
            Self for method chaining.

        Raises:
            TrainingError: If training fails.
        """
        try:
            self.env = DynamicPricingEnv(
                self.config,
                historical_data,
                rich_observation=self.rich_observation,
            )

            algorithm_class: Type[BaseAlgorithm] = self.ALGORITHM_MAP[
                self.config.model.algorithm
            ]

            model_kwargs: Dict[str, Any] = {
                "policy": "MlpPolicy",
                "env": self.env,
                "learning_rate": self.config.rl_config.learning_rate,
                "gamma": self.config.rl_config.gamma,
                "verbose": verbose,
            }

            # Algorithm-specific kwargs
            if self.config.model.algorithm != RLAlgorithm.DQN:
                model_kwargs["n_steps"] = self.config.rl_config.n_steps

            if self.config.model.algorithm == RLAlgorithm.DQN:
                model_kwargs["buffer_size"] = 10000
                model_kwargs["batch_size"] = self.config.rl_config.batch_size

            self.model = algorithm_class(**model_kwargs)
            self.model.learn(total_timesteps=self.config.rl_config.training_episodes)

            self._is_trained = True
            return self

        except Exception as e:
            raise TrainingError(
                message=str(e),
                model_type=self.config.model.algorithm.value,
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate price predictions for an input DataFrame.

        Uses vectorised observation construction instead of ``iterrows`` when
        possible.

        Args:
            df: Input DataFrame with required features.

        Returns:
            Tuple of (prices array, confidences array).

        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")

        # Build observation matrix in one pass
        obs_matrix = self._batch_to_observations(df)

        # Vectorised action prediction
        actions, _ = self.model.predict(obs_matrix, deterministic=True)
        prices = np.array(
            [self._action_to_price(int(a)) for a in actions], dtype=np.float64
        )

        # Batch confidence estimation
        confidences = self._batch_confidence(obs_matrix)

        return prices, confidences

    def predict_single(self, observation: np.ndarray) -> Tuple[float, float]:
        """Predict price for a single observation.

        Args:
            observation: Normalized observation array.

        Returns:
            Tuple of (price, confidence).
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")

        action, _states = self.model.predict(observation, deterministic=True)
        price = self._action_to_price(int(action))
        confidence = self._compute_confidence(observation)

        return price, confidence

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    def _compute_confidence(self, observation: np.ndarray) -> float:
        """Compute prediction confidence from the policy's action distribution.

        Strategy depends on the algorithm family:

        * **PPO / A2C**: Uses the value-function estimate as a proxy for
          how favourable the current state is.  The value is passed through
          a sigmoid to map it to (0, 1).
        * **DQN**: Uses the normalised Q-value spread
          ``(max Q − mean Q) / (max |Q| + ε)`` which measures how *decisive*
          the greedy action is.

        Falls back to a simple heuristic when the required internals are
        unavailable.

        Args:
            observation: Current observation (1-D array).

        Returns:
            Confidence score in [0, 1].
        """
        obs = observation.reshape(1, -1) if observation.ndim == 1 else observation

        try:
            algo = self.config.model.algorithm

            if algo in (RLAlgorithm.PPO, RLAlgorithm.A2C):
                return self._value_confidence(obs)
            elif algo == RLAlgorithm.DQN:
                return self._qvalue_confidence(obs)
        except Exception as exc:  # pragma: no cover
            logger.debug("Confidence estimation fell back to heuristic: %s", exc)

        # Fallback heuristic
        return self._heuristic_confidence(observation)

    def _value_confidence(self, obs: np.ndarray) -> float:
        """Confidence from the value-function estimate (PPO / A2C)."""
        # Access the value network through the policy
        import torch as th  # local import to avoid hard dep at module level

        with th.no_grad():
            obs_tensor = th.as_tensor(obs, device=self.model.device).float()
            # stable-baselines3 stores value net inside policy
            value = self.model.policy.predict_values(obs_tensor)
            value_scalar = float(value.item())

        # Sigmoid squashing to (0, 1)
        confidence = 1.0 / (1.0 + np.exp(-value_scalar))
        return float(np.clip(confidence, 0.0, 1.0))

    def _qvalue_confidence(self, obs: np.ndarray) -> float:
        """Confidence from Q-value spread (DQN).

        Computes ``(max Q − mean Q) / (max |Q| + ε)``.  A large spread
        means the agent is very confident in one action over the rest.
        """
        import torch as th

        with th.no_grad():
            obs_tensor = th.as_tensor(obs, device=self.model.device).float()
            q_values = self.model.q_net(obs_tensor)
            q_np = q_values.cpu().numpy().flatten()

        q_max = float(np.max(q_np))
        q_mean = float(np.mean(q_np))
        q_abs_max = float(np.max(np.abs(q_np))) + 1e-8

        spread = (q_max - q_mean) / q_abs_max
        return float(np.clip(spread, 0.0, 1.0))

    @staticmethod
    def _heuristic_confidence(observation: np.ndarray) -> float:
        """Fallback confidence heuristic.

        Higher inventory and more time remaining → higher confidence.
        """
        norm_inventory = float(observation[0])
        norm_time = float(observation[1]) if len(observation) > 1 else 0.5

        base = 0.7
        inventory_bonus = 0.15 * norm_inventory
        time_penalty = 0.15 * norm_time

        return float(np.clip(base + inventory_bonus - time_penalty, 0.0, 1.0))

    def _batch_confidence(self, obs_matrix: np.ndarray) -> np.ndarray:
        """Compute confidence for a batch of observations.

        Args:
            obs_matrix: (N, n_features) array.

        Returns:
            1-D array of confidence scores.
        """
        confidences = np.empty(obs_matrix.shape[0], dtype=np.float64)
        for i in range(obs_matrix.shape[0]):
            confidences[i] = self._compute_confidence(obs_matrix[i])
        return confidences

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _row_to_observation(self, row: pd.Series) -> np.ndarray:
        """Convert a DataFrame row to an environment-compatible observation.

        Handles both the legacy 2-feature and rich 7-feature sets.

        Args:
            row: Single row from input DataFrame.

        Returns:
            Normalized observation array.
        """
        inventory = row.get("inventory_level", self.config.inventory.initial_stock)
        initial = self.config.inventory.initial_stock
        norm_inventory = inventory / initial if initial > 0 else 0.5
        norm_time = float(row.get("norm_time", 0.5))

        if not self.rich_observation:
            return np.array([norm_inventory, norm_time], dtype=np.float32)

        # Rich features — extract with sensible defaults
        price_range = self.config.pricing.max_price - self.config.pricing.min_price
        ref_price = (self.config.pricing.min_price + self.config.pricing.max_price) / 2.0

        raw_price = row.get("price", ref_price)
        if price_range > 0:
            norm_price = (raw_price - self.config.pricing.min_price) / price_range
            price_gap = (raw_price - ref_price) / price_range
        else:
            norm_price = 0.5
            price_gap = 0.0

        base_demand_per_period = self.config.demand.base_demand / max(
            self.config.inventory.deadline_days or 30, 1
        )
        demand_mean = row.get("recent_demand_mean", base_demand_per_period)
        demand_std = row.get("recent_demand_std", 0.0)
        demand_trend = demand_mean / max(base_demand_per_period, 1e-6)
        demand_volatility = demand_std / max(base_demand_per_period, 1e-6)

        inv_rate = (initial - inventory) / initial if initial > 0 else 0.0

        return np.array(
            [norm_inventory, norm_time, norm_price, demand_trend,
             price_gap, inv_rate, demand_volatility],
            dtype=np.float32,
        )

    def _batch_to_observations(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised observation construction for a full DataFrame.

        Falls back to row-by-row when rich features require per-row logic
        that cannot be cleanly vectorised (e.g. missing columns).

        Args:
            df: Input DataFrame.

        Returns:
            (N, n_features) float32 array.
        """
        # Try fast vectorised path for legacy mode
        if not self.rich_observation:
            inventory = df.get("inventory_level", pd.Series(
                self.config.inventory.initial_stock, index=df.index
            ))
            initial = self.config.inventory.initial_stock
            norm_inv = (inventory / initial).clip(0, 1).astype(np.float32)
            norm_t = df.get("norm_time", pd.Series(0.5, index=df.index)).astype(np.float32)
            return np.column_stack([norm_inv.values, norm_t.values])

        # Rich mode — build column by column where possible
        initial = self.config.inventory.initial_stock
        price_range = self.config.pricing.max_price - self.config.pricing.min_price
        ref_price = (self.config.pricing.min_price + self.config.pricing.max_price) / 2.0
        max_steps = self.config.inventory.deadline_days or 30
        base_demand_pp = self.config.demand.base_demand / max(max_steps, 1)

        inventory = df.get("inventory_level", pd.Series(initial, index=df.index))
        norm_inv = (inventory / initial).clip(0, 1).astype(np.float32).values if initial > 0 \
            else np.full(len(df), 0.5, dtype=np.float32)

        norm_t = df.get("norm_time", pd.Series(0.5, index=df.index)).astype(np.float32).values

        raw_price = df.get("price", pd.Series(ref_price, index=df.index)).astype(np.float64).values
        if price_range > 0:
            norm_p = ((raw_price - self.config.pricing.min_price) / price_range).astype(np.float32)
            p_gap = ((raw_price - ref_price) / price_range).astype(np.float32)
        else:
            norm_p = np.full(len(df), 0.5, dtype=np.float32)
            p_gap = np.zeros(len(df), dtype=np.float32)

        demand_mean = df.get("recent_demand_mean", pd.Series(
            base_demand_pp, index=df.index
        )).astype(np.float64).values
        demand_std = df.get("recent_demand_std", pd.Series(
            0.0, index=df.index
        )).astype(np.float64).values

        demand_trend = (demand_mean / max(base_demand_pp, 1e-6)).astype(np.float32)
        demand_vol = (demand_std / max(base_demand_pp, 1e-6)).astype(np.float32)

        inv_rate = ((initial - inventory.clip(0, initial)) / initial).astype(np.float32).values if initial > 0 \
            else np.zeros(len(df), dtype=np.float32)

        return np.column_stack([norm_inv, norm_t, norm_p, demand_trend, p_gap, inv_rate, demand_vol])

    # ------------------------------------------------------------------
    # Price conversion
    # ------------------------------------------------------------------

    def _action_to_price(self, action: int) -> float:
        """Convert action index to actual price."""
        return self.config.pricing.min_price + action * self.config.pricing.price_step

    # ------------------------------------------------------------------
    # Feature importance (stub)
    # ------------------------------------------------------------------

    def feature_importance(self) -> Dict[str, float]:
        """Estimate feature importance / relevance.

        Returns a dictionary mapping feature names to importance scores
        normalised so they sum to 1.0.

        .. note::

            This is currently a **stub** that returns uniform importance.
            A production implementation could use SHAP, permutation
            importance, or the first-layer weight norms of the policy
            network.
        """
        names = self.feature_names
        uniform = 1.0 / len(names)
        return {name: uniform for name in names}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained model to disk using stable-baselines3 native format.

        Args:
            path: File path (extension will be added by sb3 if missing).
        """
        if self.model is None:
            raise ValueError("No model to save — train or load a model first.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info("Model saved to %s", save_path)

    def load(self, path: str) -> "RLAgent":
        """Load a trained model from disk.

        Args:
            path: Path to the saved model file.

        Returns:
            Self for method chaining.

        Raises:
            ImportError: If stable-baselines3 is not installed.
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required to load models")

        algorithm_class = self.ALGORITHM_MAP[self.config.model.algorithm]
        self.env = DynamicPricingEnv(
            self.config, rich_observation=self.rich_observation
        )
        self.model = algorithm_class.load(path, env=self.env)
        self._is_trained = True
        logger.info("Model loaded from %s", path)

        return self
