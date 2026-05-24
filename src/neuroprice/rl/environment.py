"""Custom Gym environment for dynamic pricing.

This module implements a Gymnasium environment that simulates
a dynamic pricing scenario based on the stochastic models
described in the academic literature on revenue management.

Supports two observation modes:
- **rich** (default): 7 features including demand trends, volatility,
  price gap, and inventory depletion rate.
- **legacy**: 2 features (inventory ratio, time ratio) matching the
  original simple observation space.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from neuroprice.models import StrategyConfig

# ---------------------------------------------------------------------------
# Feature name constants — shared with the agent module
# ---------------------------------------------------------------------------
LEGACY_FEATURE_NAMES: List[str] = ["norm_inventory", "norm_time"]

RICH_FEATURE_NAMES: List[str] = [
    "norm_inventory",       # 0  current inventory / initial stock
    "norm_time",            # 1  current step / max steps
    "norm_price",           # 2  current price normalised to [0, 1]
    "recent_demand_trend",  # 3  rolling mean of recent demand (normalised)
    "price_gap_from_ref",   # 4  (price − ref_price) / price_range
    "inventory_rate",       # 5  inventory depletion rate (normalised)
    "demand_volatility",    # 6  std of recent demand (normalised)
]


class DynamicPricingEnv(gym.Env):
    """Gymnasium environment for dynamic pricing with a rich observation space.

    State features depend on the *rich_observation* flag:

    **rich** (default, 7 features):
        0. ``norm_inventory`` – inventory / initial_stock
        1. ``norm_time`` – time_step / max_steps
        2. ``norm_price`` – (price − min_price) / (max_price − min_price)
        3. ``recent_demand_trend`` – rolling mean demand / base_demand
        4. ``price_gap_from_ref`` – (price − ref_price) / price_range
        5. ``inventory_rate`` – inventory consumed so far / initial_stock
        6. ``demand_volatility`` – rolling std demand / base_demand

    **legacy** (2 features):
        0. ``norm_inventory``
        1. ``norm_time``

    Action: select a price from a discretised price grid.
    Reward: revenue from sales with shaping penalties for inventory waste
    (amplified near deadline) and large price jumps (smoothness).

    Attributes:
        config: Strategy configuration object.
        inventory: Current inventory level.
        time_step: Current time step in the episode.
        max_steps: Maximum time steps per episode.
        demand_history: Deque of recent (raw) demand observations.
        price_history: List of prices chosen each step.

    Example:
        >>> env = DynamicPricingEnv(config)
        >>> obs, info = env.reset()
        >>> action = 5  # Select 5th price level
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: StrategyConfig,
        historical_data: Optional[Any] = None,
        render_mode: Optional[str] = None,
        rich_observation: bool = True,
        demand_window: int = 10,
        smoothness_penalty_weight: float = 0.01,
        waste_penalty_weight: float = 0.05,
    ) -> None:
        """Initialize the dynamic pricing environment.

        Args:
            config: Strategy configuration containing pricing and demand params.
            historical_data: Optional historical data for warm-starting.
            render_mode: Render mode (currently only ``'human'`` supported).
            rich_observation: If *True* (default) use the 7-feature rich
                observation; otherwise fall back to the legacy 2-feature space.
            demand_window: Number of recent periods kept for trend / volatility
                calculations.
            smoothness_penalty_weight: Weight applied to the absolute price
                change between consecutive steps.
            waste_penalty_weight: Base weight for the inventory-waste penalty
                (scaled up as deadline approaches).
        """
        super().__init__()
        self.config = config
        self.historical_data = historical_data
        self.render_mode = render_mode
        self.rich_observation = rich_observation
        self.demand_window = demand_window
        self.smoothness_penalty_weight = smoothness_penalty_weight
        self.waste_penalty_weight = waste_penalty_weight

        # ---- Price grid -------------------------------------------------
        self.n_prices = (
            int(
                (config.pricing.max_price - config.pricing.min_price)
                / config.pricing.price_step
            )
            + 1
        )
        self.action_space = spaces.Discrete(self.n_prices)

        # ---- Observation space ------------------------------------------
        self.n_features = len(RICH_FEATURE_NAMES) if rich_observation else len(LEGACY_FEATURE_NAMES)
        self.feature_names = RICH_FEATURE_NAMES if rich_observation else LEGACY_FEATURE_NAMES
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # ---- Episode state (will be reset) ------------------------------
        self.inventory: int = config.inventory.initial_stock
        self.time_step: int = 0
        self.max_steps: int = config.inventory.deadline_days or 30
        self._episode_reward: float = 0.0

        # Rich-mode trackers
        self.demand_history: deque = deque(maxlen=demand_window)
        self.price_history: List[float] = []

        # Pre-compute constants for normalisation
        self._ref_price = (config.pricing.min_price + config.pricing.max_price) / 2.0
        self._price_range = config.pricing.max_price - config.pricing.min_price
        self._base_demand_per_period = config.demand.base_demand / max(self.max_steps, 1)

    # ------------------------------------------------------------------
    # Core Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options (unused).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        self.inventory = self.config.inventory.initial_stock
        self.time_step = 0
        self._episode_reward = 0.0
        self.demand_history.clear()
        self.price_history.clear()

        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment.

        Args:
            action: Index of the price level to set.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        price = self._action_to_price(action)
        self.price_history.append(price)

        # Simulate demand
        demand = self._simulate_demand(price)
        self.demand_history.append(float(demand))

        # Sales limited by available inventory
        units_sold = min(demand, self.inventory)
        revenue = units_sold * price

        # Track inventory before update for rate calculation
        prev_inventory = self.inventory
        self.inventory -= units_sold
        self.time_step += 1
        self._episode_reward += revenue

        # ---- Reward shaping -----------------------------------------------
        reward = revenue

        # 1. Inventory waste penalty – stronger as deadline nears
        if self.inventory > 0:
            time_fraction = self.time_step / max(self.max_steps, 1)
            waste_scale = self.waste_penalty_weight * (1.0 + 2.0 * time_fraction)
            reward -= waste_scale * self.inventory

        # 2. Price smoothness penalty – discourage large jumps
        if len(self.price_history) >= 2:
            delta = abs(self.price_history[-1] - self.price_history[-2])
            reward -= self.smoothness_penalty_weight * delta

        # Check termination
        terminated = self.inventory <= 0 or self.time_step >= self.max_steps

        # Salvage value at episode end
        if terminated and self.inventory > 0:
            salvage = self.inventory * self.config.inventory.salvage_value
            reward += salvage
            self._episode_reward += salvage

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,  # truncated
            self._get_info(units_sold=units_sold, price=price),
        )

    # ------------------------------------------------------------------
    # Price ↔ action helpers
    # ------------------------------------------------------------------

    def _action_to_price(self, action: int) -> float:
        """Convert discrete action index to actual price."""
        return self.config.pricing.min_price + action * self.config.pricing.price_step

    def _price_to_action(self, price: float) -> int:
        """Convert actual price to nearest discrete action index."""
        action = int(
            (price - self.config.pricing.min_price) / self.config.pricing.price_step
        )
        return max(0, min(action, self.n_prices - 1))

    # ------------------------------------------------------------------
    # Demand simulation
    # ------------------------------------------------------------------

    def _simulate_demand(self, price: float) -> int:
        """Simulate demand using log-linear elasticity model with Poisson noise.

        D = D_base * (P / P_ref)^elasticity

        Seasonality (sin wave) is applied when enabled in config.

        Args:
            price: Current price.

        Returns:
            Simulated demand (non-negative integer).
        """
        base_demand = self.config.demand.base_demand
        elasticity = self.config.demand.elasticity

        if self._ref_price > 0 and price > 0:
            expected_demand = base_demand * (price / self._ref_price) ** elasticity
        else:
            expected_demand = base_demand

        period_demand = expected_demand / self.max_steps

        # Seasonality
        if self.config.demand.seasonality.enabled:
            period = self.config.demand.seasonality.period
            seasonality_factor = 1 + 0.2 * np.sin(2 * np.pi * self.time_step / period)
            period_demand *= seasonality_factor

        demand = self.np_random.poisson(period_demand) if period_demand > 0 else 0
        return max(0, demand)

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Build the current observation vector.

        Returns the rich 7-feature vector by default, or the legacy 2-feature
        vector when *rich_observation=False*.
        """
        if self.rich_observation:
            return self._rich_observation()
        return self._legacy_observation()

    def _legacy_observation(self) -> np.ndarray:
        """2-feature observation: [norm_inventory, norm_time]."""
        return np.array(
            [self._norm_inventory(), self._norm_time()],
            dtype=np.float32,
        )

    def _rich_observation(self) -> np.ndarray:
        """7-feature observation with demand context and price state."""
        norm_inv = self._norm_inventory()
        norm_t = self._norm_time()
        norm_p = self._norm_price()

        # Demand trend and volatility from recent history
        if len(self.demand_history) > 0:
            arr = np.array(self.demand_history, dtype=np.float32)
            trend = float(np.mean(arr)) / max(self._base_demand_per_period, 1e-6)
            volatility = float(np.std(arr)) / max(self._base_demand_per_period, 1e-6)
        else:
            trend = 0.0
            volatility = 0.0

        # Price gap from reference
        if self._price_range > 0 and len(self.price_history) > 0:
            price_gap = (self.price_history[-1] - self._ref_price) / self._price_range
        else:
            price_gap = 0.0

        # Inventory depletion rate
        initial = self.config.inventory.initial_stock
        if initial > 0:
            inv_rate = (initial - self.inventory) / initial
        else:
            inv_rate = 0.0

        return np.array(
            [norm_inv, norm_t, norm_p, trend, price_gap, inv_rate, volatility],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _norm_inventory(self) -> float:
        """Normalise inventory to [0, 1]."""
        initial = self.config.inventory.initial_stock
        return self.inventory / initial if initial > 0 else 0.0

    def _norm_time(self) -> float:
        """Normalise time step to [0, 1]."""
        return self.time_step / max(self.max_steps, 1)

    def _norm_price(self) -> float:
        """Normalise current price to [0, 1]."""
        if self._price_range <= 0 or len(self.price_history) == 0:
            return 0.5
        return (self.price_history[-1] - self.config.pricing.min_price) / self._price_range

    # ------------------------------------------------------------------
    # Info & render
    # ------------------------------------------------------------------

    def _get_info(
        self,
        units_sold: int = 0,
        price: float = 0.0,
    ) -> Dict[str, Any]:
        """Return auxiliary info dictionary."""
        info: Dict[str, Any] = {
            "inventory": self.inventory,
            "time_step": self.time_step,
            "units_sold": units_sold,
            "price": price,
            "episode_reward": self._episode_reward,
            "n_features": self.n_features,
        }
        if len(self.demand_history) > 0:
            info["recent_demand_mean"] = float(np.mean(self.demand_history))
            info["recent_demand_std"] = float(np.std(self.demand_history))
        return info

    def render(self) -> None:
        """Render the environment state (console output)."""
        if self.render_mode == "human":
            print(
                f"Step {self.time_step}/{self.max_steps} | "
                f"Inventory: {self.inventory} | "
                f"Total Reward: {self._episode_reward:.2f}"
            )
