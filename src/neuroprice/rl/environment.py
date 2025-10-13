"""Custom Gym environment for dynamic pricing.

This module implements a Gymnasium environment that simulates
a dynamic pricing scenario based on the stochastic models
described in the academic literature on revenue management.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from neuroprice.models import StrategyConfig


class DynamicPricingEnv(gym.Env):
    """Custom Gym environment for dynamic pricing.

    This environment simulates a pricing scenario where:
    - State: (inventory_level, time_remaining, optional features)
    - Action: Select a price from discretized price levels
    - Reward: Revenue from sales

    The demand model uses price elasticity to simulate realistic
    customer behavior, with stochastic elements based on Poisson
    arrival processes.

    Attributes:
        config: Strategy configuration object
        inventory: Current inventory level
        time_step: Current time step in the episode
        max_steps: Maximum time steps per episode

    Example:
        >>> env = DynamicPricingEnv(config)
        >>> obs, info = env.reset()
        >>> action = 5  # Select 5th price level
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: StrategyConfig,
        historical_data: Optional[Any] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize the dynamic pricing environment.

        Args:
            config: Strategy configuration containing pricing and demand params
            historical_data: Optional historical data for warm-starting
            render_mode: Render mode (currently only 'human' supported)
        """
        super().__init__()
        self.config = config
        self.historical_data = historical_data
        self.render_mode = render_mode

        # Calculate number of discrete price levels
        self.n_prices = (
            int(
                (config.pricing.max_price - config.pricing.min_price)
                / config.pricing.price_step
            )
            + 1
        )

        # Action space: discrete price selection
        self.action_space = spaces.Discrete(self.n_prices)

        # Observation space: normalized inventory + time + optional features
        # Using 2 core features: normalized inventory and normalized time
        n_features = 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_features,),
            dtype=np.float32,
        )

        # Initialize episode state
        self.inventory = config.inventory.initial_stock
        self.time_step = 0
        self.max_steps = config.inventory.deadline_days or 30
        self._episode_reward = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        self.inventory = self.config.inventory.initial_stock
        self.time_step = 0
        self._episode_reward = 0.0

        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment.

        Args:
            action: Index of the price level to set

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to actual price
        price = self._action_to_price(action)

        # Simulate demand based on price
        demand = self._simulate_demand(price)

        # Calculate sales (limited by inventory)
        units_sold = min(demand, self.inventory)
        revenue = units_sold * price

        # Update inventory and time
        self.inventory -= units_sold
        self.time_step += 1
        self._episode_reward += revenue

        # Check termination conditions
        terminated = self.inventory <= 0 or self.time_step >= self.max_steps

        # Add salvage value if episode ends with remaining inventory
        if terminated and self.inventory > 0:
            salvage = self.inventory * self.config.inventory.salvage_value
            revenue += salvage
            self._episode_reward += salvage

        return (
            self._get_observation(),
            revenue,
            terminated,
            False,  # truncated
            self._get_info(units_sold=units_sold, price=price),
        )

    def _action_to_price(self, action: int) -> float:
        """Convert discrete action to actual price.

        Args:
            action: Action index (0 to n_prices-1)

        Returns:
            Actual price value
        """
        return self.config.pricing.min_price + action * self.config.pricing.price_step

    def _price_to_action(self, price: float) -> int:
        """Convert price to discrete action index.

        Args:
            price: Actual price value

        Returns:
            Action index
        """
        action = int(
            (price - self.config.pricing.min_price) / self.config.pricing.price_step
        )
        return max(0, min(action, self.n_prices - 1))

    def _simulate_demand(self, price: float) -> int:
        """Simulate demand based on price using elasticity model.

        Uses a log-linear demand model:
        D = D_base * (P / P_ref)^elasticity

        With Poisson stochasticity to simulate arrival process.

        Args:
            price: Current price

        Returns:
            Simulated demand (integer)
        """
        base_demand = self.config.demand.base_demand
        elasticity = self.config.demand.elasticity

        # Reference price is the midpoint of price range
        ref_price = (
            self.config.pricing.min_price + self.config.pricing.max_price
        ) / 2

        # Calculate expected demand using elasticity
        if ref_price > 0 and price > 0:
            expected_demand = base_demand * (price / ref_price) ** elasticity
        else:
            expected_demand = base_demand

        # Scale demand to per-period level
        period_demand = expected_demand / self.max_steps

        # Apply seasonality if enabled
        if self.config.demand.seasonality.enabled:
            period = self.config.demand.seasonality.period
            seasonality_factor = 1 + 0.2 * np.sin(
                2 * np.pi * self.time_step / period
            )
            period_demand *= seasonality_factor

        # Add stochasticity using Poisson distribution
        if period_demand > 0:
            demand = self.np_random.poisson(period_demand)
        else:
            demand = 0

        return max(0, demand)

    def _get_observation(self) -> np.ndarray:
        """Get current observation (normalized state).

        Returns:
            Numpy array with normalized features
        """
        # Normalize inventory to [0, 1]
        norm_inventory = (
            self.inventory / self.config.inventory.initial_stock
            if self.config.inventory.initial_stock > 0
            else 0.0
        )

        # Normalize time to [0, 1]
        norm_time = self.time_step / self.max_steps

        return np.array([norm_inventory, norm_time], dtype=np.float32)

    def _get_info(
        self,
        units_sold: int = 0,
        price: float = 0.0,
    ) -> Dict[str, Any]:
        """Get additional info dict.

        Args:
            units_sold: Units sold in current step
            price: Price used in current step

        Returns:
            Info dictionary with episode statistics
        """
        return {
            "inventory": self.inventory,
            "time_step": self.time_step,
            "units_sold": units_sold,
            "price": price,
            "episode_reward": self._episode_reward,
        }

    def render(self) -> None:
        """Render the environment state (console output)."""
        if self.render_mode == "human":
            print(
                f"Step {self.time_step}/{self.max_steps} | "
                f"Inventory: {self.inventory} | "
                f"Total Reward: {self._episode_reward:.2f}"
            )
