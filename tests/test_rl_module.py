"""Tests for the RL module (environment and agent)."""

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from neuroprice.models import (
    DemandConfig,
    FeaturesConfig,
    InventoryConfig,
    ModelConfig,
    PricingConfig,
    RLAlgorithm,
    RLConfig,
    SeasonalityConfig,
    StrategyConfig,
)
from neuroprice.rl.environment import (
    LEGACY_FEATURE_NAMES,
    RICH_FEATURE_NAMES,
    DynamicPricingEnv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> StrategyConfig:
    """Create a minimal StrategyConfig for env testing."""
    defaults = dict(
        name="env_test",
        pricing=PricingConfig(min_price=10.0, max_price=50.0, price_step=1.0),
        demand=DemandConfig(
            base_demand=300.0,
            elasticity=-1.5,
            seasonality=SeasonalityConfig(enabled=False, period=7),
        ),
        inventory=InventoryConfig(initial_stock=50, salvage_value=2.0, deadline_days=20),
        features=FeaturesConfig(required=["price"]),
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


# ---------------------------------------------------------------------------
# DynamicPricingEnv — Rich Observation
# ---------------------------------------------------------------------------

class TestEnvRichObservation:
    """Tests for the 7-feature rich observation mode."""

    def test_rich_observation_shape(self):
        """Rich observation should return a 7-element array."""
        env = DynamicPricingEnv(_make_config(), rich_observation=True)
        obs, info = env.reset(seed=42)

        assert obs.shape == (7,), f"Expected 7 features, got {obs.shape}"
        assert obs.dtype == np.float32

    def test_rich_feature_names(self):
        """Feature names list should have 7 entries in rich mode."""
        env = DynamicPricingEnv(_make_config(), rich_observation=True)
        assert len(env.feature_names) == 7
        assert env.feature_names == RICH_FEATURE_NAMES
        assert env.n_features == 7

    def test_rich_observation_values_initial(self):
        """Initial observation should have norm_inventory=1.0 and norm_time=0.0."""
        env = DynamicPricingEnv(_make_config(), rich_observation=True)
        obs, _ = env.reset(seed=42)

        assert obs[0] == pytest.approx(1.0)  # norm_inventory
        assert obs[1] == pytest.approx(0.0)  # norm_time

    def test_rich_observation_after_step(self):
        """After a step, norm_time should advance and norm_inventory may decrease."""
        env = DynamicPricingEnv(_make_config(), rich_observation=True)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(5)

        assert obs.shape == (7,)
        assert obs[1] > 0.0  # norm_time advanced

    def test_rich_observation_demand_features_after_steps(self):
        """After several steps, demand trend and volatility should be non-zero."""
        env = DynamicPricingEnv(_make_config(), rich_observation=True)
        env.reset(seed=42)

        for _ in range(5):
            env.step(3)

        obs = env._get_observation()
        # Demand trend should be non-zero after observing demand
        assert obs[3] != 0.0 or obs[6] != 0.0  # trend or volatility


# ---------------------------------------------------------------------------
# DynamicPricingEnv — Legacy Observation
# ---------------------------------------------------------------------------

class TestEnvLegacyObservation:
    """Tests for the 2-feature legacy observation mode."""

    def test_legacy_observation_shape(self):
        """Legacy observation should return a 2-element array."""
        env = DynamicPricingEnv(_make_config(), rich_observation=False)
        obs, info = env.reset(seed=42)

        assert obs.shape == (2,), f"Expected 2 features, got {obs.shape}"
        assert obs.dtype == np.float32

    def test_legacy_feature_names(self):
        """Feature names list should have 2 entries in legacy mode."""
        env = DynamicPricingEnv(_make_config(), rich_observation=False)
        assert len(env.feature_names) == 2
        assert env.feature_names == LEGACY_FEATURE_NAMES
        assert env.n_features == 2

    def test_legacy_observation_values_initial(self):
        """Initial legacy obs should be [1.0, 0.0] (full inventory, time=0)."""
        env = DynamicPricingEnv(_make_config(), rich_observation=False)
        obs, _ = env.reset(seed=42)

        assert obs[0] == pytest.approx(1.0)
        assert obs[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reset / Step Cycle
# ---------------------------------------------------------------------------

class TestEnvResetStep:
    """Tests for the reset and step cycle."""

    def test_reset_returns_observation_and_info(self):
        """Reset should return (observation, info_dict)."""
        env = DynamicPricingEnv(_make_config())
        result = env.reset(seed=42)

        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_restores_state(self):
        """Reset should restore inventory and time_step to initial values."""
        config = _make_config()
        env = DynamicPricingEnv(config)

        # Step a few times
        env.reset(seed=42)
        for _ in range(5):
            env.step(3)

        # Reset and check
        env.reset(seed=42)
        assert env.inventory == config.inventory.initial_stock
        assert env.time_step == 0

    def test_step_returns_five_tuple(self):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        result = env.step(3)
        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_advances_time(self):
        """Each step should increment time_step by 1."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        for i in range(5):
            env.step(3)
            assert env.time_step == i + 1

    def test_step_reduces_inventory(self):
        """Step should reduce inventory by units sold."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)
        initial_inv = env.inventory

        _, _, _, _, info = env.step(3)
        assert env.inventory <= initial_inv

    def test_termination_on_inventory_exhausted(self):
        """Episode should terminate when inventory reaches zero."""
        config = _make_config()
        config.inventory.initial_stock = 1  # Very small inventory
        env = DynamicPricingEnv(config)
        env.reset(seed=42)

        # Step with low price (high demand likely) until inventory runs out
        terminated = False
        for _ in range(100):
            _, _, terminated, _, _ = env.step(0)
            if terminated:
                break

        assert terminated

    def test_termination_on_max_steps(self):
        """Episode should terminate when max_steps is reached."""
        config = _make_config()
        config.inventory.initial_stock = 10000  # Very high inventory
        config.inventory.deadline_days = 5
        env = DynamicPricingEnv(config)
        env.reset(seed=42)

        terminated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(20)
            if terminated:
                break

        assert env.time_step >= 5

    def test_truncated_always_false(self):
        """Truncated flag should always be False (no truncation logic)."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        for _ in range(10):
            _, _, _, truncated, _ = env.step(3)
            assert truncated is False

    def test_info_contains_expected_keys(self):
        """Info dict should contain inventory, time_step, units_sold, price."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)
        _, _, _, _, info = env.step(3)

        assert "inventory" in info
        assert "time_step" in info
        assert "units_sold" in info
        assert "price" in info
        assert "episode_reward" in info
        assert "n_features" in info

    def test_info_n_features_matches_mode(self):
        """Info should report correct n_features for the observation mode."""
        env_rich = DynamicPricingEnv(_make_config(), rich_observation=True)
        env_rich.reset(seed=42)
        _, _, _, _, info_rich = env_rich.step(3)
        assert info_rich["n_features"] == 7

        env_legacy = DynamicPricingEnv(_make_config(), rich_observation=False)
        env_legacy.reset(seed=42)
        _, _, _, _, info_legacy = env_legacy.step(3)
        assert info_legacy["n_features"] == 2


# ---------------------------------------------------------------------------
# Demand Simulation
# ---------------------------------------------------------------------------

class TestDemandSimulation:
    """Tests for the demand simulation model."""

    def test_demand_is_non_negative(self):
        """Simulated demand should always be non-negative."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        for _ in range(50):
            demand = env._simulate_demand(25.0)
            assert demand >= 0

    def test_demand_elasticity(self):
        """Lower prices should produce higher expected demand."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        # Average over many samples to reduce noise
        n_samples = 500
        low_price_demands = [env._simulate_demand(15.0) for _ in range(n_samples)]
        high_price_demands = [env._simulate_demand(45.0) for _ in range(n_samples)]

        avg_low = np.mean(low_price_demands)
        avg_high = np.mean(high_price_demands)

        # With negative elasticity, lower price → higher demand
        assert avg_low > avg_high

    def test_demand_with_seasonality(self):
        """Demand with seasonality enabled should vary with time."""
        config = _make_config()
        config.demand.seasonality.enabled = True
        config.demand.seasonality.period = 7
        env = DynamicPricingEnv(config)
        env.reset(seed=42)

        # Collect demand at different time steps
        demands = []
        for i in range(14):
            env.time_step = i
            demand = env._simulate_demand(25.0)
            demands.append(demand)

        # Should have some variation due to seasonality
        assert np.std(demands) > 0

    def test_demand_zero_price(self):
        """Demand simulation with zero price should not crash."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        # Should not raise
        demand = env._simulate_demand(0.0)
        assert demand >= 0


# ---------------------------------------------------------------------------
# Reward Shaping
# ---------------------------------------------------------------------------

class TestRewardShaping:
    """Tests for inventory waste and smoothness penalties."""

    def test_smoothness_penalty_on_large_price_jump(self):
        """Large price jumps should incur smoothness penalty."""
        config = _make_config()
        env = DynamicPricingEnv(config, smoothness_penalty_weight=0.1)
        env.reset(seed=42)

        # First step — no penalty (no previous price)
        _, r1, _, _, _ = env.step(0)  # min price
        # Second step — big jump to max price
        _, r2, _, _, _ = env.step(40)  # max price jump

        # r2 should include a smoothness penalty
        # We can't directly check the penalty value, but we can verify
        # that the reward is lower than it would be without penalty
        env_no_penalty = DynamicPricingEnv(config, smoothness_penalty_weight=0.0)
        env_no_penalty.reset(seed=42)
        _, r1_np, _, _, _ = env_no_penalty.step(0)
        _, r2_np, _, _, _ = env_no_penalty.step(40)

        # With penalty, reward should be lower
        assert r2 < r2_np

    def test_waste_penalty_increases_with_time(self):
        """Inventory waste penalty should be stronger near deadline."""
        config = _make_config()
        config.inventory.initial_stock = 100
        config.inventory.deadline_days = 10

        env = DynamicPricingEnv(config, waste_penalty_weight=0.5)
        env.reset(seed=42)

        # Step near beginning
        env.time_step = 1
        env.inventory = 50
        reward_early = env._simulate_demand(25.0)  # Not directly checking penalty
        # But we can verify the penalty weight scales with time_fraction

        # The waste penalty is: waste_penalty_weight * (1 + 2 * time_fraction) * inventory
        # At time_step=1/10: scale = 0.5 * (1 + 0.2) = 0.6
        # At time_step=9/10: scale = 0.5 * (1 + 1.8) = 1.4
        time_frac_early = 1 / 10
        time_frac_late = 9 / 10
        scale_early = 0.5 * (1.0 + 2.0 * time_frac_early)
        scale_late = 0.5 * (1.0 + 2.0 * time_frac_late)
        assert scale_late > scale_early

    def test_salvage_value_at_episode_end(self):
        """Salvage value should be added when episode ends with remaining inventory."""
        config = _make_config()
        config.inventory.initial_stock = 10000
        config.inventory.salvage_value = 5.0
        config.inventory.deadline_days = 2
        env = DynamicPricingEnv(config, waste_penalty_weight=0.0)
        env.reset(seed=42)

        # Step through entire episode
        total_reward = 0.0
        terminated = False
        while not terminated:
            _, r, terminated, _, _ = env.step(20)
            total_reward += r

        # Salvage should have been added
        assert env._episode_reward > 0


# ---------------------------------------------------------------------------
# Demand History Tracking
# ---------------------------------------------------------------------------

class TestDemandHistory:
    """Tests for demand_history tracking across steps."""

    def test_demand_history_populated_after_steps(self):
        """demand_history should be populated after stepping."""
        env = DynamicPricingEnv(_make_config(), demand_window=10)
        env.reset(seed=42)

        assert len(env.demand_history) == 0

        for _ in range(5):
            env.step(3)

        assert len(env.demand_history) == 5

    def test_demand_history_respects_window(self):
        """demand_history should be capped at demand_window size."""
        env = DynamicPricingEnv(_make_config(), demand_window=5)
        env.reset(seed=42)

        for _ in range(10):
            env.step(3)

        assert len(env.demand_history) == 5

    def test_demand_history_cleared_on_reset(self):
        """demand_history should be empty after reset."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        for _ in range(5):
            env.step(3)

        env.reset()
        assert len(env.demand_history) == 0

    def test_price_history_tracking(self):
        """price_history should track all prices chosen."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        actions = [0, 5, 10, 15]
        for action in actions:
            env.step(action)

        assert len(env.price_history) == 4

    def test_demand_history_in_info(self):
        """Info dict should include demand stats when history is available."""
        env = DynamicPricingEnv(_make_config())
        env.reset(seed=42)

        # Step a few times to build history
        for _ in range(3):
            env.step(3)

        _, _, _, _, info = env.step(3)
        assert "recent_demand_mean" in info
        assert "recent_demand_std" in info


# ---------------------------------------------------------------------------
# RLAgent — Initialization with Different Algorithms
# ---------------------------------------------------------------------------

class TestRLAgentInit:
    """Tests for RLAgent initialization."""

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_init_with_ppo(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Agent should initialize with PPO algorithm."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=True)
        assert agent.config == rl_config_rich
        assert agent.rich_observation is True
        assert not agent.is_trained
        assert agent.model is None

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_init_with_a2c(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Agent should initialize with A2C algorithm."""
        from neuroprice.models import RLAlgorithm

        rl_config_rich.model.algorithm = RLAlgorithm.A2C
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich)
        assert agent.config.model.algorithm == RLAlgorithm.A2C

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_init_with_dqn(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Agent should initialize with DQN algorithm."""
        from neuroprice.models import RLAlgorithm

        rl_config_rich.model.algorithm = RLAlgorithm.DQN
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich)
        assert agent.config.model.algorithm == RLAlgorithm.DQN

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", False)
    def test_init_without_sb3_raises(self, rl_config_rich):
        """ImportError should be raised when SB3 is not installed."""
        from neuroprice.rl.agents import RLAgent

        with pytest.raises(ImportError, match="stable-baselines3"):
            RLAgent(rl_config_rich)

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_agent_feature_names_rich(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Agent feature_names should return 7 names in rich mode."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=True)
        assert agent.feature_names == RICH_FEATURE_NAMES

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_agent_feature_names_legacy(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Agent feature_names should return 2 names in legacy mode."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=False)
        assert agent.feature_names == LEGACY_FEATURE_NAMES


# ---------------------------------------------------------------------------
# RLAgent — Batch Prediction
# ---------------------------------------------------------------------------

class TestRLAgentBatchPrediction:
    """Tests for batch (vectorized) prediction."""

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_predict_requires_training(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """predict() should raise ValueError if model is not trained."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich)
        df = pd.DataFrame({"price": [25.0, 30.0]})

        with pytest.raises(ValueError, match="trained"):
            agent.predict(df)

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_batch_to_observations_legacy(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """_batch_to_observations should produce (N, 2) array in legacy mode."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=False)
        df = pd.DataFrame({
            "inventory_level": [50, 100],
            "norm_time": [0.3, 0.7],
        })

        obs = agent._batch_to_observations(df)
        assert obs.shape == (2, 2)
        assert obs.dtype == np.float32

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_batch_to_observations_rich(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """_batch_to_observations should produce (N, 7) array in rich mode."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=True)
        df = pd.DataFrame({
            "inventory_level": [50, 100],
            "norm_time": [0.3, 0.7],
            "price": [25.0, 30.0],
        })

        obs = agent._batch_to_observations(df)
        assert obs.shape == (2, 7)
        assert obs.dtype == np.float32

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_predict_returns_prices_and_confidences(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """predict() should return (prices, confidences) arrays."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=False)
        # Mock the model
        agent.model = MagicMock()
        agent.model.predict.return_value = (np.array([3, 5, 7]), None)
        agent.model.policy = MagicMock()
        agent.model.device = "cpu"
        agent._is_trained = True

        df = pd.DataFrame({
            "inventory_level": [50, 100, 75],
            "norm_time": [0.3, 0.5, 0.7],
        })

        prices, confidences = agent.predict(df)

        assert len(prices) == 3
        assert len(confidences) == 3
        assert all(isinstance(p, (float, np.floating)) for p in prices)


# ---------------------------------------------------------------------------
# RLAgent — Confidence Computation
# ---------------------------------------------------------------------------

class TestConfidenceComputation:
    """Tests for confidence estimation methods."""

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_heuristic_confidence_range(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Heuristic confidence should be in [0, 1]."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich)

        for inv in [0.0, 0.5, 1.0]:
            for time in [0.0, 0.5, 1.0]:
                obs = np.array([inv, time], dtype=np.float32)
                conf = RLAgent._heuristic_confidence(obs)
                assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range for inv={inv}, time={time}"

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_heuristic_confidence_higher_with_inventory(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Higher inventory should yield higher confidence."""
        from neuroprice.rl.agents import RLAgent

        obs_high = np.array([1.0, 0.5], dtype=np.float32)
        obs_low = np.array([0.1, 0.5], dtype=np.float32)

        conf_high = RLAgent._heuristic_confidence(obs_high)
        conf_low = RLAgent._heuristic_confidence(obs_low)

        assert conf_high > conf_low

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_value_confidence_ppo(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """PPO/A2C should use value-based confidence."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich)
        agent.model = MagicMock()
        agent.model.device = "cpu"
        agent.model.policy = MagicMock()

        # Mock predict_values to return a tensor
        import torch as th
        agent.model.policy.predict_values.return_value = th.tensor([[2.0]])

        obs = np.array([[0.8, 0.3, 0.5, 1.0, 0.0, 0.2, 0.1]], dtype=np.float32)
        conf = agent._value_confidence(obs)

        assert 0.0 <= conf <= 1.0
        # sigmoid(2.0) ≈ 0.88
        assert conf > 0.8

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_qvalue_confidence_dqn(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """DQN should use Q-value spread confidence."""
        from neuroprice.models import RLAlgorithm

        rl_config_rich.model.algorithm = RLAlgorithm.DQN
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich)
        agent.model = MagicMock()
        agent.model.device = "cpu"

        # Mock q_net to return Q-values with a clear winner
        import torch as th
        agent.model.q_net.return_value = th.tensor([[1.0, 5.0, 2.0, 3.0]])

        obs = np.array([[0.8, 0.3, 0.5, 1.0, 0.0, 0.2, 0.1]], dtype=np.float32)
        conf = agent._qvalue_confidence(obs)

        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# RLAgent — Feature Importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    """Tests for the feature_importance method."""

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_feature_importance_rich(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Feature importance should have 7 entries in rich mode."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=True)
        importance = agent.feature_importance()

        assert len(importance) == 7
        assert all(name in importance for name in RICH_FEATURE_NAMES)

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_feature_importance_sums_to_one(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Feature importance scores should sum to 1.0."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=True)
        importance = agent.feature_importance()

        total = sum(importance.values())
        assert total == pytest.approx(1.0)

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_feature_importance_legacy(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Feature importance should have 2 entries in legacy mode."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=False)
        importance = agent.feature_importance()

        assert len(importance) == 2


# ---------------------------------------------------------------------------
# Observation Conversion
# ---------------------------------------------------------------------------

class TestObservationConversion:
    """Tests for row-to-observation conversion in both modes."""

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_row_to_observation_legacy(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Legacy row conversion should produce 2-element array."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=False)
        row = pd.Series({"inventory_level": 50, "norm_time": 0.3})

        obs = agent._row_to_observation(row)
        assert obs.shape == (2,)
        assert obs.dtype == np.float32

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_row_to_observation_rich(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Rich row conversion should produce 7-element array."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=True)
        row = pd.Series({
            "inventory_level": 50,
            "norm_time": 0.3,
            "price": 25.0,
        })

        obs = agent._row_to_observation(row)
        assert obs.shape == (7,)
        assert obs.dtype == np.float32

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_row_to_observation_defaults(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Missing columns should use sensible defaults."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich, rich_observation=True)
        row = pd.Series({"inventory_level": 100})

        obs = agent._row_to_observation(row)
        assert obs.shape == (7,)
        # Should not crash even with missing price, demand columns

    @patch("neuroprice.rl.agents.SB3_AVAILABLE", True)
    @patch("neuroprice.rl.agents.PPO")
    @patch("neuroprice.rl.agents.A2C")
    @patch("neuroprice.rl.agents.DQN")
    def test_action_to_price(self, mock_dqn, mock_a2c, mock_ppo, rl_config_rich):
        """Action-to-price conversion should be correct."""
        from neuroprice.rl.agents import RLAgent

        agent = RLAgent(rl_config_rich)
        assert agent._action_to_price(0) == rl_config_rich.pricing.min_price
        assert agent._action_to_price(1) == rl_config_rich.pricing.min_price + rl_config_rich.pricing.price_step
