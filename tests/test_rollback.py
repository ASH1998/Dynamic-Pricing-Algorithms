"""Tests for state management and rollback functionality."""

import pytest

from neuroprice import PricingEngine
from neuroprice.exceptions import StateRollbackError
from neuroprice.state import StateManager


class TestStateManager:
    """Tests for the StateManager class."""

    def test_initial_empty(self):
        """Test that new StateManager is empty."""
        manager = StateManager()

        assert len(manager) == 0
        assert manager.list_states() == []

    def test_save_and_restore_state(self, sample_config):
        """Test saving and restoring a state."""
        manager = StateManager()

        manager.save_state("test_state", sample_config)
        restored = manager.restore_state("test_state")

        assert restored.name == sample_config.name
        assert restored.pricing.min_price == sample_config.pricing.min_price

    def test_restore_nonexistent_state(self):
        """Test that restoring nonexistent state raises error."""
        manager = StateManager()

        with pytest.raises(StateRollbackError) as exc_info:
            manager.restore_state("nonexistent")

        error = exc_info.value
        assert error.state_name == "nonexistent"
        assert "nonexistent" in str(error)

    def test_error_lists_available_states(self, sample_config):
        """Test that error message lists available states."""
        manager = StateManager()
        manager.save_state("state_1", sample_config)
        manager.save_state("state_2", sample_config)

        with pytest.raises(StateRollbackError) as exc_info:
            manager.restore_state("missing")

        message = str(exc_info.value)
        assert "state_1" in message
        assert "state_2" in message

    def test_list_states(self, sample_config):
        """Test listing saved states."""
        manager = StateManager()

        manager.save_state("alpha", sample_config)
        manager.save_state("beta", sample_config)
        manager.save_state("gamma", sample_config)

        states = manager.list_states()

        assert "alpha" in states
        assert "beta" in states
        assert "gamma" in states
        assert len(states) == 3

    def test_has_state(self, sample_config):
        """Test checking if state exists."""
        manager = StateManager()

        manager.save_state("exists", sample_config)

        assert manager.has_state("exists")
        assert not manager.has_state("missing")
        assert "exists" in manager
        assert "missing" not in manager

    def test_delete_state(self, sample_config):
        """Test deleting a state."""
        manager = StateManager()
        manager.save_state("to_delete", sample_config)

        assert manager.has_state("to_delete")

        result = manager.delete_state("to_delete")

        assert result is True
        assert not manager.has_state("to_delete")

    def test_delete_nonexistent_state(self):
        """Test deleting nonexistent state returns False."""
        manager = StateManager()

        result = manager.delete_state("nonexistent")

        assert result is False

    def test_clear_states(self, sample_config):
        """Test clearing all states."""
        manager = StateManager()
        manager.save_state("one", sample_config)
        manager.save_state("two", sample_config)

        manager.clear_states()

        assert len(manager) == 0
        assert manager.list_states() == []

    def test_state_isolation_deep_copy(self, sample_config):
        """Test that states are deep copied (isolated)."""
        manager = StateManager()

        # Save state
        manager.save_state("original", sample_config)

        # Restore and verify it's a different object
        restored = manager.restore_state("original")

        # Should be equal but not same object
        assert restored.name == sample_config.name
        assert restored is not sample_config

    def test_overwrite_state(self, sample_config, sample_strategy_dict):
        """Test that saving with same name overwrites."""
        manager = StateManager()

        manager.save_state("state", sample_config)

        # Modify and save again
        sample_strategy_dict["name"] = "modified_name"
        from neuroprice import StrategyConfig
        modified_config = StrategyConfig(**sample_strategy_dict)

        manager.save_state("state", modified_config)

        restored = manager.restore_state("state")

        assert restored.name == "modified_name"


class TestPricingEngineRollback:
    """Tests for PricingEngine rollback functionality."""

    def test_initial_state_saved(self, sample_config):
        """Test that initial state is automatically saved."""
        engine = PricingEngine(sample_config)

        states = engine.list_states()

        assert "initial" in states

    def test_rollback_to_initial(self, sample_config):
        """Test rollback to initial state."""
        engine = PricingEngine(sample_config)
        original_name = engine.config.name

        engine.rollback("initial")

        assert engine.config.name == original_name

    def test_save_and_rollback_custom_state(self, sample_config):
        """Test saving and rolling back custom state."""
        engine = PricingEngine(sample_config)

        engine.save_state("custom_checkpoint")

        states = engine.list_states()

        assert "initial" in states
        assert "custom_checkpoint" in states

    def test_rollback_nonexistent_state_raises(self, sample_config):
        """Test that rollback to nonexistent state raises error."""
        engine = PricingEngine(sample_config)

        with pytest.raises(StateRollbackError) as exc_info:
            engine.rollback("nonexistent_state")

        assert "nonexistent_state" in str(exc_info.value)
        assert "Available states" in str(exc_info.value)

    def test_multiple_states(self, sample_config):
        """Test managing multiple states."""
        engine = PricingEngine(sample_config)

        engine.save_state("state_1")
        engine.save_state("state_2")
        engine.save_state("state_3")

        states = engine.list_states()

        assert "initial" in states
        assert "state_1" in states
        assert "state_2" in states
        assert "state_3" in states

    def test_rollback_returns_self(self, sample_config):
        """Test that rollback returns self for chaining."""
        engine = PricingEngine(sample_config)

        result = engine.rollback("initial")

        assert result is engine

    def test_list_states_returns_list(self, sample_config):
        """Test that list_states returns a list."""
        engine = PricingEngine(sample_config)

        states = engine.list_states()

        assert isinstance(states, list)
        assert all(isinstance(s, str) for s in states)


class TestStateRollbackError:
    """Tests for StateRollbackError exception."""

    def test_error_message_no_states(self):
        """Test error message when no states exist."""
        error = StateRollbackError("missing", available_states=[])

        assert "missing" in str(error)
        assert "No states have been saved" in str(error)

    def test_error_message_with_states(self):
        """Test error message lists available states."""
        error = StateRollbackError(
            "missing",
            available_states=["initial", "trained", "checkpoint"],
        )

        message = str(error)

        assert "missing" in message
        assert "initial" in message
        assert "trained" in message
        assert "checkpoint" in message

    def test_error_attributes(self):
        """Test error has correct attributes."""
        error = StateRollbackError(
            "my_state",
            available_states=["a", "b"],
        )

        assert error.state_name == "my_state"
        assert error.available_states == ["a", "b"]
