"""State management for configuration rollback functionality."""

from copy import deepcopy
from typing import Any, Dict, List

from neuroprice.exceptions import StateRollbackError


class StateManager:
    """Manages configuration states for rollback functionality.

    This class allows saving and restoring configuration states,
    enabling users to experiment with different settings and
    revert to previous configurations if needed.

    Example:
        >>> manager = StateManager()
        >>> manager.save_state("initial", config)
        >>> # ... make changes ...
        >>> manager.save_state("modified", new_config)
        >>> original = manager.restore_state("initial")
    """

    def __init__(self) -> None:
        """Initialize the state manager with empty state storage."""
        self._states: Dict[str, Any] = {}

    def save_state(self, name: str, config: Any) -> None:
        """Save a configuration state with the given name.

        Args:
            name: Unique identifier for this state
            config: Configuration object to save (will be deep copied)
        """
        self._states[name] = deepcopy(config)

    def restore_state(self, name: str) -> Any:
        """Restore a previously saved configuration state.

        Args:
            name: Identifier of the state to restore

        Returns:
            Deep copy of the saved configuration

        Raises:
            StateRollbackError: If the state name doesn't exist
        """
        if name not in self._states:
            raise StateRollbackError(
                state_name=name,
                available_states=list(self._states.keys()),
            )
        return deepcopy(self._states[name])

    def list_states(self) -> List[str]:
        """List all saved state names.

        Returns:
            List of state names in order of creation
        """
        return list(self._states.keys())

    def has_state(self, name: str) -> bool:
        """Check if a state with the given name exists.

        Args:
            name: State name to check

        Returns:
            True if state exists, False otherwise
        """
        return name in self._states

    def delete_state(self, name: str) -> bool:
        """Delete a saved state.

        Args:
            name: State name to delete

        Returns:
            True if state was deleted, False if it didn't exist
        """
        if name in self._states:
            del self._states[name]
            return True
        return False

    def clear_states(self) -> None:
        """Clear all saved states."""
        self._states.clear()

    def __len__(self) -> int:
        """Return the number of saved states."""
        return len(self._states)

    def __contains__(self, name: str) -> bool:
        """Check if a state name exists."""
        return name in self._states
