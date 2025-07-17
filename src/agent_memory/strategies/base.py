from abc import ABC, abstractmethod
from typing import List, Dict

class BaseMemory(ABC):
    """
    Abstract Base Class for all memory strategies.

    This class defines the standard interface that all memory strategies must implement.
    This ensures that all memory strategies are interchangeable and can be used by the Agent class.
    """

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the memory."""
        pass

    @abstractmethod
    def get_context(self) -> str:
        """Retrieves the context from the memory."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears the memory."""
        pass
