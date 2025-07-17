from abc import ABC, abstractmethod
from typing import Any

class BaseLLM(ABC):
    """
    Abstract Base Class for all Language Model (LLM) integrations.
    Defines the common interface for interacting with different LLM providers.
    """

    @abstractmethod
    def invoke(self, prompt: str) -> Any:
        """Invokes the LLM with a given prompt and returns its response."""
        pass
