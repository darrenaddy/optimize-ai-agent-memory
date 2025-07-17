from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_memory.llms.base import BaseLLM
from .base import BaseMemory

class SequentialMemory(BaseMemory):
    """
    The most basic memory strategy. It stores the entire conversation history.
    """

    def __init__(self, llm: Optional["BaseLLM"] = None):
        super().__init__(llm=llm)
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the memory."""
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:
        """Retrieves the entire conversation history as a single string."""
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])

    def clear(self) -> None:
        """Clears the memory."""
        self.history = []

