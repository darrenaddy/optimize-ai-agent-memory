from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_memory.llms.base import BaseLLM
from collections import deque
from .base import BaseMemory

class SlidingWindowMemory(BaseMemory):
    """
    A memory strategy that keeps a fixed number of recent messages.
    """

    def __init__(self, llm: Optional["BaseLLM"] = None, window_size: int = 5):
        super().__init__(llm=llm)
        """
        Initializes the SlidingWindowMemory.

        Args:
            llm: An optional instance of a class conforming to BaseLLM.
            window_size: The number of messages to keep in the memory.
        """
        self.history: deque = deque(maxlen=window_size)

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the memory."""
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:
        """Retrieves the conversation history within the window as a single string."""
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])

    def clear(self) -> None:
        """Clears the memory."""
        self.history.clear()
