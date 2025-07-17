from typing import List, Dict, TYPE_CHECKING
from .base import BaseMemory

if TYPE_CHECKING:
    from agent_memory.llms.base import BaseLLM

class CompressionConsolidationMemory(BaseMemory):
    """
    A memory strategy that compresses older information or consolidates redundant entries.
    This implementation uses summarization for compression.
    """

    def __init__(self, llm: "BaseLLM", compression_threshold: int = 5, compression_prompt: str = "Summarize the following conversation, focusing on key information and removing redundancy:"):
        super().__init__(llm=llm)
        self.history: List[Dict[str, str]] = []
        self.compressed_memory: List[str] = []
        self.compression_threshold = compression_threshold
        self.compression_prompt = compression_prompt
        self.llm = llm

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the history and triggers compression if the threshold is met."""
        self.history.append({"role": role, "content": content})
        if len(self.history) >= self.compression_threshold:
            self._compress_and_consolidate()

    def _compress_and_consolidate(self) -> None:
        """
        Compresses the current history into a summary and adds it to compressed memory.
        Then clears the current history.
        """
        conversation_to_compress = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        prompt = f"{self.compression_prompt}\n\n{conversation_to_compress}"
        compressed_summary = self.llm.invoke(prompt).content
        self.compressed_memory.append(compressed_summary)
        self.history = [] # Clear history after compression

    def get_context(self) -> str:
        """
        Retrieves the combined context from compressed and current history.
        """
        compressed_context = "\n".join(self.compressed_memory)
        current_history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        
        if compressed_context and current_history_context:
            return f"Compressed Past:\n{compressed_context}\n\nCurrent Conversation:\n{current_history_context}"
        elif compressed_context:
            return f"Compressed Past:\n{compressed_context}"
        elif current_history_context:
            return f"Current Conversation:\n{current_history_context}"
        else:
            return ""

    def clear(self) -> None:
        """
        Clears both the current history and compressed memory.
        """
        self.history = []
        self.compressed_memory = []