from typing import List, Dict, TYPE_CHECKING
from .base import BaseMemory

if TYPE_CHECKING:
    from agent_memory.llms.base import BaseLLM

class SummarizationMemory(BaseMemory):
    """
    A memory strategy that summarizes the conversation history to keep the context concise.
    """

    def __init__(self, llm: "BaseLLM", summary_prompt: str = "Summarize the following conversation:"):
        super().__init__(llm=llm)
        self.history: List[Dict[str, str]] = []
        self.summary_prompt = summary_prompt
        self.llm = llm

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the memory."""
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:
        """Summarizes the conversation history and returns the summary."""
        if not self.history:
            return ""

        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        prompt = f"{self.summary_prompt}\n\n{conversation}"
        response = self.llm.invoke(prompt)
        return response.content

    def clear(self) -> None:
        """Clears the memory."""
        self.history = []
