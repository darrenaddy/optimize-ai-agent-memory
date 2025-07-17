from typing import List, Dict
from .base import BaseMemory
from langchain_openai import ChatOpenAI
from ..config import OPENAI_API_KEY

class SummarizationMemory(BaseMemory):
    """
    A memory strategy that summarizes the conversation history to keep the context concise.
    """

    def __init__(self, summary_prompt: str = "Summarize the following conversation:"):
        """
        Initializes the SummarizationMemory.

        Args:
            summary_prompt: The prompt to use for summarization.
        """
        self.history: List[Dict[str, str]] = []
        self.summary_prompt = summary_prompt
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY)

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
