from typing import List, Dict, TYPE_CHECKING
from .base import BaseMemory

if TYPE_CHECKING:
    from agent_memory.llms.base import BaseLLM

class HierarchicalMemory(BaseMemory):
    """
    A memory strategy that maintains two levels of memory: a short-term (recent messages)
    and a long-term (summaries of older conversations).
    """

    def __init__(self, llm: "BaseLLM", short_term_threshold: int = 4, summary_prompt: str = "Summarize the following conversation:"):
        super().__init__(llm=llm)
        self.short_term_memory: List[Dict[str, str]] = []
        self.long_term_memory: str = ""
        self.short_term_threshold = short_term_threshold
        self.summary_prompt = summary_prompt
        self.llm = llm

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the short-term memory and triggers summarization if the threshold is exceeded."""
        self.short_term_memory.append({"role": role, "content": content})
        if len(self.short_term_memory) > self.short_term_threshold:
            self._summarize()

    def _summarize(self) -> None:
        """Summarizes the short-term memory and moves it to long-term memory."""
        messages_to_summarize = self.short_term_memory[:-1] # Summarize all but the last message
        conversation_to_summarize = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_to_summarize])
        prompt = f"{self.summary_prompt}\n\n{conversation_to_summarize}"
        summary = self.llm.invoke(prompt).content
        
        if self.long_term_memory:
            self.long_term_memory += "\n" + summary
        else:
            self.long_term_memory = summary
            
        self.short_term_memory = [self.short_term_memory[-1]] # Keep only the last message

    def get_context(self) -> str:
        """Retrieves the combined context from long-term and short-term memory."""
        short_term_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.short_term_memory])
        return f"Summary of past conversation:\n{self.long_term_memory}\n\nCurrent conversation:\n{short_term_context}"

    def clear(self) -> None:
        """Clears both long-term and short-term memory."""
        self.short_term_memory = []
        self.long_term_memory = ""
