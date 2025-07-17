from typing import Any
from langchain_community.chat_models import ChatOllama
from .base import BaseLLM

class OllamaLLM(BaseLLM):
    """
    Wrapper for Ollama chat models, conforming to the BaseLLM interface.
    """
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(model=model, base_url=base_url)

    def invoke(self, prompt: str) -> Any:
        return self.llm.invoke(prompt)
