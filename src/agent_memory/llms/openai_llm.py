from typing import Any
from langchain_openai import ChatOpenAI
from ..config import OPENAI_API_KEY
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    """
    Wrapper for OpenAI's Chat models, conforming to the BaseLLM interface.
    """
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY)

    def invoke(self, prompt: str) -> Any:
        return self.llm.invoke(prompt)
