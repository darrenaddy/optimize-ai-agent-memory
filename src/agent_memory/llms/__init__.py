from .base import BaseLLM
from .openai_llm import OpenAILLM
from .ollama_llm import OllamaLLM
from ..config import LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_BASE_URL

import os
from unittest.mock import MagicMock

def get_llm() -> BaseLLM:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Return a mock LLM during testing
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Mocked LLM response"
        return mock_llm

    if LLM_PROVIDER == "ollama":
        return OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    elif LLM_PROVIDER == "openai":
        return OpenAILLM()
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
