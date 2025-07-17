import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # Default to openai
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
