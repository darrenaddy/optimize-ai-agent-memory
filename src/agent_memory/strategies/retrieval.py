from typing import List, Dict, Optional
from .base import BaseMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from ..config import OPENAI_API_KEY

class RetrievalMemory(BaseMemory):
    """
    A memory strategy that uses a retrieval-based model (RAG) to find relevant information.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Initializes the RetrievalMemory.

        Args:
            chunk_size: The size of the text chunks to create.
            chunk_overlap: The overlap between text chunks.
        """
        self.history: List[Dict[str, str]] = []
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the memory."""
        self.history.append({"role": role, "content": content})
        texts = self.text_splitter.split_text("\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history]))
        if texts:
            self.vector_store = FAISS.from_texts(texts, self.embeddings)

    def get_context(self, query: Optional[str] = None, k: int = 2) -> str:
        """Retrieves the most relevant context for a given query."""
        if query and self.vector_store:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n".join([doc.page_content for doc in docs])
        else:
            return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])

    def clear(self) -> None:
        """Clears the memory."""
        self.history = []
        self.vector_store = None
