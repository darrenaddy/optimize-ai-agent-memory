from typing import List, Dict
import torch
from transformers import BertTokenizer, BertModel
from .base import BaseMemory

class MemoryAugmentedTransformerMemory(BaseMemory):
    """
    A memory strategy that uses a pre-trained transformer model to create a compressed representation of the conversation history.
    """

    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initializes the MemoryAugmentedTransformerMemory.

        Args:
            model_name: The name of the pre-trained transformer model to use.
        """
        self.history: List[Dict[str, str]] = []
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.memory_embedding = None

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the memory and updates the memory embedding."""
        self.history.append({"role": role, "content": content})
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        inputs = self.tokenizer(conversation, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        self.memory_embedding = torch.mean(outputs.last_hidden_state, dim=1)

    def get_context(self) -> str:
        """For this strategy, the raw context is the conversation history.
           The real value is in the embedding, which isn't directly serializable to a string prompt.
           In a real application, this embedding would be used to influence the next generation step.
        """
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])

    def clear(self) -> None:
        """Clears the memory."""
        self.history = []
        self.memory_embedding = None
