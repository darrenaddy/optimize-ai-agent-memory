from typing import Type
from .llms.base import BaseLLM
from .strategies.base import BaseMemory
from .llms import get_llm

class Agent:
    """
    A conversational agent that uses a memory strategy to maintain context.
    """

    def __init__(self, memory_strategy: Type[BaseMemory], **kwargs):
        """
        Initializes the Agent.

        Args:
            memory_strategy: The class of the memory strategy to use.
            **kwargs: Additional keyword arguments to pass to the memory strategy's constructor.
        """
        self.llm = get_llm()
        self.memory = memory_strategy(llm=self.llm, **kwargs)

    def chat(self, user_input: str) -> str:
        """
        Has a conversation with the user.

        Args:
            user_input: The user's message.

        Returns:
            The agent's response.
        """
        self.memory.add_message(role="user", content=user_input)
        context = self.memory.get_context()
        
        # This is a simplified example. In a real-world scenario, you would format the context 
        # into a proper prompt before sending it to the LLM.
        response = self.llm.invoke(context)

        self.memory.add_message(role="assistant", content=response.content)
        return response.content

    def clear_memory(self) -> None:
        """Clears the agent's memory."""
        self.memory.clear()
