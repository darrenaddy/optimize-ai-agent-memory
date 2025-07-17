import os
from dotenv import load_dotenv
from agent_memory.agent import Agent
from agent_memory.strategies.sequential import SequentialMemory
from agent_memory.strategies.sliding_window import SlidingWindowMemory
from agent_memory.strategies.summarization import SummarizationMemory
from agent_memory.strategies.retrieval import RetrievalMemory
from agent_memory.strategies.memory_augmented_transformer import MemoryAugmentedTransformerMemory
from agent_memory.strategies.hierarchical import HierarchicalMemory
from agent_memory.strategies.graph_based import GraphBasedMemory
from agent_memory.strategies.compression_consolidation import CompressionConsolidationMemory
from agent_memory.strategies.os_like_memory import OSLikeMemory

# Load environment variables
load_dotenv()

def run_agent_conversation(memory_strategy_class, strategy_name, **kwargs):
    print(f"\n--- Running Agent with {strategy_name} ---")
    agent = Agent(memory_strategy=memory_strategy_class, **kwargs)

    try:
        print("Agent: Hello! How can I help you today?")
        
        responses = []
        messages = [
            "What is the capital of France?",
            "Tell me more about its history.",
            "And what about its famous landmarks?",
            "Can you summarize our conversation so far?"
        ]

        for msg in messages:
            print(f"You: {msg}")
            response = agent.chat(msg)
            responses.append(response)
            print(f"Agent: {response}")
            
        print("\n--- Final Context ---")
        # For retrieval memory, get_context needs a query
        if strategy_name == "RetrievalMemory":
            print(agent.memory.get_context(query="What did we talk about?"))
        else:
            print(agent.memory.get_context())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        agent.clear_memory()
        print(f"--- {strategy_name} Cleared ---")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set for strategies that require it
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some memory strategies will not function correctly.")
        print("Please set it in your .env file or environment variables.")

    # Run examples for each memory strategy
    run_agent_conversation(SequentialMemory, "SequentialMemory")
    run_agent_conversation(SlidingWindowMemory, "SlidingWindowMemory", window_size=3)
    run_agent_conversation(SummarizationMemory, "SummarizationMemory")
    run_agent_conversation(RetrievalMemory, "RetrievalMemory")
    run_agent_conversation(MemoryAugmentedTransformerMemory, "MemoryAugmentedTransformerMemory")
    run_agent_conversation(HierarchicalMemory, "HierarchicalMemory", short_term_threshold=2)
    run_agent_conversation(GraphBasedMemory, "GraphBasedMemory")
    run_agent_conversation(CompressionConsolidationMemory, "CompressionConsolidationMemory", compression_threshold=2)
    run_agent_conversation(OSLikeMemory, "OSLikeMemory", page_size=2, max_pages=2)
