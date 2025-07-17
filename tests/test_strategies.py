import os
import pytest
from agent_memory.strategies.sequential import SequentialMemory
from agent_memory.strategies.sliding_window import SlidingWindowMemory
from agent_memory.strategies.summarization import SummarizationMemory
from agent_memory.strategies.retrieval import RetrievalMemory
from agent_memory.strategies.hierarchical import HierarchicalMemory
from agent_memory.strategies.graph_based import GraphBasedMemory
from agent_memory.strategies.os_like_memory import OSLikeMemory


def test_sequential_memory():
    memory = SequentialMemory()
    memory.add_message(role="user", content="Hello")
    memory.add_message(role="assistant", content="Hi there!")
    context = memory.get_context()
    assert "user: Hello" in context
    assert "assistant: Hi there!" in context
    memory.clear()
    assert memory.get_context() == ""

def test_sliding_window_memory():
    memory = SlidingWindowMemory(window_size=2)
    memory.add_message(role="user", content="Message 1")
    memory.add_message(role="assistant", content="Message 2")
    memory.add_message(role="user", content="Message 3")
    context = memory.get_context()
    assert "Message 1" not in context
    assert "Message 2" in context
    assert "Message 3" in context
    memory.clear()
    assert memory.get_context() == ""


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_retrieval_memory():
    memory = RetrievalMemory()
    memory.add_message(role="user", content="What is the capital of France?")
    memory.add_message(role="assistant", content="The capital of France is Paris.")
    context = memory.get_context("What is the capital of France?")
    assert "Paris" in context
    memory.clear()
    assert memory.get_context("irrelevant query") == ""


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_memory_augmented_transformer_memory():
    memory = MemoryAugmentedTransformerMemory()
    memory.add_message(role="user", content="Hello, world!")
    assert memory.memory_embedding is not None
    memory.clear()
    assert memory.memory_embedding is None


# This test requires an OpenAI API key to be set in the environment.


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_hierarchical_memory():
    memory = HierarchicalMemory(short_term_threshold=2)
    memory.add_message(role="user", content="Message 1")
    memory.add_message(role="assistant", content="Message 2")
    memory.add_message(role="user", content="Message 3") # This should trigger summarization
    assert len(memory.short_term_memory) == 1
    assert memory.long_term_memory != ""
    memory.clear()
    assert memory.short_term_memory == []
    assert memory.long_term_memory == ""


def test_graph_based_memory():
    memory = GraphBasedMemory()
    memory.add_message(role="user", content="Hello")
    memory.add_message(role="assistant", content="Hi there!")
    context = memory.get_context()
    assert "user: Hello" in context
    assert "assistant: Hi there!" in context
    memory.clear()
    assert memory.get_context() == ""


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_compression_consolidation_memory():
    memory = CompressionConsolidationMemory(compression_threshold=2)
    memory.add_message(role="user", content="First message.")
    memory.add_message(role="assistant", content="Second message.") # This should trigger compression
    assert len(memory.history) == 0 # History should be cleared after compression
    assert len(memory.compressed_memory) == 1
    memory.add_message(role="user", content="Third message.")
    context = memory.get_context()
    assert "Compressed Past" in context
    assert "Third message" in context
    memory.clear()
    assert memory.get_context() == ""


def test_os_like_memory():
    memory = OSLikeMemory(page_size=2, max_pages=2)
    memory.add_message(role="user", content="Msg 1")
    memory.add_message(role="assistant", content="Msg 2") # Page 1 complete
    memory.add_message(role="user", content="Msg 3")
    memory.add_message(role="assistant", content="Msg 4") # Page 2 complete
    memory.add_message(role="user", content="Msg 5")
    memory.add_message(role="assistant", content="Msg 6") # Page 3 complete, Page 1 discarded
    
    context = memory.get_context()
    assert "Msg 1" not in context # Should be discarded
    assert "Msg 2" not in context # Should also be discarded as part of Page 1
    assert "Msg 3" in context
    assert "Msg 4" in context
    assert "Msg 5" in context
    
    memory.clear()
    assert memory.get_context() == ""


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_summarization_memory():
    memory = SummarizationMemory()
    memory.add_message(role="user", content="What is the capital of France?")
    memory.add_message(role="assistant", content="The capital of France is Paris.")
    context = memory.get_context()
    assert "paris" in context.lower()
    memory.clear()
    assert memory.get_context() == ""
