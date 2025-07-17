import pytest
from unittest.mock import MagicMock, patch
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
from agent_memory.llms.base import BaseLLM

# Mock the LLM to avoid actual API calls during agent tests
@pytest.fixture(autouse=True)
def mock_llm_for_tests(monkeypatch):
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.invoke.return_value.content = "Mocked LLM response"
    monkeypatch.setattr("agent_memory.llms.get_llm", lambda: mock_llm)
    return mock_llm


def test_agent_with_sequential_memory(mock_llm_for_tests):
    agent = Agent(memory_strategy=SequentialMemory)
    response = agent.chat("Hello")
    assert response == "Mocked LLM response"
    assert "user: Hello" in agent.memory.get_context()
    assert "assistant: Mocked LLM response" in agent.memory.get_context()
    agent.clear_memory()
    assert agent.memory.get_context() == ""


def test_agent_with_sliding_window_memory(mock_llm_for_tests):
    agent = Agent(memory_strategy=SlidingWindowMemory, window_size=2)
    agent.chat("Msg 1")
    agent.chat("Msg 2")
    agent.chat("Msg 3")
    context = agent.memory.get_context()
    assert "Msg 1" not in context
    assert "Msg 2" not in context
    assert "Msg 3" in context
    agent.clear_memory()
    assert agent.memory.get_context() == ""


def test_agent_with_summarization_memory(mock_llm_for_tests):
    agent = Agent(memory_strategy=SummarizationMemory)
    agent.chat("Long conversation about history.")
    context = agent.memory.get_context()
    # We can't assert the exact summary, but we can check if it's not empty
    assert context != ""
    agent.clear_memory()
    assert agent.memory.get_context() == ""


def test_agent_with_hierarchical_memory(mock_llm_for_tests):
    agent = Agent(memory_strategy=HierarchicalMemory, short_term_threshold=2)
    agent.chat("Msg 1")
    agent.chat("Msg 2")
    agent.chat("Msg 3")
    context = agent.memory.get_context()
    assert "Mocked LLM response" in context # Summary should be in context
    agent.clear_memory()
    assert agent.memory.long_term_memory == ""
    assert agent.memory.short_term_memory == []
    assert agent.memory.long_term_memory == ""


def test_agent_with_compression_consolidation_memory(mock_llm_for_tests):
    agent = Agent(memory_strategy=CompressionConsolidationMemory, compression_threshold=2)
    agent.chat("Msg A")
    agent.chat("Msg B")
    agent.chat("Msg C")
    context = agent.memory.get_context()
    assert "Mocked LLM response" in context # Compressed summary should be in context
    agent.clear_memory()
    assert agent.memory.get_context() == ""