import pytest
from unittest.mock import MagicMock, patch
from agent_memory.agent import Agent
from agent_memory.strategies.sequential import SequentialMemory
from agent_memory.strategies.sliding_window import SlidingWindowMemory
from agent_memory.strategies.summarization import SummarizationMemory

# Mock the ChatOpenAI to avoid actual API calls during agent tests
@pytest.fixture(autouse=True)
def mock_openai_chat(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Mocked AI response"
    monkeypatch.setattr("agent_memory.agent.ChatOpenAI", lambda api_key, **kwargs: mock_llm)
    monkeypatch.setattr("agent_memory.strategies.summarization.ChatOpenAI", lambda api_key, **kwargs: mock_llm)
    monkeypatch.setattr("agent_memory.strategies.retrieval.OpenAIEmbeddings", lambda api_key, **kwargs: MagicMock())
    monkeypatch.setattr("agent_memory.strategies.hierarchical.ChatOpenAI", lambda api_key, **kwargs: mock_llm)
    monkeypatch.setattr("agent_memory.strategies.compression_consolidation.ChatOpenAI", lambda api_key, **kwargs: mock_llm)


def test_agent_with_sequential_memory():
    agent = Agent(memory_strategy=SequentialMemory)
    response = agent.chat("Hello")
    assert response == "Mocked AI response"
    assert "user: Hello" in agent.memory.get_context()
    assert "assistant: Mocked AI response" in agent.memory.get_context()
    agent.clear_memory()
    assert agent.memory.get_context() == ""


def test_agent_with_sliding_window_memory():
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


def test_agent_with_summarization_memory():
    agent = Agent(memory_strategy=SummarizationMemory)
    agent.chat("Long conversation about history.")
    context = agent.memory.get_context()
    # We can't assert the exact summary, but we can check if it's not empty
    assert context != ""
    agent.clear_memory()
    assert agent.memory.get_context() == ""
