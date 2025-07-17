import os
import pytest
from unittest.mock import MagicMock, patch
from agent_memory.strategies.sequential import SequentialMemory
from agent_memory.strategies.sliding_window import SlidingWindowMemory
from agent_memory.strategies.summarization import SummarizationMemory
from agent_memory.strategies.retrieval import RetrievalMemory
from agent_memory.strategies.memory_augmented_transformer import MemoryAugmentedTransformerMemory
from agent_memory.strategies.hierarchical import HierarchicalMemory
from agent_memory.strategies.compression_consolidation import CompressionConsolidationMemory
from agent_memory.strategies.graph_based import GraphBasedMemory
from agent_memory.strategies.os_like_memory import OSLikeMemory
from agent_memory.llms.base import BaseLLM

@pytest.fixture(autouse=True)
def mock_llm_for_strategies(monkeypatch):
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.invoke.return_value.content = "Mocked LLM response for summarization/compression"
    monkeypatch.setattr("agent_memory.strategies.retrieval.OpenAIEmbeddings", MagicMock)
    monkeypatch.setattr("agent_memory.llms.get_llm", lambda: mock_llm)
    return mock_llm


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


def test_retrieval_memory(monkeypatch):
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3] # Mock embedding for query
    monkeypatch.setattr("agent_memory.strategies.retrieval.OpenAIEmbeddings", lambda **kwargs: mock_embeddings)

    with patch('langchain_community.vectorstores.faiss.FAISS.from_texts') as mock_faiss_from_texts:
        mock_instance = MagicMock()
        mock_instance.similarity_search.return_value = [MagicMock(page_content="Paris")]
        mock_faiss_from_texts.return_value = mock_instance

        memory = RetrievalMemory()
        memory.add_message(role="user", content="What is the capital of France?")
        memory.add_message(role="assistant", content="The capital of France is Paris.")
        context = memory.get_context(query="What is the capital of France?")
        assert "Paris" in context
        memory.clear()
        assert memory.get_context("irrelevant query") == ""


def test_memory_augmented_transformer_memory():
    memory = MemoryAugmentedTransformerMemory()
    memory.add_message(role="user", content="Hello, world!")
    assert memory.memory_embedding is not None
    memory.clear()
    assert memory.memory_embedding is None


# This test requires an OpenAI API key to be set in the environment.


def test_hierarchical_memory(mock_llm_for_strategies):
    memory = HierarchicalMemory(llm=mock_llm_for_strategies, short_term_threshold=2)
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


def test_compression_consolidation_memory(mock_llm_for_strategies):
    memory = CompressionConsolidationMemory(llm=mock_llm_for_strategies, compression_threshold=2)
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


def test_summarization_memory(mock_llm_for_strategies):
    memory = SummarizationMemory(llm=mock_llm_for_strategies)
    memory.add_message(role="user", content="What is the capital of France?")
    memory.add_message(role="assistant", content="The capital of France is Paris.")
    context = memory.get_context()
    assert "Mocked LLM response for summarization/compression" in context
    memory.clear()
    assert memory.get_context() == ""
