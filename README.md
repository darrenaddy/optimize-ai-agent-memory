# Optimize AI Agent Memory - Best Practices Edition

This project is a complete rewrite of the original `optimize-ai-agent-memory` repository, with a focus on professional software engineering best practices. It transforms the original collection of scripts into a modular, extensible, and testable Python library.

Many thanks to @FareedKahn-dev for his very good explanation of 9 beginner-to-advanced memory optimization techniques for AI agents in his Medium article: [Implementing 9 Techniques to Optimize AI Agent Memory](https://medium.com/gitconnected/implementing-9-techniques-to-optimize-ai-agent-memory-67d813e3d796)

## Core Improvements

- **Modular Architecture:** Each memory strategy is a self-contained class that inherits from a common `BaseMemory` interface.
- **Test-Driven:** The project includes a full suite of `pytest` tests to ensure correctness and prevent regressions.
- **Dependency Management:** Uses `poetry` for reproducible builds.
- **Configuration Management:** API keys and other settings are managed via environment variables, not hardcoded.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd optimize-ai-agent-memory
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Set up your environment:**
   - Rename `.env.example` to `.env`.
   - Add your OpenAI API key to the `.env` file. You can obtain your API key from [OpenAI platform website](https://platform.openai.com/account/api-keys).

## Usage

To see the different memory strategies in action, run the `example.py` script:

```bash
poetry run python example.py
```

**Note:** Some memory strategies (Summarization, Retrieval, Hierarchical, Compression & Consolidation) require an active OpenAI API key to function correctly. If `OPENAI_API_KEY` is not set, these examples will print a warning and may not produce meaningful output.

### Memory Strategies Implemented:

1.  **Sequential Memory:**
    *   **Concept:** This is the most straightforward memory strategy. It involves storing every piece of information (e.g., user inputs, agent responses) in a chronological list. The entire list is then passed as context to the language model for every new turn in the conversation.
    *   **Advantages:** Extremely simple to implement and guarantees that no information is lost from the conversation history. Ideal for short, focused interactions where the full context is always relevant.
    *   **Disadvantages:** Becomes highly inefficient and expensive as the conversation length grows. Large contexts consume more tokens (leading to higher costs) and can exceed the context window limits of most language models, leading to truncation and loss of older, potentially relevant, information.
    *   **Use Cases:** Simple chatbots, short Q&A sessions, or scenarios where conversation length is strictly limited.
2.  **Sliding Window Memory:**
    *   **Concept:** This strategy maintains a fixed-size window of the most recent interactions. As new messages are added, the oldest messages fall out of the window, ensuring the context provided to the LLM remains within a manageable token limit.
    *   **Advantages:** Effectively manages context length, preventing token limit overruns and controlling costs. It's simple to implement and ensures that the most recent and often most relevant information is always available.
    *   **Disadvantages:** Information outside the window is permanently lost, regardless of its importance. This can lead to the agent forgetting crucial details from earlier in the conversation if they are no longer within the active window.
    *   **Use Cases:** Conversations where only recent history is critical, such as short-term task-oriented dialogues, or when strict token limits must be enforced.
3.  **Summarization-Based Memory:**
    *   **Concept:** Instead of discarding old information, this strategy periodically summarizes past conversations using a language model. The summary then replaces the detailed older messages, keeping the overall context concise while retaining key information.
    *   **Advantages:** Reduces context length significantly, saving tokens and allowing for longer conversations. It attempts to preserve the essence of past interactions, mitigating the 'forgetting' problem of sliding windows.
    *   **Disadvantages:** Summarization itself consumes tokens and introduces latency. The quality of the summary depends heavily on the LLM's summarization capabilities, and critical details might be lost or misinterpreted during the summarization process.
    *   **Use Cases:** Long-running conversations where a high-level understanding of past interactions is sufficient, customer support chatbots, or personal assistants.
4.  **Retrieval-Based Memory (RAG):**
    *   **Concept:** This advanced strategy doesn't store the entire conversation directly in the LLM's context. Instead, it stores past interactions (or external knowledge) in a separate database (e.g., a vector store). When the agent needs context, it uses the current query to retrieve only the most relevant pieces of information from this database, which are then provided to the LLM.
    *   **Advantages:** Overcomes the context window limitation entirely, allowing access to a vast amount of information. It's highly scalable and can incorporate external, up-to-date knowledge. Reduces token usage by only providing relevant snippets.
    *   **Disadvantages:** Requires additional infrastructure (vector database, embedding models). Retrieval accuracy depends on the quality of embeddings and the relevance of the stored information. Can be complex to implement and fine-tune.
    *   **Use Cases:** Knowledge-intensive chatbots, agents needing access to large documentation bases, personalized assistants with extensive user profiles, or any application requiring up-to-date external information.
5.  **Memory-Augmented Transformer:**
    *   **Concept:** This strategy involves using a transformer model (like BERT) to encode the entire conversation history into a fixed-size vector (an embedding). This embedding then serves as a compressed representation of the memory. While the current implementation simply returns the raw conversation for context, in a real-world scenario, this embedding would be fed into another model or used to influence the generation process of the main LLM.
    *   **Advantages:** Provides a dense, fixed-size representation of potentially very long contexts, theoretically overcoming token limits. The transformer can capture complex relationships and nuances within the conversation.
    *   **Disadvantages:** Requires training or fine-tuning a transformer model for effective compression. The interpretability of the compressed memory is low. Integrating this embedding effectively into the LLM's generation process can be complex.
    *   **Use Cases:** Advanced research into memory mechanisms, scenarios where a fixed-size memory representation is crucial, or when fine-grained control over memory encoding is desired.
6.  **Hierarchical Memory:**
    *   **Concept:** This strategy combines the best aspects of sliding window and summarization. It maintains a short-term memory for recent, detailed interactions (like a sliding window) and a long-term memory for summarized older conversations. When the short-term memory reaches a certain size, its oldest parts are summarized and moved to the long-term memory.
    *   **Advantages:** Balances detail for recent interactions with conciseness for older context. Reduces token usage compared to sequential memory while retaining more information than a pure sliding window. Offers a more nuanced approach to memory management.
    *   **Disadvantages:** More complex to implement than simpler strategies. The summarization step still incurs LLM calls and potential information loss. Managing the transition between short-term and long-term memory requires careful design.
    *   **Use Cases:** Agents requiring both immediate conversational detail and a broader understanding of past interactions, such as complex customer service bots, personal assistants, or interactive storytelling agents.
7.  **Graph-Based Memory:**
    *   **Concept:** This strategy represents memories not as a linear sequence, but as a network of interconnected nodes (e.g., entities, events, facts) and edges (relationships between them). When new information is added, it's integrated into this knowledge graph. Retrieval involves traversing the graph to find relevant nodes and their connections based on the current query.
    *   **Advantages:** Enables sophisticated reasoning and inference by leveraging relationships between pieces of information. Can answer complex questions that require synthesizing information from disparate parts of the memory. Highly flexible and extensible.
    *   **Disadvantages:** Significantly more complex to implement and manage than other strategies. Requires robust entity extraction, relationship identification, and graph database technologies. Retrieval can be computationally intensive.
    *   **Use Cases:** Agents requiring deep understanding and reasoning capabilities, such as medical diagnostic assistants, legal research tools, or complex problem-solving AI.
8.  **Compression & Consolidation Memory:**
    *   **Concept:** This strategy focuses on actively reducing the size of the memory by compressing older information or consolidating redundant entries. This can involve summarization, but also techniques like identifying and merging duplicate facts, or abstracting specific instances into general rules. The goal is to maintain a rich, yet compact, representation of past interactions.
    *   **Advantages:** Significantly reduces memory footprint and token usage over time. Improves efficiency by removing noise and redundancy. Can lead to more coherent and focused context for the LLM.
    *   **Disadvantages:** Requires sophisticated algorithms for identifying redundancy and performing effective compression. Can be computationally intensive. Risk of losing subtle nuances or specific details during the consolidation process.
    *   **Use Cases:** Long-running agents that accumulate a large amount of information, agents operating under strict resource constraints, or systems where memory efficiency is paramount.
9.  **OS-Like Memory Management:**
    *   **Concept:** This is a highly conceptual strategy that draws parallels to how operating systems manage computer memory. It involves treating the agent's memory as a virtual space, using techniques like paging (dividing memory into fixed-size blocks) and swapping (moving data between fast and slow storage) to handle very large contexts efficiently. The goal is to provide the illusion of unlimited memory to the agent while managing physical memory constraints.
    *   **Advantages:** Offers a framework for managing extremely large memory contexts that would otherwise be impossible to handle. Can optimize for memory access patterns and reduce the need to load the entire history into active memory.
    *   **Disadvantages:** Very complex to implement and debug. The analogy to OS memory management is conceptual; a true implementation would require deep system-level integration. Performance can be unpredictable due to the overhead of paging and swapping.
    *   **Use Cases:** Highly specialized agents dealing with massive, long-term knowledge bases, or research into novel memory architectures for AI. This is more of a theoretical exploration than a practical, off-the-shelf solution for most applications.

## Testing

To run the test suite:

```bash
poetry run pytest
```

Some tests that interact with OpenAI APIs will be skipped if `OPENAI_API_KEY` is not set in your environment.
