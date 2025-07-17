from typing import List, Dict
import networkx as nx
from .base import BaseMemory

class GraphBasedMemory(BaseMemory):
    """
    A memory strategy that represents memories as nodes in a graph, with relationships between them.
    This allows for more complex retrieval and reasoning.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.message_count = 0

    def add_message(self, role: str, content: str) -> None:
        """Adds a message as a node in the graph."""
        node_id = f"message_{self.message_count}"
        self.graph.add_node(node_id, role=role, content=content, type="message")
        
        # Optionally, add edges to previous messages to maintain sequence
        if self.message_count > 0:
            self.graph.add_edge(f"message_{self.message_count-1}", node_id, relation="follows")
            
        self.message_count += 1

    def get_context(self, query: str = None) -> str:
        """
        Retrieves context from the graph. For simplicity, this currently returns all message nodes.
        In a real implementation, this would involve graph traversal and reasoning based on the query.
        """
        context_messages = []
        for node_id in sorted(self.graph.nodes()):
            if self.graph.nodes[node_id].get("type") == "message":
                context_messages.append(f"{self.graph.nodes[node_id]['role']}: {self.graph.nodes[node_id]['content']}")
        return "\n".join(context_messages)

    def clear(self) -> None:
        """
        Clears the graph-based memory.
        """
        self.graph.clear()
        self.message_count = 0

