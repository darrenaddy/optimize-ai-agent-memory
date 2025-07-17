from typing import List, Dict, Any
from .base import BaseMemory

class OSLikeMemory(BaseMemory):
    """
    A conceptual memory strategy that mimics operating system memory management principles,
    such as paging or virtual memory, to handle large memory contexts.
    
    This implementation uses a simple paging mechanism where messages are stored in 'pages'.
    When the number of pages exceeds a limit, older pages are 'swapped out' (discarded).
    """

    def __init__(self, page_size: int = 2, max_pages: int = 3):
        """
        Initializes the OSLikeMemory.

        Args:
            page_size: The number of messages per 'page'.
            max_pages: The maximum number of active 'pages' to keep in memory.
        """
        self.pages: List[List[Dict[str, str]]] = []
        self.current_page: List[Dict[str, str]] = []
        self.page_size = page_size
        self.max_pages = max_pages

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the current page. If the page is full, a new page is created.
           If max_pages is exceeded, the oldest page is discarded.
        """
        self.current_page.append({"role": role, "content": content})
        if len(self.current_page) >= self.page_size:
            self.pages.append(self.current_page[:])
            self.current_page = []
            
            if len(self.pages) > self.max_pages:
                self.pages.pop(0) # Discard the oldest page

    def get_context(self) -> str:
        """
        Retrieves the context from all active pages, simulating a contiguous memory space.
        """
        all_messages = []
        for page in self.pages:
            all_messages.extend(page)
        all_messages.extend(self.current_page) # Add messages from the current, uncommitted page
        
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in all_messages])

    def clear(self) -> None:
        """
        Clears all pages and the current page.
        """
        self.pages = []
        self.current_page = []
