from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseRetriever(ABC):
    """
    Abstract base class for knowledge retrievers.
    """

    @abstractmethod
    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Retrieve knowledge based on the given query.

        Args:
            query (str): The query to retrieve knowledge for.

        Returns:
            Dict[str, Any]: Retrieved knowledge.
        """
        pass