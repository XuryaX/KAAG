from abc import ABC, abstractmethod

class BaseKnowledgeRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> str:
        pass