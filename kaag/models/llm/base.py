# kaag/models/llm/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def generate_with_history(self, prompt: str, history: List[Dict[str, str]], **kwargs) -> str:
        pass