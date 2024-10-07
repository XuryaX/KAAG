from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        pass