from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        pass