from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_properties(self, properties: Any) -> Dict[str, Any]:
        pass