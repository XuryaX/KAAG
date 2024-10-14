from typing import Dict, Any, Tuple
from .base import BaseAnalyzer

class StaticAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.scenario_data = None

    def analyze(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.scenario_data:
            raise Exception("Scenario data not set")
        
    
    def update_properties(self, properties: Any) -> Dict[str, Any]:
        return super().update_properties(properties)