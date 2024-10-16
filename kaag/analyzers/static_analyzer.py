from typing import Dict, Any, Tuple
from .base import BaseAnalyzer

class StaticAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.scenario_data = None

    def analyze(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.scenario_data is None:
            raise Exception("Scenario data not set")
        
        scenario_data = self.scenario_data
        self.scenario_data = None #To ensure that the scenario data is updated in every turn.
        
        return scenario_data
        
    
    def update_properties(self, properties: Any) -> None:
        self.scenario_data = properties