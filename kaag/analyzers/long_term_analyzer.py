from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, List

class LongTermAnalyzer(BaseAnalyzer):
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: List[Dict[str, float]] = []

    def analyze(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        current_metrics = context.get('metrics', {})
        self.history.append(current_metrics)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        results = {}
        if len(self.history) == self.window_size:
            for metric in current_metrics:
                values = [h[metric] for h in self.history]
                results[f"{metric}_trend"] = (values[-1] - values[0]) / self.window_size

        return results