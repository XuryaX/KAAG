from .base_analyzer import BaseAnalyzer
from typing import Dict, Any
import re

class CustomMetricAnalyzer(BaseAnalyzer):
    def __init__(self, metrics_config: Dict[str, Dict[str, Any]]):
        self.metrics_config = metrics_config

    def analyze(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        results = {}
        for metric, config in self.metrics_config.items():
            if 'keywords' in config:
                count = sum(1 for keyword in config['keywords'] if keyword.lower() in (user_input + ai_response).lower())
                results[metric] = min(count * config.get('weight', 1), config.get('max', 1))
            elif 'regex' in config:
                matches = re.findall(config['regex'], user_input + ai_response, re.IGNORECASE)
                results[metric] = min(len(matches) * config.get('weight', 1), config.get('max', 1))
        return results