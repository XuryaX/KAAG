# kaag/core/dbn.py

from typing import Dict, Any, List
from .metrics import MetricsManager
from .stages import StageManager
import numpy as np

class DynamicBayesianNetwork:
    def __init__(self, metrics_manager: MetricsManager, stage_manager: StageManager):
        self.metrics_manager = metrics_manager
        self.stage_manager = stage_manager
        self.current_stage = self.stage_manager.get_current_stage(self.metrics_manager.get_metrics_state())
        self.conversation_history = []

    def update(self, metric_updates: Dict[str, float], user_input: str, ai_response: str):
        self.metrics_manager.update_metrics(metric_updates)
        self.conversation_history.append((user_input, ai_response))
        new_stage = self.stage_manager.get_current_stage(self.metrics_manager.get_metrics_state())
        
        if new_stage != self.current_stage:
            transition_probability = self._calculate_transition_probability(new_stage)
            if np.random.random() < transition_probability:
                self.current_stage = new_stage

    def get_current_stage(self):
        return self.current_stage

    def check_triggers(self, triggers: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = self.metrics_manager.get_metrics_state()
        for trigger in triggers:
            if eval(trigger['condition'], {**metrics, 'min': min, 'max': max}):
                return trigger
        return None

    def _calculate_transition_probability(self, new_stage):
        current_metrics = self.metrics_manager.get_metrics_state()
        stage_alignment = sum(1 for m, v in current_metrics.items() if new_stage.conditions[m][0] <= v <= new_stage.conditions[m][1])
        return stage_alignment / len(current_metrics)

    def get_context(self):
        return {
            "current_stage": self.current_stage.id,
            "metrics": self.metrics_manager.get_metrics_state(),
            "conversation_history": self.conversation_history[-5:]  # Last 5 turns
        }