import numpy as np
from typing import Dict, Any, List
from .metrics import MetricsManager
from .stages import StageManager

class DynamicBayesianNetwork:
    def __init__(self, metrics_manager: MetricsManager, stage_manager: StageManager):
        self.metrics_manager = metrics_manager
        self.stage_manager = stage_manager
        self.current_stage = self.stage_manager.get_current_stage(self.metrics_manager.get_metrics_state())
        self.conversation_history = []
        self.transition_matrix = np.ones((len(self.stage_manager.stages), len(self.stage_manager.stages))) / len(self.stage_manager.stages)

    def update(self, metric_updates: Dict[str, float], user_input: str, ai_response: str):
        self.metrics_manager.update_metrics(metric_updates)
        self.conversation_history.append((user_input, ai_response))
        new_stage = self.stage_manager.get_current_stage(self.metrics_manager.get_metrics_state())
        
        if new_stage != self.current_stage:
            transition_probability = self._calculate_transition_probability(new_stage)
            if np.random.random() < transition_probability:
                self._update_transition_matrix(self.current_stage, new_stage)
                self.current_stage = new_stage

    def get_current_stage(self):
        return self.current_stage

    def _calculate_transition_probability(self, new_stage):
        current_index = self.stage_manager.stages.index(self.current_stage)
        new_index = self.stage_manager.stages.index(new_stage)
        return self.transition_matrix[current_index, new_index]

    def _update_transition_matrix(self, old_stage, new_stage):
        old_index = self.stage_manager.stages.index(old_stage)
        new_index = self.stage_manager.stages.index(new_stage)
        self.transition_matrix[old_index, new_index] += 0.1
        self.transition_matrix[old_index] /= self.transition_matrix[old_index].sum()

    def get_context(self):
        return {
            "current_stage": self.current_stage.id,
            "metrics": self.metrics_manager.get_metrics_state(),
            "conversation_history": self.conversation_history[-5:]
        }

    def learn_from_success(self, success_score: float):
        self.stage_manager.adjust_stage_conditions(self.current_stage, self.metrics_manager.get_metrics_state(), success_score)