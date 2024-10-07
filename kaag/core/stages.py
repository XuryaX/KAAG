from typing import Dict, List, Any
import numpy as np

class Stage:
    def __init__(self, id: str, conditions: Dict[str, List[float]], instructions: str, custom_responses: Dict[str, str] = None):
        self.id = id
        self.conditions = conditions
        self.instructions = instructions
        self.custom_responses = custom_responses or {}

    def check_conditions(self, metrics: Dict[str, float]) -> float:
        condition_scores = []
        for m_id, (min_val, max_val) in self.conditions.items():
            if m_id in metrics:
                score = max(0, min(1, (metrics[m_id] - min_val) / (max_val - min_val)))
                condition_scores.append(score)
        return np.mean(condition_scores) if condition_scores else 0

    def get_response(self, key: str) -> str:
        return self.custom_responses.get(key, self.instructions)

class StageManager:
    def __init__(self, stages_config: List[Dict[str, Any]]):
        self.stages = [Stage(**stage_config) for stage_config in stages_config]

    def get_current_stage(self, metrics: Dict[str, float]) -> Stage:
        stage_scores = [(stage, stage.check_conditions(metrics)) for stage in self.stages]
        return max(stage_scores, key=lambda x: x[1])[0]

    def add_custom_stage(self, stage_config: Dict[str, Any]):
        self.stages.append(Stage(**stage_config))