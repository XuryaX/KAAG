from typing import Dict, Any, List
import numpy as np

class Metric:
    def __init__(self, id: str, min: float, max: float, initial: float, decay_factor: float = 0.95):
        self.id = id
        self.min_value = min
        self.max_value = max
        self.value = initial
        self.history = [initial]
        self.decay_factor = decay_factor

    def update(self, change: float):
        new_value = self.value * self.decay_factor + change
        self.value = max(self.min_value, min(self.max_value, new_value))
        self.history.append(self.value)

    def get_trend(self, window: int = 5) -> float:
        if len(self.history) < 2:
            return 0
        recent = self.history[-window:]
        return np.polyfit(range(len(recent)), recent, 1)[0]

class MetricsManager:
    def __init__(self, metrics_config: Dict[str, Dict[str, Any]]):
        self.metrics = {m_id: Metric(m_id, **m_config) for m_id, m_config in metrics_config.items()}

    def update_metrics(self, updates: Dict[str, float]):
        for metric_id, change in updates.items():
            if metric_id in self.metrics:
                self.metrics[metric_id].update(change)

    def get_metrics_state(self) -> Dict[str, float]:
        return {m_id: metric.value for m_id, metric in self.metrics.items()}

    def get_metrics_trends(self) -> Dict[str, float]:
        return {m_id: metric.get_trend() for m_id, metric in self.metrics.items()}

    def add_custom_metric(self, metric_id: str, config: Dict[str, Any]):
        self.metrics[metric_id] = Metric(metric_id, **config)