import matplotlib.pyplot as plt
from typing import Dict, List

class Visualizer:
    def __init__(self):
        self.metric_history = {}
        self.stage_history = []

    def update(self, metrics: Dict[str, float], current_stage: str):
        for metric, value in metrics.items():
            if metric not in self.metric_history:
                self.metric_history[metric] = []
            self.metric_history[metric].append(value)
        self.stage_history.append(current_stage)

    def plot_metrics(self):
        plt.figure(figsize=(12, 6))
        for metric, values in self.metric_history.items():
            plt.plot(values, label=metric)
        plt.legend()
        plt.title("Metric Changes Over Time")
        plt.xlabel("Turn")
        plt.ylabel("Value")
        plt.show()

    def plot_stages(self):
        plt.figure(figsize=(12, 4))
        unique_stages = list(set(self.stage_history))
        stage_indices = [unique_stages.index(stage) for stage in self.stage_history]
        plt.plot(stage_indices, marker='o')
        plt.yticks(range(len(unique_stages)), unique_stages)
        plt.title("Conversation Stage Progression")
        plt.xlabel("Turn")
        plt.ylabel("Stage")
        plt.show()