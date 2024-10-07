from typing import Dict, Any

class ExplanationGenerator:
    @staticmethod
    def generate_explanation(metrics: Dict[str, float], current_stage: str, previous_stage: str) -> str:
        explanation = f"Current stage: {current_stage}\n"
        if current_stage != previous_stage:
            explanation += f"Transitioned from: {previous_stage}\n"
        
        explanation += "Metrics influencing the decision:\n"
        for metric, value in metrics.items():
            explanation += f"- {metric}: {value}\n"
        
        return explanation