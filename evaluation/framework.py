# kaag/evaluation/framework.py

from typing import List, Dict, Any
from ..core.dbn import DynamicBayesianNetwork
from ..models.llm.base import BaseLLM
from ..analyzers.base_analyzer import BaseAnalyzer
from ..simulation.simulator import Simulator
from .metrics import (
    calculate_context_score,
    calculate_adaptation_score,
    calculate_inference_score,
    calculate_steerability_score,
    calculate_satisfaction_score,
    calculate_coherence_score,
    calculate_information_density_score
)

class EvaluationFramework:
    def __init__(self, dbn: DynamicBayesianNetwork, llm: BaseLLM, config: Dict[str, Any], analyzers: List[BaseAnalyzer]):
        self.simulator = Simulator(dbn, llm, config, analyzers)
        self.config = config
        self.previous_metrics = {}

    def evaluate_turn(self, user_input: str, expected_output: str) -> Dict[str, Any]:
        generated_output = self.simulator.run_interaction(user_input)
        current_metrics = self.simulator.dbn.metrics_manager.get_metrics_state()

        evaluation = {
            "context_score": calculate_context_score(generated_output, expected_output),
            "adaptation_score": calculate_adaptation_score(self.previous_metrics, current_metrics),
            "inference_score": calculate_inference_score(generated_output, expected_output),
            "steerability_score": calculate_steerability_score(generated_output, user_input, expected_output),
            "coherence_score": calculate_coherence_score(generated_output),
            "information_density_score": calculate_information_density_score(generated_output),
            "generated_output": generated_output
        }

        self.previous_metrics = current_metrics
        return evaluation

    def evaluate_conversation(self, transcript: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results = []
        generated_responses = []
        expected_responses = []

        for i in range(0, len(transcript) - 1, 2):
            user_input = transcript[i]['utterance']
            expected_output = transcript[i+1]['utterance'] if i+1 < len(transcript) else ""
            turn_result = self.evaluate_turn(user_input, expected_output)
            results.append(turn_result)
            generated_responses.append(turn_result['generated_output'])
            expected_responses.append(expected_output)

        overall_satisfaction = calculate_satisfaction_score(generated_responses, expected_responses)
        for result in results:
            result['satisfaction_score'] = overall_satisfaction

        return results