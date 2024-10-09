# kaag/evaluation/framework.py

from typing import List, Dict, Any
from kaag.core.dbn import DynamicBayesianNetwork
from kaag.models.llm.base import BaseLLM
from kaag.analyzers.base_analyzer import BaseAnalyzer
from kaag.simulation.simulator import Simulator
from kaag.models.retriever.static_retriever import StaticDBRetriever
from .metrics import (
    calculate_context_score,
    calculate_adaptation_score,
    calculate_inference_score,
    calculate_steerability_score
)

class EvaluationFramework:
    def __init__(self, dbn: DynamicBayesianNetwork, llm: BaseLLM, config: Dict[str, Any], analyzers: List[BaseAnalyzer]):
        static_retriever = StaticDBRetriever()
        self.simulator = Simulator(dbn, llm, static_retriever, config, analyzers)
        self.config = config
        self.previous_metrics = {}
        self.previous_stage = None

    def evaluate_turn(self, user_input: str, expected_output: str) -> Dict[str, Any]:
        generated_output = self.simulator.run_interaction(user_input)
        current_metrics = self.simulator.dbn.metrics_manager.get_metrics_state()
        current_stage = self.simulator.dbn.get_current_stage().id
        context = self.simulator.dbn.get_context()

        evaluation = {
            "context_score": calculate_context_score(generated_output, expected_output, context),
            "adaptation_score": calculate_adaptation_score(self.previous_metrics, current_metrics, {"current_stage": current_stage, "previous_stage": self.previous_stage}),
            "inference_score": calculate_inference_score(generated_output, expected_output, context),
            "steerability_score": calculate_steerability_score(generated_output, user_input, expected_output, context),
            "generated_output": generated_output
        }

        self.previous_metrics = current_metrics
        self.previous_stage = current_stage
        return evaluation

    def evaluate_conversation(self, transcript: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(transcript) - 1, 2):
            user_input = transcript[i]['utterance']
            expected_output = transcript[i+1]['utterance'] if i+1 < len(transcript) else ""
            turn_result = self.evaluate_turn(user_input, expected_output)
            results.append(turn_result)

        return results