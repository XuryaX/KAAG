# kaag/simulation/simulator.py

from typing import Dict, Any, List
from ..core.dbn import DynamicBayesianNetwork
from ..models.llm.base import BaseLLM
from ..analyzers.base_analyzer import BaseAnalyzer
from ..models.retriever.base import BaseRetriever
import logging

class Simulator:
    def __init__(self, dbn: DynamicBayesianNetwork, llm: BaseLLM, retriever: BaseRetriever, config: Dict[str, Any], analyzers: List[BaseAnalyzer]):
        self.dbn = dbn
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.analyzers = analyzers
        self.turn_count = 0
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            filename=self.config['simulation']['logging']['file_path'],
            level=getattr(logging, self.config['simulation']['logging']['level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def run_interaction(self, user_input: str) -> str:
        self.turn_count += 1
        if self.turn_count > self.config['simulation']['max_turns']:
            return "Maximum turns reached. Ending conversation."

        context = self._generate_context(user_input)
        plan = self._generate_plan(context)
        ai_response = self._generate_response(context, plan)

        analysis_results = self._analyze_interaction(user_input, ai_response, context)
        self.dbn.update(analysis_results, user_input, ai_response)

        if any(phrase in ai_response.lower() for phrase in self.config['simulation']['end_phrases']):
            logging.info("Conversation ended")
            return f"{ai_response}\nConversation ended."

        return ai_response

    def _generate_context(self, user_input: str) -> Dict[str, Any]:
        dbn_context = self.dbn.get_context()
        retrieved_knowledge = self.retriever.retrieve(user_input)
        
        return {
            **dbn_context,
            "retrieved_knowledge": retrieved_knowledge,
            "user_input": user_input
        }

    def _generate_plan(self, context: Dict[str, Any]) -> str:
        plan_prompt = f"""
        Given the following context, generate a plan for the next response:
        Current stage: {context['current_stage']}
        Metrics: {context['metrics']}
        User input: {context['user_input']}

        Plan:
        """
        return self.llm.generate(plan_prompt)

    def _generate_response(self, context: Dict[str, Any], plan: str) -> str:
        response_prompt = f"""
        Context:
        Current stage: {context['current_stage']}
        Metrics: {context['metrics']}
        User input: {context['user_input']}
        
        Plan: {plan}
        
        Generate a response based on the above context and plan:
        """
        return self.llm.generate(response_prompt)

    def _analyze_interaction(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        results = {}
        for analyzer in self.analyzers:
            try:
                results.update(analyzer.analyze(user_input, ai_response, context))
            except Exception as e:
                logging.error(f"Error in {analyzer.__class__.__name__}: {str(e)}")
        return results