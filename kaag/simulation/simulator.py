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
        ai_response = self.llm.generate(context)

        analysis_results = self._analyze_interaction(user_input, ai_response, context)
        self.dbn.update(analysis_results, user_input, ai_response)

        trigger = self.dbn.check_triggers(self.config['triggers'])
        if trigger:
            logging.info(f"Trigger activated: {trigger['action']}")
            return trigger['message']

        if any(phrase in ai_response.lower() for phrase in self.config['simulation']['end_phrases']):
            logging.info("Conversation ended")
            return f"{ai_response}\nConversation ended."

        return ai_response

    def _generate_context(self, user_input: str) -> str:
        dbn_context = self.dbn.get_context()
        retrieved_knowledge = self.retriever.retrieve(user_input)
        
        context = f"""
        Current interaction stage: {dbn_context['current_stage']}
        Stage-specific instructions: {self.dbn.current_stage.instructions}
        Global instructions: {self.config['global_instructions']}
        Current metrics state: {dbn_context['metrics']}
        Conversation history: {self._format_conversation_history(dbn_context['conversation_history'])}
        Relevant knowledge: {retrieved_knowledge}
        User's last input: {user_input}
        Your response:
        """
        return context

    def _analyze_interaction(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        results = {}
        for analyzer in self.analyzers:
            try:
                results.update(analyzer.analyze(user_input, ai_response, context))
            except Exception as e:
                logging.error(f"Error in {analyzer.__class__.__name__}: {str(e)}")
        return results

    def _format_conversation_history(self, history: List[tuple]) -> str:
        return "\n".join([f"User: {turn[0]}\nAI: {turn[1]}" for turn in history])