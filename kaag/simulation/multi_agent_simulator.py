from typing import Dict, Any, List
from ..core.dbn import DynamicBayesianNetwork
from ..models.llm.base import BaseLLM
from ..analyzers.base_analyzer import BaseAnalyzer

class MultiAgentSimulator:
    def __init__(self, agents: Dict[str, DynamicBayesianNetwork], llm: BaseLLM, config: Dict[str, Any], analyzers: List[BaseAnalyzer]):
        self.agents = agents
        self.llm = llm
        self.config = config
        self.analyzers = analyzers
        self.conversation_history = []

    def run_interaction(self, speaker: str, utterance: str) -> Dict[str, str]:
        responses = {}
        for agent_name, agent_dbn in self.agents.items():
            if agent_name != speaker:
                context = self._generate_context(agent_name, speaker, utterance)
                response = self.llm.generate(context)
                self._update_agent(agent_name, utterance, response)
                responses[agent_name] = response

        self.conversation_history.append({"speaker": speaker, "utterance": utterance, "responses": responses})
        return responses

    def _generate_context(self, agent_name: str, speaker: str, utterance: str) -> str:
        agent_dbn = self.agents[agent_name]
        current_stage = agent_dbn.get_current_stage()
        metrics_state = agent_dbn.metrics_manager.get_metrics_state()

        context = f"""
        Agent: {agent_name}
        Speaker: {speaker}
        Utterance: {utterance}
        Current interaction stage: {current_stage.id}
        Stage-specific instructions: {current_stage.instructions}
        Current metrics state: {metrics_state}
        Conversation history: {self._format_conversation_history()}
        AI response:
        """
        return context

    def _update_agent(self, agent_name: str, utterance: str, response: str):
        agent_dbn = self.agents[agent_name]
        analysis_results = self._analyze_interaction(utterance, response)
        agent_dbn.update(analysis_results)

    def _analyze_interaction(self, utterance: str, response: str) -> Dict[str, float]:
        results = {}
        for analyzer in self.analyzers:
            results.update(analyzer.analyze(utterance, response, {}))
        return results

    def _format_conversation_history(self) -> str:
        return "\n".join([f"{turn['speaker']}: {turn['utterance']}" for turn in self.conversation_history[-5:]])