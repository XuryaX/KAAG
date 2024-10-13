# kaag/core/kaag.py

from typing import Dict, Any, List
from ..llm.base import LLMInterface
from ..gim.gim import GamifiedInteractionModel
from ..dbn.dbn import DynamicBayesianNetwork
from ..analyzers.base import BaseAnalyzer
from jinja2 import Template
import importlib

class KAAG:
    def __init__(self, llm: LLMInterface, config: Dict[str, Any], template: Template):
        self.llm = llm
        self.config = config
        self.template = template
        self.dbn = DynamicBayesianNetwork()
        self.gim = GamifiedInteractionModel(self.dbn.graph)
        self.analyzers: List[BaseAnalyzer] = []
        self.interaction_state: Dict[str, Any] = {}
        self.current_node: str = self.config.get('initial_node', 'initial_contact')

        self._initialize_from_config()

    def _initialize_from_config(self):
        # Initialize DBN nodes and edges from config
        for stage in self.config['stages']:
            self.dbn.add_node(stage['id'], conditions=stage.get('conditions', {}))
        
        # Add edges (transitions between stages)
        for i in range(len(self.config['stages']) - 1):
            self.dbn.add_edge(self.config['stages'][i]['id'], self.config['stages'][i+1]['id'], lambda _: True)

        # Initialize interaction state
        self.interaction_state = {
            metric: details['initial']
            for metric, details in self.config['metrics'].items()
        }

        # Initialize analyzers dynamically
        self._load_analyzers()

    def _load_analyzers(self):
        for analyzer_config in self.config.get('analyzers', []):
            module_name, class_name = analyzer_config['class'].rsplit('.', 1)
            module = importlib.import_module(module_name)
            analyzer_class = getattr(module, class_name)
            self.analyzers.append(analyzer_class())

    def process_turn(self, user_input: str) -> str:
        # Update interaction state based on user input
        self.interaction_state['last_message'] = user_input
        self.interaction_state = self.dbn.update_interaction_state(self.interaction_state, self.analyzers)

        # Select next node using GIM
        self.current_node = self.gim.select_next_node(self.current_node, self.interaction_state)

        # Generate response using LLM with the template
        stage_instructions = next((stage['instructions'] for stage in self.config['stages'] if stage['id'] == self.current_node), '')
        prompt = self.template.render(
            persona=self.config['persona'],
            knowledge={'state': 'Current knowledge state'},  # This should be updated with actual knowledge state
            aptitude={
                'interaction_state': self.interaction_state,
                'stage_specific_instructions': stage_instructions
            },
            user_message=user_input
        )
        response = self.llm.generate(prompt)

        return response

    def get_current_state(self) -> Dict[str, Any]:
        return {
            'current_node': self.current_node,
            'interaction_state': self.interaction_state
        }