from typing import Dict, Any, List
from ..llm.base import LLMInterface
from ..gim.gim import GamifiedInteractionModel
from ..dbn.dbn import DynamicBayesianNetwork
from ..analyzers.base import BaseAnalyzer
from jinja2 import Template
import importlib
import logging

class KAAG:
    def __init__(self, llm: LLMInterface, config: Dict[str, Any], template: Template):
        self.llm = llm
        self.config = config
        self.template = template
        self.dbn = DynamicBayesianNetwork()
        self.gim = GamifiedInteractionModel(self.dbn)
        self.analyzers: List[BaseAnalyzer] = []
        self.interaction_state: Dict[str, Any] = {}
        self.current_node: str = self.config.get('initial_node', 'initial_contact')
        self.conversation_history: List[Dict[str, str]] = []
        self.logger = self._setup_logger()

        self._initialize_from_config()

    def _setup_logger(self):
        logger = logging.getLogger('KAAG')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('kaag_log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _initialize_from_config(self):
        for stage in self.config['stages']:
            self.dbn.add_node(stage['id'], is_leaf=stage.get('is_leaf', False))
        
        for stage in self.config['stages']:
            for transition in stage.get('transitions', []):
                self.dbn.add_edge(
                    stage['id'],
                    transition['to'],
                    condition=self._create_condition_function(transition.get('conditions', {})),
                    utility=self._create_utility_function(transition.get('utility', lambda x: 0))
                )

        self.interaction_state = {
            metric: details['initial']
            for metric, details in self.config['metrics'].items()
        }

        self._load_analyzers()

    def _create_condition_function(self, conditions: Dict[str, Any]):
        def condition_function(interaction_state: Dict[str, Any]) -> bool:
            return all(
                interaction_state.get(metric, 0) >= min_value and interaction_state.get(metric, 0) <= max_value
                for metric, (min_value, max_value) in conditions.items()
            )
        return condition_function

    def _create_utility_function(self, utility_config):
        if callable(utility_config):
            return utility_config
        elif isinstance(utility_config, dict) and 'function' in utility_config:
            return eval(utility_config['function'])
        elif isinstance(utility_config, (int, float)):
            return lambda x: utility_config
        else:
            return lambda x: 0

    def _load_analyzers(self):
        for analyzer_config in self.config.get('analyzers', []):
            module_name, class_name = analyzer_config['class'].rsplit('.', 1)
            module = importlib.import_module(module_name)
            analyzer_class = getattr(module, class_name)
            self.analyzers.append(analyzer_class())

    def get_analyzers(self):
        return self.analyzers

    def process_turn(self, user_input: str) -> str:
        self.interaction_state['last_message'] = user_input
        for analyzer in self.analyzers:
            self.interaction_state.update(analyzer.analyze(self.interaction_state))

        self.logger.info(f"Current State: {self.get_current_state()}")

        if self.dbn.is_leaf_node(self.current_node):
            return self._handle_leaf_node(user_input)

        next_node = self.gim.select_next_node(self.current_node, self.interaction_state)
        self.logger.info(f"Transition: {self.current_node} -> {next_node}")
        self.current_node = next_node

        stage_instructions = next((stage['instructions'] for stage in self.config['stages'] if stage['id'] == self.current_node), '')
        stage_examples = next((stage['examples'] for stage in self.config['stages'] if stage['id'] == self.current_node), '')

        formatted_examples = [
            {"user": example['user'], "assistant": example['AI']}
            for example in stage_examples
        ]

        prompt = self.template.render(
            persona=self.config['persona'],
            knowledge={'state': 'Current knowledge state'},
            aptitude={
                'interaction_state': self.interaction_state,
                'stage_specific_instructions': stage_instructions
            },
            conversation_history=self._format_conversation_history(),
            user_message=user_input,
            examples=formatted_examples
        )
        response = self.llm.generate(prompt)

        self.conversation_history.append({"user": user_input, "assistant": response})
        return response

    def _handle_leaf_node(self, user_input) -> str:
        leaf_instructions = next((stage['instructions'] for stage in self.config['stages'] if stage['id'] == self.current_node), '')
        stage_examples = next((stage['examples'] for stage in self.config['stages'] if stage['id'] == self.current_node), '')
        formatted_examples = [
            {"user": example['user'], "assistant": example['AI']}
            for example in stage_examples
        ]
        prompt = self.template.render(
            persona=self.config['persona'],
            knowledge={'state': 'Current knowledge state'},
            aptitude={
                'interaction_state': self.interaction_state,
                'stage_specific_instructions': leaf_instructions
            },
            conversation_history=self._format_conversation_history(),
            user_message=user_input,
            examples=formatted_examples
        )
        response = self.llm.generate(prompt)
        self.logger.info(f"Reached leaf node: {self.current_node}. Conversation ended.")
        return response

    def _format_conversation_history(self) -> str:
        formatted_history = ""
        for turn in self.conversation_history[-5:]:  # Only use the last 5 turns
            formatted_history += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        return formatted_history

    def get_current_state(self) -> Dict[str, Any]:
        return {
            'current_node': self.current_node,
            'interaction_state': self.interaction_state
        }