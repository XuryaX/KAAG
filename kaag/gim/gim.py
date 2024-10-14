from typing import Dict, Any
import math
import random

class GamifiedInteractionModel:
    def __init__(self, dbn: 'DynamicBayesianNetwork'):
        self.dbn = dbn

    def select_next_node(self, current_node: str, interaction_state: Dict[str, Any]) -> str:
        if self.dbn.is_leaf_node(current_node):
            return current_node

        possible_transitions = self.dbn.get_possible_transitions(current_node, interaction_state)
        if not possible_transitions:
            return current_node

        utilities = [
            self.dbn.get_utility(current_node, next_node, interaction_state)
            for next_node in possible_transitions
        ]

        # Softmax selection with temperature
        temperature = 0.5  # Adjust this value to control randomness
        exp_utilities = [math.exp(u / temperature) for u in utilities]
        total = sum(exp_utilities)
        probabilities = [eu / total for eu in exp_utilities]

        selected_node = random.choices(possible_transitions, weights=probabilities, k=1)[0]

        # Add some randomness to occasionally explore non-optimal paths
        if random.random() < 0.1:  # 10% chance to explore
            return random.choice(possible_transitions)

        return selected_node