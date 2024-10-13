from typing import Dict, Any, List
import networkx as nx

class GamifiedInteractionModel:
    def __init__(self, dbn: nx.DiGraph):
        self.dbn = dbn

    def calculate_utility(self, node: str, interaction_state: Dict[str, Any]) -> float:
        """
        Calculate the utility of a node based on the current interaction state.
        """
        node_data = self.dbn.nodes[node]
        conditions = node_data.get('conditions', {})
        utility = 0

        for metric, (min_value, max_value) in conditions.items():
            if metric in interaction_state:
                value = interaction_state[metric]
                if min_value <= value <= max_value:
                    utility += 1  # Simple scoring, can be made more sophisticated

        return utility

    def select_next_node(self, current_node: str, interaction_state: Dict[str, Any]) -> str:
        """
        Select the next node to transition to based on utility calculations.
        """
        neighbors = list(self.dbn.successors(current_node))
        utilities = [self.calculate_utility(node, interaction_state) for node in neighbors]

        if not utilities:
            return current_node  # Stay in the current node if no valid transitions

        max_utility = max(utilities)
        best_nodes = [node for node, utility in zip(neighbors, utilities) if utility == max_utility]

        # If multiple nodes have the same utility, you can implement a tie-breaking mechanism here
        return best_nodes[0]

    def update_interaction_state(self, current_state: Dict[str, Any], analyzers: List[Any]) -> Dict[str, Any]:
        """
        Update the interaction state based on the current state and analyzer outputs.
        """
        new_state = current_state.copy()

        for analyzer in analyzers:
            metric, value = analyzer.analyze(current_state)
            new_state[metric] = value

        return new_state