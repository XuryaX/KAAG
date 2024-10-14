import networkx as nx
from typing import Any, Callable, Dict, List

class DynamicBayesianNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_name: str, is_leaf: bool = False):
        self.graph.add_node(node_name, is_leaf=is_leaf)

    def add_edge(self, parent: str, child: str, condition: Callable[[Dict[str, Any]], bool], utility: Callable[[Dict[str, Any]], float]):
        self.graph.add_edge(parent, child, condition=condition, utility=utility)

    def get_possible_transitions(self, current_node: str, interaction_state: Dict[str, Any]) -> List[str]:
        return [
            child for child in self.graph.successors(current_node)
            if self.graph[current_node][child]['condition'](interaction_state)
        ]

    def get_utility(self, parent: str, child: str, interaction_state: Dict[str, Any]) -> float:
        return self.graph[parent][child]['utility'](interaction_state)

    def is_leaf_node(self, node: str) -> bool:
        return self.graph.nodes[node].get('is_leaf', False)