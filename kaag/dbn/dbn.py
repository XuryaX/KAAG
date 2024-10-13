import networkx as nx
from typing import Any, Callable, Dict, List

class DynamicBayesianNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_name: str, initial_state: Any = None, conditions: Dict[str, tuple] = None):
        self.graph.add_node(node_name, state=initial_state, conditions=conditions or {})

    def add_edge(self, parent: str, child: str, condition: Callable[[Dict[str, Any]], bool]):
        self.graph.add_edge(parent, child, condition=condition)

    def update_interaction_state(self, current_state: Dict[str, Any], analyzers: List[Any]) -> Dict[str, Any]:
        """
        The function updates the interaction state based on the current state and analyzer outputs.
        """
        new_state = current_state.copy()

        for analyzer in analyzers:
            metric, value = analyzer.analyze(current_state)
            new_state[metric] = value

        return new_state

    def update_node_state(self, node: str, new_state: Any):
        self.graph.nodes[node]['state'] = new_state

    def get_node_state(self, node: str) -> Any:
        return self.graph.nodes[node].get('state')