# tests/test_dbn.py

"""
Unit tests for the Dynamic Bayesian Network.
"""

import unittest
from dbn.dbn import (
    create_dbn,
    add_node,
    add_edge,
    update_states,
    get_node_state,
)

class TestDynamicBayesianNetwork(unittest.TestCase):
    def setUp(self):
        """Set up a basic DBN for testing."""
        self.dbn = create_dbn()
        add_node(self.dbn, 'A', initial_state=True)
        add_node(self.dbn, 'B', initial_state=False)
        add_node(self.dbn, 'C', initial_state=None)

        def condition_edge_A_B(context):
            return context.get('weather') == 'rainy'

        def condition_edge_B_C(context):
            return context.get('traffic') == 'heavy'

        add_edge(self.dbn, 'A', 'B', condition=condition_edge_A_B)
        add_edge(self.dbn, 'B', 'C', condition=condition_edge_B_C)

    def test_states_with_conditions_met(self):
        """Test state updates when conditions are met."""
        context = {'weather': 'rainy', 'traffic': 'heavy'}
        update_states(self.dbn, context)
        self.assertTrue(get_node_state(self.dbn, 'A'))
        self.assertTrue(get_node_state(self.dbn, 'B'))
        self.assertTrue(get_node_state(self.dbn, 'C'))

    def test_states_with_conditions_not_met(self):
        """Test state updates when conditions are not met."""
        context = {'weather': 'sunny', 'traffic': 'light'}
        update_states(self.dbn, context)
        self.assertTrue(get_node_state(self.dbn, 'A'))
        self.assertFalse(get_node_state(self.dbn, 'B'))
        self.assertIsNone(get_node_state(self.dbn, 'C'))

    def test_partial_conditions_met(self):
        """Test state updates when some conditions are met."""
        context = {'weather': 'rainy', 'traffic': 'light'}
        update_states(self.dbn, context)
        self.assertTrue(get_node_state(self.dbn, 'A'))
        self.assertTrue(get_node_state(self.dbn, 'B'))
        self.assertTrue(get_node_state(self.dbn, 'C') is None)

if __name__ == '__main__':
    unittest.main()
