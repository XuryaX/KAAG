import unittest
from unittest.mock import Mock, patch
from kaag.simulation.simulator import Simulator
from kaag.core.dbn import DynamicBayesianNetwork
from kaag.models.llm.base import BaseLLM
from kaag.analyzers.base_analyzer import BaseAnalyzer

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.mock_dbn = Mock(spec=DynamicBayesianNetwork)
        self.mock_llm = Mock(spec=BaseLLM)
        self.mock_analyzer = Mock(spec=BaseAnalyzer)
        self.config = {
            'simulation': {
                'max_turns': 10,
                'end_phrases': ['end', 'stop'],
                'logging': {'enabled': False}
            },
            'global_instructions': 'Test instructions'
        }
        self.simulator = Simulator(self.mock_dbn, self.mock_llm, self.config, [self.mock_analyzer])

    def test_run_interaction(self):
        self.mock_llm.generate.return_value = "AI response"
        self.mock_analyzer.analyze.return_value = {"metric": 0.5}
        self.mock_dbn.check_triggers.return_value = None
        
        response = self.simulator.run_interaction("User input")
        
        self.assertEqual(response, "AI response")
        self.mock_llm.generate.assert_called_once()
        self.mock_analyzer.analyze.assert_called_once()
        self.mock_dbn.update.assert_called_once_with({"metric": 0.5})

    def test_max_turns(self):
        self.simulator.turn_count = 10
        response = self.simulator.run_interaction("User input")
        self.assertIn("Maximum turns reached", response)

    def test_end_phrase(self):
        self.mock_llm.generate.return_value = "Let's end the conversation"
        response = self.simulator.run_interaction("User input")
        self.assertIn("Conversation ended", response)

    def test_trigger_activation(self):
        self.mock_dbn.check_triggers.return_value = {"action": "test", "message": "Trigger activated"}
        response = self.simulator.run_interaction("User input")
        self.assertEqual(response, "Trigger activated")

if __name__ == '__main__':
    unittest.main()