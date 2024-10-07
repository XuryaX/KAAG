import unittest
from kaag.core.metrics import Metric, MetricsManager
from kaag.core.stages import Stage, StageManager
from kaag.core.dbn import DynamicBayesianNetwork

class TestMetrics(unittest.TestCase):
    def test_metric_update(self):
        metric = Metric("test", 0, 100, 50)
        metric.update(10)
        self.assertEqual(metric.value, 60)
        metric.update(-20)
        self.assertEqual(metric.value, 40)
        metric.update(100)
        self.assertEqual(metric.value, 100)

class TestStages(unittest.TestCase):
    def test_stage_conditions(self):
        stage = Stage("test", {"metric1": [0, 50], "metric2": [50, 100]}, "Test instructions")
        self.assertTrue(stage.check_conditions({"metric1": 25, "metric2": 75}))
        self.assertFalse(stage.check_conditions({"metric1": 75, "metric2": 25}))

class TestDynamicBayesianNetwork(unittest.TestCase):
    def test_dbn_update(self):
        metrics_config = {
            "metric1": {"min": 0, "max": 100, "initial": 50},
            "metric2": {"min": 0, "max": 100, "initial": 50}
        }
        stages_config = [
            {"id": "stage1", "conditions": {"metric1": [0, 50], "metric2": [0, 50]}, "instructions": "Stage 1"},
            {"id": "stage2", "conditions": {"metric1": [50, 100], "metric2": [50, 100]}, "instructions": "Stage 2"}
        ]
        metrics_manager = MetricsManager(metrics_config)
        stage_manager = StageManager(stages_config)
        dbn = DynamicBayesianNetwork(metrics_manager, stage_manager)
        
        self.assertEqual(dbn.get_current_stage().id, "stage1")
        dbn.update({"metric1": 25, "metric2": 75})
        self.assertEqual(dbn.get_current_stage().id, "stage2")

if __name__ == '__main__':
    unittest.main()