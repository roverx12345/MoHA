import tempfile
import unittest
from pathlib import Path

from moha.demo import DemoJudge, DemoRunner, sample
from moha.loop import Calibrator, SearchPolicy
from moha.models import Episode, Harness, ValidationPolicy
from moha.shared import (AGGREGATION_RULE, NamespacedStore, SharedCalibrator,
                         compare_shared)
from moha.store import RunStore


def runs(samples, harness, answers, repeat=0, cost=100):
    return [Episode(item.sample_id, item.id, harness.id, repeat, list(item.video_key),
                    item.expected_answer, answer, "completed", usage={"video_tokens": cost})
            for item, answer in zip(samples, answers)]


class SharedComparisonTests(unittest.TestCase):
    def test_models_are_pooled_without_hiding_per_stack_regressions(self):
        samples = [sample(f"val{i}") for i in range(4)]
        old, new = Harness(), Harness(overview=True)
        left = {
            "a": [runs(samples, old, ["B"] * 4)],
            "b": [runs(samples, old, ["A"] * 4)],
        }
        right = {
            "a": [runs(samples, new, ["A"] * 4)],
            "b": [runs(samples, new, ["B"] * 4)],
        }
        result = compare_shared(left, right, samples, old, new,
                                ValidationPolicy(bootstrap_samples=30))
        self.assertEqual(result["aggregation"], AGGREGATION_RULE)
        self.assertEqual(result["episode_count"], 8)
        self.assertEqual(result["accuracy_delta"], 0)
        self.assertEqual(result["by_stack"]["a"]["accuracy_delta"], 1)
        self.assertEqual(result["by_stack"]["b"]["accuracy_delta"], -1)
        self.assertEqual(result["reason"], "insufficient_gain")

    def test_missing_cost_in_any_stack_blocks_shared_promotion(self):
        samples = [sample("val")]
        old, new = Harness(), Harness(overview=True)
        left = {"a": [runs(samples, old, ["B"])], "b": [runs(samples, old, ["B"])]}
        right = {"a": [runs(samples, new, ["A"])], "b": [runs(samples, new, ["A"])]}
        right["b"][0][0].usage = {}
        result = compare_shared(left, right, samples, old, new,
                                ValidationPolicy(bootstrap_samples=10))
        self.assertEqual(result["reason"], "missing_measured_cost")
        self.assertFalse(result["accepted"])


class SharedLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "run"

    def test_one_harness_is_selected_and_frozen_for_all_stacks(self):
        with RunStore(self.root, {"shared": "unit"}) as store:
            calibrators = {}
            for stack_id in ("model-a", "model-b"):
                namespace = NamespacedStore(store, f"models/{stack_id}")
                calibrators[stack_id] = Calibrator(
                    runners=[DemoRunner()], judges=[DemoJudge()], store=namespace,
                    calibration=[sample("cal")],
                    validation=[sample("val0"), sample("val1")],
                    search=SearchPolicy(max_rounds=2),
                    validation_policy=ValidationPolicy(bootstrap_samples=20),
                    perception_calibration=True)
            result = SharedCalibrator(calibrators=calibrators, store=store).run()
            frozen = store.read("frozen_harness.json")
        self.assertTrue(result["harness"]["overview"])
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["history"][0]["validation"]["stack_count"], 2)
        self.assertEqual(result["perception_calibration"]["selected_policy_id"], "fps1_scale1")
        self.assertEqual(result["perception_calibration"]["rows"][0]["episodes"], 4)
        self.assertEqual(frozen["schema"], "moha_shared_frozen_v1")
        self.assertEqual(frozen["stack_ids"], ["model-a", "model-b"])
        self.assertTrue((self.root / "models/model-a/episodes").is_dir())
        self.assertTrue((self.root / "models/model-b/episodes").is_dir())

    def test_namespaces_cannot_escape_shared_run(self):
        with RunStore(self.root, {}) as store:
            with self.assertRaises(ValueError):
                NamespacedStore(store, "../outside")


if __name__ == "__main__":
    unittest.main()
