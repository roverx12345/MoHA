import copy
import os
import tempfile
import unittest
from pathlib import Path

from moha.demo import DemoJudge, DemoRunner, sample
from moha.loop import Calibrator, SearchPolicy
from moha.lineage import import_shared_fork, plan_shared_fork
from moha.models import Episode, Harness, ValidationPolicy
from moha.shared import (AGGREGATION_RULE, NamespacedStore, SharedCalibrator,
                         compare_shared)
from moha.store import RunStore
from moha.supervise import supervise_shared


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

    def test_shared_supervisor_preserves_ordered_stack_arguments(self):
        output, state = self.root, Path(self.tmp.name) / "supervisor"
        output.mkdir()
        (output / "manifest.json").write_text('{"identity_hash":"shared-unit"}')
        first, second = Path(self.tmp.name) / "a.json", Path(self.tmp.name) / "b.json"
        first.write_text("{}")
        second.write_text("{}")
        calls = []

        def popen(command, **_):
            calls.append(command)
            class Process:
                pid = 123
                def wait(self):
                    (output / "checkpoint.json").write_text('{"status":"completed"}')
                    return 0
            return Process()

        code = supervise_shared([f"a={first}", f"b={second}"], output, state,
                                popen=popen, sleeper=lambda _: None)
        self.assertEqual(code, 0)
        self.assertEqual(calls[0][3], "shared-resume")
        self.assertEqual(calls[0][4:8], ["--stack", f"a={first}", "--stack", f"b={second}"])


class SharedForkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def identity(self, commit, source_files, endpoint, *, concurrency=None):
        stacks = []
        for stack_id in ("a", "b"):
            config = {
                "observer": {"model": "omni", "base_url": endpoint},
                "models": {"planner": {"spec": {"model": stack_id,
                                                    "base_url": endpoint}}},
                "constant": "same",
            }
            if concurrency is not None:
                config["planner_max_inflight_per_endpoint"] = concurrency
            stacks.append({"stack_id": stack_id, "identity": {
                "config": config,
                "source": {"commit": commit, "dirty": False,
                           "source_hash": commit, "files": source_files,
                           "environment": {"python": "unit"}},
                "runtime": {"commit": "flat", "dirty": False,
                            "source_hash": "flat", "files": {}},
                "input_hashes": {"calibration": "c", "validation": "v"},
            }})
        return {"implementation": "moha_shared_calibration_v1",
                "aggregation": "unit", "validation_schedule": "unit",
                "stack_order": ["a", "b"], "stacks": stacks}

    def test_fork_binds_and_hardlinks_completed_artifacts(self):
        old_files = {"src/moha/bridge.py": "old", "src/moha/shared.py": "same"}
        new_files = {"src/moha/bridge.py": "new", "src/moha/shared.py": "same",
                     "src/moha/lineage.py": "new"}
        parent_identity = self.identity("old", old_files, "http://old/v1")
        child_base = self.identity("new", new_files,
                                   "http://local/v1,http://remote/v1", concurrency=1)
        parent_root, child_root = self.root / "parent", self.root / "child"
        checkpoint = {"round": 0, "attempt": 0, "harness": Harness().to_dict(),
                      "history": [], "status": "running", "phase": None}
        with RunStore(parent_root, parent_identity) as parent:
            parent.write("checkpoint.json", checkpoint)
            parent.write("models/a/episodes/unit.json", {"status": "completed"}, immutable=True)
            plan = plan_shared_fork(parent, child_base)
            child_identity = copy.deepcopy(child_base)
            child_identity["lineage"] = plan["descriptor"]
            with RunStore(child_root, child_identity) as child:
                result = import_shared_fork(parent, child, plan)
                self.assertEqual(result["status"], "forked")
                self.assertEqual(child.read("checkpoint.json"), checkpoint)
                self.assertEqual(child.read("lineage.json")["artifact_count"], 1)
        parent_file = parent_root / "models/a/episodes/unit.json"
        child_file = child_root / "models/a/episodes/unit.json"
        self.assertEqual(os.stat(parent_file).st_ino, os.stat(child_file).st_ino)

    def test_fork_rejects_source_changes_outside_lineage_boundary(self):
        parent_identity = self.identity("old", {"src/moha/runtime.py": "old"},
                                        "http://old/v1")
        child_identity = self.identity("new", {"src/moha/runtime.py": "new"},
                                       "http://new/v1", concurrency=1)
        with RunStore(self.root / "parent", parent_identity) as parent:
            parent.write("checkpoint.json", {"round": 0, "attempt": 0,
                "harness": Harness().to_dict(), "history": [], "status": "running", "phase": None})
            with self.assertRaisesRegex(ValueError, "outside the approved lineage boundary"):
                plan_shared_fork(parent, child_identity)


if __name__ == "__main__":
    unittest.main()
