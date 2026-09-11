import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from moha.catalog import catalog
from moha.demo import DemoJudge, DemoRunner, sample
from moha.execution import ExecutionPolicy
from moha.loop import Calibrator, SearchPolicy
from moha.models import Harness, ValidationPolicy
from moha.perception import DEFAULT_POLICY, policy_grid, select_policy, validate_result
from moha.store import RunStore


class SamplingPolicyTests(unittest.TestCase):
    def test_density_rounding_and_cap_boundaries(self):
        for rate, threshold in ((0.5, 256), (1, 128), (2, 64)):
            policy = ExecutionPolicy(target_fps=rate)
            self.assertEqual(policy.sampling_request(threshold)["target_frames"], 128)
            self.assertFalse(policy.sampling_request(threshold)["frame_cap_hit"])
            self.assertTrue(policy.sampling_request(threshold + 0.01)["frame_cap_hit"])
            self.assertEqual(policy.sampling_request(0.01)["target_frames"], 1)
        self.assertEqual([ExecutionPolicy(target_fps=f).sampling_request(25)["target_frames"]
                          for f in (0.5, 1, 2)], [13, 25, 50])

    def test_auto_retains_one_fps_and_fixed_primitives_remain_explicit(self):
        for duration in (0.2, 8, 25, 60, 300):
            self.assertEqual(ExecutionPolicy().sampling_request(duration),
                             ExecutionPolicy(target_fps=1).sampling_request(duration))
        fixed = ExecutionPolicy(frames=64)
        self.assertEqual(fixed.sampling_request(8)["target_frames"], 64)
        self.assertIsNone(fixed.sampling_request(8)["target_fps"])
        self.assertFalse(any(c.coordinate.endswith(".frames") for c in catalog().values()))

    def test_rate_and_fixed_frames_cannot_silently_override_each_other(self):
        for rate in (True, 0, 3, float("nan"), float("inf"), "1"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                ExecutionPolicy(target_fps=rate)
        for settings in ({"default": {"frames": 32, "target_fps": 2}},
                         {"default": {"target_fps": 2}, "text": {"frames": 64}},
                         {"default": {"frames": 64}, "text": {"target_fps": 2}}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Harness.from_dict({"execution": settings})
        h = Harness.from_dict({"execution": {"default": {"target_fps": 2}, "text": {"source_scale": 0.5}}})
        self.assertEqual(h.execution_for_goal("text").sampling_request(25)["target_frames"], 50)
        self.assertEqual(Harness.from_dict(h.to_dict()), h)

    def test_nine_policies_preserve_structure_and_default_identity(self):
        base = Harness(memory=True, verification=True, specialists=("ocr",), max_steps=12)
        grid = policy_grid(base)
        self.assertEqual(len({c["harness_id"] for c in grid}), 9)
        self.assertEqual(next(c for c in grid if c["id"] == DEFAULT_POLICY)["harness_id"], base.id)
        for config in grid:
            h = Harness.from_dict(config["harness"])
            self.assertEqual(replace(h, execution=base.execution), base)
            for goal in ("general", "sequence", "text", "speech"):
                self.assertEqual(h.execution_for_goal(goal).sampling_rate, config["target_fps"])
                self.assertEqual(h.execution_for_goal(goal).source_scale, config["source_scale"])
                self.assertEqual(h.execution_for_goal(goal).priority, "balanced")
        with self.assertRaises(ValueError):
            policy_grid(Harness.from_dict({"execution": {"default": {"frames": 64}}}))

    def test_validation_utility_ties_and_cost_constraints(self):
        rows = [{"id": DEFAULT_POLICY, "accuracy": 0.5, "mean_cost": 100},
                {"id": "a", "accuracy": 0.5, "mean_cost": 50},
                {"id": "b", "accuracy": 0.75, "mean_cost": 200}]
        self.assertEqual(select_policy(rows[:2], ValidationPolicy())["id"], DEFAULT_POLICY)
        self.assertEqual(select_policy(rows, ValidationPolicy())["id"], "b")
        self.assertEqual(select_policy(rows, ValidationPolicy(max_cost_ratio=1))["id"], DEFAULT_POLICY)
        self.assertEqual(select_policy(rows, ValidationPolicy(cost_penalty=1, cost_scale=100))["id"], "a")
        with self.assertRaises(ValueError):
            select_policy([rows[0], {**rows[1], "mean_cost": None}], ValidationPolicy())


class RateRunner(DemoRunner):
    def __init__(self, interrupt=False):
        self.calls, self.interrupt = [], interrupt

    def run(self, harness, s, repeat):
        self.calls.append((s.sample_id, harness.id, repeat))
        policy = harness.execution_for_goal("general")
        if self.interrupt and s.sample_id == "val1" and policy.sampling_rate == 0.5 and policy.source_scale == 0.5:
            self.interrupt = False
            raise RuntimeError("synthetic interruption during final selection")
        result = super().run(harness, s, repeat)
        if s.sample_id.startswith("val"):
            win = policy.sampling_rate == 2 and policy.source_scale == 0.75
            result.answer = "A" if harness.overview and (win or s.sample_id == "val0") else "B"
        return result


class PerceptionLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "run"

    def loop(self, store, runner, **kwargs):
        return Calibrator(runners=[runner], judges=[DemoJudge()], store=store,
            calibration=[sample("cal")], validation=[sample(f"val{i}") for i in range(3)],
            search=SearchPolicy(max_rounds=2), validation_policy=ValidationPolicy(bootstrap_samples=20),
            perception_calibration=True, **kwargs)

    def test_selects_all_nine_on_validation_after_structure_without_judge_gate(self):
        runner = RateRunner()
        with RunStore(self.root, {}) as store:
            loop = self.loop(store, runner)
            self.assertFalse(any(c.coordinate.startswith("execution.") for c in loop.catalog.values()))
            result = loop.run()
            final = result["perception_calibration"]
            self.assertEqual(final["selected_policy_id"], "fps2_scale0.75")
            self.assertEqual(len(final["rows"]), 9)
            self.assertEqual({r["episodes"] for r in final["rows"]}, {3})
            self.assertEqual(validate_result(final, Harness(overview=True), loop.validation_policy).id,
                             store.read("frozen_harness.json")["harness_id"])
            # The default is reacquired for final selection, rather than mixing
            # promotion and final-stage episode caches with the same harness ID.
            self.assertEqual(Counter(runner.calls)[("val0", Harness(overview=True).id, 0)], 2)
            self.assertEqual(sum(s == "cal" for s, _, _ in runner.calls), 2)
        before = list(runner.calls)
        with RunStore(self.root, {}, resume=True) as store:
            self.assertEqual(self.loop(store, runner).run(), result)
        self.assertEqual(runner.calls, before)

    def test_interruption_preserves_structure_and_reuses_completed_final_episodes(self):
        runner = RateRunner(interrupt=True)
        with RunStore(self.root, {}) as store:
            with self.assertRaises(RuntimeError):
                self.loop(store, runner).run()
            self.assertEqual(store.read("checkpoint.json")["phase"], "perception")
            self.assertIsNone(store.read("frozen_harness.json"))
        cal_calls = sum(s == "cal" for s, _, _ in runner.calls)
        with RunStore(self.root, {}, resume=True) as store:
            result = self.loop(store, runner).run()
        self.assertEqual(result["perception_calibration"]["selected_policy_id"], "fps2_scale0.75")
        self.assertEqual(sum(s == "cal" for s, _, _ in runner.calls), cal_calls)
        first = policy_grid(Harness(overview=True))[0]["harness_id"]
        self.assertEqual(Counter(runner.calls)[("val0", first, 0)], 1)
        self.assertEqual(Counter(runner.calls)[("val1", first, 0)], 2)

    def test_final_selection_is_not_exported_as_completed_when_cost_is_missing(self):
        class MissingCost(RateRunner):
            def run(self, h, s, repeat):
                result = super().run(h, s, repeat)
                if h.execution_for_goal("general").sampling_rate == 0.5:
                    result.usage = {}
                return result
        with RunStore(self.root, {}) as store:
            with self.assertRaisesRegex(ValueError, "measured cost"):
                self.loop(store, MissingCost()).run()
            self.assertIsNone(store.read("frozen_harness.json"))
            self.assertIsNone(store.read("result.json"))

    def test_specialist_probes_still_receive_execution_alternatives(self):
        class Judge(DemoJudge):
            def diagnose(self, *args):
                return {"status": "valid", "failure": "observer", "failed_capability": "ocr",
                        "candidate_id": None, "evidence_steps": [1]}
            def recommend(self, payload, diagnosis, available):
                self.seen = diagnosis["observer_resolution"]
                return {"status": "valid", "candidate_id": None, "proposal_reason": "test"}
        class Resolver:
            def resolve(self, sample, harness, episode, diagnosis, available):
                self.seen = [c.coordinate for c in available]
                return {"status": "no_rescue", "goal_type": "text"}
        judge, resolver = Judge(), Resolver()
        with RunStore(self.root, {}) as store:
            loop = self.loop(store, RateRunner(), resolvers=[resolver])
            loop.judges = (judge,)
            current = Harness()
            episodes = loop.batch(current, loop.calibration, 0, "calibration")
            loop.diagnose(current, episodes, list(loop.catalog.values()))
            self.assertTrue(any(c.endswith(".target_fps") for c in resolver.seen))
            self.assertEqual(judge.seen["status"], "no_rescue")

    def test_deferred_sampling_diagnosis_does_not_enter_completed_probe_recommendation(self):
        class Judge(DemoJudge):
            def diagnose(self, *args):
                return {"status": "valid", "failure": "observer", "failed_capability": None,
                        "candidate_id": None, "evidence_steps": [1]}
            def recommend(self, *args):
                raise AssertionError("deferred sampling has no completed counterfactual probe")
        with RunStore(self.root, {}) as store:
            loop = self.loop(store, RateRunner())
            loop.judges = (Judge(),)
            episodes = loop.batch(Harness(), loop.calibration, 0, "calibration")
            result, = loop.diagnose(Harness(), episodes, list(loop.catalog.values()))
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["observer_resolution"]["status"], "deferred")
            self.assertIsNone(result["candidate_id"])
            self.assertTrue(result["proposal_reason"])
            self.assertNotIn("observer_proposal", result)


if __name__ == "__main__":
    unittest.main()
