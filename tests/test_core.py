import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from moha.catalog import catalog
from moha.demo import DemoJudge, DemoRunner, sample, run_demo
from moha.evaluate import checked, compare
from moha.loop import Calibrator, SearchPolicy
from moha.models import Episode, Harness, ValidationPolicy, check_splits
from moha.records import usage_from
from moha.store import RunStore


class ContractTests(unittest.TestCase):
    def test_shared_start_and_roundtrip(self):
        base = Harness()
        self.assertFalse(any((base.overview, base.memory, base.verification, base.retrieval_guard, base.specialists)))
        self.assertEqual(Harness.from_dict(base.to_dict()), base)
        self.assertEqual(base.id, Harness.from_dict(base.to_dict()).id)

    def test_invalid_values(self):
        for kwargs in ({"max_steps": True}, {"history_tokens": 0}, {"memory": 1}, {"specialists": ("count",)},
                       {"execution": (("text", "default"),)}, {"execution": (("default", "oops"),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Harness(**kwargs)
        for kwargs in ({"repeats": 0}, {"min_gain": float("nan")}, {"max_cost_ratio": float("inf")}, {"require_positive_interval": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ValidationPolicy(**kwargs)

    def test_video_split_not_question_split(self):
        with self.assertRaises(ValueError):
            check_splits([sample("one")], [replace(sample("two"), video_id="one")])
        with self.assertRaises(ValueError):
            check_splits([sample("one")], [sample("one")])
        with self.assertRaises(ValueError):
            replace(sample("one"), video_id=None)

    def test_catalog_is_one_coordinate_and_runnable_data(self):
        base = Harness()
        for item in catalog().values():
            changed = item.apply(base).to_dict()
            self.assertLessEqual(sum(changed[k] != v for k, v in base.to_dict().items()), 1)

    def test_independent_coordinates_inherit_default_settings(self):
        base = Harness.from_dict({"execution": {"default": {"frames": 64, "source_scale": 0.75}}})
        mode = catalog()["observer.execution.text.priority.spatial"]
        changed = mode.apply(base)
        self.assertEqual(changed.execution_for_goal("text").to_dict(),
            {"frames": 64, "source_scale": 0.75, "priority": "spatial"})
        self.assertEqual(changed.execution_for_goal("sequence"), base.execution_for_goal("sequence"))
        self.assertFalse(catalog()["observer.execution.text.target_fps.2.0"].available(base))
        self.assertFalse(mode.available(changed))
        self.assertEqual(Harness.from_dict(changed.to_dict()), changed)


class AccountingTests(unittest.TestCase):
    def test_receipts_are_deduplicated_but_session_ledger_not_multiplied(self):
        receipt = {"receipt_id": "r1", "usage": {"sampled_frames": 10}}
        result = {"observer_execution_receipt": receipt, "backend_result": {"observer_execution_receipt": receipt}}
        events = [{"kind": "tool_result", "tool": "observe", "result": result}, {"kind": "planner"}]
        usage = usage_from(events, {"budget_ledger": {"frames_used": 12, "video_tokens_used": 512, "look_used": 2}})
        self.assertEqual(usage["observer_calls"], 1)
        self.assertEqual(usage["sampled_frames"], 12)
        self.assertEqual(usage["video_tokens"], 512)
        self.assertEqual(usage["sensory_looks"], 2)

    def test_missing_cost_is_unknown(self):
        usage = usage_from([{"kind": "tool_result", "tool": "observe", "result": {}}], None)
        self.assertIsNone(usage["observer_calls"])
        self.assertIsNone(usage["video_tokens"])

    def test_conflicting_receipts_fail(self):
        with self.assertRaises(ValueError):
            usage_from([{"kind": "tool_result", "result": {
                "observer_execution_receipt": {"receipt_id": "x"},
                "nested": {"observer_execution_receipt": {"receipt_id": "x", "usage": {"sampled_frames": 5}}}}}], None)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.samples = [sample(f"val{i}") for i in range(8)]
        self.old, self.new = Harness(), Harness(overview=True)

    def runs(self, harness, answers, costs=100, repeats=1):
        return [[Episode(s.sample_id, s.id, harness.id, r, list(s.video_key), s.expected_answer,
                         answers[r][i] if repeats > 1 else answers[i], "completed", usage={"video_tokens": costs})
                 for i, s in enumerate(self.samples)] for r in range(repeats)]

    def test_paper_gate_strict_margin_and_matched_cap_not_same_spend(self):
        old = self.runs(self.old, ["B"] * 8)
        new = self.runs(self.new, ["A"] * 8, 200)
        self.assertTrue(compare(old, new, self.samples, self.old, self.new, ValidationPolicy())['accepted'])
        self.assertFalse(compare(old, new, self.samples, self.old, self.new, ValidationPolicy(min_gain=1))['accepted'])
        self.assertEqual(compare(old, new, self.samples, self.old, self.new, ValidationPolicy(max_cost_ratio=1))['reason'], 'cost_limit')

    def test_complete_pair_identity_is_required(self):
        data = self.runs(self.old, ["B"] * 8)[0]
        for mutated in (data[:-1], data + [data[0]], [replace(data[0], expected_answer="B")] + data[1:],
                        [replace(data[0], harness_id=self.new.id)] + data[1:], [replace(data[0], repeat=2)] + data[1:],
                        [replace(data[0], status="error")] + data[1:]):
            with self.subTest(mutated=mutated[0].sample_id), self.assertRaises(ValueError):
                checked(mutated, self.samples, self.old, 0)

    def test_missing_cost_cannot_promote(self):
        old, new = self.runs(self.old, ["B"] * 8), self.runs(self.new, ["A"] * 8)
        new[0][0].usage = {}
        self.assertEqual(compare(old, new, self.samples, self.old, self.new, ValidationPolicy())["reason"], "missing_measured_cost")

    def test_invalid_final_answer_is_incorrect_with_its_full_measured_cost(self):
        old = self.runs(self.old, ["A"] * 8)
        new = self.runs(self.new, ["A"] * 8)
        new[0][0] = replace(new[0][0], status="invalid_final_answer", usage={"video_tokens": 900})
        verdict = compare(old, new, self.samples, self.old, self.new, ValidationPolicy())
        self.assertEqual(verdict["new_accuracy"], 7 / 8)
        self.assertEqual(verdict["new_cost"], 200)
        self.assertEqual(verdict["paired"]["correct_to_wrong"], 1)
        self.assertFalse(verdict["accepted"])
        for status in ("error", "provider_error", "invalid_verification", "unknown_status"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                checked([replace(new[0][0], status=status)] + new[0][1:], self.samples, self.new, 0)

    def test_optional_repeats_gate_detects_regression(self):
        old = self.runs(self.old, [["A"] * 8, ["B"] * 8, ["B"] * 8], repeats=3)
        new = self.runs(self.new, [["B"] * 8, ["A"] * 8, ["A"] * 8], repeats=3)
        policy = ValidationPolicy(repeats=3, require_each_repeat_nonnegative=True)
        verdict = compare(old, new, self.samples, self.old, self.new, policy)
        self.assertEqual(verdict["reason"], "inconsistent_repeats")
        self.assertEqual(verdict["paired"]["wrong_to_correct"], 16)
        self.assertEqual(verdict["paired"]["correct_to_wrong"], 8)

    def test_optional_confidence_gate(self):
        old, new = self.runs(self.old, ["B"] * 8), self.runs(self.new, ["A"] + ["B"] * 7)
        result = compare(old, new, self.samples, self.old, self.new, ValidationPolicy(require_positive_interval=True))
        self.assertEqual(result["reason"], "uncertain_gain")


class StoreAndLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"

    def loop(self, store, **kwargs):
        params = dict(runners=[DemoRunner()], judges=[kwargs.pop("judge", DemoJudge())], store=store,
                      calibration=[sample("cal")], validation=[sample("val")],
                      validation_policy=ValidationPolicy(bootstrap_samples=30))
        params.update(kwargs)
        return Calibrator(**params)

    def test_run_is_exclusive_immutable_and_identity_bound(self):
        with RunStore(self.root, {"source": 1}) as store:
            store.write("x.json", {"v": 1}, immutable=True)
            with self.assertRaises(ValueError):
                store.write("x.json", {"v": 2}, immutable=True)
            with self.assertRaises(ValueError):
                store.write("../outside.json", {})
            with self.assertRaises(BlockingIOError):
                RunStore(self.root, {"source": 1}, resume=True)
        with self.assertRaises(ValueError):
            RunStore(self.root, {"source": 2}, resume=True)

    def test_demo_end_to_end_and_completed_resume(self):
        result = run_demo(self.root)
        self.assertTrue(result["harness"]["overview"])
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(run_demo(self.root, resume=True), result)
        self.assertTrue((self.root / "frozen_harness.json").is_file())

    def test_episode_interrupt_resume_reuses_completed_samples(self):
        class Interrupted(DemoRunner):
            calls = []
            fail = True
            def run(runner, harness, s, repeat):
                runner.calls.append((s.sample_id, harness.id))
                if s.sample_id == "val" and harness.overview and runner.fail:
                    runner.fail = False
                    raise RuntimeError("synthetic interruption")
                return super().run(harness, s, repeat)
        runner = Interrupted()
        with RunStore(self.root, {}) as store:
            with self.assertRaises(RuntimeError):
                self.loop(store, runners=[runner]).run()
        with RunStore(self.root, {}, resume=True) as store:
            result = self.loop(store, runners=[runner]).run()
        self.assertEqual(runner.calls.count(("val", Harness().id)), 1)
        self.assertEqual(runner.calls.count(("val", Harness(overview=True).id)), 2)
        self.assertTrue(result["history"][0]["validation"]["accepted"])

    def test_failed_diagnosis_does_not_poison_immutable_resume_slot(self):
        class OnceBad(DemoJudge):
            bad = True
            def diagnose(judge, *args):
                if judge.bad:
                    judge.bad = False
                    return {"status": "error"}
                return super().diagnose(*args)
        judge = OnceBad()
        with RunStore(self.root, {}) as store:
            with self.assertRaises(RuntimeError):
                self.loop(store, judge=judge).run()
        with RunStore(self.root, {}, resume=True) as store:
            self.assertEqual(self.loop(store, judge=judge).run()["status"], "completed")

    def test_invalid_diagnoses_stop_before_validation(self):
        class Invalid:
            def diagnose(self, _, available):
                return {"status": "error"}
        with RunStore(self.root, {}) as store:
            with self.assertRaises(RuntimeError):
                self.loop(store, judge=Invalid()).run()
            self.assertEqual(len(list((self.root / "episodes").glob("*.json"))), 1)

    def test_abstention_needs_patience_and_stable_profile(self):
        class Abstains(DemoJudge):
            def diagnose(self, *args):
                return {**super().diagnose(*args), "candidate_id": None, "proposal_reason": "uncertain"}
        with RunStore(self.root, {}) as store:
            result = self.loop(store, judge=Abstains()).run()
        self.assertEqual(result["stop_reason"], "converged")
        self.assertEqual(result["round"], 2)
        self.assertEqual(result["profile_distance"], 0)

    def test_changing_failure_profile_is_not_convergence(self):
        class Alternates:
            n = 0
            def diagnose(self, payload, available):
                self.n += 1
                return {"status": "valid", "failure": "orientation" if self.n % 2 else "retrieval", "candidate_id": None}
        with RunStore(self.root, {}) as store:
            result = self.loop(store, judge=Alternates(), search=SearchPolicy(max_rounds=3)).run()
        self.assertEqual(result["stop_reason"], "round_budget")

    def test_rejection_uses_next_rank_without_rejudging_then_refreshes_after_promotion(self):
        from moha.evidence import unpack
        class Votes:
            calls = []
            def diagnose(judge, payload, available):
                data = unpack(payload)
                judge.calls.append((data["sample_id"], data["harness"]["overview"]))
                candidate = "planner.module.memory_basic" if data["sample_id"] == "cal1" else "planner.module.overview"
                return {"status": "valid", "failure": "verification", "candidate_id": candidate}
        class Runner(DemoRunner):
            def run(self, harness, s, repeat):
                e = super().run(harness, s, repeat)
                if s.sample_id.startswith("cal"):
                    e.answer = "B"  # Keep calibration failures after the validation gain.
                return e
        judge = Votes()
        with RunStore(self.root, {}) as store:
            result = self.loop(store, runners=[Runner()], judge=judge,
                               calibration=[sample("cal1"), sample("cal2")],
                               search=SearchPolicy(max_rounds=2)).run()
        self.assertEqual([h["candidate"] for h in result["history"]],
                         ["planner.module.memory_basic", "planner.module.overview"])
        self.assertEqual([h["validation"]["accepted"] for h in result["history"]], [False, True])
        self.assertEqual(judge.calls, [("cal1", False), ("cal2", False), ("cal1", True), ("cal2", True)])

    def test_observer_probe_precedes_final_proposal_and_never_sees_validation(self):
        from moha.evidence import unpack
        calls = []
        class Judge:
            def diagnose(self, payload, available):
                calls.append(("diagnose", unpack(payload)["sample_id"]))
                return {"status": "valid", "failure": "observer", "candidate_id": None}
            def recommend(self, payload, diagnosis, available):
                calls.append(("recommend", unpack(payload)["sample_id"]))
                self_status = diagnosis["observer_resolution"]["status"]
                assert self_status == "execution_rescue"
                return {"status": "valid", "candidate_id": "planner.module.overview", "proposal_reason": "test"}
        class Resolver:
            def resolve(self, sample, *args):
                calls.append(("probe", sample.sample_id))
                return {"status": "execution_rescue"}
        with RunStore(self.root, {}) as store:
            self.loop(store, judge=Judge(), resolvers=[Resolver()]).run()
        self.assertEqual(calls, [("diagnose", "cal"), ("probe", "cal"), ("recommend", "cal")])


if __name__ == "__main__":
    unittest.main()
