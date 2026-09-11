"""Concurrency, stable routing and recovery without model calls."""
import tempfile
import threading
import unittest
from concurrent.futures import wait as real_wait
from pathlib import Path
from unittest.mock import patch
from moha.demo import DemoJudge, DemoRunner, sample
from moha.evidence import unpack
from moha.loop import Calibrator
from moha.models import Harness
from moha.store import RunStore


class Runner(DemoRunner):
    def __init__(self):
        self.calls = []

    def run(self, harness, item, repeat):
        self.calls.append(item.sample_id)
        return super().run(harness, item, repeat)


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = RunStore(Path(self.tmp.name) / "run", {})
        self.addCleanup(self.store.close)
        self.samples = [sample(f"cal{i}") for i in range(4)]

    def calibrator(self, runners, **kwargs):
        judges = kwargs.pop("judges", None)
        return Calibrator(runners=runners, judges=judges if judges is not None else [kwargs.pop("judge", DemoJudge())], store=self.store,
                          calibration=self.samples, validation=[sample("val")], **kwargs)

    def test_two_lanes_overlap_keep_each_runner_serial_and_return_manifest_order(self):
        barrier = threading.Barrier(2, timeout=3)
        class Concurrent(Runner):
            def __init__(self):
                super().__init__()
                self.lock = threading.Lock()
            def run(self, *args):
                if not self.lock.acquire(blocking=False):
                    raise AssertionError("one lane ran multiple episodes concurrently")
                try:
                    barrier.wait()  # A serial implementation cannot reach this barrier.
                    return super().run(*args)
                finally:
                    self.lock.release()
        runners = [Concurrent(), Concurrent()]
        episodes = self.calibrator(runners).batch(Harness(), self.samples, 0, "calibration")
        self.assertEqual([e.sample_id for e in episodes], [s.sample_id for s in self.samples])
        self.assertEqual([r.calls for r in runners], [["cal0", "cal2"], ["cal1", "cal3"]])
        self.assertEqual(self.store.read("progress.json")["completed"], 4)
        self.assertEqual(self.store.read("progress.json")["parallel_lanes"], 2)
        self.assertEqual(len(list((self.store.root / "episodes").glob("*.json"))), 4)
        self.assertTrue(all(self.store.read(f"workers/{i}.json")["status"] == "completed" for i in range(2)))

    def test_cached_holes_and_candidate_validation_keep_the_original_lanes(self):
        runners = [Runner(), Runner()]
        cal = self.calibrator(runners)
        cal.batch(Harness(), self.samples[:1], 0, "validation")
        cal.batch(Harness(), self.samples, 0, "validation")
        self.assertEqual([r.calls for r in runners], [["cal0", "cal2"], ["cal1", "cal3"]])
        for r in runners:
            r.calls.clear()
        cal.batch(Harness(memory=True), self.samples, 0, "validation")
        self.assertEqual([r.calls for r in runners], [["cal0", "cal2"], ["cal1", "cal3"]])
        for r in runners:
            r.calls.clear()
        # Same completed batch is a cache-only resume, still checked and ordered.
        episodes = cal.batch(Harness(memory=True), self.samples, 0, "validation")
        self.assertEqual([r.calls for r in runners], [[], []])
        self.assertEqual([e.sample_id for e in episodes], [s.sample_id for s in self.samples])

    def test_error_stops_new_submissions_and_saves_the_other_inflight_episode(self):
        barrier, release = threading.Barrier(2, timeout=3), threading.Event()
        class Broken(Runner):
            def run(self, *args):
                episode = super().run(*args)
                barrier.wait()
                episode.status = "error"
                return episode
        class Inflight(Runner):
            def run(self, *args):
                barrier.wait()
                if not release.wait(3):
                    raise AssertionError("coordinator did not observe the failed lane")
                return super().run(*args)
        def observe_failure(*args, **kwargs):
            done, pending = real_wait(*args, **kwargs)
            if any(f.exception() is not None for f in done):
                release.set()
            return done, pending
        runners = [Broken(), Inflight()]
        with patch("moha.loop.wait", side_effect=observe_failure), self.assertRaises(RuntimeError):
            self.calibrator(runners).batch(Harness(), self.samples, 0, "calibration")
        self.assertEqual([r.calls for r in runners], [["cal0"], ["cal1"]])
        self.assertEqual(self.store.read("workers/0.json")["status"], "error")
        self.assertEqual(self.store.read("workers/1.json")["status"], "completed")
        self.assertEqual(len(list((self.store.root / "episodes").glob("*.json"))), 1)
        self.assertEqual(len(list((self.store.root / "errors").glob("*.json"))), 1)
        resumed = [Runner(), Runner()]
        episodes = self.calibrator(resumed).batch(Harness(), self.samples, 0, "calibration")
        self.assertEqual([r.calls for r in resumed], [["cal0", "cal2"], ["cal3"]])
        self.assertEqual(len(episodes), 4)

    def test_invalid_final_answer_continues_the_batch_and_is_not_replayed_on_resume(self):
        class InvalidFinal(Runner):
            def run(self, harness, item, repeat):
                episode = super().run(harness, item, repeat)
                if item.sample_id == "cal0":
                    episode.status = "invalid_final_answer"
                    episode.raw = {"verification_gate": {"audit_status": "valid"},
                                   "final_finish_reason": "length", "original_output": "Unfinished explanation"}
                return episode
        runners = [InvalidFinal(), Runner()]
        calibrator = self.calibrator(runners)
        episodes = calibrator.batch(Harness(verification=True), self.samples, 0, "validation")
        self.assertEqual([r.calls for r in runners], [["cal0", "cal2"], ["cal1", "cal3"]])
        self.assertEqual((episodes[0].status, episodes[0].correct), ("invalid_final_answer", False))
        self.assertEqual(self.store.read("progress.json")["completed"], 4)
        self.assertEqual(len(list((self.store.root / "episodes").glob("*.json"))), 4)
        self.assertFalse(list((self.store.root / "errors").glob("*.json")))
        replay = calibrator.batch(Harness(verification=True), self.samples, 0, "validation")
        self.assertEqual([e.to_dict() for e in replay], [e.to_dict() for e in episodes])
        self.assertEqual([r.calls for r in runners], [["cal0", "cal2"], ["cal1", "cal3"]])

    def test_observer_probes_use_the_calibration_samples_lane(self):
        class Judge(DemoJudge):
            def diagnose(self, *args):
                return {"status": "valid", "failure": "observer", "candidate_id": None}
            def recommend(self, *args):
                return {"status": "valid", "candidate_id": None, "proposal_reason": "No change"}
        class Resolver:
            def __init__(self):
                self.calls = []
            def resolve(self, item, *args):
                self.calls.append(item.sample_id)
                return {"status": "inconclusive"}
        resolvers = [Resolver(), Resolver()]
        cal = self.calibrator([Runner(), Runner()], judge=Judge(), resolvers=resolvers)
        episodes = cal.batch(Harness(), self.samples, 0, "calibration")
        cal.diagnose(Harness(), episodes, [])
        self.assertEqual([r.calls for r in resolvers], [["cal0", "cal2"], ["cal1", "cal3"]])

    def test_invalid_lane_or_duplicate_sample_fails_before_inference(self):
        runner = Runner()
        for runners in ([], [runner, runner]):
            with self.assertRaises(ValueError):
                self.calibrator(runners)
        with self.assertRaises(ValueError):
            self.calibrator([runner], resolvers=[])
        cal = self.calibrator([runner])
        with self.assertRaises(ValueError):
            cal.batch(Harness(), [self.samples[0], self.samples[0]], 0, "calibration")
        self.assertEqual(runner.calls, [])

    def test_eight_diagnoses_overlap_with_independent_judges_and_ordered_votes(self):
        self.samples = [sample(f"cal{i}") for i in range(16)]
        barrier = threading.Barrier(8, timeout=3)
        class Concurrent(DemoJudge):
            def __init__(self):
                self.calls = []
                self.lock = threading.Lock()
            def diagnose(self, payload, available):
                if not self.lock.acquire(blocking=False):
                    raise AssertionError("a Judge client was shared concurrently")
                try:
                    self.calls.append(unpack(payload)["sample_id"])
                    barrier.wait()  # Requires eight actual simultaneous calls.
                    return super().diagnose(payload, available)
                finally:
                    self.lock.release()
        judges = [Concurrent() for _ in range(8)]
        cal = self.calibrator([Runner(), Runner()], judges=judges)
        episodes = cal.batch(Harness(), self.samples, 0, "calibration")
        results = cal.diagnose(Harness(), episodes, [])
        self.assertEqual([r["sample_id"] for r in results], [s.sample_id for s in self.samples])
        self.assertEqual([len(j.calls) for j in judges], [2] * 8)
        self.assertEqual(self.store.read("progress.json")["parallel_workers"], 8)
        self.assertEqual(self.store.read("progress.json")["completed"], 16)
        self.assertEqual(cal.diagnose(Harness(), episodes, []), results)
        self.assertEqual([len(j.calls) for j in judges], [2] * 8)  # All cached.
        with self.assertRaisesRegex(ValueError, "completed artifact"):
            self.calibrator([Runner(), Runner()], judges=[DemoJudge()])

    def test_out_of_order_failures_keep_manifest_order_and_probe_lane(self):
        # Correct samples and out-of-order completion must not renumber probes.
        barrier = threading.Barrier(2, timeout=3)
        second_done = threading.Event()
        class ObserverJudge(DemoJudge):
            def diagnose(self, payload, available):
                barrier.wait()
                if unpack(payload)["sample_id"] == "cal1" and not second_done.wait(3):
                    raise AssertionError("second trace was not processed independently")
                return {"status": "valid", "failure": "observer", "candidate_id": None}
            def recommend(self, payload, result, available):
                if unpack(payload)["sample_id"] == "cal2":
                    second_done.set()
                return {"status": "valid", "candidate_id": None, "proposal_reason": "No change"}
        class Resolver:
            def __init__(self):
                self.calls = []
            def resolve(self, item, *args):
                self.calls.append(item.sample_id)
                return {"status": "inconclusive"}
        resolvers = [Resolver(), Resolver()]
        cal = self.calibrator([Runner(), Runner()], judges=[ObserverJudge(), ObserverJudge()], resolvers=resolvers)
        episodes = cal.batch(Harness(), self.samples, 0, "calibration")
        episodes[0].answer = episodes[3].answer = "A"
        results = cal.diagnose(Harness(), episodes, [])
        self.assertEqual([r["sample_id"] for r in results], ["cal1", "cal2"])
        self.assertEqual([r.calls for r in resolvers], [["cal2"], ["cal1"]])

    def test_four_workers_serialize_probes_on_each_original_service(self):
        barrier = threading.Barrier(4, timeout=3)
        probes = threading.Barrier(2, timeout=3)
        class ObserverJudge(DemoJudge):
            def diagnose(self, *args):
                barrier.wait()
                return {"status": "valid", "failure": "observer", "candidate_id": None}
            def recommend(self, *args):
                return {"status": "valid", "candidate_id": None, "proposal_reason": "No change"}
        class Resolver:
            def __init__(self):
                self.lock, self.calls = threading.Lock(), []
            def resolve(self, item, *args):
                if not self.lock.acquire(blocking=False):
                    raise AssertionError("same probe service/client used concurrently")
                try:
                    self.calls.append(item.sample_id)
                    probes.wait()  # The two original lanes can still overlap.
                    return {"status": "inconclusive"}
                finally:
                    self.lock.release()
        resolvers = [Resolver(), Resolver()]
        cal = self.calibrator([Runner(), Runner()], judges=[ObserverJudge() for _ in range(4)], resolvers=resolvers)
        cal.diagnose(Harness(), cal.batch(Harness(), self.samples, 0, "calibration"), [])
        self.assertEqual([sorted(r.calls) for r in resolvers], [["cal0", "cal2"], ["cal1", "cal3"]])

    def test_diagnosis_exception_stops_submissions_and_preserves_inflight_cache(self):
        self.samples = [sample(f"cal{i}") for i in range(8)]
        barrier, release = threading.Barrier(4, timeout=3), threading.Event()
        calls = []
        class Interrupted(DemoJudge):
            def diagnose(self, payload, available):
                calls.append(unpack(payload)["sample_id"])
                barrier.wait()
                if unpack(payload)["sample_id"] == "cal0":
                    raise RuntimeError("synthetic interruption")
                if not release.wait(3):
                    raise AssertionError("coordinator did not observe failure")
                return super().diagnose(payload, available)
        def observe_failure(*args, **kwargs):
            done, pending = real_wait(*args, **kwargs)
            if any(f.exception() is not None for f in done):
                release.set()
            return done, pending
        cal = self.calibrator([Runner()], judges=[Interrupted() for _ in range(4)])
        episodes = cal.batch(Harness(), self.samples, 0, "calibration")
        with patch("moha.loop.wait", side_effect=observe_failure), self.assertRaises(RuntimeError):
            cal.diagnose(Harness(), episodes, [])
        self.assertEqual(sorted(calls), ["cal0", "cal1", "cal2", "cal3"])
        self.assertEqual(len(list((self.store.root / "diagnoses").glob("*.json"))), 3)
        resumed_calls = []
        class Resumed(DemoJudge):
            def diagnose(self, payload, available):
                resumed_calls.append(unpack(payload)["sample_id"])
                return super().diagnose(payload, available)
        resumed = self.calibrator([Runner()], judges=[Resumed() for _ in range(4)])
        self.assertEqual(len(resumed.diagnose(Harness(), episodes, [])), 8)
        self.assertEqual(sorted(resumed_calls), ["cal0", "cal4", "cal5", "cal6", "cal7"])

    def test_duplicate_judge_or_empty_pool_fails_before_inference(self):
        judge = DemoJudge()
        for judges in ([], [judge, judge]):
            with self.assertRaisesRegex(ValueError, "independent judge"):
                self.calibrator([Runner()], judges=judges)
