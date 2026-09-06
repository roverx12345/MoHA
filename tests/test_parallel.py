"""Concurrency, stable routing and recovery without model calls."""
import tempfile
import threading
import unittest
from concurrent.futures import wait as real_wait
from pathlib import Path
from unittest.mock import patch
from moha.demo import DemoJudge, DemoRunner, sample
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
        return Calibrator(runners=runners, judge=kwargs.pop("judge", DemoJudge()), store=self.store,
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
