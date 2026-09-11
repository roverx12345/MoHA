import errno
import fcntl
import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError

from moha.demo import DemoJudge, sample
from moha.failures import ExecutionFailure, classify_failure
from moha.loop import Calibrator
from moha.models import Harness
from moha.store import RunStore
from moha.supervise import supervise


class FailureTests(unittest.TestCase):
    def test_timeout_chain_is_preserved_without_exception_text(self):
        cause = TimeoutError("credential=not-for-logs; request=private")
        error = RuntimeError("provider details must not enter the journal")
        error.__cause__ = cause
        result = classify_failure(error)
        self.assertTrue(result["retryable"])
        self.assertEqual(result["category"], "timeout")
        self.assertNotIn("private", json.dumps(result))
        self.assertNotIn("credential", json.dumps(result))
        self.assertEqual(classify_failure(ExecutionFailure(result)), result)

    def test_only_transient_http_and_network_failures_are_retryable(self):
        for code in (408, 429, 500, 502, 503, 504):
            with HTTPError("private", code, "private", {}, None) as error:
                self.assertTrue(classify_failure(error)["retryable"])
        for code in (400, 401, 403, 404, 422):
            with HTTPError("private", code, "private", {}, None) as error:
                self.assertFalse(classify_failure(error)["retryable"])
        self.assertTrue(classify_failure(URLError(TimeoutError()))["retryable"])
        self.assertTrue(classify_failure(OSError(errno.ECONNRESET, "private"))["retryable"])
        for error in (ValueError("bad schema"), PermissionError(), FileNotFoundError(),
                      RuntimeError("planner HTTP 503; not a provider exception"), URLError("unknown")):
            self.assertFalse(classify_failure(error)["retryable"])


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output, self.control = self.root / "run", self.root / "control"
        self.output.mkdir()
        self.config = self.root / "config.json"
        self.config.write_text('{"planner_retries":2}')
        (self.output / "manifest.json").write_text('{"identity_hash":"unit"}')
        self.cache = self.output / "cached-episode.json"
        self.cache.write_bytes(b'original completed episode\n')
        self.calls, self.delays = [], []

    def launch(self, results):
        def popen(command, **kwargs):
            self.calls.append(command)
            code, checkpoint = results[len(self.calls) - 1]
            output = self.output
            class Process:
                pid = 1000 + len(self.calls)
                def wait(self):
                    if checkpoint is not None:
                        (output / "checkpoint.json").write_text(json.dumps(checkpoint))
                    return code
            return Process()
        return popen

    def run_supervisor(self, results, **kwargs):
        return supervise(self.config, self.output, self.control, popen=self.launch(results),
                         sleeper=self.delays.append, **kwargs)

    def test_transient_failure_resumes_same_run_and_preserves_cache(self):
        failed = {"status": "error", "failure": {"retryable": True, "category": "timeout"}}
        code = self.run_supervisor([(1, failed), (0, {"status": "completed"})])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0], self.calls[1])
        self.assertIn("resume", self.calls[0])
        self.assertEqual(self.delays, [30])
        self.assertEqual(self.cache.read_bytes(), b'original completed episode\n')
        state = json.loads((self.control / "state.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual([a["decision"] for a in state["attempts"]], ["retry", "completed"])

    def test_restart_budget_is_persistent_and_not_reset_by_reinvocation(self):
        failed = {"status": "error", "failure": {"retryable": True}}
        results = [(1, failed)] * 4
        self.assertEqual(self.run_supervisor(results), 1)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.delays, [30, 60, 120])
        self.assertEqual(self.run_supervisor([]), 1)
        self.assertEqual(len(self.calls), 4)
        state = json.loads((self.control / "state.json").read_text())
        self.assertEqual(state["reason"], "restart_limit_reached")

    def test_nonretryable_failure_stops_immediately(self):
        self.assertEqual(self.run_supervisor([(1, {"status": "error", "failure": {"retryable": False}})]), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.delays)

    def test_old_failure_cannot_restart_a_startup_or_identity_error(self):
        path = self.output / "checkpoint.json"
        path.write_text('{"status":"error","failure":{"retryable":true}}')
        os.utime(path, (1, 1))
        self.assertEqual(self.run_supervisor([(1, None)]), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(json.loads((self.control / "state.json").read_text())["reason"], "unclassified_exit")

    def test_missing_exit_and_duplicate_supervisors_cannot_start_a_child(self):
        self.control.mkdir()
        with (self.control / ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_supervisor([])
        self.assertEqual(self.calls, [])


try:
    from flat.core.errors import ProviderError, ProviderResponseError
    from test_runtime import Planner, Service
    from test_verification_gate import final
    from moha.runtime import EpisodeRunner
except ImportError:
    ProviderError = None


@unittest.skipIf(ProviderError is None, "pinned runtime required")
class RuntimeRecoveryTests(unittest.TestCase):
    def timeout(self):
        error = ProviderError("planner transport failed after 3 attempt(s): TimeoutError")
        error.__cause__ = TimeoutError("must remain private")
        return error

    def test_provider_http_is_classified_but_response_contract_errors_are_not_retried(self):
        self.assertTrue(classify_failure(ProviderError("planner HTTP 503; response_sha256=unit"))["retryable"])
        self.assertFalse(classify_failure(ProviderError("planner HTTP 401; response_sha256=unit"))["retryable"])
        self.assertFalse(classify_failure(ProviderResponseError("malformed output"))["retryable"])

    def test_planner_timeout_reaches_checkpoint_as_retryable(self):
        with tempfile.TemporaryDirectory() as tmp, RunStore(Path(tmp) / "run", {}) as store:
            runner = EpisodeRunner(Service(), Planner([self.timeout()]), store=store)
            loop = Calibrator(runners=[runner], judges=[DemoJudge()], store=store,
                              calibration=[sample("cal")], validation=[sample("val")])
            with self.assertRaises(ExecutionFailure):
                loop.run()
            checkpoint = store.read("checkpoint.json")
            self.assertTrue(checkpoint["failure"]["retryable"])
            error_file, = (store.root / "errors").glob("episode-*.json")
            episode = json.loads(error_file.read_text())
            self.assertTrue(episode["raw"]["failure"]["retryable"])
            self.assertNotIn("must remain private", error_file.read_text())
            self.assertEqual(len(list((store.root / "episodes").glob("*.json"))), 0)

    def test_audit_failure_uses_separate_client_and_blocks_episode_replay(self):
        planner, audit_planner = Planner([final()]), Planner([self.timeout()])
        result = EpisodeRunner(Service(), planner, audit_planner=audit_planner).run(
            Harness(verification=True), sample("cal"), 0)
        self.assertEqual(result.status, "error")
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(audit_planner.calls), 1)
        self.assertFalse(result.raw["failure"]["retryable"])
        self.assertEqual(result.raw["failure"]["automatic_resume_blocked"], "verification_already_started")


if __name__ == "__main__":
    unittest.main()
