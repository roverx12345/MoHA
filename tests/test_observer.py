"""Real media/provider/registry integration, with offline model responses."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from moha.demo import sample
from moha.evidence import diagnosis_view, unpack
from moha.models import Harness
from moha.runtime import EpisodeRunner
from test_runtime import VIDEO_OS_AVAILABLE, Planner, call


GOOD = {"facts": [{"fact": "A test pattern is visible.", "modalities": ["video"],
    "support_time_seconds": [0.2, 0.8], "spatial_scope": "global"}],
    "uncertainties": [], "missing": [], "requested_refinement": None}
ANSWER = {"role": "assistant", "content": '{"status":"answered","answer":"A"}'}


def response(value=GOOD, *, finish="stop", usage=True, tokens=80, input_tokens=600):
    from flat.providers.client import TransportResponse
    body = {"id": "offline", "model": "Qwen3-Omni-30B-A3B-Instruct",
        "choices": [{"message": {"content": value if isinstance(value, str) else json.dumps(value)},
                     "finish_reason": finish}]}
    if usage:
        body["usage"] = {"prompt_tokens": input_tokens, "completion_tokens": tokens,
                         "total_tokens": input_tokens + tokens}
    return TransportResponse(status=200, body=json.dumps(body).encode())


class Transport:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def post(self, **kwargs):
        self.calls.append(json.loads(kwargs["body"]))
        return self.replies.pop(0)


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires the pinned Video OS runtime")
class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_tmp = tempfile.TemporaryDirectory()
        cls.media = Path(cls.media_tmp.name) / "unit.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc2=size=320x240:rate=24", "-f", "lavfi", "-i", "sine=frequency=440",
            "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.media)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.media_tmp.cleanup()

    def service(self, replies, *, original=False, backend="qwen3omni", budget=None, media=None):
        from moha.observer import ObserverService
        from flat.providers.core import VideoOSPerceptionService, AssetCatalog
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        transport = Transport(replies)
        service = (VideoOSPerceptionService if original else ObserverService)(
            catalog=AssetCatalog({"unit": media or self.media}, allowed_roots=[(media or self.media).parent]),
            output_root=tmp.name, perception_backend=backend, perception_api_key="offline-test",
            base_url="http://localhost:1/v1", transport=transport, provider_retries=0,
            image_retries=0, budget=budget)
        return service, transport

    def observe(self, service, session=None):
        session = session or service.begin_episode("unit")["session_id"]
        result = service.inspect_window(session, start_seconds=1, end_seconds=2.5,
            inspection_goal="Describe the test pattern changes.", fps=4, resolution=224)
        return session, result

    def test_source_relative_modes_reach_actual_media_and_audited_receipts(self):
        from dataclasses import replace
        from flat.providers.core import default_perception_budget
        from flat.agent.observer_registry import ObserverHarnessConfig, ObserverGoal
        from moha.observer import PolicyExecution, PolicyObserverRegistry
        from moha.execution import ExecutionPolicy
        from moha.probes import realized_signature
        large = Path(self.media_tmp.name) / "large.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=1920x1080:rate=24", "-t", "1", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(large)], check=True)
        budget = replace(default_perception_budget(), f_view=128, p_view=2073600,
                         p_call=2073600*128, b_video=1000)
        signatures = []
        for mode in ("temporal", "spatial", "balanced"):
            service, transport = self.service([response()], budget=budget, media=large)
            session = service.begin_episode("unit")["session_id"]
            registry = PolicyObserverRegistry()
            registry.register("omni", service)
            result = registry.observe(config=ObserverHarnessConfig(),
                execution=PolicyExecution(policy=ExecutionPolicy(frames=128, priority=mode), modalities=("video",)),
                session_id=session, window=(0,1), goal=ObserverGoal(type="general",target="pattern"), receipt_id=mode)
            receipt = result["observer_execution_receipt"]
            actual = receipt["realized_execution"]
            self.assertEqual(receipt["window"],[0,1])
            self.assertEqual(actual["resolution"],actual["allocation"]["resolution"])
            self.assertEqual(actual["sampled_frames"],actual["allocation"]["frames"])
            self.assertEqual(len(actual["frame_timestamps_seconds"]),actual["sampled_frames"])
            self.assertIsNotNone(realized_signature(receipt))
            self.assertLessEqual(actual["input_token_accounting"]["budgeted_video_tokens"],1000)
            self.assertEqual(len(transport.calls),1)
            signatures.append(realized_signature(receipt))
        self.assertEqual(len(set(signatures)),3)

    def test_successful_first_call_is_wire_identical_and_preserves_claims(self):
        # Includes repeated and conflicting claims: no post-hoc merging/filtering.
        output = copy.deepcopy(GOOD)
        output["facts"] += [copy.deepcopy(output["facts"][0]),
            {**output["facts"][0], "fact": "The pattern is absent."}]
        original, before = self.service([response(output)], original=True)
        service, after = self.service([response(output)])
        _, old = self.observe(original)
        session, result = self.observe(service)
        self.assertEqual(before.calls, after.calls)
        self.assertEqual(len(result["observation"]["facts"]), 3)
        self.assertEqual(result["budget"], old["budget"])
        self.assertNotIn("observer_output_recoveries", service.receipt(session))

    def test_truncated_json_retries_same_media_once_with_accounting_and_saved_audits(self):
        from moha.observer import REPAIR_PROMPT
        service, transport = self.service([response('{"facts":[', finish="length", tokens=2048), response()])
        session, result = self.observe(service)
        self.assertEqual(len(transport.calls), 2)
        first, second = transport.calls
        self.assertEqual(first["messages"][1:], second["messages"][1:])
        self.assertEqual(second["messages"][0]["content"], first["messages"][0]["content"] + REPAIR_PROMPT)
        self.assertNotIn("max_completion_tokens", first)
        self.assertEqual(second["max_completion_tokens"], 2048)
        self.assertEqual(result["observation"]["facts"][0]["support_time_seconds"], [1.2, 1.8])
        receipt = service.receipt(session)
        self.assertEqual(receipt["provider_totals"], {"calls": 2, "input_tokens": 1200,
                                                   "output_tokens": 2128, "total_tokens": 3328})
        self.assertEqual(receipt["budget_ledger"]["look_used"], 2)
        recovery = receipt["observer_output_recoveries"][0]
        self.assertEqual(recovery["status"], "recovered")
        self.assertEqual(recovery["additional_calls"], 1)
        self.assertEqual([a["reason"] for a in recovery["attempts"]], ["output_truncated", None])
        root = service._session(session).root
        for attempt in recovery["attempts"]:
            request = json.loads((root / "request_audits" / (attempt["call_id"] + ".json")).read_text())
            output = json.loads((root / "provider_outputs" / (attempt["call_id"] + ".json")).read_text())
            self.assertEqual(request["request_sha256"], output["audit"]["request_sha256"])
        normal, _ = self.service([response()])
        normal_session, _ = self.observe(normal)
        normal_ledger = normal.receipt(normal_session)["budget_ledger"]
        for key in ("frames_used", "video_tokens_used", "audio_seconds_opened", "video_seconds_opened"):
            self.assertEqual(receipt["budget_ledger"][key], 2 * normal_ledger[key])

    def test_well_formed_but_length_finished_output_is_retried(self):
        service, transport = self.service([response(finish="length"), response()])
        self.observe(service)
        self.assertEqual(len(transport.calls), 2)

    def test_schema_invalid_output_is_repaired_and_a_lower_explicit_cap_is_preserved(self):
        from dataclasses import replace
        service, transport = self.service([response({"facts": "invalid"}), response()])
        session = service.begin_episode("unit")["session_id"]
        backend = service._session(session).perception
        backend.spec = replace(backend.spec, max_completion_tokens=1024)
        self.observe(service, session)
        self.assertEqual([c["max_completion_tokens"] for c in transport.calls], [1024, 1024])
        self.assertEqual(backend.spec.max_completion_tokens, 1024)

    def test_time_outside_window_is_retried_not_clamped(self):
        output = copy.deepcopy(GOOD)
        output["facts"][0]["support_time_seconds"] = [52, 53]
        service, transport = self.service([response(output), response()])
        session, result = self.observe(service)
        self.assertEqual(result["observation"]["facts"][0]["support_time_seconds"], [1.2, 1.8])
        recovery = service.receipt(session)["observer_output_recoveries"][0]
        self.assertEqual(recovery["attempts"][0]["reason"], "support_time_out_of_window")
        self.assertEqual(len(transport.calls), 2)

    def test_compact_observer_uses_same_recovery_and_host_time_mapping(self):
        # The compact schema permits a root fact array and supplies optional fields.
        output = [{"fact": "Pattern", "support_time_seconds": [0.2, 0.8]}]
        service, transport = self.service([response("{"), response(output)], backend="qwen2.5omni")
        _, result = self.observe(service)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(result["observation"]["facts"][0]["support_time_seconds"], [1.2, 1.8])

    def test_exhaustion_clears_pending_state_and_planner_can_continue_without_bad_evidence(self):
        service, transport = self.service([response("TRUNCATED_BAD_FACT", finish="length"),
            response("TRUNCATED_BAD_FACT", finish="length"), response()])
        observe = call("observe", {"start_seconds": 1, "end_seconds": 2.5,
            "goal": {"type": "sequence", "target": "pattern changes"}})
        planner = Planner([observe, observe, ANSWER])
        episode = EpisodeRunner(service, planner).run(Harness(memory=True), sample("unit"), 0)
        self.assertEqual(episode.status, "completed", episode.raw)
        self.assertEqual(len(transport.calls), 3)
        self.assertNotIn("TRUNCATED_BAD_FACT", json.dumps(planner.calls))
        error = next(e["result"] for e in episode.events if e["kind"] == "tool_result")
        self.assertEqual(error["error_type"], "ObserverOutputError")
        self.assertTrue(error["recoverable"])
        self.assertEqual(episode.usage["observer_retry_calls"], 1)
        self.assertEqual(episode.usage["provider_totals"]["calls"], 3)
        self.assertEqual(episode.usage["sensory_looks"], 3)
        view = unpack(diagnosis_view(episode, sample("unit"), Harness(memory=True)))
        self.assertEqual(view["observer_output_recoveries"][0]["status"], "exhausted")

    def test_network_missing_usage_and_budget_errors_remain_fatal(self):
        from dataclasses import replace
        from flat.providers.core import default_perception_budget
        from flat.core.errors import ProviderError
        from flat.providers.client import TransportResponse
        failures = [TransportResponse(status=503, body=b"unavailable"), response(usage=False),
                    response("{", usage=False), response("{", input_tokens=1000000)]
        for reply in failures:
            with self.subTest(reply=reply.status):
                service, transport = self.service([reply], budget=replace(default_perception_budget("qwen3omni"), c_sensor_max=100000))
                with self.assertRaises(ProviderError):
                    self.observe(service)
                self.assertEqual(len(transport.calls), 1)

    def test_transport_failure_on_retry_is_not_converted_to_observer_evidence_error(self):
        from flat.core.errors import ProviderError
        from flat.providers.client import TransportResponse
        service, transport = self.service([response("{"), TransportResponse(status=503, body=b"offline")])
        session = service.begin_episode("unit")["session_id"]
        with self.assertRaises(ProviderError):
            self.observe(service, session)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(service.receipt(session)["observer_output_recoveries"][0]["status"], "aborted")

    def test_probe_unusable_output_is_inconclusive_and_does_not_offer_a_rescue(self):
        from moha.probes import ObserverResolver, ProbeRunner
        observe = call("observe", {"start_seconds": 1, "end_seconds": 2.5,
            "goal": {"type": "sequence", "target": "pattern changes"}})
        original_service, _ = self.service([response()])
        episode = EpisodeRunner(original_service, Planner([observe, ANSWER])).run(Harness(), sample("unit"), 0)
        service, transport = self.service([response("{"), response("{")])
        resolver = ObserverResolver(ProbeRunner(service), object())
        step = next(e["step"] for e in episode.events if e["kind"] == "tool_result")
        result = resolver.resolve(sample("unit"), Harness(), episode, {"evidence_steps": [step]})
        self.assertEqual(result["status"], "inconclusive", result)
        self.assertEqual(result["probes"][0]["status"], "unusable")
        self.assertEqual(len(transport.calls), 2)

    def test_non_observation_schema_is_not_retried(self):
        from flat.core.errors import ProviderResponseError
        from flat.providers.core import VideoOSPerceptionService
        service, _ = self.service([])
        with patch.object(VideoOSPerceptionService, "_save_call", side_effect=ProviderResponseError("unusable")) as base:
            with self.assertRaises(ProviderResponseError):
                service._save_call(type("Session", (), {"perception": None})(),
                    mode_label="image_ocr", request=None, output_schema={}, schema_name="image_ocr_output")
        self.assertEqual(base.call_count, 1)
