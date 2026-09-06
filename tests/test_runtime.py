"""Offline integration against the actual Video OS registry and provider wire code."""
import copy
import importlib.util
import json
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
from moha.demo import sample
from moha.models import Harness
from moha.runtime import EpisodeRunner


VIDEO_OS_AVAILABLE = importlib.util.find_spec("video_os") is not None


class Service:
    perception_model = "unit-omni"
    def __init__(self):
        self.calls = []
    def begin_episode(self, asset_id):
        self.calls.append(("begin", {"asset_id": asset_id}))
        return {"session_id": "unit-session", "media": {"duration_seconds": 60.0, "has_audio": True}}
    def get_state(self, session_id):
        return {"session_id": session_id, "budget": {"look_used": 1}, "state": {"mode": "THINK"}}
    def search(self, session_id, **kwargs):
        self.calls.append(("search", kwargs))
        return {"candidates": [{"start_seconds": 10.0, "end_seconds": 20.0}]}
    def overview(self, session_id, **kwargs):
        self.calls.append(("overview", kwargs))
        return {"summary": "coarse orientation"}
    def inspect_window(self, session_id, **kwargs):
        self.calls.append(("observe", kwargs))
        fps, resolution = kwargs["fps"], kwargs["resolution"]
        return {"observation": {"observation_id": "obs1", "facts": [{"fact": "A person jumps.", "support_time_seconds": [12, 13]}]},
                "view": {"view_id": "v1", "resolution": [resolution*2, resolution], "encoded_video_fps": fps,
                         "modalities": ["video", "audio"], "input_token_accounting": {"sampled_frames": int(10*fps)}},
                "budget": {"look_used": 1}}
    def receipt(self, session_id):
        return {"session_id": session_id, "budget_ledger": {"frames_used": 10, "video_tokens_used": 256,
            "audio_seconds_opened": 10, "video_seconds_opened": 10, "look_used": 1}}


class Planner:
    def __init__(self, messages):
        self.messages, self.calls = messages, []
    def call(self, **kwargs):
        from video_os.agent.harness import PlannerResponse
        self.calls.append(copy.deepcopy(kwargs))
        message = self.messages.pop(0)
        if isinstance(message, Exception):
            raise message
        return PlannerResponse.from_assistant_message(message)


def call(name, arguments, id="call"):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def script(final=None):
    return [call("video_player_search", {"query": "jumping person", "top_k": 3}, "search"),
            call("video_player_observe", {"start_seconds": 10, "end_seconds": 20,
                                         "goal": {"type": "general", "target": "person action"}}, "observe"),
            {"role": "assistant", "content": json.dumps(final or {"status": "answered", "answer": "A"})}]


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "run these integration tests in the Video OS environment")
class RuntimeTests(unittest.TestCase):
    def test_real_registry_search_observe_answer(self):
        service, planner = Service(), Planner(script())
        result = EpisodeRunner(service, planner).run(Harness(), sample("cal"), 0)
        self.assertEqual(result.status, "completed", result.raw)
        self.assertTrue(result.correct)
        self.assertEqual([c[0] for c in service.calls], ["begin", "search", "observe"])
        self.assertEqual(result.usage["observer_calls"], 1)
        self.assertEqual(result.usage["video_tokens"], 256)
        self.assertEqual(len(planner.calls), 3)
        context = [e for e in result.events if e["kind"] == "context"][-1]
        self.assertEqual(context["visible_observations"][0]["observation_id"], "obs1")
        self.assertNotIn("expected_answer", str(planner.calls))
        self.assertEqual({x["function"]["name"] for x in planner.calls[0]["tools"]}, {"video_player_search", "video_player_observe"})

    def test_modules_and_execution_reach_existing_runtime(self):
        service, planner = Service(), Planner(script())
        harness = Harness(overview=True, memory=True, verification=True,
                          execution=(("default", "default"), ("general", "dense_temporal")))
        result = EpisodeRunner(service, planner).run(harness, sample("cal"), 0)
        self.assertEqual(result.status, "completed", result.raw)
        self.assertIn("overview", [c[0] for c in service.calls])
        observed = next(c[1] for c in service.calls if c[0] == "observe")
        self.assertEqual(observed["fps"], 2.0)
        context = json.loads(planner.calls[-1]["messages"][-1]["content"])
        self.assertTrue(context["evidence_memory"])
        self.assertIn("advisory_verification", context)

    def test_planner_cleanup_preserves_full_audit_and_does_not_enable_memory(self):
        from moha.context import PLANNER_CONTEXT_POLICY
        service, planner = Service(), Planner(script())
        result = EpisodeRunner(service, planner).run(Harness(), sample("cal"), 0)
        self.assertEqual(result.status, "completed")
        sent = json.loads(planner.calls[-1]["messages"][-2]["content"])
        self.assertNotIn("observations", sent["player_state"])
        self.assertNotIn("observer_execution_receipt", sent)
        self.assertEqual(sent["observation"]["facts"][0]["fact"], "A person jumps.")
        raw = next(e["result"] for e in result.events if e["kind"] == "tool_result" and e["tool"] == "video_player_observe")
        self.assertIn("observer_execution_receipt", raw)
        self.assertTrue(raw["player_state"]["observations"])
        self.assertNotIn("evidence_memory", planner.calls[-1]["messages"][-1]["content"])
        self.assertEqual(result.raw["planner_context_policy"], PLANNER_CONTEXT_POLICY)
        self.assertEqual([e for e in result.events if e["kind"] == "context"][-1]["messages"], planner.calls[-1]["messages"])

    def test_cleanup_runs_before_bounded_history_selection(self):
        from moha.context import planner_messages
        from video_os.agent.harness import _planner_history_messages
        from test_context import envelope, message
        original = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        for i in range(5):
            value = envelope("visible fact " + str(i))
            value["player_state"]["observations"] *= 100
            original.extend([call("video_player_observe", {}, str(i)), message(value, str(i))])
        raw_selected, raw_audit = _planner_history_messages(original, token_limit=6000, max_turns=8)
        selected, audit = _planner_history_messages(planner_messages(original), token_limit=6000, max_turns=8)
        self.assertGreater(audit["history_turns_kept"], raw_audit["history_turns_kept"])
        self.assertIn("visible fact 2", str(selected))
        self.assertNotIn("visible fact 2", str(raw_selected))

    def test_budget_exhaustion_does_not_get_free_final_call(self):
        service, planner = Service(), Planner(script())
        result = EpisodeRunner(service, planner).run(Harness(max_steps=2), sample("cal"), 0)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertIsNone(result.answer)
        self.assertEqual(len(planner.calls), 2)

    def test_explicit_abstention_and_natural_terminal_answer(self):
        for message, status, answer in [({"status": "abstained", "answer": None}, "abstained", None),
                                        ({"status": "answered", "answer": "B"}, "completed", "B")]:
            result = EpisodeRunner(Service(), Planner(script(message))).run(Harness(), sample("cal"), 0)
            self.assertEqual((result.status, result.answer), (status, answer))
        messages = script()
        messages[-1]["content"] = "The answer is A."
        self.assertEqual(EpisodeRunner(Service(), Planner(messages)).run(Harness(), sample("cal"), 0).answer, "A")

    def test_planner_outage_preserves_auditable_error_episode(self):
        from moha.store import RunStore
        with tempfile.TemporaryDirectory() as tmp, RunStore(Path(tmp)/"run", {}) as store:
            result = EpisodeRunner(Service(), Planner([ConnectionError("outage")]), store=store).run(Harness(), sample("cal"), 0)
            self.assertEqual(result.status, "error")
            self.assertTrue(list(store.root.glob("attempts/*/events/*.json")))
            self.assertTrue(list(store.root.glob("attempts/*/result.json")))

    def test_observer_outage_cannot_enter_promotion_as_wrong_answer(self):
        from video_os.core.errors import ProviderError
        class Broken(Service):
            def inspect_window(self, *args, **kwargs):
                raise ProviderError("offline outage")
        planner = Planner(script())
        result = EpisodeRunner(Broken(), planner).run(Harness(), sample("cal"), 0)
        self.assertEqual(result.status, "error")
        self.assertEqual(len(planner.calls), 2)
        self.assertIn("perception_receipt", result.raw)

    def test_invalid_tool_request_is_feedback_not_semantic_rewrite(self):
        messages = script()
        arguments = json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])
        arguments["goal"]["reference"] = "invalid nonrelation reference"
        messages[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
        service, planner = Service(), Planner(messages)
        result = EpisodeRunner(service, planner).run(Harness(), sample("cal"), 0)
        self.assertEqual(result.status, "completed")
        self.assertNotIn("observe", [c[0] for c in service.calls])
        self.assertTrue(json.loads(planner.calls[-1]["messages"][-2]["content"])["isError"])

    def test_probe_uses_real_observer_api_without_planner_or_search(self):
        from moha.probes import ProbeRunner
        original = {"observer_id": "omni", "observer_model": "unit-omni", "window": [10, 20], "candidate_id": "s1_c1",
            "goal": {"type": "general", "target": "person action"},
            "requested_execution": {"fps": 1, "resolution": 384, "modalities": ["video", "audio"], "prompt_profile": "generic"}}
        service = Service()
        result = ProbeRunner(service).observe(sample("cal"), original, "dense_temporal")
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([c[0] for c in service.calls], ["begin", "observe"])
        self.assertEqual(service.calls[1][1]["start_seconds"], 10)
        self.assertEqual(service.calls[1][1]["end_seconds"], 20)
        self.assertEqual(service.calls[1][1]["fps"], 2)
        self.assertNotIn("expected_answer", str(service.calls))


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "run these integration tests in the Video OS environment")
class WireTests(unittest.TestCase):
    def test_text_boundary_keeps_video_metadata_in_actual_user_message(self):
        from video_os.core.dispatch import sanitize_gpt_text_payload
        from moha.evidence import messages_view, pack, unpack
        content = json.dumps({"initial": {"media": {"duration_seconds": 132.655599,
                             "has_audio": True}}, "task": {"question": "What happens last?"}})
        messages = [{"role": "user", "content": content}]
        payload = pack({"messages": messages_view(messages), "tool_result": {"artifact_id": "raw-media-handle"}})
        sent = unpack(sanitize_gpt_text_payload(payload))
        self.assertEqual(sent["messages"][0]["content"], content)
        self.assertEqual(json.loads(sent["messages"][0]["content"])["initial"]["media"]["duration_seconds"], 132.655599)
        self.assertNotIn("artifact_id", sent["tool_result"])

    def test_schema_reaches_real_adapter_wire_and_invalid_field_can_repair(self):
        from video_os.core.budget import BudgetContract
        from video_os.core.dispatch import ProviderRole
        from video_os.providers.client import OpenAICompatibleAdapter, ProviderSpec, TransportResponse
        from moha.bridge import TextRoleClient
        from moha.roles import Judge
        from test_roles import diagnosis, payload
        from moha.catalog import catalog
        class Transport:
            calls = []
            def post(self, **kwargs):
                self.calls.append(json.loads(kwargs["body"]))
                value = {"wrong_key": "x"} if len(self.calls) == 1 else diagnosis()
                body = {"id": "unit", "model": "unit", "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
                return TransportResponse(status=200, body=json.dumps(body).encode())
        budget = BudgetContract(c_text_max=None, c_sensor_max=None, b_control=None, b_video=8192,
            b_state=None, b_task=None, f_view=32, p_view=147456, p_call=1572864, k_look=8, k_compare=2, f_episode=256, b_video_episode=65536)
        transport = Transport()
        adapter = OpenAICompatibleAdapter(spec=ProviderSpec(role=ProviderRole.GPT_TEXT, model="unit", response_format_mode="json_schema"),
                                          api_key="unit-key", budget=budget, transport=transport)
        result = Judge(TextRoleClient(adapter)).diagnose(payload(), list(catalog().values()))
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.calls[0]["response_format"]["type"], "json_schema")
        self.assertIn("candidate_id", transport.calls[0]["response_format"]["json_schema"]["schema"]["properties"])


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "run these integration tests in the Video OS environment")
class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
        self.config["specialists"] = []
        self.config.pop("image", None)
        self.config.pop("asr", None)
        self.config["media_root"] = str(self.root)
        for role in ("planner", "judge"):
            self.config["models"][role]["key"] = {"local": True}
        for role in ("calibration", "validation"):
            video = self.root / f"{role}.mp4"
            video.write_bytes(role.encode())
            row = {"sample_id": role, "video_id": role, "media_relpath": video.name,
                   "media_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "expected_answer": "A",
                   "task": {"question": "Synthetic test", "options": {"A": "yes", "B": "no"}}}
            manifest = {"manifest_version": "video_os_direct_eval_input_v1", "selection_split": role,
                        "dataset": "unit", "samples": [row]}
            path = self.root / f"{role}.json"
            path.write_text(json.dumps(manifest))
            self.config[role+"_manifest"] = str(path)
        self.config_path = self.root / "config.json"
        self.save()

    def save(self):
        self.config_path.write_text(json.dumps(self.config))

    def prepare(self):
        from moha.bridge import prepare
        with patch("moha.bridge.source_identity", return_value={"dirty": False, "source_hash": "unit"}), \
             patch("urllib.request.urlopen", side_effect=AssertionError("doctor must not call an endpoint")):
            return prepare(self.config_path, Path(__file__).parents[1])

    def test_doctor_manifest_validation_and_build_without_model_calls(self):
        from moha.bridge import build
        from moha.store import RunStore
        prepared = self.prepare()
        self.assertEqual(len(prepared["calibration"]), 1)
        self.assertNotIn("observer.specialist.ocr", prepared["allowed_ids"])
        self.assertNotIn("planner.module.retrieval_basic", prepared["allowed_ids"])
        with RunStore(self.root/"run", prepared["identity"]) as store, \
             patch("urllib.request.urlopen", side_effect=AssertionError("build must not call an endpoint")):
            calibrator = build(prepared, store)
            self.assertEqual(calibrator.runner.service.perception_model, "Qwen3-Omni-30B-A3B-Instruct")
            self.assertEqual(calibrator.runner.service.asr_backend, "qwen3omni")
            self.assertIsNotNone(calibrator.resolver)

    def test_modified_video_fails_manifest_hash(self):
        (self.root/"calibration.mp4").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.prepare()

    def test_specialist_requires_executable_backend(self):
        self.config["specialists"] = ["ocr"]
        self.save()
        with self.assertRaisesRegex(ValueError, "OCR requires"):
            self.prepare()
        self.config["specialists"] = ["asr"]
        self.save()
        with self.assertRaisesRegex(ValueError, "ASR requires"):
            self.prepare()

    def test_specialists_build_separate_models_and_keep_shared_initial_harness(self):
        from moha.bridge import build
        from moha.store import RunStore
        template = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
        for field in ("specialists", "image", "asr"):
            self.config[field] = template[field]
        self.config["image"]["key"] = {"local": True}
        self.save()
        prepared = self.prepare()
        self.assertEqual(prepared["initial"].specialists, ())
        self.assertIn("observer.specialist.ocr", prepared["allowed_ids"])
        self.assertIn("observer.specialist.asr", prepared["allowed_ids"])
        with RunStore(self.root / "specialists-run", prepared["identity"]) as store:
            with patch("urllib.request.urlopen", side_effect=AssertionError("no model calls during build")):
                runner = build(prepared, store).runner
                self.assertEqual(runner.asr_backend, "whisper")
                self.assertEqual(runner.service.asr_perception_model, "whisper-large-v3-turbo")
                self.assertEqual(runner.service.ocr_perception_model, "Qwen/Qwen3.5-4B")
                self.assertEqual(runner.service.image_base_url, "https://api2.aigcbest.top/v1")
                from video_os.core.schema import SchemaRegistry
                adapter = runner.service.whisper_backend_factory(self.root, SchemaRegistry(), prepared["budget"])
                self.assertIsNone(adapter.default_language)
                self.assertEqual(adapter.endpoint(), "http://127.0.0.1:8093/v1/audio/transcriptions")

    def test_asr_rejects_chat_options_that_the_transcription_adapter_cannot_apply(self):
        template = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
        self.config["asr"] = template["asr"]
        self.config["asr"]["spec"]["max_completion_tokens"] = 100
        self.save()
        with self.assertRaisesRegex(ValueError, "ASR spec"):
            self.prepare()

    def test_shared_start_cannot_be_silently_replaced(self):
        self.config["initial"]["overview"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "shared Initial-Omni"):
            self.prepare()

    def test_old_selector_configuration_is_rejected_instead_of_silently_used(self):
        self.config["models"]["selector"] = copy.deepcopy(self.config["models"]["judge"])
        self.save()
        with self.assertRaisesRegex(ValueError, "selection is deterministic"):
            self.prepare()

    def test_cli_doctor_has_no_inference_side_effect(self):
        from moha.__main__ import main
        with patch("moha.bridge.source_identity", return_value={"dirty": True, "source_hash": "unit"}), \
             patch("urllib.request.urlopen", side_effect=AssertionError("no inference")), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["doctor", "--config", str(self.config_path)]), 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["clean_worktree"])


if __name__ == "__main__":
    unittest.main()
