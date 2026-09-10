"""Exact temporal support through the real registry, routes and probe runner."""
import copy
import json
import unittest
from moha.demo import sample
from moha.models import Harness
from moha.runtime import EpisodeRunner
from moha.records import receipts
from test_runtime import VIDEO_OS_AVAILABLE, Service, Planner, call


GOAL = {"type": "sequence", "target": "what happens before and after the action"}
ANSWER = {"role": "assistant", "content": '{"status":"answered","answer":"A"}'}


def observe(start, end, goal=None):
    return call("video_player_observe", {"start_seconds": start, "end_seconds": end,
                                         "goal": goal or GOAL})


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires the pinned Video OS runtime")
class WindowTests(unittest.TestCase):
    def registry(self, service=None):
        from moha.tools import WindowPlayerRegistry
        from flat.agent.harness import VideoToolRegistry
        service = service or Service()
        started = service.begin_episode("unit")
        return WindowPlayerRegistry(VideoToolRegistry(service, started["session_id"]),
            started=started, initial_state=service.get_state(started["session_id"]),
            observer_profile="semantic_omni"), service

    def test_direct_observe_without_search_reaches_wire_and_receipt(self):
        from moha.tools import PLANNER_TOOL_POLICY
        service, planner = Service(), Planner([observe(5, 35), ANSWER])
        episode = EpisodeRunner(service, planner).run(Harness(), sample("unit"), 0)
        self.assertEqual(episode.status, "completed", episode.raw)
        self.assertEqual([c[0] for c in service.calls], ["begin", "observe"])
        wire = service.calls[-1][1]
        self.assertEqual((wire["start_seconds"], wire["end_seconds"], wire["experiment_render"]["requested_frames"]), (5, 35, 30))
        self.assertNotIn("resolution", wire)
        receipt = receipts(episode.events)[0]
        self.assertEqual(receipt["window"], [5, 35])
        self.assertIsNone(receipt["candidate_id"])
        self.assertEqual(receipt["goal"], GOAL)
        self.assertEqual(episode.raw["planner_tool_policy"], PLANNER_TOOL_POLICY)
        self.assertIsNone(episode.raw["player_state"]["active_candidate_id"])
        self.assertEqual(episode.raw["player_state"]["visited_windows"], [[5, 35]])
        sent = json.loads(planner.calls[-1]["messages"][-2]["content"])
        self.assertEqual(sent["observation_context"]["window"], [5, 35])
        self.assertIsNone(sent["observation_context"]["candidate_id"])
        schemas = {s["function"]["name"]: s["function"] for s in planner.calls[0]["tools"]}
        self.assertEqual(set(schemas), {"video_player_search", "video_player_observe"})
        parameters = schemas["video_player_observe"]["parameters"]
        self.assertEqual(set(parameters["properties"]), {"start_seconds", "end_seconds", "goal"})
        self.assertEqual(set(parameters["required"]), set(parameters["properties"]))
        self.assertFalse(parameters["additionalProperties"])

    def test_search_candidate_can_be_expanded_or_left_for_another_region(self):
        class ShortSearch(Service):
            def search(self, session_id, **kwargs):
                self.calls.append(("search", kwargs))
                return {"candidates": [{"start_seconds": 10, "end_seconds": 12}]}
        service = ShortSearch()
        planner = Planner([call("video_player_search", {"query": "action"}),
                           observe(5, 25), observe(40, 55), ANSWER])
        episode = EpisodeRunner(service, planner).run(Harness(), sample("unit"), 0)
        self.assertEqual(episode.status, "completed", episode.raw)
        windows = [(c[1]["start_seconds"], c[1]["end_seconds"]) for c in service.calls if c[0] == "observe"]
        self.assertEqual(windows, [(5, 25), (40, 55)])
        self.assertEqual([r["window"] for r in receipts(episode.events)], [[5, 25], [40, 55]])
        self.assertEqual(episode.raw["player_state"]["visited_windows"], [[5, 25], [40, 55]])

    def test_fractional_end_and_subsecond_support_are_not_shifted_or_rounded(self):
        class Fractional(Service):
            def begin_episode(self, asset_id):
                started = super().begin_episode(asset_id)
                started["media"]["duration_seconds"] = 60.123457
                return started
        service, planner = Fractional(), Planner([observe(59.923456, 60.123457), ANSWER])
        episode = EpisodeRunner(service, planner).run(Harness(), sample("unit"), 0)
        self.assertEqual(episode.status, "completed", episode.raw)
        self.assertEqual(receipts(episode.events)[0]["window"], [59.923456, 60.123457])
        self.assertEqual(episode.raw["player_state"]["current_window"], [59.923456, 60.123457])

    def test_invalid_windows_or_controls_leave_state_and_backend_unchanged(self):
        registry, service = self.registry()
        invalid = [{"start_seconds": a, "end_seconds": b, "goal": GOAL}
                   for a, b in [(-1, 5), (5, 5), (6, 5), (0, 61), (True, 5),
                                (0, "5"), (float("nan"), 5), (0, float("inf"))]]
        valid = {"start_seconds": 5, "end_seconds": 35, "goal": GOAL}
        invalid += [{**valid, key: value} for key, value in
                    [("fps", 8), ("resolution", 768), ("observer", "other"),
                     ("candidate_id", "s1_c1"), ("sampling_policy", "dense")]]
        invalid += [{"candidate_id": "s1_c1", "goal": GOAL},
                    {"start_seconds": 0, "goal": GOAL}, {**valid, "goal": None},
                    {**valid, "goal": {**GOAL, "reference": "invalid for sequence"}}]
        before = copy.deepcopy(registry.snapshot())
        calls = copy.deepcopy(service.calls)
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                registry.invoke("video_player_observe", arguments)
            self.assertEqual(registry.snapshot(), before)
            self.assertEqual(service.calls, calls)

    def test_wrong_session_and_unexposed_tools_cannot_bypass_window_contract(self):
        registry, service = self.registry()
        args = {"start_seconds": 0, "end_seconds": 10, "goal": GOAL}
        with self.assertRaises(PermissionError):
            registry.invoke("video_player_observe", args, session_id="another")
        with self.assertRaises(ValueError):
            registry.invoke("video_inspect_window", {**args, "fps": 8})
        self.assertEqual([x[0] for x in service.calls], ["begin"])

    def test_bad_window_is_repairable_planner_feedback(self):
        planner = Planner([observe(0, 61), observe(0, 30), ANSWER])
        service = Service()
        episode = EpisodeRunner(service, planner).run(Harness(), sample("unit"), 0)
        self.assertEqual(episode.status, "completed", episode.raw)
        self.assertTrue(json.loads(planner.calls[1]["messages"][-2]["content"])["isError"])
        self.assertEqual([c[0] for c in service.calls], ["begin", "observe"])

    def test_execution_adaptation_changes_sampling_while_preserving_window(self):
        service, planner = Service(), Planner([observe(0, 60), ANSWER])
        harness = Harness.from_dict({"execution": {"default": {}, "sequence": {"frames": 128}}})
        episode = EpisodeRunner(service, planner).run(harness, sample("unit"), 0)
        self.assertEqual(episode.status, "completed", episode.raw)
        receipt = receipts(episode.events)[0]
        self.assertEqual(receipt["window"], [0, 60])
        self.assertEqual(receipt["requested_execution"]["frames"], 128)

    def test_failed_observer_records_the_selected_window_without_a_candidate(self):
        from flat.core.errors import ProviderError
        class Broken(Service):
            def inspect_window(self, *args, **kwargs):
                raise ProviderError("offline provider failure")
        episode = EpisodeRunner(Broken(), Planner([observe(3, 43)])).run(Harness(), sample("unit"), 0)
        self.assertEqual(episode.status, "error")
        receipt = receipts(episode.events)[0]
        self.assertEqual(receipt["window"], [3, 43])
        self.assertIsNone(receipt["candidate_id"])
        self.assertEqual(receipt["goal"], GOAL)
        self.assertEqual(receipt["realized_execution"]["target_frames"], 40)
        self.assertEqual(receipt["realized_execution"]["target_fps"], 1)
        self.assertIsNone(receipt["realized_execution"]["realized_frames"])

    def test_rate_reaches_explicit_frame_wire_and_actual_receipt(self):
        for fps, expected in ((0.5, 13), (1, 25), (2, 50)):
            service = Service()
            harness = Harness.from_dict({"execution": {"default": {"target_fps": fps}}})
            episode = EpisodeRunner(service, Planner([observe(5, 30), ANSWER])).run(harness, sample("unit"), 0)
            self.assertEqual(episode.status, "completed", episode.raw)
            wire = service.calls[-1][1]
            self.assertNotIn("fps", wire)
            self.assertEqual(wire["experiment_render"]["requested_frames"], expected)
            r = receipts(episode.events)[0]["realized_execution"]
            self.assertEqual(r["window_duration"], 25)
            self.assertEqual(r["target_fps"], fps)
            self.assertEqual(r["target_frames"], expected)
            self.assertEqual(r["realized_frames"], expected)
            self.assertEqual(r["realized_fps"], expected / 25)
            self.assertEqual(r["realized_resolution"], [640, 360])
            self.assertFalse(r["frame_cap_hit"])

    def test_receipt_uses_realized_frames_when_other_budgets_bind(self):
        class Limited(Service):
            def plan_observer_execution(self, session_id, window, policy):
                from test_execution import renderer, media
                from moha.execution import allocate
                return allocate(renderer(b_video=512), media(), window, policy)
        harness = Harness.from_dict({"execution": {"default": {"target_fps": 2}}})
        episode = EpisodeRunner(Limited(), Planner([observe(0, 60), ANSWER])).run(harness, sample("unit"), 0)
        self.assertEqual(episode.status, "completed", episode.raw)
        r = receipts(episode.events)[0]["realized_execution"]
        self.assertEqual(r["target_frames"], 120)
        self.assertLess(r["realized_frames"], 120)
        self.assertEqual(r["realized_fps"], r["sampled_frames"] / 60)
        self.assertFalse(r["frame_cap_hit"])

    def test_probe_reuses_direct_window_without_retrieval(self):
        from moha.probes import ProbeRunner
        episode = EpisodeRunner(Service(), Planner([observe(5, 45), ANSWER])).run(Harness(), sample("unit"), 0)
        original = receipts(episode.events)[0]
        service = Service()
        from moha.observer import PolicyExecution
        from moha.execution import ExecutionPolicy
        result = ProbeRunner(service).observe(sample("unit"), original, PolicyExecution(policy=ExecutionPolicy(frames=128)))
        self.assertEqual(result["status"], "completed", result)
        receipt = result["result"]["observer_execution_receipt"]
        self.assertEqual(receipt["window"], [5, 45])
        self.assertIsNone(receipt["candidate_id"])
        self.assertEqual(receipt["goal"], original["goal"])
        self.assertEqual([c[0] for c in service.calls], ["begin", "observe"])

    def test_specialist_routes_preserve_the_planner_window(self):
        class Specialists(Service):
            def read_text(self, session_id, **kwargs):
                self.calls.append(("ocr", kwargs))
                return {"observation": {"observation_id": "ocr", "facts": [{"fact": "OWNERS MANUAL"}]},
                        "image_analysis": {"text": "OWNERS MANUAL"}, "budget": {"look_used": 1}}
            def asr_window(self, session_id, **kwargs):
                self.calls.append(("asr", kwargs))
                return {"observation": {"observation_id": "asr", "facts": [{"fact": "Hello"}]},
                        "asr": {"segments": [{"text": "Hello"}]}, "budget": {"look_used": 1}}
        for specialist, goal_type in [("ocr", "text"), ("asr", "speech")]:
            with self.subTest(specialist=specialist):
                service = Specialists()
                planner = Planner([observe(5, 25, {"type": goal_type, "target": "read words"}), ANSWER])
                episode = EpisodeRunner(service, planner).run(Harness(specialists=(specialist,)), sample("unit"), 0)
                self.assertEqual(episode.status, "completed", episode.raw)
                self.assertEqual([c[0] for c in service.calls], ["begin", specialist])
                receipt = receipts(episode.events)[0]
                self.assertEqual(receipt["window"], [5, 25])
                self.assertIsNone(receipt["candidate_id"])
                self.assertTrue(receipt["specialist_active"])
