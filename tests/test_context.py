import copy
import json
import unittest
from moha.context import planner_messages, tool_context
from moha.models import canonical


def message(value, call_id="call"):
    return {"role": "tool", "tool_call_id": call_id, "content": canonical(value)}


def envelope(text="The car is red."):
    return {"tool": "observe", "isError": False,
            "observation": {"observation_id": "same", "facts": [
                {"fact_id": "same.fact_1", "fact": text, "support_time_seconds": [10, 11]}],
                "missing": ["No readable plate"], "uncertainties": ["Color affected by lighting"]},
            "observer_execution_receipt": {"receipt_id": "AUDIT_ONLY", "observer_model": "unit",
                "candidate_id": "s1_c1", "window": [10, 12], "goal": {"type": "attribute", "target": "car"},
                "realized_execution": {"fps": 1, "resolution": 288, "sampled_frames": 2, "modalities": ["video"]}},
            "player_state": {"current_window": [10, 12], "visited_windows": [[10, 12]],
                "last_observation_id": "ORPHAN", "observations": [
                    {"observation_id": "ORPHAN", "fact_ids": ["ORPHAN.fact_1"], "fact_count": 1}],
                "search": {"last_search_id": "s1", "candidates": [
                    {"candidate_id": "s0_c1", "start_seconds": 0, "end_seconds": 2},
                    {"candidate_id": "s1_c1", "start_seconds": 10, "end_seconds": 12}],
                    "history": [{"search_id": "s1", "candidate_ids": ["s1_c1"]}]}},
            "result": {"player_state": {"current_window": [10, 12]}, "evidence": {"claim": "Other claim"}},
            "budget": {"look_used": 1}, "state": {"budget": {"look_used": 1}, "state": {"step": 1}},
            "view": {"time_range_seconds": [10, 12], "frame_timestamps_seconds": [10, 11]},
            "backend_result": {"time_mapping": {"model_time_origin_source_seconds": 10}}}


class ContextTests(unittest.TestCase):
    def test_orphan_audits_removed_but_facts_scope_and_latest_actions_remain(self):
        original = envelope()
        before = copy.deepcopy(original)
        result = tool_context(original)
        self.assertNotIn("ORPHAN", canonical(result))
        self.assertNotIn("AUDIT_ONLY", canonical(result))
        self.assertEqual(result["observation"], original["observation"])
        self.assertNotIn("view", result)
        self.assertNotIn("backend_result", result)
        self.assertEqual(result["navigation"]["search"]["candidates"], original["player_state"]["search"]["candidates"])
        self.assertEqual(result["navigation"]["search"]["history"], original["player_state"]["search"]["history"])
        self.assertEqual(result["navigation"]["visited_windows"], [[10, 12]])
        self.assertEqual(result["observation_context"]["sampling"]["sampled_frames"], 2)
        self.assertEqual(result["observation_context"]["sampling"]["frame_timestamps_seconds"], [10, 11])
        self.assertEqual(result["observation_context"]["instruction"], "car")
        self.assertEqual(result["observation_context"]["evidence_type"], "attribute")
        self.assertNotIn("goal", result["observation_context"])
        self.assertEqual(tool_context(result), result)
        self.assertEqual(result["evidence"], {"claim": "Other claim"})
        for key in ("result", "budget", "state", "player_state"):
            self.assertNotIn(key, result)
        self.assertEqual(original, before)

    def test_only_latest_state_is_replayed_and_older_search_handles_survive(self):
        search = envelope()
        search["tool"] = "search"
        source = [message(search, "search"), message(envelope(), "observe")]
        result = [json.loads(m["content"]) for m in planner_messages(source)]
        self.assertEqual(result[0]["candidates"], [{"candidate_id": "s1_c1", "start_seconds": 10, "end_seconds": 12}])
        for key in ("state", "budget", "player_state"):
            self.assertNotIn(key, result[0])
            self.assertNotIn(key, result[1])
        self.assertNotIn("navigation", result[0])
        self.assertIn("navigation", result[1])
        self.assertEqual(source[0]["tool_call_id"], "search")

    def test_conflicting_claims_with_the_same_id_are_never_overwritten(self):
        source = [message(envelope("red")), message(envelope("blue"))]
        result = [json.loads(m["content"]) for m in planner_messages(source)]
        self.assertEqual([x["observation"]["facts"][0]["fact"] for x in result], ["red", "blue"])
        self.assertEqual([x["observation"]["observation_id"] for x in result], ["same", "same"])

    def test_metadata_alone_cannot_resurrect_missing_evidence(self):
        value = envelope()
        value.pop("observation")
        result = tool_context(value)
        self.assertNotIn("observation", result)
        self.assertNotIn("ORPHAN", canonical(result))
        self.assertNotIn("evidence_memory", result)

    def test_protocol_error_does_not_remove_the_last_available_state(self):
        error = {"isError": True, "error_type": "ValueError", "message": "candidate_id is unknown"}
        source = [message(envelope()), message(error)]
        result = [json.loads(m["content"]) for m in planner_messages(source)]
        self.assertIn("navigation", result[0])
        self.assertEqual(result[1], error)

    def test_unstructured_messages_and_tool_calls_keep_their_original_content(self):
        source = [{"role": "system", "content": "Keep this"}, {"role": "user", "content": "Question"},
                  {"role": "assistant", "tool_calls": [{"id": "c", "function": {"arguments": "{}"}}]},
                  {"role": "tool", "content": "not JSON"}, {"role": "tool", "content": "null"},
                  {"role": "tool", "content": [{"type": "text", "text": "tool text"}]}]
        self.assertEqual(planner_messages(source), source)

    def test_prefetched_overview_keeps_navigation_content_without_stale_state(self):
        overview = {**envelope(), "overview": {"summary": "A red car appears later."}}
        initial = {"task": {"question": "Which color?"}, "initial": {"media": {"duration_seconds": 30}},
                   "overview": overview}
        source = [{"role": "system", "content": "system"}, {"role": "user", "content": canonical(initial)},
                  message(envelope())]
        result = json.loads(planner_messages(source)[1]["content"])
        self.assertEqual(result["task"], initial["task"])
        self.assertEqual(result["initial"], initial["initial"])
        self.assertEqual(result["overview"]["overview"], overview["overview"])
        self.assertNotIn("navigation", result["overview"])

    def test_initial_and_all_tool_inputs_exclude_host_telemetry(self):
        telemetry = {"action_advice": ["Use next_keyframe and step_frames."],
                     "video_token_episode_advisory_limit": 65536, "frames_used": 128}
        initial = {"task": {"question": "Which color?"}, "initial": {
            "media": {"duration_seconds": 30, "has_audio": True, "frame_rate": 25},
            "sensory_budget": telemetry, "budget": {"video_tokens_used": 50}}}
        result = envelope()
        result["state"]["sensory_budget"] = telemetry
        result["receipt"] = {"provider": "private endpoint", "request_sha256": "private hash"}
        projected = planner_messages([{"role": "system", "content": "system"},
            {"role": "user", "content": canonical(initial)}, message(result)])
        self.assertEqual(json.loads(projected[1]["content"])["initial"],
                         {"media": {"duration_seconds": 30, "has_audio": True}})
        sent = canonical(projected)
        for hidden in ("action_advice", "next_keyframe", "step_frames", "advisory_limit",
                       "video_tokens_used", "request_sha256", "private endpoint", "sensory_budget"):
            self.assertNotIn(hidden, sent)
        self.assertEqual(json.loads(projected[2]["content"])["observation"], result["observation"])

    def test_mirrored_evidence_deduplicates_only_identical_records(self):
        original = envelope()
        original["evidence"] = {"claim": "Other claim"}
        original["backend_result"]["evidence"] = {"claim": "Conflicting claim"}
        projected = tool_context(original)
        self.assertEqual(projected["evidence"], {"claim": "Other claim"})
        self.assertEqual(projected["additional_results"], [{"evidence": {"claim": "Conflicting claim"}}])
        self.assertEqual(tool_context(projected), projected)


if __name__ == "__main__":
    unittest.main()
