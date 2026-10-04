import copy
import json
import unittest
from dataclasses import replace
from moha.demo import sample
from moha.models import Episode, Harness, canonical
from moha.evidence import diagnosis_view, filter_judge_input, messages_view, unpack
from moha.roles import Judge
from moha.catalog import catalog
from test_roles import FakeClient, diagnosis


def fixture():
    s, h = sample("calibration_only"), Harness()
    first = {"observation_id": "same_id", "facts": [{"fact": "The car is red."}], "view_id": "same_view"}
    later = {"observation_id": "same_id", "facts": [{"fact": "The car is blue."}], "view_id": "same_view"}
    initial = [{"role": "system", "content": "Actual planner instructions."},
               {"role": "user", "content": canonical({"task": s.task, "budget": {"steps": 16}})}]
    seen_red = initial + [{"role": "tool", "content": canonical({"observation": first})}]
    seen_blue = initial + [{"role": "tool", "content": canonical({"observation": later})}]
    events = [
        {"kind": "context", "step": 1, "messages": initial, "visible_observations": []},
        {"kind": "tool_call", "step": 1, "tool": "observe", "call_id": "c1",
         "arguments": {"goal": {"type": "general", "target": "car color"}}},
        {"kind": "tool_result", "step": 1, "tool": "observe", "call_id": "c1",
         "result": {"observation": first, "backend_result": {"observation": first}}},
        {"kind": "context", "step": 2, "messages": seen_red, "visible_observations": [first]},
        {"kind": "tool_call", "step": 3, "tool": "observe", "call_id": "c2",
         "arguments": {"goal": {"type": "general", "target": "car color"}}},
        {"kind": "tool_result", "step": 3, "tool": "observe", "call_id": "c2",
         "result": {"observation": later}},
        {"kind": "context", "step": 4, "messages": seen_blue, "visible_observations": [later]},
        {"kind": "terminal", "step": 5, "message": {"role": "assistant", "content": '{"answer":"B"}'}}]
    e = Episode(s.sample_id, s.id, h.id, 0, list(s.video_key), "A", "B", "completed", events=events,
                raw={"messages": initial, "tool_schemas": [{"name": "observe"}]})
    return e, s, h


class EvidenceTests(unittest.TestCase):
    def test_historical_shared_json_remains_readable(self):
        encoded = {"encoding": "shared_json", "shared": {"shared_1": {"facts": ["red"]}},
                   "data": {"a": {"$ref": "shared_1"}, "b": {"facts": ["blue"]},
                            "literal": {"$literal": {"$ref": "shared_1"}}}}
        self.assertEqual(unpack(encoded), {"a": {"facts": ["red"]}, "b": {"facts": ["blue"]},
                                          "literal": {"$ref": "shared_1"}})

    def test_judge_receives_actual_context_and_both_conflicting_claims(self):
        e, s, h = fixture()
        before = copy.deepcopy(e.to_dict())
        view = unpack(diagnosis_view(e, s, h))
        self.assertEqual(view["harness"], h.to_dict())
        self.assertEqual(view["tool_schemas"], e.raw["tool_schemas"])
        contexts = [x for x in view["events"] if x["kind"] == "context"]
        self.assertEqual(contexts[1]["messages"], messages_view(e.events[3]["messages"]))
        self.assertEqual(contexts[2]["messages"][-1]["content"]["observation"]["facts"][0]["fact"], "The car is blue.")
        self.assertIn("The car is red.", canonical(view))
        self.assertIn("The car is blue.", canonical(view))
        self.assertEqual(e.to_dict(), before)
        self.assertNotIn("used_by_final_reasoning", view)

    def test_non_json_messages_and_json_scalars_keep_their_meaning(self):
        texts = ['{"x":1}\n[Context note] Earlier history omitted.', 'null', '"literal"']
        result = messages_view([{"role": "tool", "content": t} for t in texts])
        self.assertEqual(result[0]["content"], texts[0])
        self.assertEqual(result[0]["content_encoding"], "text")
        self.assertIsNone(result[1]["content"])
        self.assertEqual(result[2]["content"], "literal")

    def test_missing_context_is_unknown_and_wrong_identity_is_rejected(self):
        e, s, h = fixture()
        e.raw = {}
        e.events[0].pop("messages")
        gaps = unpack(diagnosis_view(e, s, h))["recording_gaps"]
        self.assertTrue(gaps["tool_schemas_missing"])
        self.assertEqual(gaps["context_steps_without_messages"], [1])
        with self.assertRaises(ValueError):
            diagnosis_view(e, sample("wrong"), h)
        with self.assertRaises(ValueError):
            diagnosis_view(e, s, Harness(memory=True))

    def test_judge_filters_metadata_without_rewriting_evidence_or_source(self):
        e, s, h = fixture()
        e.usage = {"sampled_frames": 80, "provider_totals": {"input_tokens": 5000}}
        e.events[0]["history_audit"] = {"history_tokens": 900, "history_token_limit": 1000,
                                       "history_compacted": True, "history_turns_dropped": 2}
        # Schema names and model-authored data are not execution metadata.
        e.raw["tool_schemas"] = [{"properties": {"usage": {"type": "string"}}}]
        e.events[2]["result"]["observation"]["facts"][0].update(
            usage="The sign literally says usage.", frame_count="Text on the sign.")
        e.events[2]["audit"] = {"observer_execution_receipt": {
            "window": [10, 20], "instruction": "Read the sign.",
            "usage": {"input_tokens": 5000}, "receipt_id": "internal",
            "realized_execution": {"sampled_frames": 5, "fps": .5, "resolution": 384,
                "frame_timestamps_seconds": [10, 12, 14, 16, 18], "frame_cap_hit": False,
                "allocation": {"tokens": 5000}, "source_sha256": "internal"}}}
        e.events[-1]["message"]["content"] = '{"answer":"B","usage":"model authored","frame_count":5}'
        before = copy.deepcopy(e.to_dict())
        original = diagnosis_view(e, s, h)
        client = FakeClient(diagnosis(evidence_steps=[1]))
        result = Judge(client).diagnose(original, list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        sent = client.requests[0]["payload"]
        self.assertNotIn("encoding", sent)
        self.assertNotIn("shared", sent)
        self.assertNotIn("usage", sent)
        self.assertNotIn("shared_N", client.requests[0]["prompt"])
        self.assertEqual(sent["tool_schemas"], original["tool_schemas"])
        self.assertEqual(sent["initial_messages"], original["initial_messages"])
        self.assertEqual([(v["step"], v["kind"]) for v in sent["events"]],
                         [(v["step"], v["kind"]) for v in original["events"]])
        for source, filtered in zip(original["events"], sent["events"]):
            if source["kind"] == "context":
                self.assertNotIn("messages", filtered)
                self.assertEqual(filtered["visible_observations"], source["visible_observations"])
            elif source["kind"] == "tool_call":
                self.assertEqual(filtered, source)
            elif source["kind"] == "tool_result":
                self.assertEqual(filtered["result"]["observation"], source["result"]["observation"])
            elif "message" in source:
                self.assertEqual(filtered["message"], source["message"])
        self.assertEqual(sent["events"][0]["history_audit"],
                         {"history_compacted": True, "history_turns_dropped": 2})
        receipt = sent["events"][2]["audit"]["observer_execution_receipt"]
        self.assertEqual(receipt, {"window": [10, 20], "instruction": "Read the sign.",
            "realized_execution": {"fps": .5, "resolution": 384,
                "frame_timestamps_seconds": [10, 12, 14, 16, 18], "frame_cap_hit": False}})
        self.assertEqual(e.to_dict(), before)
        self.assertEqual(original, diagnosis_view(e, s, h))

    def test_context_without_visibility_keeps_messages_and_filter_only_deletes(self):
        e, s, h = fixture()
        e.events[3].pop("visible_observations")
        original = diagnosis_view(e, s, h)
        filtered = filter_judge_input(original)
        self.assertEqual(filtered["events"][3]["messages"], original["events"][3]["messages"])

        def assert_deletions_only(before, after):
            if isinstance(before, dict):
                self.assertEqual(list(after), [k for k in before if k in after])
                for key in after:
                    assert_deletions_only(before[key], after[key])
            elif isinstance(before, list):
                self.assertEqual(len(after), len(before))
                for left, right in zip(before, after):
                    assert_deletions_only(left, right)
            else:
                self.assertEqual(after, before)
        assert_deletions_only(original, filtered)

    def test_observer_recommendation_keeps_complete_context_and_actual_probes(self):
        e, s, h = fixture()
        d = {"status": "valid", **diagnosis(evidence_steps=[1]),
             "observer_resolution": {"status": "execution_rescue", "probes": [
                 {"status": "completed", "usage": {"input_tokens": 5000},
                  "result": {"observation": {"facts": ["Actual recovered evidence"]}}}],
                 "verdicts": [{"status": "valid", "reason": "Comparison reason", "attempts": ["AUDIT_NOT_NEEDED"]}]}}
        client = FakeClient({"candidate_id": None, "proposal_reason": "unresolved conflict"})
        result = Judge(client).recommend(diagnosis_view(e, s, h), d, list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        sent = unpack(client.requests[0]["payload"])
        self.assertNotIn("encoding", sent)
        self.assertNotIn("usage", sent["observer_resolution"]["probes"][0])
        self.assertIn("The car is red.", canonical(sent))
        self.assertIn("The car is blue.", canonical(sent))
        self.assertIn("Actual recovered evidence", canonical(sent))
        self.assertIn("Comparison reason", canonical(sent))
        self.assertNotIn("AUDIT_NOT_NEEDED", canonical(sent))
        self.assertEqual(len([x for x in sent["events"] if x["kind"] == "context"]), 3)


if __name__ == "__main__":
    unittest.main()
