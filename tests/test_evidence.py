import copy
import json
import unittest
from dataclasses import replace
from moha.demo import sample
from moha.models import Episode, Harness, canonical
from moha.evidence import diagnosis_view, messages_view, pack, unpack
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
    def test_shared_json_roundtrip_preserves_variants_and_literal_references(self):
        repeated = {"facts": ["red" * 300], "nested": {"value": 1}}
        value = {"a": repeated, "b": repeated, "changed": {"facts": ["blue" * 300]},
                 "literal": {"$ref": "shared_1"}, "also_literal": {"$literal": {"$ref": "x"}}}
        encoded = pack(value)
        self.assertEqual(unpack(encoded), value)
        self.assertLess(len(canonical(encoded)), len(canonical(value)))
        self.assertTrue(encoded["shared"])

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

    def test_judge_validates_cited_steps_inside_shared_payload(self):
        e, s, h = fixture()
        client = FakeClient(diagnosis(evidence_steps=[1]))
        result = Judge(client).diagnose(diagnosis_view(e, s, h), list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        self.assertEqual(client.requests[0]["payload"]["encoding"], "shared_json")

    def test_observer_recommendation_keeps_complete_context_and_actual_probes(self):
        e, s, h = fixture()
        d = {"status": "valid", **diagnosis(evidence_steps=[1]),
             "observer_resolution": {"status": "execution_rescue", "probes": [
                 {"status": "completed", "result": {"observation": {"facts": ["Actual recovered evidence"]}}}],
                 "verdicts": [{"status": "valid", "reason": "Comparison reason", "attempts": ["AUDIT_NOT_NEEDED"]}]}}
        client = FakeClient({"candidate_id": None, "proposal_reason": "unresolved conflict"})
        result = Judge(client).recommend(diagnosis_view(e, s, h), d, list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        sent = unpack(client.requests[0]["payload"])
        self.assertIn("The car is red.", canonical(sent))
        self.assertIn("The car is blue.", canonical(sent))
        self.assertIn("Actual recovered evidence", canonical(sent))
        self.assertIn("Comparison reason", canonical(sent))
        self.assertNotIn("AUDIT_NOT_NEEDED", canonical(sent))
        self.assertEqual(len([x for x in sent["events"] if x["kind"] == "context"]), 3)


if __name__ == "__main__":
    unittest.main()
