import copy
import json
import unittest
from dataclasses import replace
from moha.demo import sample
from moha.models import Episode, Harness, canonical
from moha.evidence import diagnosis_view, messages_view, pack, unpack, selection_trace, representative_traces
from moha.roles import Judge, Selector
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
        {"kind": "tool_call", "step": 1, "tool": "video_player_observe", "call_id": "c1",
         "arguments": {"goal": {"type": "general", "target": "car color"}}},
        {"kind": "tool_result", "step": 1, "tool": "video_player_observe", "call_id": "c1",
         "result": {"observation": first, "backend_result": {"observation": first}}},
        {"kind": "context", "step": 2, "messages": seen_red, "visible_observations": [first]},
        {"kind": "tool_call", "step": 3, "tool": "video_player_observe", "call_id": "c2",
         "arguments": {"goal": {"type": "general", "target": "car color"}}},
        {"kind": "tool_result", "step": 3, "tool": "video_player_observe", "call_id": "c2",
         "result": {"observation": later}},
        {"kind": "context", "step": 4, "messages": seen_blue, "visible_observations": [later]},
        {"kind": "terminal", "step": 5, "message": {"role": "assistant", "content": '{"answer":"B"}'}}]
    e = Episode(s.sample_id, s.id, h.id, 0, list(s.video_key), "A", "B", "completed", events=events,
                raw={"messages": initial, "tool_schemas": [{"name": "video_player_observe"}]})
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
        result = messages_view([{"role": "user", "content": t} for t in texts])
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
            selection_trace(e, s, Harness(memory=True), {})

    def test_judge_validates_cited_steps_inside_shared_payload(self):
        e, s, h = fixture()
        client = FakeClient(diagnosis(evidence_steps=[1]))
        result = Judge(client).diagnose(diagnosis_view(e, s, h))
        self.assertEqual(result["status"], "valid")
        self.assertEqual(client.requests[0]["payload"]["encoding"], "shared_json")

    def test_selector_keeps_later_counterevidence_and_final_context(self):
        e, s, h = fixture()
        d = {"status": "valid", "sample_id": s.sample_id, **diagnosis(evidence_steps=[1])}
        d["trace_evidence"] = selection_trace(e, s, h, d)
        client = FakeClient({"candidate_id": None, "reason": "unresolved conflict"})
        result = Selector(client).select([d], h, list(catalog().values()), [
            {"candidate": "x", "from": "a", "to": "b", "validation": {
                "accepted": False, "test_label": "HELDOUT_LABEL_MUST_NOT_LEAK"}}])
        self.assertEqual(result["status"], "valid")
        sent = unpack(client.requests[0]["payload"])
        example = sent["calibration_evidence"]["examples"][0]["trace"]
        self.assertEqual(example["sample_id"], s.sample_id)
        self.assertIn("The car is red.", canonical(example))
        self.assertIn("The car is blue.", canonical(example))
        self.assertIn(4, [x["step"] for x in example["context_excerpts"]])
        self.assertNotIn("HELDOUT_LABEL_MUST_NOT_LEAK", canonical(sent))

    def test_full_context_excerpts_are_bounded_without_dropping_claims(self):
        e, s, h = fixture()
        e.events += [{"kind": "context", "step": step, "messages": e.raw["messages"]}
                     for step in range(6, 17)]
        trace = selection_trace(e, s, h, diagnosis(evidence_steps=[1, 6, 8, 10, 12]))
        self.assertEqual([c["step"] for c in trace["context_excerpts"]], [1, 2, 16])
        self.assertIn(10, trace["context_steps_omitted"])
        self.assertIn("The car is red.", canonical(trace))
        self.assertIn("The car is blue.", canonical(trace))

    def test_selector_probe_evidence_is_not_only_a_rescue_label(self):
        e, s, h = fixture()
        d = diagnosis(observer_resolution={"status": "execution_rescue", "probes": [
            {"status": "completed", "preset": "dense_temporal", "result": {
                "observation": {"facts": [{"fact": "Actual recovered evidence"}]}}}],
            "verdicts": [{"status": "valid", "preset": "dense_temporal",
                          "reason": "Comparison reason", "attempts": [{"secret": "AUDIT_NOT_NEEDED"}]}]})
        trace = selection_trace(e, s, h, d)
        self.assertIn("Actual recovered evidence", canonical(trace))
        self.assertIn("Comparison reason", canonical(trace))
        self.assertNotIn("AUDIT_NOT_NEEDED", canonical(trace))

    def test_examples_are_bounded_and_execution_rescue_has_its_own_example(self):
        traces = [{"status": "valid", "failure": "observer", "trace_evidence": {
            "sample_id": f"s{i}", "timeline": list(range(i))}} for i in range(5)]
        traces.append({**traces[0], "observer_resolution": {"status": "execution_rescue"}})
        examples = representative_traces(traces)["examples"]
        self.assertEqual(len(examples), 2)
        ordinary = next(x for x in examples if x["failure"] == "observer")
        self.assertEqual(ordinary["family_sample_count"], 5)
        self.assertEqual(ordinary["trace"]["sample_id"], "s2")


if __name__ == "__main__":
    unittest.main()
