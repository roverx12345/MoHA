import copy
import unittest
from moha.catalog import catalog
from moha.demo import sample
from moha.models import Episode, Harness
from moha.probes import ObserverResolver, realized_signature
from moha.roles import DIAGNOSIS_SCHEMA, Judge, Selector
from moha.evidence import unpack


def diagnosis(**kwargs):
    return {"failure": "observer", "confidence": 0.8, "reason": "Observed evidence contradicts calibration reference.",
            "evidence_steps": [2], "failed_capability": None, "residual_reason": "", **kwargs}


def payload():
    return {"events": [{"kind": "tool_call", "step": 2, "tool": "video_player_observe",
                         "arguments": {"goal": {"type": "text", "target": "score"}}}]}


class FakeClient:
    def __init__(self, *outputs):
        self.outputs, self.requests = list(outputs), []

    def call(self, prompt, value, schema, name):
        self.requests.append({"prompt": prompt, "payload": copy.deepcopy(value), "schema": schema, "name": name})
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output, {"usage": {"input_tokens": 10}, "raw_text": str(output)}


class RoleTests(unittest.TestCase):
    def test_exact_schema_is_in_actual_prompt(self):
        client = FakeClient(diagnosis())
        result = Judge(client).diagnose(payload())
        self.assertEqual(result["status"], "valid")
        self.assertIn('"failure"', client.requests[0]["prompt"])
        self.assertIn("OUTPUT JSON SCHEMA", client.requests[0]["prompt"])
        self.assertEqual(client.requests[0]["schema"], DIAGNOSIS_SCHEMA)

    def test_wrong_taxonomy_field_gets_one_exact_repair(self):
        client = FakeClient({"primary_failure": "observer"}, diagnosis())
        result = Judge(client).diagnose(payload())
        self.assertEqual(result["status"], "valid")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(client.requests[0]["schema"], client.requests[1]["schema"])
        self.assertIn("previous_output", client.requests[1]["payload"])

    def test_repair_failure_is_not_unresolved_semantics(self):
        client = FakeClient({"failure_label": "observer"}, {"failure_label": "observer"})
        result = Judge(client).diagnose(payload())
        self.assertEqual(result["status"], "error")
        self.assertNotIn("failure", result)

    def test_network_errors_do_not_trigger_taxonomy_repair(self):
        client = FakeClient(ConnectionError("outage"))
        self.assertEqual(Judge(client).diagnose(payload())["status"], "error")
        self.assertEqual(len(client.requests), 1)

    def test_evidence_steps_must_exist(self):
        client = FakeClient(diagnosis(evidence_steps=[999]), diagnosis(evidence_steps=[999]))
        self.assertEqual(Judge(client).diagnose(payload())["status"], "error")

    def test_capability_needs_corresponding_typed_request(self):
        client = FakeClient(diagnosis(failed_capability="asr"), diagnosis(failed_capability="asr"))
        self.assertEqual(Judge(client).diagnose(payload())["status"], "error")
        self.assertEqual(Judge(FakeClient(diagnosis(failed_capability="ocr"))).diagnose(payload())["status"], "valid")

    def test_direct_trace_failure_does_not_call_judge(self):
        client = FakeClient()
        result = Judge(client).diagnose({"events": [{"kind": "terminal", "step": 1}]})
        self.assertEqual(result["failure"], "orientation")
        self.assertEqual(client.requests, [])

    def test_retrieved_but_never_inspected_is_direct_trace_evidence(self):
        client = FakeClient()
        result = Judge(client).diagnose({"events": [
            {"kind": "tool_call", "step": 1, "tool": "video_player_search"},
            {"kind": "tool_result", "step": 1, "tool": "video_player_search",
             "result": {"player_state": {"search": {"candidates": [{"candidate_id": "c1"}]}}}},
            {"kind": "terminal", "step": 2}]})
        self.assertEqual(result["failure"], "candidate_selection")
        self.assertEqual(client.requests, [])

    def test_selector_sees_only_calibration_and_acceptance_history(self):
        client = FakeClient({"candidate_id": "planner.module.memory_basic", "reason": "retain evidence"})
        history = [{"candidate": "x", "from": "h0", "to": "h1", "validation": {
            "accepted": False, "new_accuracy": 0.4, "heldout_secret": "DO_NOT_EXPOSE"}}]
        result = Selector(client).select([{"status": "valid", **diagnosis()}], Harness(), list(catalog().values()), history)
        self.assertEqual(result["status"], "valid")
        value = unpack(client.requests[0]["payload"])
        self.assertNotIn("DO_NOT_EXPOSE", str(value))
        self.assertIn("failure_profile", value)
        self.assertNotIn("new_accuracy", str(value))

    def test_selector_cannot_invent_catalog_entry(self):
        client = FakeClient({"candidate_id": "invented", "reason": "x"}, {"candidate_id": "invented", "reason": "x"})
        self.assertEqual(Selector(client).select([], Harness(), list(catalog().values()), [])["status"], "error")


def receipt(fps=1):
    return {"receipt_id": "original", "observer_id": "omni", "observer_model": "test-omni", "window": [10, 20],
            "goal": {"type": "general", "target": "action"}, "requested_execution": {"fps": 1.0, "resolution": 384, "modalities": ["video"], "prompt_profile": "generic"},
            "realized_execution": {"fps": fps, "resolution": 384, "sampled_frames": 10*fps,
                                   "modalities": ["video"], "prompt_profile": "generic"}}


class FakeProbe:
    def __init__(self, change=True, fail=False):
        self.calls, self.change, self.fail = [], change, fail

    def observe(self, sample, original, preset):
        self.calls.append((sample, original, preset))
        return {"status": "error" if self.fail else "completed", "preset": preset,
                "result": {"observer_execution_receipt": receipt(2 if preset != "baseline" and self.change else 1)}}


class ProbeTests(unittest.TestCase):
    def resolve(self, runner, client):
        s, harness = sample("cal"), Harness()
        e = Episode(s.sample_id, s.id, harness.id, 0, list(s.video_key), "A", "B", "completed",
                    events=[{"kind": "tool_result", "step": 2, "result": {"observer_execution_receipt": receipt()}}])
        return ObserverResolver(runner, client, p_view=384**2).resolve(s, harness, e, diagnosis())

    def test_fixed_request_fresh_control_and_alternative_rescue(self):
        runner = FakeProbe()
        result = self.resolve(runner, FakeClient({"baseline_supports_goal": False, "alternative_supports_goal": True, "reason": "required action recovered"}))
        self.assertEqual(result["status"], "execution_rescue")
        self.assertEqual([c[2] for c in runner.calls], ["baseline", "dense_temporal"])
        self.assertEqual(runner.calls[0][1], runner.calls[1][1])

    def test_baseline_also_recovers_not_execution_rescue(self):
        result = self.resolve(FakeProbe(), FakeClient({"baseline_supports_goal": True, "alternative_supports_goal": True, "reason": "both have evidence"}))
        self.assertEqual(result["status"], "inconclusive")

    def test_ineffective_render_skips_rescue_judge(self):
        client = FakeClient()
        result = self.resolve(FakeProbe(change=False), client)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(client.requests, [])

    def test_probe_errors_not_capability_failures(self):
        result = self.resolve(FakeProbe(fail=True), FakeClient())
        self.assertEqual(result["status"], "error")

    def test_no_rescue_requires_known_negative_both_controls(self):
        result = self.resolve(FakeProbe(), FakeClient({"baseline_supports_goal": False, "alternative_supports_goal": False, "reason": "both lack evidence"}))
        self.assertEqual(result["status"], "no_rescue")
        result = self.resolve(FakeProbe(), FakeClient({"baseline_supports_goal": None, "alternative_supports_goal": False, "reason": "baseline unknown"}))
        self.assertEqual(result["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
