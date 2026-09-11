import copy
import unittest
from moha.catalog import catalog
from moha.demo import sample
from moha.models import Episode, Harness
from moha.probes import ObserverResolver, realized_signature
from moha.roles import DIAGNOSIS_SCHEMA, Judge, rank_candidates
from moha.evidence import unpack
from moha.execution import EXECUTION_POLICY


def diagnosis(**kwargs):
    return {"failure": "observer", "confidence": 0.8, "reason": "Observed evidence contradicts calibration reference.",
            "evidence_steps": [2], "failed_capability": None, "residual_reason": "", "candidate_id": None, "proposal_reason": "await probes", **kwargs}


def payload():
    return {"events": [{"kind": "tool_call", "step": 2, "tool": "video_player_observe",
                         "arguments": {"goal": {"type": "text", "target": "score"}}}]}


class FakeClient:
    def __init__(self, *outputs):
        self.outputs, self.requests = list(outputs), []

    def call(self, prompt, value, schema, name):
        self.requests.append({"prompt": prompt, "payload": copy.deepcopy(value), "schema": schema, "name": name})
        output = self.outputs[0] if name == "moha_probe_verdict" else self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output, {"usage": {"input_tokens": 10}, "raw_text": str(output)}


class RoleTests(unittest.TestCase):
    def test_exact_schema_is_in_actual_prompt(self):
        client = FakeClient(diagnosis())
        result = Judge(client).diagnose(payload(), list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        self.assertIn('"failure"', client.requests[0]["prompt"])
        self.assertIn("OUTPUT JSON SCHEMA", client.requests[0]["prompt"])
        self.assertTrue(set(DIAGNOSIS_SCHEMA["properties"]) < set(client.requests[0]["schema"]["properties"]))

    def test_wrong_taxonomy_field_gets_one_exact_repair(self):
        client = FakeClient({"primary_failure": "observer"}, diagnosis())
        result = Judge(client).diagnose(payload(), list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(client.requests[0]["schema"], client.requests[1]["schema"])
        self.assertIn("previous_output", client.requests[1]["payload"])

    def test_repair_failure_is_not_unresolved_semantics(self):
        client = FakeClient({"failure_label": "observer"}, {"failure_label": "observer"})
        result = Judge(client).diagnose(payload(), list(catalog().values()))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("failure", result)

    def test_network_errors_do_not_trigger_taxonomy_repair(self):
        client = FakeClient(ConnectionError("outage"))
        self.assertEqual(Judge(client).diagnose(payload(), list(catalog().values()))["status"], "error")
        self.assertEqual(len(client.requests), 1)

    def test_evidence_steps_must_exist(self):
        client = FakeClient(diagnosis(evidence_steps=[999]), diagnosis(evidence_steps=[999]))
        self.assertEqual(Judge(client).diagnose(payload(), list(catalog().values()))["status"], "error")

    def test_schema_distinguishes_recorded_steps_from_event_indices(self):
        trace = payload()
        trace["events"][0]["index"] = 900
        trace["events"] += [{"kind": "planner", "step": 7, "index": 901},
                            {"kind": "context", "step": 2}, {"kind": "metadata"}]
        client = FakeClient(diagnosis(evidence_steps=[900]), diagnosis(evidence_steps=[2]))
        result = Judge(client).diagnose(trace, list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        self.assertEqual(client.requests[0]["schema"]["properties"]["evidence_steps"]["items"]["enum"], [2, 7])
        self.assertEqual(client.requests[0]["schema"], client.requests[1]["schema"])
        self.assertEqual(result["attempts"][0]["output"]["evidence_steps"], [900])
        self.assertEqual(result["evidence_steps"], [2])

    def test_null_proposal_reason_is_required_on_wire_and_not_fabricated(self):
        client = FakeClient(diagnosis(proposal_reason=""), diagnosis(proposal_reason="Probes must come first."))
        result = Judge(client).diagnose(payload(), list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        self.assertIsNone(result["candidate_id"])
        self.assertEqual(result["proposal_reason"], "Probes must come first.")
        self.assertEqual(result["attempts"][0]["output"]["proposal_reason"], "")
        self.assertEqual(client.requests[0]["schema"]["properties"]["proposal_reason"]["minLength"], 1)
        self.assertIn("candidate_id is null", client.requests[0]["prompt"])
        invalid = diagnosis(proposal_reason=" ")
        self.assertEqual(Judge(FakeClient(invalid, invalid)).diagnose(payload(), list(catalog().values()))["status"], "error")

    def test_no_recorded_steps_allows_only_empty_citations(self):
        client = FakeClient(diagnosis(failure="unresolved", evidence_steps=[], residual_reason="No trace evidence."))
        result = Judge(client).diagnose({"events": []}, list(catalog().values()))
        self.assertEqual(result["status"], "valid")
        self.assertEqual(client.requests[0]["schema"]["properties"]["evidence_steps"]["maxItems"], 0)

    def test_capability_needs_corresponding_typed_request(self):
        client = FakeClient(diagnosis(failed_capability="asr"), diagnosis(failed_capability="asr"))
        self.assertEqual(Judge(client).diagnose(payload(), list(catalog().values()))["status"], "error")
        self.assertEqual(Judge(FakeClient(diagnosis(failed_capability="ocr"))).diagnose(payload(), list(catalog().values()))["status"], "valid")

    def test_no_evidence_trace_still_needs_model_proposal_not_fixed_routing(self):
        client = FakeClient(diagnosis(failure="orientation", evidence_steps=[1],
                                    candidate_id="planner.module.memory_basic", proposal_reason="hypothesis"))
        result = Judge(client).diagnose({"events": [{"kind": "terminal", "step": 1}]}, list(catalog().values()))
        self.assertEqual(result["candidate_id"], "planner.module.memory_basic")
        self.assertEqual(len(client.requests), 1)

    def test_initial_observer_and_unresolved_must_abstain(self):
        for failure in ("observer", "unresolved"):
            invalid = diagnosis(failure=failure, candidate_id="planner.module.memory_basic")
            client = FakeClient(invalid, invalid)
            self.assertEqual(Judge(client).diagnose(payload(), list(catalog().values()))["status"], "error")

    def test_judge_cannot_invent_or_choose_unavailable_catalog_entry(self):
        for candidate in ("invented", "planner.module.memory_basic", "observer.specialist.ocr"):
            d = diagnosis(failure="verification", candidate_id=candidate)
            self.assertEqual(Judge(FakeClient(d, d)).diagnose(payload(), [catalog()["planner.module.overview"]])["status"], "error")

    def test_observer_specialist_admission_matches_this_trace_and_probe(self):
        available = list(catalog().values())
        for status, goal, allowed in [("no_rescue", "text", True), ("execution_rescue", "text", False),
                                      ("inconclusive", "text", False), ("no_rescue", "speech", False)]:
            d = {"status": "valid", **diagnosis(failed_capability="ocr"),
                 "observer_resolution": {"status": status, "goal_type": goal}}
            client = FakeClient({"candidate_id": "observer.specialist.ocr" if allowed else None,
                                 "proposal_reason": "local probe evidence"})
            result = Judge(client).recommend(payload(), d, available)
            self.assertEqual(result["status"], "valid")
            ids = [c["id"] for c in unpack(client.requests[0]["payload"])["available"]]
            self.assertEqual("observer.specialist.ocr" in ids, allowed)
            self.assertNotIn("observer.specialist.asr", ids)

    def test_uniform_votes_ties_and_unavailable_candidates(self):
        a, b = "planner.module.memory_basic", "planner.module.overview"
        votes = [{"status": "valid", "sample_id": str(i), **diagnosis(failure="verification",
                    candidate_id=candidate, confidence=confidence)}
                 for i, (candidate, confidence) in enumerate([(a, .01), (a, .01), (b, 1), (None, 1)])]
        ranked = rank_candidates(votes, list(catalog().values()))
        self.assertEqual(ranked["candidate_id"], a)
        self.assertEqual([x["support_count"] for x in ranked["ranking"]], [2, 1])
        # Confidence never breaks a tie, nor does the order of input or catalog.
        self.assertEqual(rank_candidates(votes[1:], list(reversed(list(catalog().values()))))["candidate_id"], a)
        self.assertEqual(rank_candidates(votes, [catalog()[b]])["candidate_id"], b)
        with self.assertRaises(ValueError):
            rank_candidates(votes + [votes[0]], list(catalog().values()))

    def test_unresolved_error_and_unprobed_specialist_have_no_vote(self):
        rows = [{"sample_id": "a", "status": "valid", **diagnosis(failure="unresolved", candidate_id="planner.module.overview")},
                {"sample_id": "b", "status": "error", "candidate_id": "planner.module.overview"},
                {"sample_id": "c", "status": "valid", **diagnosis(failed_capability="ocr", candidate_id="observer.specialist.ocr")}]
        self.assertIsNone(rank_candidates(rows, list(catalog().values()))["candidate_id"])


class ExecutionEligibilityTests(unittest.TestCase):
    def test_execution_vote_requires_that_specific_probed_rescue(self):
        from moha.roles import eligible_for_trace
        candidate = catalog()["observer.execution.text.target_fps.2.0"]
        available = [candidate]
        for resolution in ({}, {"status":"inconclusive"}, {"status":"execution_rescue","candidate_id":"different"}):
            self.assertEqual(eligible_for_trace(available,diagnosis(observer_resolution=resolution)),[])
        self.assertEqual(eligible_for_trace(available,diagnosis(observer_resolution={
            "status":"execution_rescue","candidate_id":candidate.id})),available)


def receipt(fps=1):
    return {"receipt_id": "original", "observer_id": "omni", "observer_model": "test-omni", "window": [10, 20],
            "goal": {"type": "general", "target": "action"}, "requested_execution": {"policy_version": EXECUTION_POLICY, "frames": "auto", "source_scale": 1.0, "priority": "balanced", "modalities": ["video"], "prompt_profile": "generic"},
            "realized_execution": {"fps": fps, "resolution": 384, "sampled_frames": 10*fps,
                                   "modalities": ["video"], "prompt_profile": "generic", "source_sha256": "unit-source",
                                   "frame_timestamps_seconds": [10+i/fps for i in range(10*fps)]}}


class FakeProbe:
    def __init__(self, change=True, fail=False):
        self.calls, self.change, self.fail = [], change, fail

    def plan(self, sample, original, execution):
        policy = execution.policy
        return {"source_sha256": "unit-source", "window": original["window"],
                "frames": policy.sampling_request(10)["target_frames"], "resolution": [policy.source_scale, policy.priority]}

    def observe(self, sample, original, execution, candidate_id="baseline"):
        self.calls.append((sample, original, candidate_id))
        return {"status": "error" if self.fail else "completed", "candidate_id": candidate_id,
                "perception_receipt": {},
                "result": {"observer_execution_receipt": receipt(2 if candidate_id != "baseline" and self.change else 1)}}


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
        self.assertEqual([c[2] for c in runner.calls], ["baseline", "observer.execution.general.priority.spatial"])
        self.assertEqual(runner.calls[0][1], runner.calls[1][1])

    def test_baseline_also_recovers_not_execution_rescue(self):
        result = self.resolve(FakeProbe(), FakeClient({"baseline_supports_goal": True, "alternative_supports_goal": True, "reason": "both have evidence"}))
        self.assertEqual(result["status"], "inconclusive")

    def test_ineffective_render_skips_rescue_judge(self):
        client = FakeClient()
        result = self.resolve(FakeProbe(change=False), client)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(client.requests, [])

    def test_planned_noops_do_not_render_or_call_either_model(self):
        class Noop(FakeProbe):
            def plan(self, sample, original, execution):
                return {"source_sha256": "unit-source", "window": original["window"], "frames": 1, "resolution": [2,2]}
        runner, client = Noop(), FakeClient()
        result = self.resolve(runner, client)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(runner.calls, [])
        self.assertEqual(client.requests, [])
        self.assertEqual(len(result["skipped"]), 6)

    def test_only_available_candidates_can_be_probed(self):
        runner, client = FakeProbe(), FakeClient()
        s, harness = sample("cal"), Harness()
        e = Episode(s.sample_id,s.id,harness.id,0,list(s.video_key),"A","B","completed",
            events=[{"kind":"tool_result","step":2,"result":{"observer_execution_receipt":receipt()}}])
        result = ObserverResolver(runner,client).resolve(s,harness,e,diagnosis(), [])
        self.assertEqual(result["status"],"inconclusive")
        self.assertEqual(runner.calls,[])

    def test_changed_source_is_rejected_before_any_model_call(self):
        class Changed(FakeProbe):
            def plan(self, *args):
                return {**super().plan(*args), "source_sha256": "changed"}
        runner, client = Changed(), FakeClient()
        result = self.resolve(runner,client)
        self.assertEqual(result["status"],"error")
        self.assertEqual(runner.calls,[])
        self.assertEqual(client.requests,[])

    def test_probe_errors_not_capability_failures(self):
        result = self.resolve(FakeProbe(fail=True), FakeClient())
        self.assertEqual(result["status"], "error")

    def test_no_rescue_requires_known_negative_both_controls(self):
        result = self.resolve(FakeProbe(), FakeClient({"baseline_supports_goal": False, "alternative_supports_goal": False, "reason": "both lack evidence"}))
        self.assertEqual(result["status"], "no_rescue")
        result = self.resolve(FakeProbe(), FakeClient({"baseline_supports_goal": None, "alternative_supports_goal": False, "reason": "baseline unknown"}))
        self.assertEqual(result["status"], "inconclusive")

    def test_specialist_probe_cannot_use_an_unrelated_cited_goal(self):
        s, harness = sample("cal"), Harness()
        e = Episode(s.sample_id, s.id, harness.id, 0, list(s.video_key), "A", "B", "completed",
                    events=[{"kind": "tool_result", "step": 2,
                             "result": {"observer_execution_receipt": receipt()}}])
        runner, client = FakeProbe(), FakeClient()
        result = ObserverResolver(runner, client).resolve(s, harness, e, diagnosis(failed_capability="ocr"))
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(runner.calls, [])

    def test_matching_typed_goal_is_probed_even_after_an_unrelated_receipt(self):
        s, harness = sample("cal"), Harness()
        typed = receipt()
        typed.update(receipt_id="typed", goal={"type": "text", "target": "score"})
        e = Episode(s.sample_id, s.id, harness.id, 0, list(s.video_key), "A", "B", "completed",
                    events=[{"kind": "tool_result", "step": 2,
                             "result": {"observer_execution_receipt": r}} for r in (receipt(), typed)])
        runner = FakeProbe()
        client = FakeClient({"baseline_supports_goal": False, "alternative_supports_goal": False,
                             "reason": "both controls lack text evidence"})
        result = ObserverResolver(runner, client).resolve(s, harness, e, diagnosis(failed_capability="ocr"))
        self.assertEqual(result["status"], "no_rescue")
        self.assertEqual(result["goal_type"], "text")
        self.assertTrue(all(c[1]["receipt_id"] == "typed" for c in runner.calls))


if __name__ == "__main__":
    unittest.main()
