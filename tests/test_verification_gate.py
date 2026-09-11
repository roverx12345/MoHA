import copy
import json
import unittest
from moha.demo import sample
from moha.models import Harness
from moha.runtime import EpisodeRunner
from moha.verification import diagnosis_messages, parse_audit
from test_runtime import Service as BaseService, Planner, call, VIDEO_OS_AVAILABLE


class Service(BaseService):
    def inspect_window(self, session_id, **kwargs):
        result = super().inspect_window(session_id, **kwargs)
        result["observation"]["facts"][0]["support_time_seconds"] = [kwargs["start_seconds"], kwargs["end_seconds"]]
        return result


def final(label="A"):
    return {"role": "assistant", "content": json.dumps({"status": "answered", "answer": label})}


def audit(status="insufficient", label=None):
    return {"role": "assistant", "content": json.dumps({"support_status": status,
        "unsupported_assumptions": ["The inference is not established by the observation."],
        "contradictory_evidence": [], "best_supported_option": label,
        "diagnosis": "Use only the original scoped observations; the inference may be unsupported."})}


def observe(index=0):
    return call("observe", {"start_seconds": index, "end_seconds": index+2,
                "goal": {"type": "general", "target": "person action"}}, str(index))


class AuditContractTests(unittest.TestCase):
    def test_verification_requires_two_total_calls_and_is_one_catalog_coordinate(self):
        from moha.catalog import catalog
        with self.assertRaises(ValueError):
            Harness(verification=True, max_steps=1)
        self.assertEqual(Harness(max_steps=1).max_steps, 1)
        base = Harness()
        intervention = catalog()["planner.module.verification_basic"]
        target = intervention.apply(base)
        self.assertEqual(target.max_steps, base.max_steps)
        self.assertTrue(target.verification)
        self.assertEqual({k for k in target.to_dict() if target.to_dict()[k] != base.to_dict()[k]}, {"verification"})
        self.assertIn("finalize", intervention.description)

    def test_focus_ids_cannot_hide_contrary_evidence_from_final_audit(self):
        records = [{"observation": {"observation_id": name, "facts": [{"fact": fact}],
                    "missing": ["The cause is not shown."], "uncertainties": ["Identity unclear."]},
                    "observation_context": {"window": [0, 2]}}
                   for name, fact in [("one", "White marks persist."), ("two", "The woman wears clown makeup.")]]
        task = {"question": "Why?", "options": {"A": "Salt", "B": "Makeup"}, "expected_answer": "SECRET_SENTINEL"}
        candidate = {"status": "answered", "answer": "A"}
        messages = diagnosis_messages(task, records, "Ignore the second observation.", ["one"],
                                      candidate_answer=candidate, planner_hypotheses="It must be salt.")
        data = json.loads(messages[1]["content"])
        self.assertEqual(data["candidate_answer"], candidate)
        self.assertEqual(data["options"], task["options"])
        self.assertEqual(data["focus_source_ids"], ["one"])
        self.assertEqual(len(data["observations"]), 2)
        self.assertEqual(data["observations"][1]["observation"], records[1]["observation"])
        self.assertEqual(data["planner_hypotheses"]["status"], "unverified")
        self.assertNotIn("SECRET_SENTINEL", str(messages))

    def test_audit_validation_does_not_repair_invalid_labels_types_or_duplicates(self):
        valid = json.loads(audit()["content"])
        self.assertEqual(parse_audit(json.dumps(valid), {"A", "B"}), valid)
        invalid = [None, "plain prose", '{"support_status":"supported","support_status":"insufficient"}']
        for key, value in [("support_status", "maybe"), ("unsupported_assumptions", "none"),
                           ("contradictory_evidence", [3]), ("best_supported_option", {"answer": "A"}),
                           ("best_supported_option", "Z"), ("diagnosis", "")]:
            invalid.append(json.dumps({**valid, key: value}))
        for content in invalid:
            with self.subTest(content=content), self.assertRaises((ValueError, TypeError)):
                parse_audit(content, {"A", "B"})


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires pinned Video OS")
class VerificationGateTests(unittest.TestCase):
    def test_candidate_at_seven_is_audited_at_eight_and_finalized_at_nine(self):
        planner = Planner([observe(i*3) for i in range(6)] + [final("B"), audit("contradicted", "A"), final("A")])
        result = EpisodeRunner(Service(), planner).run(Harness(verification=True, max_steps=16), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(result.usage["model_calls"], 9)
        self.assertEqual(result.usage["planner_calls"], 8)
        self.assertEqual(result.usage["verification_calls"], 1)
        gate = result.raw["verification_gate"]
        self.assertEqual((gate["trigger"], gate["step"]), ("pre_submit", 8))
        candidate = next(e for e in result.events if e["kind"] == "candidate_answer")
        self.assertEqual((candidate["step"], candidate["answer"]["answer"]), (7, "B"))
        fresh = json.loads(planner.calls[7]["messages"][1]["content"])
        self.assertEqual(fresh["candidate_answer"]["answer"], "B")
        last = planner.calls[-1]
        self.assertEqual((last["tools"], last["tool_choice"]), ([], "none"))
        context = json.loads(last["messages"][-1]["content"])
        self.assertEqual(context["remaining_planner_calls"], 1)
        self.assertEqual(context["finalization"]["observations"], fresh["observations"])
        self.assertEqual(context["finalization"]["verification"]["audit"]["best_supported_option"], "A")
        self.assertEqual(result.events[-1]["kind"], "terminal")

    def test_non_submitting_memory_loop_gets_step_fifteen_audit_and_sixteen_final(self):
        messages = [observe(0), observe(3)] + [call("memory_read", {"ledger": "result"}, str(i)) for i in range(12)]
        messages += [audit(), final()]
        planner = Planner(messages)
        result = EpisodeRunner(Service(), planner).run(Harness(memory=True, verification=True, max_steps=16), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual((result.usage["model_calls"], result.usage["verification_calls"]), (16, 1))
        self.assertEqual(result.usage["planner_calls"], 15)
        self.assertEqual(result.raw["verification_gate"]["trigger"], "budget_floor")
        self.assertEqual(result.raw["verification_gate"]["step"], 15)
        self.assertEqual(len(json.loads(planner.calls[14]["messages"][1]["content"])["observations"]), 2)
        self.assertEqual(planner.calls[14]["tools"], [])
        self.assertEqual(planner.calls[15]["tools"], [])
        self.assertTrue(any(e["kind"] == "memory_control" and e["no_novelty"] for e in result.events))

    def test_manual_verification_counts_once_and_blocks_following_tools_in_same_response(self):
        request = call("verify_fresh", {}, "verify")
        request["tool_calls"] += observe()["tool_calls"] + call("verify_fresh", {}, "again")["tool_calls"]
        service, planner = Service(), Planner([request, audit(), final()])
        result = EpisodeRunner(service, planner).run(Harness(verification=True, max_steps=16), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(result.usage["model_calls"], 3)
        self.assertEqual(result.usage["verification_calls"], 1)
        self.assertEqual(result.raw["verification_gate"]["trigger"], "planner_request")
        self.assertNotIn("observe", [x[0] for x in service.calls])
        results = [e for e in result.events if e["kind"] == "tool_result"]
        self.assertEqual(len(results), 3)
        self.assertTrue(all(e["result"]["isError"] for e in results[1:]))

    def test_abstention_is_audited_and_final_decision_is_not_forced_by_verifier(self):
        abstain = {"role": "assistant", "content": '{"status":"abstained","answer":null}'}
        planner = Planner([abstain, audit("supported", "B"), final("A")])
        result = EpisodeRunner(Service(), planner).run(Harness(verification=True), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(result.usage["model_calls"], 3)
        self.assertEqual(result.raw["verification_gate"]["candidate_answer"]["status"], "abstained")

    def test_two_call_budget_starts_with_audit_without_gifting_a_planning_call(self):
        planner = Planner([audit(), final()])
        result = EpisodeRunner(Service(), planner).run(Harness(verification=True, max_steps=2), sample("cal"), 0)
        self.assertEqual(result.usage["model_calls"], 2)
        self.assertEqual(result.usage["planner_calls"], 1)
        self.assertEqual(result.raw["verification_gate"]["step"], 1)
        self.assertEqual(result.status, "completed", result.raw)

    def test_invalid_audit_is_reported_once_and_never_retried(self):
        for malformed in (None, "not a JSON audit", '{"support_status":"supported"}'):
            with self.subTest(malformed=malformed):
                planner = Planner([final(), {"role": "assistant", "content": malformed}, final()])
                result = EpisodeRunner(Service(), planner).run(Harness(verification=True), sample("cal"), 0)
                self.assertEqual(result.status, "completed", result.raw)
                self.assertEqual(result.usage["model_calls"], 3)
                self.assertEqual(result.usage["verification_calls"], 1)
                verification = json.loads(planner.calls[-1]["messages"][-1]["content"])["finalization"]["verification"]
                self.assertEqual(verification["audit_status"], "invalid")
                self.assertIsNone(verification["audit"])
                self.assertEqual(verification["diagnosis"], malformed)

    def test_post_audit_mode_has_only_one_response_and_never_executes_tools(self):
        for response in ({"role": "assistant", "content": "Still thinking."}, observe()):
            with self.subTest(response=response):
                service, planner = Service(), Planner([final(), audit(), response])
                result = EpisodeRunner(service, planner).run(Harness(verification=True), sample("cal"), 0)
                self.assertEqual((result.status, result.answer), ("invalid_final_answer", None), result.raw)
                self.assertEqual(result.usage["model_calls"], 3)
                self.assertNotIn("observe", [x[0] for x in service.calls])
                self.assertEqual(planner.calls[-1]["tools"], [])

    def test_verification_final_does_not_invoke_an_extra_answer_model(self):
        class Extractor:
            calls = 0
            def call(self, **kwargs):
                self.calls += 1
                raise AssertionError("no extra compute after verification")
        extractor = Extractor()
        planner = Planner([final(), audit(), {"role": "assistant", "content": "The evidence is unclear."}])
        result = EpisodeRunner(Service(), planner, extractor=extractor).run(Harness(verification=True), sample("cal"), 0)
        self.assertEqual(extractor.calls, 0)
        self.assertEqual(result.usage["model_calls"], 3)
        self.assertEqual(result.raw["terminal_answer_status"], "invalid")
