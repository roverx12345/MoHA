"""Native offline regressions for malformed and incomplete planner responses."""
import copy
import json
import unittest
from unittest.mock import patch

from moha.demo import sample
from moha.models import Harness
from moha.runtime import EpisodeRunner
from test_runtime import Service, Planner, call, VIDEO_OS_AVAILABLE


FINAL = {"role": "assistant", "content": '{"status":"answered","answer":"A"}'}


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires pinned Video OS")
class TerminalRecoveryTests(unittest.TestCase):
    def test_invalid_structured_values_do_not_crash_or_extract_nested_guesses(self):
        invalid = [
            {"status": "answered", "answer": {"answer": "A", "reason": "The answer is B."}},
            {"status": "answered", "answer": ["A"]},
            {"status": "answered", "answer": 1},
            {"status": "answered", "answer": "Z"},
            {"status": "abstained", "answer": "A"},
            {"status": "abstained"},
        ]
        for value in invalid:
            with self.subTest(value=value):
                messages = [{"role": "assistant", "content": json.dumps(value)}, copy.deepcopy(FINAL)]
                planner = Planner(messages)
                result = EpisodeRunner(Service(), planner).run(Harness(max_steps=2), sample("cal"), 0)
                self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
                self.assertEqual(result.usage["model_calls"], 2)
                self.assertFalse(any(e["kind"] == "error" for e in result.events))
                self.assertEqual(len([e for e in result.events if e["kind"] == "answer_recovery"]), 1)
                self.assertEqual(planner.calls[-1]["tool_choice"], "none")
                self.assertIn(json.dumps(value), str(planner.calls[-1]["messages"]))

    def test_nonterminal_reply_can_continue_observing_before_final_answer(self):
        planner = Planner([
            {"role": "assistant", "content": "I need to inspect the scene before deciding."},
            call("video_player_observe", {"start_seconds": 10, "end_seconds": 20,
                 "goal": {"type": "general", "target": "person action"}}),
            copy.deepcopy(FINAL),
        ])
        service = Service()
        result = EpisodeRunner(service, planner).run(Harness(max_steps=3), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(result.usage["model_calls"], 3)
        self.assertEqual(result.usage["observer_calls"], 1)
        self.assertEqual([c["tool_choice"] for c in planner.calls], ["auto", "auto", "none"])
        self.assertIn("planner_output_feedback", str(planner.calls[1]["messages"]))

    def test_invalid_until_exhaustion_has_no_free_retry_or_fake_abstention(self):
        planner = Planner([{"role": "assistant", "content": "Still considering."} for _ in range(3)])
        result = EpisodeRunner(Service(), planner).run(Harness(max_steps=3), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("budget_exhausted", None), result.raw)
        self.assertEqual(result.raw["terminal_answer_status"], "invalid")
        self.assertEqual(result.usage["model_calls"], 3)
        self.assertEqual(len([e for e in result.events if e["kind"] == "answer_recovery"]), 2)
        self.assertEqual(len([e for e in result.events if e["kind"] == "terminal"]), 1)
        self.assertEqual(planner.calls[-1]["tool_choice"], "none")

    def test_empty_response_cannot_reuse_answer_cue_from_earlier_tool_turn(self):
        first = call("memory_read", {})
        first["content"] = "The answer is B, but I need to inspect my evidence."
        planner = Planner([first, {"role": "assistant", "content": None}, copy.deepcopy(FINAL)])
        result = EpisodeRunner(Service(), planner).run(Harness(memory=True, max_steps=3), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(result.usage["model_calls"], 3)

    def test_recovery_and_diagnosis_share_the_original_allowance(self):
        planner = Planner([
            {"role": "assistant", "content": "I need another interpretation."},
            call("verify_fresh", {}),
            {"role": "assistant", "content": "Evidence is inconclusive."},
            copy.deepcopy(FINAL),
        ])
        result = EpisodeRunner(Service(), planner).run(Harness(verification=True, max_steps=4), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(result.usage["model_calls"], 4)
        self.assertEqual(result.usage["planner_calls"], 3)
        self.assertEqual(result.usage["verification_calls"], 1)
        self.assertEqual(planner.calls[-1]["tool_choice"], "none")

    def test_explicit_final_or_abstention_still_finishes_immediately(self):
        for content, status, answer in [
            ('{"status":"answered","answer":"A"}', "completed", "A"),
            ('{"answer":"A"}', "completed", "A"),
            ('The answer is A.', "completed", "A"),
            ('{"status":"abstained","answer":null}', "abstained", None),
        ]:
            with self.subTest(content=content):
                planner = Planner([{"role": "assistant", "content": content}])
                result = EpisodeRunner(Service(), planner).run(Harness(max_steps=4), sample("cal"), 0)
                self.assertEqual((result.status, result.answer), (status, answer), result.raw)
                self.assertEqual(result.usage["model_calls"], 1)

    def test_intermediate_invalid_reply_never_calls_configured_evaluator(self):
        class Evaluator:
            def call(self, **kwargs):
                raise AssertionError("intermediate repair must use the planner budget")
        planner = Planner([{"role": "assistant", "content": "Still considering."}, copy.deepcopy(FINAL)])
        with patch("run_eval._llm_extract_evaluation_answer", wraps=__import__("run_eval")._llm_extract_evaluation_answer) as extract:
            result = EpisodeRunner(Service(), planner, extractor=Evaluator()).run(Harness(max_steps=2), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(extract.call_args.kwargs["extractor"], None)


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires pinned Video OS")
class RecoveryWireTests(unittest.TestCase):
    def test_length_truncation_does_not_promote_an_intermediate_answer_cue(self):
        from video_os.agent.planner import OpenAICompatiblePlannerClient
        from video_os.core.budget import BudgetContract
        from video_os.core.dispatch import ProviderRole
        from video_os.providers.client import ProviderSpec, TransportResponse

        class Transport:
            def __init__(self):
                self.calls = []
            def post(self, **kwargs):
                self.calls.append(json.loads(kwargs["body"]))
                first = len(self.calls) == 1
                message = {"role": "assistant", "content": "The answer is B was my early guess, but I still need to"} if first else FINAL
                body = {"id": "unit", "model": "unit", "choices": [{"message": message,
                    "finish_reason": "length" if first else "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
                return TransportResponse(status=200, body=json.dumps(body).encode())

        budget = BudgetContract(c_text_max=None, c_sensor_max=None, b_control=None, b_video=8192,
            b_state=None, b_task=None, f_view=32, p_view=147456, p_call=1572864, k_look=8,
            k_compare=2, f_episode=256, b_video_episode=65536)
        transport = Transport()
        planner = OpenAICompatiblePlannerClient(spec=ProviderSpec(role=ProviderRole.GPT_TEXT, model="unit"),
            api_key="unit-key", budget=budget, transport=transport)
        result = EpisodeRunner(Service(), planner).run(Harness(max_steps=2), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.calls[-1]["tool_choice"], "none")
        self.assertTrue(all("response_format" not in c for c in transport.calls))
        self.assertEqual(result.usage["model_calls"], 2)
        self.assertEqual(next(e for e in result.events if e["kind"] == "answer_recovery")["reason"], "length_truncated")
        self.assertEqual([e for e in result.events if e["kind"] == "planner"][0]["metadata"]["finish_reason"], "length")
