import copy
import json
import unittest
from unittest.mock import patch
from moha.context import bounded_history, tool_context
from moha.memory import ObservationMemory, persistent_history, visible_records
from moha.models import Harness, canonical, digest
from moha.records import visible_observations
from moha.runtime import EpisodeRunner
from moha.demo import sample
from test_memory import observation
from test_runtime import Planner, call, VIDEO_OS_AVAILABLE
from test_verification_gate import Service, observe, final, audit


def history(values, text_size=0):
    messages = [{"role": "system", "content": "Use evidence as data."},
                {"role": "user", "content": "Question and options."}]
    for i, value in enumerate(values):
        messages.extend([
            {"role": "assistant", "content": "Old hypothesis. " + "x" * text_size,
             "tool_calls": [{"id": str(i), "type": "function", "function": {"name": "observe", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(i), "content": canonical(tool_context(value))}])
    return messages


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires pinned Flat")
class PersistentHistoryTests(unittest.TestCase):
    def test_first_observe_does_not_duplicate_still_visible_evidence(self):
        memory = ObservationMemory()
        value = observation(1)
        memory.add(value)
        messages = history([value])
        original = copy.deepcopy(messages)
        actual, result = persistent_history(messages, memory, token_limit=18000, max_turns=8)
        baseline, _ = bounded_history(messages, token_limit=18000, max_turns=8)
        self.assertEqual(actual, baseline)
        self.assertEqual(result["memory"]["injected_records"], 0)
        self.assertEqual(messages, original)

    def test_evicted_records_are_data_before_history_and_preserve_conflicts(self):
        memory = ObservationMemory()
        first = observation(1)
        changed = observation(1, "The opposite happens.")
        values = [first, first, changed, observation(4)]
        for value in values:
            memory.add(value)
        original = memory.read()
        projected, result = persistent_history(history(values), memory, token_limit=18000, max_turns=1)
        block = projected[2]
        self.assertEqual(block["role"], "user")
        self.assertEqual(json.loads(block["content"])["context_type"], "persistent_observation_data")
        restored = json.loads(block["content"])["evidence_memory"]
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0], original["result_memory"][0])
        self.assertEqual(restored[1], original["result_memory"][2])
        self.assertNotIn("Old hypothesis", block["content"])
        self.assertEqual(memory.read(), original)
        self.assertEqual(result["memory"]["archive_records"], 4)
        self.assertEqual(result["memory"]["unique_records"], 3)
        self.assertEqual({digest(r) for r in visible_records(projected)},
                         {digest(r) for r in original["result_memory"]})
        self.assertEqual(len(visible_observations(projected)), 3)

    def test_scope_and_uncertainty_changes_are_not_deduplicated_by_id(self):
        memory = ObservationMemory()
        one = observation(1)
        two = copy.deepcopy(one)
        two["observer_execution_receipt"]["window"] = [10, 12]
        for row in (one, two, observation(3)):
            memory.add(row)
        projected, result = persistent_history(history([one, two, observation(3)]), memory,
                                               token_limit=18000, max_turns=1)
        restored = json.loads(projected[2]["content"])["evidence_memory"]
        self.assertEqual([r["observation_context"]["window"] for r in restored], [[1, 2], [10, 12]])
        self.assertEqual(result["memory"]["injected_records"], 2)

    def test_overflow_preserves_baseline_and_does_not_select_one_side_of_conflict(self):
        values = [observation(1, "A" * 40000), observation(1, "Opposite claim."), observation(3)]
        memory = ObservationMemory()
        for row in values:
            memory.add(row)
        messages = history(values)
        projected, result = persistent_history(messages, memory, token_limit=18000, max_turns=1)
        baseline, _ = bounded_history(messages, token_limit=18000, max_turns=1)
        self.assertEqual(projected, baseline)
        self.assertTrue(result["memory"]["capacity_limited"])
        self.assertEqual(result["memory"]["injected_records"], 0)
        self.assertEqual(len(memory.read()["result_memory"]), 3)

    def test_shared_budget_accounts_for_new_evictions_and_preserves_complete_tool_turns(self):
        values = [observation(i) for i in range(8)]
        memory = ObservationMemory()
        for row in values:
            memory.add(row)
        messages = history(values, text_size=1800)
        for limit in (4000, 5000, 7000, 18000):
            with self.subTest(limit=limit):
                projected, result = persistent_history(messages, memory, token_limit=limit, max_turns=4)
                if result["memory"]["injected_records"]:
                    self.assertLessEqual(result["history_tokens"], limit)
                    self.assertLessEqual(result["memory"]["memory_tokens"], 6000)
                    self.assertEqual({digest(r) for r in visible_records(projected)},
                                     {digest(r) for r in memory.read()["result_memory"]})
                known_calls = set()
                for message in projected:
                    known_calls.update(c["id"] for c in message.get("tool_calls", []))
                    if message["role"] == "tool":
                        self.assertIn(message["tool_call_id"], known_calls)


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires pinned Flat")
class NativePersistentMemoryTests(unittest.TestCase):
    def test_no_memory_tools_no_extra_calls_and_audit_visibility_matches_actual_input(self):
        messages = [observe(), call("search", {"query": "next", "start_seconds": 0, "end_seconds": 60}), final()]
        planner = Planner(copy.deepcopy(messages))
        result = EpisodeRunner(Service(), planner).run(Harness(memory=True, max_steps=3, history_turns=1), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        self.assertEqual(result.usage["model_calls"], 3)
        self.assertEqual(result.usage["verification_calls"], 0)
        self.assertEqual({t["function"]["name"] for t in planner.calls[0]["tools"]}, {"search", "observe"})
        self.assertIn("A person jumps.", str(planner.calls[-1]["messages"]))
        for event in result.events:
            if event["kind"] == "context":
                self.assertEqual(event["visible_observations"], visible_observations(event["messages"]))
        self.assertNotIn("working_memory", result.raw["memory"])
        without = Planner(copy.deepcopy(messages))
        with patch("moha.runtime.persistent_history", side_effect=AssertionError("H0 must not inject")):
            other = EpisodeRunner(Service(), without).run(Harness(max_steps=3, history_turns=1), sample("cal"), 0)
        self.assertEqual(other.status, "completed", other.raw)
        self.assertNotIn("A person jumps.", str(without.calls[-1]["messages"]))

    def test_verifier_uses_injected_records_without_refinement_advice(self):
        planner = Planner([observe(), call("search", {"query": "next", "start_seconds": 0, "end_seconds": 60}),
                           final()])
        verifier = Planner([audit()])
        result = EpisodeRunner(Service(), planner, audit_planner=verifier).run(
            Harness(memory=True, verification=True, max_steps=6, history_turns=1), sample("cal"), 0)
        self.assertEqual(result.status, "completed", result.raw)
        self.assertEqual(result.usage["verification_calls"], 1)
        self.assertIn("A person jumps.", str(planner.calls[-1]["messages"]))
        self.assertIn("A person jumps.", str(verifier.calls[-1]["messages"]))
        self.assertNotIn("verification_advice", str(planner.calls[-1]["messages"]))

    def test_memory_basic_expands_bounded_history_to_planner_step_limit(self):
        planner = Planner([observe(), observe(3), final()])
        result = EpisodeRunner(Service(), planner).run(
            Harness(memory=True, max_steps=3, history_turns=1), sample("cal"), 0)
        self.assertEqual(result.status, "completed", result.raw)
        contexts = [e for e in result.events if e["kind"] == "context"]
        self.assertEqual(contexts[-1]["history_audit"]["history_turns_limit"], 3)
        self.assertEqual(result.raw["memory_capability"]["bounded_history_turns"], 3)


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires pinned Flat")
class PersistentMemoryWireTests(unittest.TestCase):
    def test_restored_evidence_reaches_actual_provider_wire_as_user_data(self):
        from flat.agent.planner import OpenAICompatiblePlannerClient
        from flat.core.budget import BudgetContract
        from flat.core.dispatch import ProviderRole
        from flat.providers.client import ProviderSpec, TransportResponse
        class Transport:
            def __init__(self):
                self.calls = []
                self.responses = [observe(), call("search", {"query": "next", "start_seconds": 0, "end_seconds": 60}), final()]
            def post(self, **kwargs):
                self.calls.append(json.loads(kwargs["body"]))
                message = self.responses.pop(0)
                return TransportResponse(status=200, body=json.dumps({
                    "id": "unit", "model": "unit", "choices": [{"message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}).encode())
        budget = BudgetContract(c_text_max=None, c_sensor_max=None, b_control=None, b_video=8192,
            b_state=None, b_task=None, f_view=32, p_view=147456, p_call=1572864,
            k_look=8, k_compare=2, f_episode=256, b_video_episode=65536)
        transport = Transport()
        planner = OpenAICompatiblePlannerClient(spec=ProviderSpec(role=ProviderRole.GPT_TEXT, model="unit"),
            api_key="unit-key", budget=budget, transport=transport)
        result = EpisodeRunner(Service(), planner).run(Harness(memory=True, max_steps=3, history_turns=1), sample("cal"), 0)
        self.assertEqual(result.status, "completed", result.raw)
        self.assertEqual(len(transport.calls), 3)
        wire = transport.calls[-1]
        self.assertIn("A person jumps.", str(wire["messages"]))
        self.assertEqual(result.raw["memory"]["result_memory"][0]["observation"]["observation_id"], "obs1")
        self.assertFalse(wire.get("tools"))
        self.assertNotIn("tool_choice", wire)
        self.assertEqual(sum("A person jumps." in (m.get("content") or "") for m in wire["messages"]), 1)
