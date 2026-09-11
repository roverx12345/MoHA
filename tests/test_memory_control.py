import copy
import json
import unittest
from moha.control import NoNoveltyController
from moha.memory import ObservationMemory
from moha.models import Harness
from moha.demo import sample
from moha.runtime import EpisodeRunner
from test_memory import observation
from test_runtime import Planner, call, VIDEO_OS_AVAILABLE
from test_verification_gate import Service, observe, final


class NoveltyTests(unittest.TestCase):
    def test_versions_preserve_duplicates_conflicts_and_separate_note_changes(self):
        memory = ObservationMemory()
        initial = memory.versions()
        memory.add(observation(1))
        first = memory.versions()
        self.assertNotEqual(first["result_memory_version"], initial["result_memory_version"])
        self.assertEqual(first["working_memory_version"], initial["working_memory_version"])
        memory.add(observation(1))
        duplicate = memory.versions()
        self.assertNotEqual(duplicate["result_memory_version"], first["result_memory_version"])
        memory.add(observation(1, "A contradictory claim."))
        self.assertNotEqual(memory.versions()["result_memory_version"], duplicate["result_memory_version"])
        before_note = memory.versions()
        memory.note("A hypothesis", ["obs1"])
        self.assertEqual(memory.versions()["result_memory_version"], before_note["result_memory_version"])
        self.assertNotEqual(memory.versions()["working_memory_version"], before_note["working_memory_version"])
        self.assertEqual(len(memory.read("result")["result_memory"]), 3)

    def test_repeated_visible_payload_is_suppressed_then_masked_without_mutation(self):
        memory = ObservationMemory()
        memory.add(observation(1))
        payload = memory.read("result")
        original = copy.deepcopy(payload)
        guard = NoNoveltyController()
        first = guard.read(payload, memory.versions(), [])
        second = guard.read(payload, memory.versions(), [payload])
        third = guard.read(payload, memory.versions(), [payload])
        self.assertEqual(first["result_memory"], original["result_memory"])
        self.assertTrue(second["no_novelty"])
        self.assertNotIn("result_memory", second)
        self.assertNotIn("result_memory", third)
        self.assertTrue(third["memory_read_masked"])
        self.assertEqual(payload, original)
        self.assertEqual(memory.read("result"), original)

    def test_different_source_selection_is_not_falsely_suppressed(self):
        memory = ObservationMemory()
        memory.add(observation(1))
        memory.add(observation(2))
        guard = NoNoveltyController()
        one = memory.read("result", ["obs1"])
        two = memory.read("result", ["obs2"])
        guard.read(one, memory.versions(), [])
        result = guard.read(two, memory.versions(), [one])
        self.assertFalse(result["no_novelty"])
        self.assertEqual(result["result_memory"], two["result_memory"])

    def test_evicted_payload_can_be_restored_without_a_ledger_change(self):
        memory = ObservationMemory()
        memory.add(observation(1))
        payload = memory.read("result")
        guard = NoNoveltyController()
        guard.read(payload, memory.versions(), [])
        guard.read(payload, memory.versions(), [payload])
        guard.read(payload, memory.versions(), [payload])
        self.assertTrue(guard.masked)
        result = guard.read(payload, memory.versions(), [])
        self.assertFalse(guard.masked)
        self.assertFalse(result["no_novelty"])
        self.assertTrue(result["restored_after_eviction"])
        self.assertEqual(result["result_memory"], payload["result_memory"])


@unittest.skipUnless(VIDEO_OS_AVAILABLE, "requires pinned Video OS")
class NativeMemoryControlTests(unittest.TestCase):
    def test_tool_is_masked_after_repeated_reads_and_restored_after_note_update(self):
        planner = Planner([observe(), call("memory_read", {"ledger": "result"}),
            call("memory_read", {"ledger": "result"}), call("memory_read", {"ledger": "result"}),
            call("memory_note", {"text": "A new unverified hypothesis."}),
            call("memory_read", {"ledger": "both"}), final()])
        result = EpisodeRunner(Service(), planner).run(Harness(memory=True, max_steps=8), sample("cal"), 0)
        self.assertEqual((result.status, result.answer), ("completed", "A"), result.raw)
        names = lambda c: {t["function"]["name"] for t in c["tools"]}
        self.assertNotIn("memory_read", names(planner.calls[4]))
        self.assertIn("memory_read", names(planner.calls[5]))
        self.assertEqual(result.usage["verification_calls"], 0)
        reads = [e["result"] for e in result.events if e["kind"] == "tool_result" and e["tool"] == "memory_read"]
        self.assertEqual([r["no_novelty"] for r in reads], [False, True, True, False])
        self.assertNotIn("result_memory", reads[1])
        self.assertEqual(len(result.raw["memory"]["result_memory"]), 1)
        self.assertEqual(len(result.raw["memory"]["working_memory"]), 1)

    def test_actual_bounded_context_eviction_allows_original_evidence_recovery(self):
        planner = Planner([observe(), call("memory_read", {"ledger": "result"}),
            call("search", {"query": "another scene", "top_k": 1}),
            call("memory_read", {"ledger": "result"}), final()])
        result = EpisodeRunner(Service(), planner).run(Harness(memory=True, max_steps=5, history_turns=1), sample("cal"), 0)
        self.assertEqual(result.status, "completed", result.raw)
        reads = [e["result"] for e in result.events if e["kind"] == "tool_result" and e["tool"] == "memory_read"]
        self.assertTrue(reads[-1]["restored_after_eviction"])
        self.assertEqual(reads[-1]["result_memory"], reads[0]["result_memory"])
        self.assertIn("A person jumps.", str(planner.calls[-1]["messages"]))

    def test_new_observation_reenables_read_and_keeps_old_claims(self):
        planner = Planner([observe(0), call("memory_read", {}), call("memory_read", {}),
            call("memory_read", {}), observe(3), call("memory_read", {}), final()])
        result = EpisodeRunner(Service(), planner).run(Harness(memory=True, max_steps=8), sample("cal"), 0)
        self.assertEqual(result.status, "completed", result.raw)
        self.assertNotIn("memory_read", {t["function"]["name"] for t in planner.calls[4]["tools"]})
        self.assertIn("memory_read", {t["function"]["name"] for t in planner.calls[5]["tools"]})
        reads = [e["result"] for e in result.events if e["kind"] == "tool_result" and e["tool"] == "memory_read"]
        self.assertEqual(len(reads[-1]["result_memory"]), 2)
        self.assertEqual(reads[-1]["result_memory"][0], reads[0]["result_memory"][0])
