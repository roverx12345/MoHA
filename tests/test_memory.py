import copy
import unittest
from moha.memory import ObservationMemory


def observation(index, text="A scoped claim."):
    return {"observation": {"observation_id": f"obs{index}",
                "facts": [{"fact": text, "support_time_seconds": [index, index + 1]}],
                "missing": ["The preceding action is not visible."],
                "uncertainties": ["The identity is unclear."],
                "requested_refinement": "Observe the preceding interval."},
            "observer_execution_receipt": {"window": [index, index + 1],
                "goal": {"type": "sequence", "target": "What happened before?"}}}


class MemoryTests(unittest.TestCase):
    def test_long_observation_scope_and_caveats_roundtrip_without_mutation(self):
        source = observation(1, "Long uncertain claim. " * 1000)
        original = copy.deepcopy(source)
        memory = ObservationMemory()
        memory.add(source)
        source["observation"]["facts"].clear()
        entries = memory.read("result")["result_memory"]
        self.assertEqual(entries[0]["observation"], original["observation"])
        self.assertEqual(entries[0]["observation_context"]["window"], [1, 2])
        entries[0]["observation"]["missing"].clear()
        self.assertEqual(memory.read("result")["result_memory"][0]["observation"], original["observation"])

    def test_duplicates_same_id_conflicts_and_empty_results_are_preserved(self):
        memory = ObservationMemory()
        first = observation(1)
        memory.add(first)
        memory.add(first)
        other = copy.deepcopy(first)
        other["observation"]["uncertainties"] = ["Contradictory observation."]
        memory.add(other)
        memory.add({"observation": {"observation_id": "empty", "facts": []}})
        self.assertEqual(len(memory.read("result")["result_memory"]), 4)
        selected = memory.read("result", ["obs1"])["result_memory"]
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[-1]["observation"], other["observation"])

    def test_working_notes_stay_separate_and_read_exactly(self):
        memory = ObservationMemory()
        memory.add(observation(1))
        before = memory.read("result")
        note = "  hypothesis\nThis is not an observation.  "
        result = memory.note(note, ["obs1"])
        result["text"] = "changed externally"
        self.assertEqual(memory.read("working")["working_memory"][0]["text"], note)
        self.assertEqual(memory.read("result"), before)
        self.assertEqual(memory.inventory()["working_notes"], 1)

    def test_unknown_reference_and_empty_note_do_not_modify_ledger(self):
        memory = ObservationMemory()
        with self.assertRaises(ValueError): memory.note("note", ["unknown"])
        with self.assertRaises(ValueError): memory.note(" ")
        with self.assertRaises(ValueError): memory.read(source_ids="obs1")
        self.assertEqual(memory.read(), {"result_memory": [], "working_memory": []})

    def test_errors_and_audit_state_are_not_source_observations(self):
        memory = ObservationMemory()
        memory.add({**observation(1), "isError": True})
        memory.add({"player_state": {"observations": [observation(2)["observation"]]}})
        self.assertEqual(memory.read("result")["result_memory"], [])
