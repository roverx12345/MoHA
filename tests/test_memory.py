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

    def test_non_evidence_content_is_not_archived_and_no_notes_api_exists(self):
        memory = ObservationMemory()
        memory.add({"working_note": {"text": "Answer D is certain."}})
        memory.add({"working_memory": [{"text": "Another hypothesis."}]})
        self.assertEqual(memory.read(), {"result_memory": []})
        self.assertFalse(hasattr(memory, "note"))
        with self.assertRaises(ValueError): memory.read("working")
        with self.assertRaises(ValueError): memory.read(source_ids=["unknown"])
        with self.assertRaises(ValueError): memory.read(source_ids="obs1")

    def test_additional_conflicting_public_observations_are_archived(self):
        memory = ObservationMemory()
        first = observation(1)
        other = observation(1, "A contradictory claim.")["observation"]
        first["additional_results"] = [{"observation": other}]
        memory.add(first)
        records = memory.read()["result_memory"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["observation"], other)
        self.assertEqual(records[1]["observation_context"], records[0]["observation_context"])

    def test_errors_and_audit_state_are_not_source_observations(self):
        memory = ObservationMemory()
        memory.add({**observation(1), "isError": True})
        memory.add({"player_state": {"observations": [observation(2)["observation"]]}})
        self.assertEqual(memory.read("result")["result_memory"], [])
