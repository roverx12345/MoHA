import copy
import json
import unittest
from moha.memory import ObservationMemory
from moha.models import canonical
from moha.records import visible_observations


def observation(index, text="A scoped claim."):
    return {"observation": {"observation_id": f"obs{index}",
                "facts": [{"fact_id": f"fact{index}", "fact": text, "support_time_seconds": [index, index + 1]}],
                "missing": ["The preceding action is not visible in this window."],
                "uncertainties": ["The identity of the person is unclear."],
                "requested_refinement": "Observe the preceding interval."},
            "observer_execution_receipt": {"window": [index, index + 1],
                "goal": {"type": "sequence", "target": "What happened before?"},
                "realized_execution": {"sampled_frames": 2, "fps": 1}}}


class MemoryTests(unittest.TestCase):
    def test_preserves_long_claim_with_scope_and_all_caveats(self):
        source = observation(1, "This detailed action was not confirmed. " * 30)
        memory = ObservationMemory()
        memory.add(source)
        entries, audit = memory.snapshot(token_limit=10000, count=len)
        self.assertEqual(entries[0]["observation"], source["observation"])
        scope = entries[0]["observation_context"]
        self.assertEqual(scope["window"], [1, 2])
        self.assertEqual(scope["goal"], source["observer_execution_receipt"]["goal"])
        self.assertEqual(audit["omitted_observations"], 0)
        self.assertNotIn("observer_execution_receipt", entries[0])

    def test_zero_fact_observation_keeps_negative_results(self):
        source = observation(1)
        source["observation"]["facts"] = []
        memory = ObservationMemory()
        memory.add(source)
        entries, _ = memory.snapshot(token_limit=10000, count=len)
        self.assertEqual(entries[0]["observation"], source["observation"])

    def test_capacity_keeps_early_and_recent_whole_records_in_order(self):
        memory = ObservationMemory()
        for index in range(8):
            memory.add(observation(index))
        all_entries, _ = memory.snapshot(token_limit=100000, count=len)
        limit = len(canonical([all_entries[0], all_entries[1], all_entries[-1]]))
        entries, audit = memory.snapshot(token_limit=limit, count=len)
        self.assertEqual(entries, [all_entries[0], all_entries[1], all_entries[-1]])
        self.assertEqual(audit["omitted_observations"], 5)
        self.assertLessEqual(audit["tokens"], limit)
        self.assertEqual(memory.snapshot(token_limit=100000, count=len)[0], all_entries)

    def test_oversized_observation_is_omitted_whole_without_orphan_ids(self):
        memory = ObservationMemory()
        memory.add(observation(1, "x" * 10000))
        memory.add(observation(2))
        entries, audit = memory.snapshot(token_limit=1000, count=len)
        self.assertEqual([x["observation"]["observation_id"] for x in entries], ["obs2"])
        self.assertNotIn("obs1", canonical(entries))
        self.assertEqual(audit["omitted_observations"], 1)
        self.assertLessEqual(audit["tokens"], 1000)

    def test_dedup_is_by_full_record_never_id_or_fact_text_alone(self):
        source = observation(1)
        memory = ObservationMemory()
        memory.add(source)
        memory.add(copy.deepcopy(source))
        source["observation"]["missing"] = ["A conflicting qualification."]
        memory.add(source)
        source["observer_execution_receipt"]["window"] = [20, 21]
        memory.add(source)
        entries, _ = memory.snapshot(token_limit=10000, count=len)
        self.assertEqual(len(entries), 3)
        self.assertNotEqual(entries[0]["observation"]["missing"], entries[1]["observation"]["missing"])
        self.assertNotEqual(entries[1]["observation_context"], entries[2]["observation_context"])

    def test_input_and_sent_snapshots_cannot_mutate_memory(self):
        source = observation(1)
        original = copy.deepcopy(source["observation"])
        memory = ObservationMemory()
        memory.add(source)
        source["observation"]["facts"].clear()
        first, _ = memory.snapshot(token_limit=10000, count=len)
        first[0]["observation"]["missing"].clear()
        self.assertEqual(memory.snapshot(token_limit=10000, count=len)[0][0]["observation"], original)

    def test_errors_and_audit_only_ids_are_not_evidence(self):
        memory = ObservationMemory()
        memory.add({**observation(1), "isError": True})
        memory.add({"observation": {"observation_id": "empty", "facts": []}})
        memory.add({"player_state": {"observations": [observation(2)["observation"]]}})
        entries, audit = memory.snapshot(token_limit=100, count=len)
        self.assertEqual(entries, [])
        self.assertEqual(audit["available_observations"], 0)

    def test_visibility_includes_memory_and_same_id_conflicts(self):
        first = observation(1)["observation"]
        other = {**first, "uncertainties": ["Contradictory report."]}
        messages = [{"role": "tool", "content": json.dumps({"observation": other})},
                    {"role": "user", "content": json.dumps({"evidence_memory": [
                        {"observation": first}, {"observation": other}]})},
                    {"role": "user", "content": json.dumps({"observation": observation(9)["observation"]})}]
        self.assertEqual(visible_observations(messages), [other, first])
