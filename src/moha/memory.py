"""Append-only result and working ledgers; reads return original records."""
from __future__ import annotations
import copy
from .context import tool_context

MEMORY_POLICY = "moha_two_ledgers_v1"


class ObservationMemory:
    def __init__(self):
        self.result_memory = []
        self.working_memory = []

    def add(self, result):
        if result.get("isError"):
            return
        value = tool_context(result, keep_state=False)
        observation = value.get("observation")
        if not isinstance(observation, dict):
            return
        scope = value.get("observation_context", {})
        if "window" not in scope and value.get("view", {}).get("time_range_seconds") is not None:
            scope["window"] = value["view"]["time_range_seconds"]
        self.result_memory.append({"observation": observation, "observation_context": scope})

    def read(self, ledger="both", source_ids=None):
        if ledger not in {"result", "working", "both"}:
            raise ValueError("ledger must be result, working or both")
        records = self.result_memory
        if source_ids is not None:
            if not isinstance(source_ids, list) or any(not isinstance(x, str) for x in source_ids):
                raise ValueError("source_ids must be a list of strings")
            known = {r["observation"].get("observation_id") for r in records}
            if set(source_ids) - known:
                raise ValueError("unknown observation source ID")
            records = [r for r in records if r["observation"].get("observation_id") in source_ids]
        value = {}
        if ledger in {"result", "both"}:
            value["result_memory"] = records
        if ledger in {"working", "both"}:
            value["working_memory"] = self.working_memory
        return copy.deepcopy(value)

    def note(self, text, source_ids=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("working note must be nonempty text")
        self.read("result", source_ids)
        note = {"note_id": len(self.working_memory) + 1, "text": text,
                "source_ids": copy.deepcopy(source_ids or [])}
        self.working_memory.append(note)
        return copy.deepcopy(note)

    def inventory(self):
        return {"result_records": len(self.result_memory), "working_notes": len(self.working_memory),
                "source_ids": list(dict.fromkeys(r["observation"].get("observation_id")
                                    for r in self.result_memory if r["observation"].get("observation_id")))}


def memory_tools():
    source_ids = {"type": "array", "items": {"type": "string"},
                  "description": "Optional observation IDs. Omit to read all sources."}
    return [
        {"type": "function", "function": {"name": "memory_read",
         "description": "Read original result observations or your working notes verbatim, including older observations omitted from conversation history.",
         "parameters": {"type": "object", "properties": {
             "ledger": {"type": "string", "enum": ["result", "working", "both"]},
             "source_ids": source_ids}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "memory_note",
         "description": "Append a working note exactly as written. Notes are your reasoning, not source observations.",
         "parameters": {"type": "object", "properties": {"text": {"type": "string"},
             "source_ids": source_ids}, "required": ["text"], "additionalProperties": False}}},
    ]
