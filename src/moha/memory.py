"""Bounded, verbatim observation memory; no summarization or evidence merging."""
from __future__ import annotations
import copy
from .context import tool_context
from .models import canonical


MEMORY_POLICY = "moha_whole_observations_v1"


class ObservationMemory:
    def __init__(self):
        self._entries = []
        self._seen = set()

    def add(self, result):
        if result.get("isError"):
            return
        value = tool_context(result, keep_state=False)
        observation = value.get("observation")
        if not isinstance(observation, dict) or not any(
                observation.get(k) for k in ("facts", "missing", "uncertainties", "requested_refinement")):
            return
        # Keep the same observation text supplied to the planner, together with
        # its support and goal. IDs alone, state and raw media handles are not memory.
        scope = value.get("observation_context", {})
        if "window" not in scope and value.get("view", {}).get("time_range_seconds") is not None:
            scope["window"] = value["view"]["time_range_seconds"]
        entry = {"observation": observation, "observation_context": scope}
        key = canonical(entry)
        if key not in self._seen:
            self._entries.append(entry)
            self._seen.add(key)

    def snapshot(self, *, token_limit, count):
        """Prefer two early anchors, then recent records; never split a record.

        The acquired log is bounded by episode calls. Only this selected view is
        sent to the planner. Token counting is supplied by the pinned runtime.
        """
        selected = []
        order = list(range(min(2, len(self._entries)))) + list(range(len(self._entries) - 1, 1, -1))
        for index in order:
            proposed = sorted([*selected, index])
            if count(canonical([self._entries[i] for i in proposed])) <= token_limit:
                selected = proposed
        entries = copy.deepcopy([self._entries[i] for i in selected])
        return entries, {"policy": MEMORY_POLICY, "available_observations": len(self._entries),
                         "retained_observations": len(entries),
                         "omitted_observations": len(self._entries) - len(entries),
                         "tokens": count(canonical(entries)), "token_limit": token_limit}
