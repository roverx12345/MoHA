"""Lossless observation archive and automatic, budgeted evidence restoration."""
from __future__ import annotations
import copy
import json
from .context import bounded_history, tool_context
from .models import canonical, digest

MEMORY_POLICY = "moha_persistent_evidence_injection_v1"
MEMORY_CAPABILITY = {
    "mode": "automatic_context_injection", "max_memory_tokens": 6000,
    "placement": "before_recent_history", "selection": "all_missing_or_none",
    "budget": "shared_history_allowance", "bounded_history": "planner_step_limit",
    "tools": [],
}


def result_records(value):
    """Read public observations and explicit additional results, never notes/state."""
    if not isinstance(value, dict) or value.get("isError"):
        return []
    rows = []
    if isinstance(value.get("observation"), dict):
        rows.append({"observation": copy.deepcopy(value["observation"]),
                     "observation_context": copy.deepcopy(value.get("observation_context", {}))})
    for extra in value.get("additional_results", []):
        if isinstance(extra, dict):
            scoped = {"observation_context": value.get("observation_context", {}), **extra}
            rows.extend(result_records(scoped))
    return rows


def visible_records(messages):
    """Full scoped records in actual model input, including restored evidence."""
    records = []
    for message in messages:
        if message.get("role") not in {"tool", "user"} or not isinstance(message.get("content"), str):
            continue
        try:
            value = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("isError"):
            continue
        if message["role"] == "tool":
            records.extend(result_records(value))
            # Read-only compatibility with immutable historical tool traces.
            if isinstance(value.get("result_memory"), list):
                records.extend(copy.deepcopy(value["result_memory"]))
        elif isinstance(value.get("evidence_memory"), list):
            records.extend(copy.deepcopy(value["evidence_memory"]))
    return records


class ObservationMemory:
    def __init__(self):
        self.result_memory = []

    def add(self, result):
        self.result_memory.extend(result_records(tool_context(result, keep_state=False)))

    def read(self, ledger="result", source_ids=None):
        if ledger != "result":
            raise ValueError("persistent memory stores only original result observations")
        records = self.result_memory
        if source_ids is not None:
            if not isinstance(source_ids, list) or any(not isinstance(x, str) for x in source_ids):
                raise ValueError("source_ids must be a list of strings")
            known = {r["observation"].get("observation_id") for r in records}
            if set(source_ids) - known:
                raise ValueError("unknown observation source ID")
            records = [r for r in records if r["observation"].get("observation_id") in source_ids]
        return {"result_memory": copy.deepcopy(records)}


def persistent_history(messages, memory, *, token_limit, max_turns):
    """Restore every missing record atomically, or preserve the baseline on overflow.

    The archive is untouched. Only exact scoped copies are deduplicated in the
    view. Reducing the recent-history allowance can evict additional evidence;
    iterate until every such record is included, without partial conflict groups
    or a relevance/recency ranking. The pinned newest-turn advisory exception
    remains intact, but never grants a free memory block above the total allowance.
    """
    from flat.core.context import ConservativeTokenCounter
    counter = ConservativeTokenCounter()
    original, original_audit = bounded_history(messages, token_limit=token_limit, max_turns=max_turns)
    records = memory.read()["result_memory"]
    unique = {digest(record): record for record in records}
    cap = min(MEMORY_CAPABILITY["max_memory_tokens"], token_limit)
    audit = {"policy": MEMORY_POLICY, "archive_records": len(records), "unique_records": len(unique),
             "max_memory_tokens": cap, "history_token_limit": token_limit,
             "memory_tokens": 0, "injected_records": 0, "injected_record_hashes": [],
             "capacity_limited": False, "missing_records": 0, "status": "already_visible"}
    bounded, history_audit = original, original_audit
    reserve = 0
    for _ in range(len(unique) + 2):
        visible = {digest(record) for record in visible_records(bounded)}
        missing = {key: record for key, record in unique.items() if key not in visible}
        audit["missing_records"] = len(missing)
        if not missing:
            return original, {**original_audit, "memory": audit}
        message = {"role": "user", "content": canonical({
            "context_type": "persistent_observation_data",
            "evidence_memory": list(missing.values()),
        })}
        tokens = counter.count(canonical([message]))
        audit["requested_memory_tokens"] = tokens
        if tokens > cap:
            audit.update(capacity_limited=True, status="memory_capacity_exceeded")
            break
        if history_audit["history_tokens"] + tokens <= token_limit:
            restored = bounded[:2] + [message] + bounded[2:]
            audit.update(status="restored", memory_tokens=tokens, injected_records=len(missing),
                         injected_record_hashes=list(missing))
            return restored, {**history_audit, "memory": audit,
                "history_tokens": history_audit["history_tokens"] + tokens,
                "recent_history_tokens": history_audit["history_tokens"],
                "message_count": len(restored)}
        if tokens <= reserve:
            audit.update(capacity_limited=True, status="protected_history_exceeds_allowance")
            break
        reserve = tokens
        bounded, history_audit = bounded_history(messages, token_limit=token_limit-reserve, max_turns=max_turns)
    else:
        raise RuntimeError("persistent evidence allocation did not converge")
    audit["missing_records"] = len(set(unique) - {digest(r) for r in visible_records(original)})
    return original, {**original_audit, "memory": audit}
