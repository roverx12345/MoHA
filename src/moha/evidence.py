"""Calibration evidence views. References deduplicate data, not causal claims."""
from __future__ import annotations
import copy
import json
from collections import Counter
from .models import canonical, digest


def pack(value):
    """Losslessly share repeated JSON containers, keeping unique content inline."""
    counts, sizes = Counter(), {}

    def count(item):
        if isinstance(item, (dict, list)):
            key = digest(item)
            counts[key] += 1
            sizes[key] = len(canonical(item).encode())
            for child in item.values() if isinstance(item, dict) else item:
                count(child)
    count(value)
    shared, ids = {}, {}

    def encode(item):
        if not isinstance(item, (dict, list)):
            return item
        key = digest(item)
        reusable = counts[key] > 1 and sizes[key] >= 512
        if reusable and key in ids:
            return {"$ref": ids[key]}
        if isinstance(item, dict):
            encoded = {k: encode(v) for k, v in item.items()}
            # Escape literal reference-shaped data so expansion is unambiguous.
            if set(encoded) in ({"$ref"}, {"$literal"}):
                encoded = {"$literal": encoded}
        else:
            encoded = [encode(v) for v in item]
        if reusable:
            ref = f"shared_{len(ids) + 1}"
            ids[key] = ref
            shared[ref] = encoded
            return {"$ref": ref}
        return encoded
    data = encode(value)
    return {"encoding": "shared_json", "data": data, "shared": shared}


def unpack(payload):
    if payload.get("encoding") != "shared_json":
        return payload
    def expand(item):
        if isinstance(item, list):
            return [expand(v) for v in item]
        if isinstance(item, dict):
            if set(item) == {"$literal"}:
                return {k: expand(v) for k, v in item["$literal"].items()}
            if set(item) == {"$ref"}:
                return expand(payload["shared"][item["$ref"]])
            return {k: expand(v) for k, v in item.items()}
        return item
    return expand(payload["data"])


def messages_view(messages):
    """Decode tool/assistant JSON; keep system and user messages as original text."""
    result = []
    for message in messages:
        value = copy.deepcopy(message)
        content = value.get("content")
        if isinstance(content, str):
            # The text-only provider strips a top-level media key. The original
            # user text can contain benign video metadata (duration, fps, size).
            if value.get("role") in {"system", "user"}:
                value["content_encoding"] = "text"
                result.append(value)
                continue
            try:
                value["content"] = json.loads(content)
                value["content_encoding"] = "decoded_json"
            except json.JSONDecodeError:
                value["content_encoding"] = "text"
        result.append(value)
    return result


def _check(episode, sample, harness):
    if episode.sample_hash != sample.id or episode.harness_id != harness.id:
        raise ValueError("evidence source differs from sample/harness")


def diagnosis_view(episode, sample, harness):
    _check(episode, sample, harness)
    events = []
    for event in episode.events:
        if event["kind"] not in {"tool_call", "tool_result", "terminal", "context", "planner", "error"}:
            continue
        item = {k: copy.deepcopy(v) for k, v in event.items() if k != "metadata"}
        if "messages" in item:
            item["messages"] = messages_view(item["messages"])
        if "message" in item:
            item["message"] = messages_view([item["message"]])[0]
        events.append(item)
    raw = episode.raw
    recoveries = raw.get("perception_receipt", {}).get("observer_output_recoveries", [])
    return pack({"sample_id": sample.sample_id, "task": sample.task,
        "expected_answer": sample.expected_answer, "answer": episode.answer, "status": episode.status,
        "harness": harness.to_dict(), "usage": episode.usage,
        "initial_messages": messages_view(raw.get("messages", [])[:2]),
        "tool_schemas": raw.get("tool_schemas"), "events": events,
        **({"observer_output_recoveries": copy.deepcopy(recoveries)} if recoveries else {}),
        "recording_gaps": {
            "tool_schemas_missing": not bool(raw.get("tool_schemas")),
            "initial_messages_missing": not bool(raw.get("messages")),
            "context_steps_without_messages": [e["step"] for e in episode.events
                if e["kind"] == "context" and "messages" not in e]}})
