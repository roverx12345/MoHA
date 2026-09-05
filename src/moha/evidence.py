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
    return pack({"sample_id": sample.sample_id, "task": sample.task,
        "expected_answer": sample.expected_answer, "answer": episode.answer, "status": episode.status,
        "harness": harness.to_dict(), "usage": episode.usage,
        "initial_messages": messages_view(raw.get("messages", [])[:2]),
        "tool_schemas": raw.get("tool_schemas"), "events": events,
        "recording_gaps": {
            "tool_schemas_missing": not bool(raw.get("tool_schemas")),
            "initial_messages_missing": not bool(raw.get("messages")),
            "context_steps_without_messages": [e["step"] for e in episode.events
                if e["kind"] == "context" and "messages" not in e]}})


def tool_evidence(result):
    """Keep claims and execution evidence; omit repeated backend/state wrappers."""
    backend = result.get("backend_result") or {}
    fields = ("isError", "error_type", "message", "observation", "observer_execution_receipt", "overview")
    value = {k: copy.deepcopy(result.get(k, backend.get(k))) for k in fields
             if k in result or k in backend}
    search = backend.get("search") or result.get("search")
    if search:
        value["search"] = copy.deepcopy(search)
    state = result.get("player_state") or {}
    if state.get("search", {}).get("candidates"):
        value["available_candidates"] = copy.deepcopy(state["search"]["candidates"])
    for key in ("time_mapping", "presentation"):
        if key in backend:
            value[key] = copy.deepcopy(backend[key])
    view = backend.get("view") or result.get("view") or {}
    value["view"] = {k: copy.deepcopy(view[k]) for k in (
        "view_id", "time_range_seconds", "frame_timestamps_seconds", "resolution",
        "spatial_region_normalized", "modalities") if k in view}
    return value


def selection_trace(episode, sample, harness, diagnosis):
    _check(episode, sample, harness)
    cited = sorted(set(diagnosis.get("evidence_steps", [])))
    context_steps = sorted(e["step"] for e in episode.events if e["kind"] == "context")
    # Keep all actions/claims below. Bound only the extra full context excerpts.
    selected = {cited[0], cited[0] + 1} if cited else set()
    if context_steps:
        selected.add(context_steps[-1])
    timeline, contexts = [], []
    for e in episode.events:
        row = {k: copy.deepcopy(e[k]) for k in ("index", "kind", "step", "tool", "call_id") if k in e}
        if e["kind"] == "tool_call":
            row["arguments"] = copy.deepcopy(e.get("arguments"))
        elif e["kind"] == "tool_result":
            row["result"] = tool_evidence(e.get("result", {}))
        elif e["kind"] == "context":
            row["context"] = copy.deepcopy(e.get("context", {}))
            row["history_audit"] = copy.deepcopy(e.get("history_audit", {}))
            row["visible_observation_ids"] = [o["observation_id"] for o in e.get("visible_observations", [])]
            if e["step"] in selected:
                contexts.append({"step": e["step"], "messages_recorded": "messages" in e,
                                 "messages": messages_view(e.get("messages", []))})
        elif e["kind"] in {"planner", "terminal"}:
            message = e.get("message", {})
            # Actions are already listed; retain all returned public text.
            if message.get("content"):
                row["message"] = messages_view([message])[0]
            elif e["kind"] != "terminal":
                continue
        else:
            continue
        timeline.append(row)
    resolution = diagnosis.get("observer_resolution", {})
    probes = [{"preset": p.get("preset"), "status": p.get("status"),
               "result": tool_evidence(p.get("result", {})), "error_type": p.get("error_type")}
              for p in resolution.get("probes", [])]
    verdicts = [{k: v[k] for k in ("preset", "status", "baseline_supports_goal",
                    "alternative_supports_goal", "reason") if k in v}
                for v in resolution.get("verdicts", [])]
    return {"sample_id": sample.sample_id, "sample_hash": sample.id, "episode_hash": digest(episode.to_dict()),
        "task": sample.task, "expected_answer": sample.expected_answer,
        "answer": episode.answer, "status": episode.status, "usage": episode.usage,
        "tool_schemas": episode.raw.get("tool_schemas"),
        "cited_steps": cited, "timeline": timeline, "context_excerpts": contexts,
        "context_steps_omitted": [s for s in context_steps if s not in selected],
        "all_tool_actions_and_observation_claims_retained": True,
        "observer_probes": probes, "observer_probe_verdicts": verdicts}


def representative_traces(diagnoses):
    """One actual calibration example per failure family/capability, deterministically."""
    groups = {}
    for d in diagnoses:
        if d.get("status") == "valid" and d.get("trace_evidence"):
            family = ("observer_execution" if d.get("observer_resolution", {}).get("status") == "execution_rescue"
                      else d["failure"])
            groups.setdefault((family, d.get("failed_capability")), []).append(d["trace_evidence"])
    examples = []
    for (failure, capability), traces in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        ordered = sorted(traces, key=lambda t: (len(t["timeline"]), t["sample_id"]))
        examples.append({"failure": failure, "failed_capability": capability, "family_sample_count": len(traces),
                         "trace": ordered[(len(ordered) - 1) // 2]})
    return {"selection_rule": "one median-length recorded trace per failure family/capability; not a correctness ranking",
            "examples": examples}
