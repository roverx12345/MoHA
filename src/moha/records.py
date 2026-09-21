"""Normalize once at the boundary. All downstream consumers use these events."""
from __future__ import annotations
import math
import json
import copy
from .models import Episode, Harness, Sample, digest


def observation_request(receipt):
    """Read current instructions or an immutable historical typed request."""
    if "instruction" in receipt:
        return {k: copy.deepcopy(receipt[k]) for k in ("instruction", "evidence_type", "reference")
                if k in receipt}
    old = receipt.get("goal", {})
    return {public: copy.deepcopy(old[internal]) for public, internal in
            (("instruction", "target"), ("evidence_type", "type"), ("reference", "reference"))
            if internal in old}


def observer_audit(receipt):
    """Keep execution/probe provenance separate from the public tool result."""
    value = copy.deepcopy(receipt)
    value.update(observation_request(value))
    value.pop("goal", None)
    value.pop("budget_after", None)  # The session ledger already records accounting.
    if value.get("candidate_id") in (None, ""):
        value.pop("candidate_id", None)
    if isinstance(value.get("receipt_id"), str):
        value["receipt_id"] = value["receipt_id"].removeprefix("player-")
    value["receipt_version"] = "moha_observer_audit_v1"
    return value


def numeric(value):
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 else None)


def receipts(events: list[dict]) -> list[dict]:
    found = {}

    def walk(value):
        if isinstance(value, dict):
            item = value.get("observer_execution_receipt")
            if isinstance(item, dict):
                key = item.get("receipt_id") or item.get("observation_id") or digest(item)
                # The same call can be mirrored in an envelope and backend result.
                if key in found and found[key] != item:
                    raise ValueError(f"conflicting observer receipt {key}")
                found[key] = item
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for event in events:
        if event["kind"] == "tool_result":
            walk(event.get("result", {}))
            walk(event.get("audit", {}))
    return list(found.values())


def usage_from(events: list[dict], session: dict | None) -> dict:
    items = receipts(events)
    ledger = (session or {}).get("budget_ledger") or {}
    usage = {"observer_calls": len(items), "planner_calls": sum(e["kind"] == "planner" for e in events)}
    usage["verification_calls"] = sum(e["kind"] == "verification_request" for e in events)
    usage["model_calls"] = usage["planner_calls"] + usage["verification_calls"]
    usage["sensory_looks"] = numeric(ledger.get("look_used"))
    fields = {"sampled_frames": ("frames_used", ("sampled_frames", "frame_count")),
              "video_tokens": ("video_tokens_used", ("video_tokens", "vision_tokens")),
              "audio_seconds": ("audio_seconds_opened", ("audio_seconds",)),
              "video_seconds": ("video_seconds_opened", ("video_seconds",))}
    missing_receipt = any(e["kind"] == "tool_result" and e.get("tool") in {"observe", "video_player_observe"}
                          and not e.get("result", {}).get("isError")
                          and not receipts([e]) for e in events)
    if missing_receipt:
        usage["observer_calls"] = None
    for output, (ledger_key, keys) in fields.items():
        value = numeric(ledger.get(ledger_key))
        if value is None:
            values = []
            for item in items:
                maps = [item.get("usage") or {}, item.get("realized_execution") or {}]
                candidates = [numeric(m.get(k)) for m in maps for k in keys]
                values.append(next((x for x in candidates if x is not None), None))
            value = (sum(values) if not missing_receipt and all(v is not None for v in values) else None)
        usage[output] = value
    usage["provider_totals"] = (session or {}).get("provider_totals")
    recoveries = (session or {}).get("observer_output_recoveries", [])
    if recoveries:
        # observer_calls counts logical requests; retries are additionally
        # charged in the sensory ledger and included in provider_totals.
        usage["observer_retry_calls"] = sum(r["additional_calls"] for r in recoveries)
    return usage


def normalize(sample: Sample, harness: Harness, repeat: int, raw: dict) -> Episode:
    events = raw["events"]
    answer = raw.get("answer")
    if isinstance(answer, dict):
        answer = answer.get("answer")
    if answer is not None and answer not in sample.task["options"]:
        answer = None
    status = raw.get("status", "error")
    usage = usage_from(events, raw.get("perception_receipt"))
    usage["planner_calls"] = raw.get("planner_calls_used", usage["planner_calls"])
    usage["verification_calls"] = raw.get("verification_calls_used", usage["verification_calls"])
    usage["model_calls"] = raw.get("model_calls_used", usage["model_calls"])
    return Episode(sample.sample_id, sample.id, harness.id, repeat, list(sample.video_key),
                   sample.expected_answer, answer, status, events,
                   usage, raw)


def visible_observations(messages):
    """Record precisely which compact observations reach each planner call."""
    found = {}
    def walk(value):
        if isinstance(value, dict):
            observation = value.get("observation")
            if isinstance(observation, dict) and observation.get("observation_id"):
                # The same ID can occur with changed or conflicting content.
                found[digest(observation)] = observation
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for message in messages:
        if message.get("role") in {"tool", "user"} and isinstance(message.get("content"), str):
            try:
                value = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if message["role"] == "tool":
                walk(value)
            elif isinstance(value, dict):
                walk(value.get("evidence_memory", []))
                # Verification advice repeats exactly the original evidence supplied
                # to the single audit, including only records visible on that call.
                for key in ("finalization", "verification_advice"):
                    advice = value.get(key)
                    if isinstance(advice, dict):
                        walk(advice.get("observations", []))
    return list(found.values())
