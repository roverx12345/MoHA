"""Normalize once at the boundary. All downstream consumers use these events."""
from __future__ import annotations
import math
import json
from .models import Episode, Harness, Sample, digest


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
    return list(found.values())


def usage_from(events: list[dict], session: dict | None) -> dict:
    items = receipts(events)
    ledger = (session or {}).get("budget_ledger") or {}
    usage = {"observer_calls": len(items), "planner_calls": sum(e["kind"] == "planner" for e in events)}
    usage["sensory_looks"] = numeric(ledger.get("look_used"))
    fields = {"sampled_frames": ("frames_used", ("sampled_frames", "frame_count")),
              "video_tokens": ("video_tokens_used", ("video_tokens", "vision_tokens")),
              "audio_seconds": ("audio_seconds_opened", ("audio_seconds",)),
              "video_seconds": ("video_seconds_opened", ("video_seconds",))}
    missing_receipt = any(e["kind"] == "tool_result" and e.get("tool") == "video_player_observe"
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
    return usage


def normalize(sample: Sample, harness: Harness, repeat: int, raw: dict) -> Episode:
    events = raw["events"]
    answer = raw.get("answer")
    if isinstance(answer, dict):
        answer = answer.get("answer")
    if answer is not None and answer not in sample.task["options"]:
        answer = None
    status = raw.get("status", "error")
    return Episode(sample.sample_id, sample.id, harness.id, repeat, list(sample.video_key),
                   sample.expected_answer, answer, status, events,
                   usage_from(events, raw.get("perception_receipt")), raw)


def visible_observations(messages):
    """Record precisely which compact observations reach each planner call."""
    found = {}
    def walk(value):
        if isinstance(value, dict):
            observation = value.get("observation")
            if isinstance(observation, dict) and observation.get("observation_id"):
                found[observation["observation_id"]] = observation
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for message in messages:
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            try:
                walk(json.loads(message["content"]))
            except json.JSONDecodeError:
                continue
    return list(found.values())
