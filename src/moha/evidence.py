"""Calibration evidence views; Judge input filtering only deletes fields."""
from __future__ import annotations
import copy
import json


# These are execution/accounting fields, never observation text. Keep the full
# originals in episode/probe artifacts. No summaries, references or value rewrites.
JUDGE_OMIT_FIELDS = frozenset({
    "usage", "provider_totals", "input_token_accounting", "sensory_budget",
    "budget_after", "allocation", "packing", "execution_profile", "presentation",
    "planner_geometry", "experiment_render", "observer_experiment", "interface_metrics",
    "source_sha256", "render_config_sha256", "renderer_version", "session_id",
    "receipt_id", "receipt_version", "raw_output_sha256",
    "sampled_frames", "frame_count", "base_requested_frames", "requested_frames",
    "target_frames", "realized_frames", "frames_used", "frame_episode_advisory_limit",
    "frames_over_episode_advisory", "decoded_pixels_used",
    "video_tokens", "video_tokens_used", "video_token_episode_advisory_limit",
    "video_tokens_over_episode_advisory", "history_tokens", "history_token_limit",
    "full_message_count", "message_count", "projection", "preflight",
})


def filter_judge_input(value):
    """Return the same tree with audit fields and repeated context dialogue removed."""
    if isinstance(value, list):
        return [filter_judge_input(item) for item in value]
    if not isinstance(value, dict):
        return value
    # Arbitrary model/user text and semantic evidence may use the same field names.
    if value.get("role") in ("system", "user", "assistant"):
        return copy.deepcopy(value)
    protected = {"task", "arguments", "goal", "instruction", "reference", "observation",
                 "visible_observations", "facts", "uncertainties", "missing",
                 "requested_refinement", "tool_schemas", "tools", "harness", "available"}
    omitted = JUDGE_OMIT_FIELDS
    # Use the existing per-call visibility record verbatim. Older incomplete
    # records without that field must keep their messages as visibility evidence.
    if value.get("kind") == "context" and "visible_observations" in value:
        omitted = omitted | {"messages"}
    return {key: copy.deepcopy(item) if key in protected else filter_judge_input(item)
            for key, item in value.items() if key not in omitted}


def unpack(payload):
    """Read historical shared-json inputs; new requests are ordinary JSON."""
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
    return {"sample_id": sample.sample_id, "task": sample.task,
        "expected_answer": sample.expected_answer, "answer": episode.answer, "status": episode.status,
        "harness": harness.to_dict(), "usage": episode.usage,
        "initial_messages": messages_view(raw.get("messages", [])[:2]),
        "tool_schemas": raw.get("tool_schemas"), "events": events,
        **({"observer_output_recoveries": copy.deepcopy(recoveries)} if recoveries else {}),
        "recording_gaps": {
            "tool_schemas_missing": not bool(raw.get("tool_schemas")),
            "initial_messages_missing": not bool(raw.get("messages")),
            "context_steps_without_messages": [e["step"] for e in episode.events
                if e["kind"] == "context" and "messages" not in e]}}
