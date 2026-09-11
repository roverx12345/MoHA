"""Planner-facing projection of recorded envelopes; never a memory store."""
from __future__ import annotations
import copy
import json
from .models import canonical


PLANNER_CONTEXT_POLICY = "moha_planner_context_v4"
STATE_FIELDS = ("player_state", "navigation")
HISTORY_NOTICE = (
    "\n\n[Context note] Some earlier planner/tool turns were omitted. Use the "
    "retained observations and explicit evidence memory, if supplied. Each "
    "observation is scoped to its window and instruction; missing evidence within that "
    "view does not establish absence across the video. Latest navigation state "
    "records the current and previously inspected windows. Newer observations do not "
    "automatically override earlier or conflicting evidence."
)


def fields(value, names):
    return {k: copy.deepcopy(value[k]) for k in names if k in value}


def search_context(value):
    return fields(value, ("query", "keywords", "modality", "bounds", "candidates",
                          "last_search_id", "history", "available", "reason"))


def tool_context(result, *, keep_state=True):
    """Expose a small evidence contract; retain full envelopes only in records."""
    value = fields(result, ("tool", "isError", "status", "reason", "reason_code", "error_type",
                            "error", "message", "recoverable", "candidates", "audit", "audit_status",
                            "result_memory", "working_memory", "working_note", "no_novelty",
                            "restored_after_eviction", "memory_read_masked", "result_records",
                            "working_notes", "source_ids", "ledger"))
    if value.get("tool") in ("video_player_search", "video_player_observe"):
        value["tool"] = value["tool"].removeprefix("video_player_")
    # Parsed audit already contains its diagnosis. Invalid raw output remains
    # explicit, with its invalid status, but provider receipts never reach input.
    if "diagnosis" in result and not isinstance(result.get("audit"), dict):
        value["diagnosis"] = copy.deepcopy(result["diagnosis"])

    # Flat can mirror the same evidence in three envelopes. Deduplicate only
    # identical copies; distinct or conflicting records must remain available.
    for layer in (result, result.get("result", {}), result.get("backend_result", {})):
        if not isinstance(layer, dict):
            continue
        for key in ("observation", "evidence", "asr", "image_analysis", "overview", "search"):
            if key not in layer:
                continue
            item = search_context(layer[key]) if key == "search" and isinstance(layer[key], dict) else copy.deepcopy(layer[key])
            if key not in value:
                value[key] = item
            elif value[key] != item:
                extra = {key: item}
                if extra not in value.setdefault("additional_results", []):
                    value["additional_results"].append(extra)
    if isinstance(result.get("additional_results"), list):
        value.setdefault("additional_results", []).extend(copy.deepcopy(result["additional_results"]))

    scope = fields(result.get("observation_context", {}),
                   ("window", "goal", "instruction", "evidence_type", "reference", "sampling"))
    receipt = result.get("observer_execution_receipt")
    if isinstance(receipt, dict):
        scope.update(fields(receipt, ("window", "goal")))
        scope["sampling"] = fields(receipt.get("realized_execution", {}),
                                   ("fps", "resolution", "sampled_frames", "modalities"))
    goal = scope.pop("goal", None)
    if isinstance(goal, dict):
        scope.update({public: copy.deepcopy(goal[internal])
                      for public, internal in (("instruction", "target"), ("evidence_type", "type"),
                                               ("reference", "reference")) if internal in goal})
    view = result.get("view", {})
    if isinstance(view, dict) and isinstance(value.get("observation"), dict):
        if "window" not in scope and "time_range_seconds" in view:
            scope["window"] = copy.deepcopy(view["time_range_seconds"])
        sampling = scope.setdefault("sampling", {})
        sampling.update(fields(view, ("frame_timestamps_seconds", "modalities")))
    if scope:
        value["observation_context"] = scope

    player = result.get("player_state", result.get("navigation"))
    if isinstance(player, dict):
        search = player.get("search", {})
        if not keep_state and value.get("tool") == "search":
            candidates = search.get("candidates", [])
            latest = next((h for h in reversed(search.get("history", []))
                           if h.get("search_id") == search.get("last_search_id")), {})
            ids = latest.get("candidate_ids")
            value["candidates"] = copy.deepcopy([c for c in candidates if ids is None or c.get("candidate_id") in ids])
        if keep_state:
            navigation = fields(player, ("current_window", "visited_windows"))
            if isinstance(search, dict) and search:
                navigation["search"] = search_context(search)
            if navigation:
                value["navigation"] = navigation
    return value


def planner_messages(messages):
    """Project before the existing token/turn cap; keep raw messages untouched."""
    result = copy.deepcopy(messages)
    tools = {}
    for index, message in enumerate(result):
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            value = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            tools[index] = value
    # Protocol errors may have no state; retain the most recent supplied snapshot.
    latest = max((i for i, value in tools.items() if any(k in value for k in STATE_FIELDS)), default=None)
    for index, value in tools.items():
        result[index]["content"] = canonical(tool_context(value, keep_state=index == latest))
    # Optional prefetched overview is an initial navigation result, not history
    # memory. Its full envelope is also recorded as a tool_result event.
    if len(result) > 1 and result[1].get("role") == "user" and isinstance(result[1].get("content"), str):
        try:
            initial = json.loads(result[1]["content"])
        except json.JSONDecodeError:
            initial = None
        if isinstance(initial, dict) and isinstance(initial.get("initial"), dict):
            # Perception budgets, renderer hints and unavailable low-level tools
            # are host accounting, not instructions for the two-tool planner.
            initial["initial"] = {"media": fields(initial["initial"].get("media", {}),
                                                   ("duration_seconds", "has_audio"))}
            if isinstance(initial.get("overview"), dict):
                initial["overview"] = tool_context(initial["overview"], keep_state=latest is None)
            result[1]["content"] = canonical(initial)
    return result


def bounded_history(messages, *, token_limit, max_turns):
    """Reuse the pinned history cap with a notice that preserves evidence conflicts."""
    from flat.agent.harness import _planner_history_messages
    projected = planner_messages(messages)
    bounded, audit = _planner_history_messages(projected, token_limit=token_limit, max_turns=max_turns)
    if audit["history_compacted"] and isinstance(projected[1].get("content"), str):
        bounded[1]["content"] = projected[1]["content"] + HISTORY_NOTICE
    audit["projection"] = PLANNER_CONTEXT_POLICY
    return bounded, audit
