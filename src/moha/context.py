"""Planner-facing projection of recorded envelopes; never a memory store."""
from __future__ import annotations
import copy
import json
from .models import canonical


PLANNER_CONTEXT_POLICY = "moha_planner_context_v2"
STATE_FIELDS = ("player_state", "state", "budget")
HISTORY_NOTICE = (
    "\n\n[Context note] Some earlier planner/tool turns were omitted. Use the "
    "retained observations and explicit evidence memory, if supplied. Each "
    "observation is scoped to its window and goal; missing evidence within that "
    "view does not establish absence across the video. Latest player/budget state "
    "records current navigation and resources. Newer observations do not "
    "automatically override earlier or conflicting evidence."
)


def tool_context(result, *, keep_state=True):
    """Keep evidence and actionable handles; remove audit-only state history."""
    value = copy.deepcopy(result)
    player = value.get("player_state")
    if isinstance(player, dict):
        # Older search replies still need their returned handles when their
        # repeated state snapshots are removed. Never trim the latest catalog.
        if not keep_state and value.get("tool") == "video_player_search":
            search = player.get("search", {})
            candidates = search.get("candidates", [])
            latest = next((h for h in reversed(search.get("history", []))
                           if h.get("search_id") == search.get("last_search_id")), {})
            ids = latest.get("candidate_ids")
            value["candidates"] = [c for c in candidates if ids is None or c.get("candidate_id") in ids]
        for key in ("observations", "last_observation_id", "harness_version", "interface_version",
                    "state_schema_version", "observer_profile"):
            player.pop(key, None)

    # The dispatcher puts a second, reduced player snapshot inside result.
    # Its other fields may contain evidence and must remain intact.
    nested = value.get("result")
    if isinstance(nested, dict) and isinstance(player, dict):
        nested.pop("player_state", None)
        if not nested:
            value.pop("result")

    receipt = value.pop("observer_execution_receipt", None)
    if isinstance(receipt, dict):
        scope = {k: receipt[k] for k in ("candidate_id", "window", "goal") if k in receipt}
        realized = receipt.get("realized_execution", {})
        scope["sampling"] = {k: realized[k] for k in ("fps", "resolution", "sampled_frames", "modalities")
                             if k in realized}
        value["observation_context"] = scope

    if not keep_state:
        for key in STATE_FIELDS:
            value.pop(key, None)
    else:
        state = value.get("state")
        if isinstance(state, dict):
            if state.get("budget") == value.get("budget"):
                state.pop("budget", None)
            runtime = state.get("state")
            if isinstance(runtime, dict):
                for key in ("evidence_ids", "notebook_ids"):
                    runtime.pop(key, None)
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
        if isinstance(initial, dict) and isinstance(initial.get("overview"), dict):
            initial["overview"] = tool_context(initial["overview"], keep_state=latest is None)
            result[1]["content"] = canonical(initial)
    return result


def bounded_history(messages, *, token_limit, max_turns):
    """Reuse the pinned history cap with a notice that preserves evidence conflicts."""
    from video_os.agent.harness import _planner_history_messages
    projected = planner_messages(messages)
    bounded, audit = _planner_history_messages(projected, token_limit=token_limit, max_turns=max_turns)
    if audit["history_compacted"] and isinstance(projected[1].get("content"), str):
        bounded[1]["content"] = projected[1]["content"] + HISTORY_NOTICE
    audit["projection"] = PLANNER_CONTEXT_POLICY
    return bounded, audit
