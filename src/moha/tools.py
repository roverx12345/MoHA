"""Planner-selected time windows over the pinned Video OS observer execution."""
from __future__ import annotations
import math
from video_os.agent.player import VideoPlayerRegistry, PlayerProtocolError
from video_os.agent.observer_registry import ObserverGoal


PLANNER_TOOL_POLICY = "moha_semantic_windows_v1"


class WindowPlayerRegistry(VideoPlayerRegistry):
    """Keep search and typed observe; the planner owns the temporal support."""

    def schemas(self):
        schemas = super().schemas()
        for item in schemas:
            function = item["function"]
            if function["name"] == "video_player_search":
                function["description"] = (
                    "Search the video for candidate moments. Returned source-time windows "
                    "are navigation hints, not evidence or limits on observation. Use their "
                    "timestamps to choose a window for video_player_observe."
                )
            elif function["name"] == "video_player_observe":
                function["description"] = (
                    "Observe a source-time window for one typed evidence goal. Choose its "
                    "start and end directly; search is optional and its candidate windows "
                    "may be expanded to include preceding or following events. The harness "
                    "controls frame rate, resolution, sampling and observer routing. Require "
                    "0 <= start_seconds < end_seconds <= video duration."
                )
                parameters = function["parameters"]
                goal = parameters["properties"]["goal"]
                goal["description"] = "Typed evidence demand for the selected source-time window."
                parameters["properties"] = {
                    "start_seconds": {"type": "number", "minimum": 0,
                                      "description": "Window start in seconds from the video start."},
                    "end_seconds": {"type": "number", "minimum": 0,
                                    "description": "Window end in source seconds, at most the video duration."},
                    "goal": goal,
                }
                parameters["required"] = ["start_seconds", "end_seconds", "goal"]
        return schemas

    def snapshot(self):
        return {**super().snapshot(), "interface_version": PLANNER_TOOL_POLICY}

    def _envelope(self, action, *, backend_tool=None, backend_result=None):
        # The pinned Player helper stringifies its legacy candidate argument.
        # Clear that unused field before it is copied into either receipt mirror.
        if action == "video_player_observe" and backend_result is not None:
            receipt = backend_result.get("observer_execution_receipt")
            if isinstance(receipt, dict):
                receipt["candidate_id"] = None
        return super()._envelope(action, backend_tool=backend_tool, backend_result=backend_result)

    def invoke(self, name, arguments, *, session_id=None):
        if name == "video_player_search":
            return super().invoke(name, arguments, session_id=session_id)
        if name != "video_player_observe":
            raise PlayerProtocolError(f"unknown planner tool {name!r}")
        if session_id is not None and session_id != self.session_id:
            raise PermissionError("player action session does not match the active episode")
        if set(arguments) != {"start_seconds", "end_seconds", "goal"}:
            raise PlayerProtocolError("observe accepts exactly start_seconds, end_seconds and goal")
        start, end = arguments["start_seconds"], arguments["end_seconds"]
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
               for t in (start, end)):
            raise PlayerProtocolError("observation timestamps must be finite numbers")
        if not 0 <= start < end <= self.state.duration_seconds:
            raise PlayerProtocolError(
                f"require 0 <= start_seconds < end_seconds <= {self.state.duration_seconds}"
            )
        # Validate the semantic request before changing state or consuming media.
        try:
            goal = ObserverGoal.from_mapping(arguments["goal"], allow_coverage=False,
                                             require_relation_reference=False)
        except (TypeError, ValueError) as exc:
            raise PlayerProtocolError(str(exc)) from exc
        # Player's navigation setter recentres, rounds and imposes a 0.5 s floor.
        # Direct observation preserves the exact valid support chosen by the planner.
        self.state.window_start_seconds, self.state.window_end_seconds = float(start), float(end)
        self.state.mode, self.state.active_candidate_id = "local", None
        window = [float(start), float(end)]
        if not self.state.visited_windows or self.state.visited_windows[-1] != window:
            self.state.visited_windows.append(window)
        self.state.note_non_search_progress()
        self.state.set_focus(None, modality="visual")
        self.state.record_action("observe")
        try:
            return self._observe_initial_omni(name, "", goal.to_dict(include_coverage=False))
        except Exception as exc:
            receipt = getattr(exc, "observer_execution_receipt", None)
            if isinstance(receipt, dict):
                receipt["candidate_id"] = None
                # The pinned player numbers successful observations only. Give
                # failed requests their own IDs so the next success cannot collide.
                receipt["receipt_id"] += f"-failed-{self.state.action_count}"
            raise
