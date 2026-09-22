"""Planner-selected time windows over the pinned Flat observer execution."""
from __future__ import annotations
import math
import copy
from flat.agent.player import VideoPlayerRegistry, PlayerProtocolError, normalize_initial_omni_search_query
from flat.agent.observer_registry import ObserverGoal
from .models import Harness
from .execution import FRAME_CAP
from .observer import PolicyExecution, PolicyObserverRegistry
from .records import observer_audit


PLANNER_TOOL_POLICY = "moha_scoped_search_instructions_v5"
MIN_OBSERVE_WINDOW_SECONDS = 0.5


class WindowPlayerRegistry(VideoPlayerRegistry):
    """Keep search and instructed observe; the planner owns temporal support."""

    def __init__(self, backend, *, harness=None, **kwargs):
        self.moha_harness = harness or Harness()
        registry = PolicyObserverRegistry()
        # VideoToolRegistry exposes the existing service used by specialist routes.
        registry.register("omni", backend.video_os)
        super().__init__(backend, **kwargs)
        self._observer_registry = registry

    def _execution_for_goal(self, goal):
        return PolicyExecution(policy=self.moha_harness.execution_for_goal(goal.type),
            modalities=self._fixed_observer_execution.modalities,
            prompt_profile=self._fixed_observer_execution.prompt_profile)

    def observer_contract(self):
        contract = super().observer_contract()
        if contract is not None:
            contract["execution"] = PolicyExecution(policy=self.moha_harness.execution_for_goal("default"),
                modalities=self._fixed_observer_execution.modalities,
                prompt_profile=self._fixed_observer_execution.prompt_profile).to_dict()
            contract["execution_policy"] = self.moha_harness.to_dict()["execution"]
        return contract

    def schemas(self):
        schemas = super().schemas()
        default_policy = self.moha_harness.execution_for_goal("default")
        max_window = default_policy.max_window_seconds
        if max_window is None:
            window_limit = (
                "The active profile uses a fixed frame count, so there is no "
                "rate-derived maximum window; the host still enforces the media "
                "and episode budgets."
            )
        else:
            window_limit = (
                f"The active profile samples at {default_policy.sampling_rate:g} FPS "
                f"with a hard {FRAME_CAP}-frame cap, so one observe window may span "
                f"at most {max_window:g} seconds. If you request a longer interval, "
                "the host returns only the leading allowed source-time interval and "
                "explicitly reports the returned window; use another observe for "
                "the remainder."
            )
        for item in schemas:
            function = item["function"]
            if function["name"] == "video_player_search":
                function["name"] = "search"
                function["description"] = (
                    "Search only the explicit source-time range for candidate moments. "
                    "For a global search, set start_seconds=0 and end_seconds to the video duration. "
                    "An empty result never widens the range automatically. Returned windows "
                    "are navigation hints, not evidence or limits on observation. Use their "
                    "timestamps to choose a window for observe."
                )
                parameters = function["parameters"]
                parameters["properties"].update({
                    "start_seconds": {"type": "number", "minimum": 0,
                        "description": "Search range start in source-video seconds."},
                    "end_seconds": {"type": "number", "minimum": 0,
                        "description": "Search range end in source-video seconds, at most the video duration."},
                })
                parameters["required"] = ["query", "start_seconds", "end_seconds"]
            elif function["name"] == "video_player_observe":
                function["name"] = "observe"
                function["description"] = (
                    "Give the observer a concrete instruction or question about a source-time window. Choose its "
                    "start and end directly; search is optional and its candidate windows "
                    "may be expanded to include preceding or following events. The harness "
                    "controls frame rate, resolution, sampling and observer routing. Require "
                    "0 <= start_seconds < end_seconds <= video duration. " + window_limit
                )
                parameters = function["parameters"]
                goal = parameters["properties"]["goal"]
                parameters["properties"] = {
                    "start_seconds": {"type": "number", "minimum": 0,
                                      "description": "Window start in seconds from the video start."},
                    "end_seconds": {"type": "number", "minimum": 0,
                                    "description": (
                                        "Window end in source seconds, at most the video duration. "
                                        "The observation window must be at least 0.5 seconds long. "
                                        + window_limit
                                    )},
                    "instruction": {**goal["properties"]["target"], "description":
                        "A direct instruction or question for the observer: specify what to inspect "
                        "and which visible or audible facts to report. Request timing or order when "
                        "relevant, and uncertainty when evidence is unclear. Do not ask it to infer "
                        "unobservable intentions or choose the overall task answer."},
                    "evidence_type": {**goal["properties"]["type"], "description":
                        "Kind of evidence requested; the harness uses this for observer routing."},
                    "reference": {**goal["properties"]["reference"], "description":
                        "Optional comparison reference, valid only for evidence_type='relation'."},
                }
                parameters["required"] = ["start_seconds", "end_seconds", "instruction", "evidence_type"]
        return schemas

    def snapshot(self):
        return {**super().snapshot(), "interface_version": PLANNER_TOOL_POLICY}

    def _envelope(self, action, *, backend_tool=None, backend_result=None):
        # The pinned Player helper stringifies its legacy candidate argument.
        # Clear that unused field before it is copied into either receipt mirror.
        if action == "observe" and backend_result is not None:
            receipt = backend_result.get("observer_execution_receipt")
            if isinstance(receipt, dict):
                backend_result["observer_execution_receipt"] = observer_audit(receipt)
        public_action = "search" if action == "video_player_search" else action
        return super()._envelope(public_action, backend_tool=backend_tool, backend_result=backend_result)

    def invoke(self, name, arguments, *, session_id=None):
        if session_id is not None and session_id != self.session_id:
            raise PermissionError("tool request session does not match the active episode")
        if name == "search":
            return self._search(arguments)
        if name != "observe":
            raise PlayerProtocolError(f"unknown planner tool {name!r}")
        required = {"start_seconds", "end_seconds", "instruction", "evidence_type"}
        if not required <= set(arguments) or set(arguments) - required - {"reference"}:
            raise PlayerProtocolError("observe requires start_seconds, end_seconds, instruction and "
                                      "evidence_type; only reference is optional")
        start, end = self._window(arguments)
        if end - start + 1e-9 < MIN_OBSERVE_WINDOW_SECONDS:
            raise PlayerProtocolError(
                "observe window must be at least "
                f"{MIN_OBSERVE_WINDOW_SECONDS:g} seconds"
            )
        # Validate the semantic request before changing state or consuming media.
        try:
            # Adapt the public instruction to Flat's internal routing type.
            internal = {"target": arguments["instruction"], "type": arguments["evidence_type"]}
            if "reference" in arguments:
                internal["reference"] = arguments["reference"]
            goal = ObserverGoal.from_mapping(internal, allow_coverage=False,
                                             require_relation_reference=False)
        except (TypeError, ValueError) as exc:
            message = str(exc).replace("goal.target", "instruction").replace("goal.reference", "reference")
            message = message.replace("observer goal type", "evidence_type").replace("relation goals", "relation evidence")
            raise PlayerProtocolError(message) from exc
        execution = self._execution_for_goal(goal)
        max_window = execution.policy.max_window_seconds
        returned_end = end
        adjustment = None
        if max_window is not None and end - start > max_window + 1e-9:
            returned_end = min(end, start + max_window)
            adjustment = {
                "policy": "rate_frame_cap_window_v1",
                "requested_window": [start, end],
                "returned_window": [start, returned_end],
                "max_window_seconds": max_window,
                "target_fps": execution.policy.sampling_rate,
                "frame_cap": FRAME_CAP,
                "message": (
                    f"The requested observe window [{start:g}, {end:g}] seconds "
                    f"exceeds the active {execution.policy.sampling_rate:g} FPS / "
                    f"{FRAME_CAP}-frame limit. Only [{start:g}, {returned_end:g}] "
                    "seconds was observed; request another window for the remainder."
                ),
            }
        # Preserve the effective source support. Overlong requests are clipped
        # to the leading rate/cap-compatible interval and reported explicitly.
        self.state.window_start_seconds, self.state.window_end_seconds = float(start), float(returned_end)
        self.state.mode, self.state.active_candidate_id = "local", None
        window = [float(start), float(returned_end)]
        if not self.state.visited_windows or self.state.visited_windows[-1] != window:
            self.state.visited_windows.append(window)
        self.state.note_non_search_progress()
        self.state.set_focus(None, modality="visual")
        self.state.record_action("observe")
        try:
            result = self._observe_initial_omni(name, "", goal.to_dict(include_coverage=False))
            if adjustment is not None:
                result["window_adjustment"] = copy.deepcopy(adjustment)
                result["message"] = adjustment["message"]
                receipt = result.get("observer_execution_receipt")
                if isinstance(receipt, dict):
                    receipt["requested_window"] = copy.deepcopy(adjustment["requested_window"])
                    receipt["window_adjustment"] = copy.deepcopy(adjustment)
                result["player_state"] = self.snapshot()
            return result
        except Exception as exc:
            receipt = getattr(exc, "observer_execution_receipt", None)
            if isinstance(receipt, dict):
                receipt.pop("candidate_id", None)
                receipt["receipt_id"] += f"-failed-{self.state.action_count}"
            raise

    def _window(self, arguments):
        start, end = arguments["start_seconds"], arguments["end_seconds"]
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
               for t in (start, end)):
            raise PlayerProtocolError("timestamps must be finite numbers")
        if not 0 <= start < end <= self.state.duration_seconds:
            raise PlayerProtocolError(
                f"require 0 <= start_seconds < end_seconds <= {self.state.duration_seconds}"
            )
        return float(start), float(end)

    def _search(self, arguments):
        required = {"query", "start_seconds", "end_seconds"}
        if not required <= set(arguments) or set(arguments) - required - {"top_k"}:
            raise PlayerProtocolError("search requires query, start_seconds and end_seconds; only top_k is optional")
        start, end = self._window(arguments)
        query = normalize_initial_omni_search_query(arguments["query"])
        top_k = arguments.get("top_k", 3)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k not in {1, 2, 3, 5}:
            raise PlayerProtocolError("top_k must be one of 1, 2, 3, or 5")
        bounds = {"start_seconds": start, "end_seconds": end}
        if self.state.retrieval_guard_active:
            result = self._search_unavailable_result("search")
            result["search"].update(bounds=bounds, query=query, available=False)
            return result
        backend_result = self.backend.invoke_natural_language_search(query, top_k=top_k,
            start_seconds=start, end_seconds=end, session_id=self.session_id)
        result = self._attach_recorded_search("search", backend_tool="video_search",
            backend_result=backend_result, query=query, top_k=top_k)
        result["search"]["bounds"] = bounds
        if not result["search"]["candidates"]:
            result["search"]["reason"] = "no_candidates_in_requested_range"
        if not backend_result.get("isError"):
            self.state.search_history[-1]["bounds"] = copy.deepcopy(bounds)
        result["player_state"] = self.snapshot()
        return result
