"""One audited repair of unusable observations over the same bounded media."""
from __future__ import annotations
import copy
from dataclasses import replace, dataclass, field
from flat.agent.observer_registry import FixedObserverExecution, ObserverRegistry
from .execution import ExecutionPolicy, EXECUTION_POLICY, allocate
from flat.core.errors import ProviderResponseError
from flat.providers.core import (VideoOSPerceptionService, _default_compact_modalities,
    _normalize_compact_observation, map_view_relative_times_to_source)


OBSERVER_RECOVERY_POLICY = "moha_observer_one_repair_v1"
REPAIR_PROMPT = """\nThe previous response was unusable. Re-inspect the SAME supplied media
for the SAME evidence goal and return one complete JSON object matching the schema.
Report goal-relevant events and changes. Merge a continuously unchanged action or
state into one supported interval; do not emit identical descriptions for each
sampled frame. Preserve genuinely separate repeated actions and conflicting evidence.
Do not invent details to fill time slots. Keep facts concise and include the required
uncertainty/missing fields. All support times are relative to this view, whose first
instant is 0; use only the valid time range in view_metadata. Do not answer the task.
Finish the entire JSON within the unchanged output budget."""


class ObserverOutputError(ValueError):
    """No usable evidence after the one allowed format/time repair."""


def invalid_output_reason(result, request, schema_name):
    if result.finish_reason == "length":
        return "output_truncated"
    metadata = request.payload.get("view_metadata", {})
    contract = metadata.get("model_output_coordinate_contract", {})
    if contract.get("time") != "view_relative_seconds_host_maps_to_source":
        return None
    parsed = result.parsed
    if schema_name == "observation_output_compact":
        parsed, _ = _normalize_compact_observation(parsed,
            default_modalities=_default_compact_modalities(request.content_blocks))
    try:
        # Reuse the host validator, but leave the original result untouched.
        # The normal runtime still performs the actual source-time mapping.
        map_view_relative_times_to_source(parsed,
            source_time_range=contract["valid_time_range_seconds"])
    except ProviderResponseError:
        return "support_time_out_of_window"
    return None


class ObserverService(VideoOSPerceptionService):
    def plan_observer_execution(self, session_id, window, policy):
        with self._lock:
            session = self._session(session_id)
            return allocate(session.renderer, session.media, window, policy)

    def _save_call(self, session, *, mode_label, request, output_schema, schema_name, backend=None):
        provider = backend or session.perception
        if mode_label not in {"look", "verify"} or schema_name not in {
                "observation_output", "observation_output_compact"}:
            return super()._save_call(session, mode_label=mode_label, request=request,
                output_schema=output_schema, schema_name=schema_name, backend=provider)
        recovery = None
        try:
            for attempt in range(2):
                try:
                    call_id, result = super()._save_call(session, mode_label=mode_label,
                        request=request, output_schema=output_schema, schema_name=schema_name, backend=provider)
                except ProviderResponseError as exc:
                    result = exc.receipt
                    # Transport/envelope errors, missing accounting and budget
                    # violations remain infrastructure errors, not format repairs.
                    if result is None or result.usage.input_tokens is None:
                        raise
                    limit = provider.budget.c_sensor_max
                    if limit is not None and result.usage.input_tokens > limit:
                        raise
                    if result.finish_reason not in {None, "stop", "length"}:
                        raise
                    call_id = exc.call_id
                    expected = provider.request_audit(request, output_schema=output_schema,
                                                       schema_name=schema_name)["request_sha256"]
                    if result.request_sha256 != expected:
                        raise RuntimeError("saved failed-call audit differs from dispatched request") from exc
                    # The base service saves the raw artifact but excludes failed
                    # JSON from totals. Count that already-consumed call exactly once.
                    with session.call_lock:
                        session.provider_results.append(result)
                    reason = "output_truncated" if result.finish_reason == "length" else "invalid_observation_json"
                else:
                    reason = invalid_output_reason(result, request, schema_name)
                if recovery is not None:
                    recovery["attempts"].append({"call_id": call_id, "reason": reason})
                if reason is None:
                    if recovery is not None:
                        recovery["status"] = "recovered"
                    return call_id, result
                if recovery is None:
                    metadata = request.payload["view_metadata"]
                    recovery = {"policy": OBSERVER_RECOVERY_POLICY, "status": "retrying",
                        "view_id": metadata.get("view_id"),
                        "model_time_range_seconds": metadata.get("time_range_seconds"),
                        "inspection_goal": request.payload.get("inspection_goal"),
                        "attempts": [{"call_id": call_id, "reason": reason}], "additional_calls": 0}
                    if not hasattr(session, "observer_output_recoveries"):
                        session.observer_output_recoveries = []
                    session.observer_output_recoveries.append(recovery)
                if attempt:
                    recovery["status"] = "exhausted"
                    raise ObserverOutputError(
                        "Observer output remained unusable after one repair of the same window "
                        f"({reason}). No observation was accepted. This is an observer output "
                        "failure, not evidence that the requested event is absent. Choose another "
                        "observation or answer/abstain using existing evidence.")
                prompt = request.control_prompt + REPAIR_PROMPT
                session.ledger.validate_control(control_tokens=session.context.counter.count(prompt))
                request = replace(request, control_prompt=prompt,
                    estimated_input_tokens=request.estimated_input_tokens + session.context.counter.count(REPAIR_PROMPT))
                provider = copy.copy(provider)
                provider.spec = replace(provider.spec, retries=0,
                    max_completion_tokens=min(provider.spec.max_completion_tokens or 2048, 2048))
                # Audit/preflight before charging and dispatching the one retry.
                provider.request_audit(request, output_schema=output_schema, schema_name=schema_name)
                session.ledger.charge("look", session.runtime.pending.charge)
                recovery["additional_calls"] = 1
        except Exception:
            if recovery is not None and recovery["status"] == "retrying":
                recovery["status"] = "aborted"
            raise
        finally:
            if recovery is not None:
                session.trace.add_event("observer_output_recovery", copy.deepcopy(recovery))

    def receipt(self, session_id):
        result = super().receipt(session_id)
        with self._lock:
            recoveries = getattr(self._session(session_id), "observer_output_recoveries", [])
            if recoveries:
                result["observer_output_recoveries"] = copy.deepcopy(recoveries)
        return result


@dataclass(frozen=True)
class PolicyExecution(FixedObserverExecution):
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def to_dict(self):
        return {"policy_version": EXECUTION_POLICY, **self.policy.to_dict(),
                "modalities": list(self.modalities), "prompt_profile": self.prompt_profile}

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop("policy_version", None) != EXECUTION_POLICY:
            raise ValueError("observer receipt requires the current source-relative execution policy")
        modalities = tuple(data.pop("modalities"))
        prompt = data.pop("prompt_profile")
        return cls(policy=ExecutionPolicy(**data), modalities=modalities, prompt_profile=prompt)


class _PlannedBackend:
    """Per-call adapter; never mutate a shared service or its sampling defaults."""
    def __init__(self, service, plan):
        self.service, self.plan = service, plan
        self.perception_model = service.perception_model

    def inspect_window(self, session_id, **kwargs):
        # These legacy sentinels are not execution controls for this policy.
        kwargs.pop("fps", None)
        kwargs.pop("resolution", None)
        return self.service.inspect_window(session_id, **kwargs,
            experiment_render=self.plan["experiment_render"])


class PolicyObserverRegistry(ObserverRegistry):
    def failure_receipt(self, **kwargs):
        receipt = super().failure_receipt(**kwargs)
        start, end = kwargs["window"]
        request = kwargs["execution"].policy.sampling_request(end - start)
        receipt["realized_execution"].update(request, realized_frames=None,
            realized_fps=None, realized_resolution=None)
        return receipt

    def observe(self, *, execution, **kwargs):
        if not isinstance(execution, PolicyExecution):
            raise TypeError("MoHA requires a source-relative execution policy")
        observer_id = kwargs["config"].route(kwargs["goal"])
        service = self.get(observer_id)
        plan = service.plan_observer_execution(kwargs["session_id"], kwargs["window"], execution.policy)
        registry = ObserverRegistry()
        registry.register(observer_id, _PlannedBackend(service, plan))
        result = registry.observe(execution=execution, **kwargs)
        receipt = result["observer_execution_receipt"]
        view = result.get("view", {})
        realized = receipt["realized_execution"]
        realized.update({k: copy.deepcopy(view.get(k)) for k in
            ("resolution", "frame_timestamps_seconds", "decoded_frame_indices", "source_sha256")})
        # The provider exposes a reduced view; the allocator read this fingerprint
        # from the same active session metadata before rendering.
        realized["source_sha256"] = plan["source_sha256"]
        realized["allocation"] = plan
        realized.update({k: plan[k] for k in ("window_duration", "target_fps", "requested_frames",
            "target_frames", "frame_cap", "frame_cap_hit")})
        frames = realized.get("sampled_frames")
        realized["realized_frames"] = frames
        realized["realized_fps"] = frames / plan["window_duration"] if frames is not None else None
        realized["realized_resolution"] = copy.deepcopy(realized["resolution"])
        # Report source-time sampling density, separately from encoded playback FPS.
        realized["fps"] = realized["realized_fps"]
        # Retain processor-specific resize/token accounting separately from the
        # actual rendered resolution: processor minimum pixels can undo savings.
        realized["input_token_accounting"] = copy.deepcopy(view.get("input_token_accounting", {}))
        return result
