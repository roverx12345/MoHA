"""One audited repair of unusable observations over the same bounded media."""
from __future__ import annotations
import copy
from dataclasses import replace
from video_os.core.errors import ProviderResponseError
from video_os.providers.core import (VideoOSPerceptionService, _default_compact_modalities,
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
