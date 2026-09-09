"""Validation-only sampling-rate selection after structural adaptation is frozen."""
from __future__ import annotations
from dataclasses import asdict, replace
from .evaluate import compare
from .execution import FRAME_CAP, SAMPLING_RATES, CHOICES
from .models import Harness, ValidationPolicy
from .records import numeric, receipts


SELECTION_RULE = "max_utility_ties_default_then_cost_then_id_v1"
DEFAULT_POLICY = "fps1_scale1"


def policy_grid(base: Harness):
    if base.execution != Harness().execution:
        raise ValueError("final perception calibration requires unchanged H0 execution during structural adaptation")
    configs = []
    for rate in SAMPLING_RATES:
        for scale in CHOICES["source_scale"]:
            # auto is the canonical 1 FPS baseline, including its original hash.
            settings = {}
            if rate != 1:
                settings["target_fps"] = rate
            if scale != 1:
                settings["source_scale"] = scale
            harness = replace(base, execution=(("default", tuple(settings.items())),))
            configs.append({"id": f"fps{rate:g}_scale{scale:g}", "target_fps": rate,
                "source_scale": scale, "priority": "balanced", "frame_cap": FRAME_CAP,
                "harness": harness.to_dict(), "harness_id": harness.id})
    return configs


def select_policy(rows, policy: ValidationPolicy):
    baseline = next(row for row in rows if row["id"] == DEFAULT_POLICY)
    if any(numeric(row.get(k)) is None for row in rows for k in ("accuracy", "mean_cost")):
        raise ValueError("perception selection requires complete accuracy and measured cost")
    eligible = [row for row in rows if row["id"] == DEFAULT_POLICY or policy.max_cost_ratio is None
                or row["mean_cost"] <= baseline["mean_cost"] * policy.max_cost_ratio + 1e-9]
    def rank(row):
        utility = row["accuracy"] - policy.cost_penalty * row["mean_cost"] / policy.cost_scale
        return (-utility, row["id"] != DEFAULT_POLICY, row["mean_cost"], row["id"])
    return min(eligible, key=rank)


def execution_summary(runs):
    calls = [r for run in runs for episode in run for r in receipts(episode.events)
             if r.get("observer_id") == "omni" and not r.get("specialist_active")]
    successful = [r.get("realized_execution", {}) for r in calls if not r.get("error")]
    fields = ("window_duration", "target_fps", "target_frames", "realized_frames", "realized_fps")
    def mean(values):
        return sum(values) / len(values) if values else None
    recorded = [r for r in successful if all(numeric(r.get(k)) is not None for k in fields)
                and isinstance(r.get("realized_resolution"), (list, tuple))
                and len(r["realized_resolution"]) == 2 and type(r.get("frame_cap_hit")) is bool]
    return {"generalist_calls": len(calls), "failed_calls": len(calls) - len(successful),
            "recorded_calls": len(recorded), "missing_records": len(successful) - len(recorded),
            "means": {k: mean([r[k] for r in recorded]) for k in fields},
            "mean_resolution": [mean([r["realized_resolution"][i] for r in recorded]) for i in (0, 1)],
            "frame_cap_hit_count": sum(r["frame_cap_hit"] for r in recorded),
            "frame_cap_hit_fraction": mean([int(r["frame_cap_hit"]) for r in recorded]),
            "below_target_count": sum(r["realized_frames"] < r["target_frames"] for r in recorded)}


def calibrate_perception(calibrator, base: Harness):
    """Use the existing lane scheduler, strict episodes and resumable split cache."""
    configs = policy_grid(base)
    policy, samples, store = calibrator.validation_policy, calibrator.validation, calibrator.store
    plan = {"schema": "moha_perception_calibration_v1", "structural_harness_id": base.id,
            "structural_harness": base.to_dict(), "configs": configs,
            "validation_samples": [s.id for s in samples], "validation_policy": asdict(policy),
            "selection_rule": SELECTION_RULE, "default_policy_id": DEFAULT_POLICY,
            "scope": "paired validation episodes; planner settings fixed, observation windows may differ",
            "schedule": "default_first_then_grid_rotated_by_repeat_v1"}
    store.write("perception/plan.json", plan, immutable=True)
    existing = store.read("perception/result.json")
    if existing is not None:
        validate_result(existing, base, policy)
        return existing
    ordered = sorted(configs, key=lambda c: c["id"] != DEFAULT_POLICY)
    runs = {c["id"]: [] for c in configs}
    audits = {c["id"]: [] for c in configs}
    for repeat in range(policy.repeats):
        offset = repeat % len(ordered)
        for config in ordered[offset:] + ordered[:offset]:
            harness = Harness.from_dict(config["harness"])
            episodes = calibrator.batch(harness, samples, repeat, "perception_validation")
            audit = execution_summary([episodes])
            if audit["missing_records"]:
                raise ValueError("perception validation contains observations without realized sampling records")
            audits[config["id"]].append(audit)
            # Full traces stay in the immutable episode files; paired comparisons
            # only need outcomes, identities and measured costs.
            runs[config["id"]].append([replace(e, events=[], raw={}) for e in episodes])
    baseline = next(c for c in configs if c["id"] == DEFAULT_POLICY)
    rows = []
    for config in configs:
        episodes = [e for run in runs[config["id"]] for e in run]
        costs = [numeric(e.usage.get(policy.cost_metric)) for e in episodes]
        if any(cost is None for cost in costs):
            raise ValueError("perception selection requires measured cost for every validation episode")
        accuracy, cost = sum(e.correct for e in episodes) / len(episodes), sum(costs) / len(costs)
        row = {k: v for k, v in config.items() if k != "harness"}
        row.update(accuracy=accuracy, correct=sum(e.correct for e in episodes), episodes=len(episodes),
            mean_cost=cost, cost_metric=policy.cost_metric,
            utility=accuracy - policy.cost_penalty * cost / policy.cost_scale,
            realized_execution_by_repeat=audits[config["id"]],
            versus_default=compare(runs[DEFAULT_POLICY], runs[config["id"]], samples,
                Harness.from_dict(baseline["harness"]), Harness.from_dict(config["harness"]), policy))
        store.write(f"perception/scores/{config['id']}.json", row, immutable=True)
        rows.append(row)
    winner = select_policy(rows, policy)
    selected = next(c for c in configs if c["id"] == winner["id"])
    result = {"schema": "moha_perception_calibration_v1", "status": "completed",
              "structural_harness_id": base.id, "selection_rule": SELECTION_RULE,
              "default_policy_id": DEFAULT_POLICY, "selected_policy_id": winner["id"],
              "selected_harness": selected["harness"], "selected_harness_id": selected["harness_id"],
              "sample_count": len(samples), "repeats": policy.repeats, "rows": rows,
              "note": "Validation model selection, not independent test performance. Paired intervals are descriptive; structural promotion thresholds do not gate this argmax."}
    store.write("perception/result.json", result, immutable=True)
    return result


def validate_result(result, base, policy):
    """Reconstruct the selected configuration when resuming or exporting it."""
    configs = {c["id"]: c for c in policy_grid(base)}
    rows = result.get("rows", [])
    if (result.get("schema") != "moha_perception_calibration_v1" or result.get("status") != "completed"
            or result.get("structural_harness_id") != base.id or result.get("selection_rule") != SELECTION_RULE
            or len(rows) != len(configs) or {r["id"] for r in rows} != set(configs)
            or any(r["harness_id"] != configs[r["id"]]["harness_id"] for r in rows)):
        raise ValueError("incomplete or inconsistent final perception calibration")
    selected = configs[select_policy(rows, policy)["id"]]
    if (result.get("selected_policy_id") != selected["id"]
            or result.get("selected_harness_id") != selected["harness_id"]
            or result.get("selected_harness") != selected["harness"]):
        raise ValueError("selected perception policy differs from the validation ranking")
    return Harness.from_dict(selected["harness"])
