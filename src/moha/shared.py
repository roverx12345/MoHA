"""Joint calibration of one Harness across multiple planner model stacks."""
from __future__ import annotations

import copy
import random
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from .evaluate import checked, compare
from .failures import classify_failure
from .models import Episode, Harness, ValidationPolicy
from .perception import DEFAULT_POLICY, SELECTION_RULE, execution_summary, policy_grid, select_policy
from .records import numeric
from .roles import failure_profile, rank_candidates


AGGREGATION_RULE = "one_vote_per_failed_model_sample_trace_equal_model_episode_pool_v1"
VALIDATION_SCHEDULE = "alternate_configuration_rotate_stack_by_repeat_v1"


class NamespacedStore:
    """Give one model stack an isolated view of a shared locked RunStore."""

    def __init__(self, parent, prefix: str):
        if not isinstance(prefix, str) or not prefix or prefix.startswith("/") or ".." in Path(prefix).parts:
            raise ValueError("invalid model stack artifact namespace")
        self.parent = parent
        self.prefix = prefix.rstrip("/")
        self.root = parent.root / self.prefix
        self.identity_hash = parent.identity_hash

    def _name(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def read(self, name: str):
        return self.parent.read(self._name(name))

    def write(self, name: str, value, *, immutable: bool = False):
        return self.parent.write(self._name(name), value, immutable=immutable)

    def record_error(self, stage: str, payload: dict):
        return self.parent.record_error(f"{self.prefix.replace('/', '-')}-{stage}", payload)


def validate_shared_prepared(stacks: dict[str, dict]) -> None:
    if len(stacks) < 2:
        raise ValueError("shared calibration requires at least two model stacks")
    reference_id = next(iter(stacks))
    reference = stacks[reference_id]
    for stack_id, prepared in stacks.items():
        if not isinstance(stack_id, str) or not stack_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in stack_id):
            raise ValueError("stack IDs may contain only letters, numbers, dot, underscore and hyphen")
        checks = {
            "initial H0": prepared["initial"] == reference["initial"],
            "search policy": prepared["search"] == reference["search"],
            "validation policy": prepared["policy"] == reference["policy"],
            "calibration split": [s.id for s in prepared["calibration"]] == [s.id for s in reference["calibration"]],
            "validation split": [s.id for s in prepared["validation"]] == [s.id for s in reference["validation"]],
            "budget": prepared["config"]["budget"] == reference["config"]["budget"],
            "candidate catalog": prepared["allowed_ids"] == reference["allowed_ids"],
            "runtime": prepared["identity"]["runtime"]["source_hash"] == reference["identity"]["runtime"]["source_hash"],
            "source": prepared["identity"]["source"]["source_hash"] == reference["identity"]["source"]["source_hash"],
            "input hashes": prepared["identity"]["input_hashes"] == reference["identity"]["input_hashes"],
            "Judge": prepared["config"]["models"]["judge"] == reference["config"]["models"]["judge"],
            "observer": prepared["config"]["observer"] == reference["config"]["observer"],
            "specialist set": prepared["config"].get("specialists", []) == reference["config"].get("specialists", []),
            "OCR backend": prepared["config"].get("image") == reference["config"].get("image"),
            "ASR backend": prepared["config"].get("asr") == reference["config"].get("asr"),
            "final perception calibration": prepared["config"].get("perception_calibration", False) is True
                and reference["config"].get("perception_calibration", False) is True,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"shared stack {stack_id} differs from {reference_id}: {', '.join(failed)}")


def shared_identity(stacks: dict[str, dict], *, lineage: dict | None = None) -> dict:
    validate_shared_prepared(stacks)
    identity = {
        "implementation": "moha_shared_calibration_v1",
        "aggregation": AGGREGATION_RULE,
        "validation_schedule": VALIDATION_SCHEDULE,
        "stack_order": list(stacks),
        "stacks": [{"stack_id": stack_id, "identity": prepared["identity"]}
                   for stack_id, prepared in stacks.items()],
    }
    if lineage is not None:
        if lineage.get("schema") != "moha_shared_lineage_v1":
            raise ValueError("unknown shared calibration lineage schema")
        identity["lineage"] = copy.deepcopy(lineage)
    return identity


def compare_shared(old_by_stack: dict[str, list[list[Episode]]],
                   new_by_stack: dict[str, list[list[Episode]]], samples,
                   old: Harness, new: Harness, policy: ValidationPolicy) -> dict:
    """Paired validation with equal weight per model/sample/repeat episode.

    Bootstrap units preserve the outcomes of every stack for a sampled video and
    preserve run-wide variation by resampling complete repeat blocks.
    """
    if not samples or len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("validation samples must be nonempty and unique")
    if not old_by_stack or list(old_by_stack) != list(new_by_stack):
        raise ValueError("shared comparisons require the same ordered model stacks")
    stack_ids = list(old_by_stack)
    groups_by_video = defaultdict(list)
    for index, sample in enumerate(samples):
        groups_by_video[sample.video_key].append(index)
    groups = list(groups_by_video.values())
    gains = {stack_id: [] for stack_id in stack_ids}
    counts = {"wrong_to_correct": 0, "correct_to_wrong": 0,
              "correct_to_correct": 0, "wrong_to_wrong": 0}
    old_correct = new_correct = 0
    old_costs, new_costs = [], []
    missing_cost = False
    by_stack = {}
    for stack_id in stack_ids:
        if len(old_by_stack[stack_id]) != policy.repeats or len(new_by_stack[stack_id]) != policy.repeats:
            raise ValueError("incorrect shared repeat count")
        by_stack[stack_id] = compare(old_by_stack[stack_id], new_by_stack[stack_id], samples,
                                     old, new, policy)
        for repeat in range(policy.repeats):
            left = checked(old_by_stack[stack_id][repeat], samples, old, repeat)
            right = checked(new_by_stack[stack_id][repeat], samples, new, repeat)
            row = []
            for a, b in zip(left, right):
                old_correct += int(a.correct)
                new_correct += int(b.correct)
                counts[f"{'correct' if a.correct else 'wrong'}_to_{'correct' if b.correct else 'wrong'}"] += 1
                ca = numeric(a.usage.get(policy.cost_metric))
                cb = numeric(b.usage.get(policy.cost_metric))
                if ca is None or cb is None:
                    missing_cost = True
                    penalty = 0
                else:
                    old_costs.append(ca)
                    new_costs.append(cb)
                    penalty = policy.cost_penalty * (cb - ca) / policy.cost_scale
                row.append(int(b.correct) - int(a.correct) - penalty)
            gains[stack_id].append(row)
    n = len(stack_ids) * len(samples) * policy.repeats
    base = {
        "old_accuracy": old_correct / n,
        "new_accuracy": new_correct / n,
        "accuracy_delta": (new_correct - old_correct) / n,
        "paired": counts,
        "sample_count_per_stack": len(samples),
        "episode_count": n,
        "video_count": len(groups),
        "repeats": policy.repeats,
        "stack_count": len(stack_ids),
        "stack_ids": stack_ids,
        "aggregation": AGGREGATION_RULE,
        "cost_metric": policy.cost_metric,
        "by_stack": by_stack,
    }
    if missing_cost:
        return {**base, "accepted": False, "reason": "missing_measured_cost"}
    old_cost, new_cost = sum(old_costs) / n, sum(new_costs) / n
    repeat_gains = [sum(gains[stack_id][repeat][index]
                        for stack_id in stack_ids for index in range(len(samples)))
                    / (len(stack_ids) * len(samples)) for repeat in range(policy.repeats)]
    rng = random.Random(0)
    bootstrap = []
    for _ in range(policy.bootstrap_samples):
        indices = [index for _ in groups for index in groups[rng.randrange(len(groups))]]
        repeat_ids = [rng.randrange(policy.repeats) for _ in range(policy.repeats)]
        value = sum(gains[stack_id][repeat][index]
                    for stack_id in stack_ids for repeat in repeat_ids for index in indices)
        bootstrap.append(value / (len(stack_ids) * len(indices) * policy.repeats))
    bootstrap.sort()
    tail = (1 - policy.confidence) / 2
    low = bootstrap[int(tail * (len(bootstrap) - 1))]
    high = bootstrap[int((1 - tail) * (len(bootstrap) - 1))]
    gain = sum(repeat_gains) / len(repeat_gains)
    reason = "accepted"
    if old == new:
        reason = "same_configuration"
    elif policy.max_cost_ratio is not None and new_cost > old_cost * policy.max_cost_ratio + 1e-9:
        reason = "cost_limit"
    elif gain <= policy.min_gain:
        reason = "insufficient_gain"
    elif policy.require_positive_interval and low <= 0:
        reason = "uncertain_gain"
    elif policy.require_each_repeat_nonnegative and min(repeat_gains) < 0:
        reason = "inconsistent_repeats"
    return {**base, "accepted": reason == "accepted", "reason": reason,
            "utility_delta": gain, "interval": [low, high], "confidence": policy.confidence,
            "interval_method": "percentile_bootstrap_repeat_video_and_model_pool_v1",
            "repeat_gains": repeat_gains, "old_cost": old_cost, "new_cost": new_cost}


def calibrate_shared_perception(shared, base: Harness) -> dict:
    configs = policy_grid(base)
    policy, samples, store = shared.policy, shared.validation, shared.store
    stack_ids = list(shared.calibrators)
    plan = {
        "schema": "moha_shared_perception_calibration_v1",
        "structural_harness_id": base.id,
        "structural_harness": base.to_dict(),
        "configs": configs,
        "validation_samples": [s.id for s in samples],
        "validation_policy": asdict(policy),
        "stack_ids": stack_ids,
        "aggregation": AGGREGATION_RULE,
        "selection_rule": SELECTION_RULE,
        "default_policy_id": DEFAULT_POLICY,
        "scope": "one shared policy selected from pooled paired validation episodes across model stacks",
        "schedule": "default_first_then_grid_rotated_by_repeat_and_stack_v1",
    }
    store.write("perception/plan.json", plan, immutable=True)
    existing = store.read("perception/result.json")
    if existing is not None:
        validate_shared_perception_result(existing, base, policy, stack_ids)
        return existing
    ordered = sorted(configs, key=lambda config: config["id"] != DEFAULT_POLICY)
    runs = {config["id"]: {stack_id: [] for stack_id in stack_ids} for config in configs}
    audits = {config["id"]: {stack_id: [] for stack_id in stack_ids} for config in configs}
    for repeat in range(policy.repeats):
        offset = repeat % len(ordered)
        for config in ordered[offset:] + ordered[:offset]:
            harness = Harness.from_dict(config["harness"])
            stack_order = stack_ids[repeat % len(stack_ids):] + stack_ids[:repeat % len(stack_ids)]
            for stack_id in stack_order:
                episodes = shared.calibrators[stack_id].batch(
                    harness, samples, repeat, "shared_perception_validation")
                audit = execution_summary([episodes])
                if audit["missing_records"]:
                    raise ValueError("shared perception validation contains observations without realized sampling records")
                audits[config["id"]][stack_id].append(audit)
                runs[config["id"]][stack_id].append([replace(e, events=[], raw={}) for e in episodes])
                store.write("progress.json", {"phase": "perception", "config_id": config["id"],
                    "stack_id": stack_id, "repeat": repeat, "status": "completed"})
    baseline = next(config for config in configs if config["id"] == DEFAULT_POLICY)
    rows = []
    for config in configs:
        stack_rows = {}
        pooled = []
        for stack_id in stack_ids:
            episodes = [episode for run in runs[config["id"]][stack_id] for episode in run]
            costs = [numeric(episode.usage.get(policy.cost_metric)) for episode in episodes]
            if any(cost is None for cost in costs):
                raise ValueError("shared perception selection requires measured cost for every validation episode")
            accuracy = sum(episode.correct for episode in episodes) / len(episodes)
            cost = sum(costs) / len(costs)
            stack_rows[stack_id] = {"accuracy": accuracy, "correct": sum(e.correct for e in episodes),
                "episodes": len(episodes), "mean_cost": cost,
                "utility": accuracy - policy.cost_penalty * cost / policy.cost_scale,
                "realized_execution_by_repeat": audits[config["id"]][stack_id]}
            pooled.extend(episodes)
        costs = [numeric(episode.usage.get(policy.cost_metric)) for episode in pooled]
        accuracy = sum(episode.correct for episode in pooled) / len(pooled)
        cost = sum(costs) / len(costs)
        row = {k: v for k, v in config.items() if k != "harness"}
        row.update(accuracy=accuracy, correct=sum(e.correct for e in pooled), episodes=len(pooled),
            mean_cost=cost, cost_metric=policy.cost_metric,
            utility=accuracy - policy.cost_penalty * cost / policy.cost_scale,
            aggregation=AGGREGATION_RULE, by_stack=stack_rows,
            versus_default=compare_shared(runs[DEFAULT_POLICY], runs[config["id"]], samples,
                Harness.from_dict(baseline["harness"]), Harness.from_dict(config["harness"]), policy))
        store.write(f"perception/scores/{config['id']}.json", row, immutable=True)
        rows.append(row)
    winner = select_policy(rows, policy)
    selected = next(config for config in configs if config["id"] == winner["id"])
    result = {
        "schema": "moha_shared_perception_calibration_v1",
        "status": "completed",
        "structural_harness_id": base.id,
        "selection_rule": SELECTION_RULE,
        "aggregation": AGGREGATION_RULE,
        "stack_ids": stack_ids,
        "default_policy_id": DEFAULT_POLICY,
        "selected_policy_id": winner["id"],
        "selected_harness": selected["harness"],
        "selected_harness_id": selected["harness_id"],
        "sample_count_per_stack": len(samples),
        "repeats": policy.repeats,
        "rows": rows,
        "note": "Shared validation model selection, not independent test performance.",
    }
    store.write("perception/result.json", result, immutable=True)
    return result


def validate_shared_perception_result(result, base, policy, stack_ids):
    configs = {config["id"]: config for config in policy_grid(base)}
    rows = result.get("rows", [])
    if (result.get("schema") != "moha_shared_perception_calibration_v1"
            or result.get("status") != "completed"
            or result.get("structural_harness_id") != base.id
            or result.get("selection_rule") != SELECTION_RULE
            or result.get("aggregation") != AGGREGATION_RULE
            or result.get("stack_ids") != stack_ids
            or len(rows) != len(configs) or {row["id"] for row in rows} != set(configs)
            or any(row["harness_id"] != configs[row["id"]]["harness_id"] for row in rows)):
        raise ValueError("incomplete or inconsistent shared perception calibration")
    selected = configs[select_policy(rows, policy)["id"]]
    if (result.get("selected_policy_id") != selected["id"]
            or result.get("selected_harness_id") != selected["harness_id"]
            or result.get("selected_harness") != selected["harness"]):
        raise ValueError("selected shared perception policy differs from the validation ranking")
    return Harness.from_dict(selected["harness"])


class SharedCalibrator:
    """One search state and one promoted Harness shared by all model stacks."""

    def __init__(self, *, calibrators: dict[str, object], store):
        if len(calibrators) < 2:
            raise ValueError("shared calibration requires at least two calibrators")
        self.calibrators = calibrators
        self.store = store
        first = next(iter(calibrators.values()))
        self.initial = Harness.from_dict(first.state["harness"])
        self.search, self.policy = first.search, first.validation_policy
        self.calibration, self.validation = first.calibration, first.validation
        self.catalog, self.p_view = first.catalog, first.p_view
        for stack_id, calibrator in calibrators.items():
            checks = {
                "initial H0": Harness.from_dict(calibrator.state["harness"]) == self.initial,
                "search policy": calibrator.search == self.search,
                "validation policy": calibrator.validation_policy == self.policy,
                "calibration split": calibrator.calibration == self.calibration,
                "validation split": calibrator.validation == self.validation,
                "catalog": calibrator.catalog == self.catalog,
                "p_view": calibrator.p_view == self.p_view,
                "perception calibration": calibrator.perception_calibration is True,
            }
            failed = [name for name, ok in checks.items() if not ok]
            if failed:
                raise ValueError(f"shared calibrator {stack_id} differs: {', '.join(failed)}")
        experiment = {
            "schema": "moha_shared_experiment_v1",
            "stack_ids": list(calibrators),
            "initial": self.initial.to_dict(),
            "search": asdict(self.search),
            "validation": asdict(self.policy),
            "calibration": [sample.id for sample in self.calibration],
            "heldout": [sample.id for sample in self.validation],
            "catalog": [item.to_dict() for item in self.catalog.values()],
            "aggregation": AGGREGATION_RULE,
            "validation_schedule": VALIDATION_SCHEDULE,
            "perception_grid": policy_grid(self.initial),
        }
        self.store.write("experiment.json", experiment, immutable=True)
        self.state = self.store.read("checkpoint.json") or {
            "round": 0, "attempt": 0, "harness": self.initial.to_dict(), "history": [],
            "no_progress_rounds": 0, "previous_profile": None, "profile": None,
            "status": "running", "stop_reason": None,
        }

    def _save(self):
        self.store.write("checkpoint.json", self.state)

    def _advance_round(self, promoted):
        current, previous = self.state["profile"], self.state["previous_profile"]
        distance = (sum(abs(current.get(key, 0) - previous.get(key, 0))
                        for key in set(current) | set(previous)) if previous is not None else None)
        self.state.update(round=self.state["round"] + 1, attempt=0, previous_profile=current,
                          profile_distance=distance,
                          no_progress_rounds=0 if promoted else self.state["no_progress_rounds"] + 1)
        self._save()
        return (self.state["no_progress_rounds"] >= self.search.patience
                and distance is not None and distance < self.search.failure_stability)

    def _recommendations(self, current, available):
        diagnoses, by_stack = [], {}
        for stack_id, calibrator in self.calibrators.items():
            calibrator.state["round"] = self.state["round"]
            episodes = calibrator.batch(current, self.calibration, self.state["round"], "shared_calibration")
            local = calibrator.diagnose(current, episodes, available)
            by_stack[stack_id] = local
            for diagnosis in local:
                item = copy.deepcopy(diagnosis)
                item["stack_id"] = stack_id
                item["source_sample_id"] = diagnosis["sample_id"]
                item["sample_id"] = f"{stack_id}::{diagnosis['sample_id']}"
                diagnoses.append(item)
            self.store.write("progress.json", {"phase": "calibration", "round": self.state["round"],
                "stack_id": stack_id, "status": "completed"})
        return diagnoses, by_stack

    def _validate(self, current, candidate):
        stack_ids = list(self.calibrators)
        left = {stack_id: [] for stack_id in stack_ids}
        right = {stack_id: [] for stack_id in stack_ids}
        for repeat in range(self.policy.repeats):
            order = [(current, left, "baseline"), (candidate, right, "candidate")]
            for harness, target, label in (order if repeat % 2 == 0 else reversed(order)):
                rotated = stack_ids[repeat % len(stack_ids):] + stack_ids[:repeat % len(stack_ids)]
                for stack_id in rotated:
                    target[stack_id].append(self.calibrators[stack_id].batch(
                        harness, self.validation, repeat, "shared_validation"))
                    self.store.write("progress.json", {"phase": "validation", "round": self.state["round"],
                        "attempt": self.state["attempt"], "configuration": label,
                        "stack_id": stack_id, "repeat": repeat, "status": "completed"})
        return compare_shared(left, right, self.validation, current, candidate, self.policy)

    def run(self) -> dict:
        if self.state["status"] == "completed":
            return self._finish(self.state["stop_reason"])
        self.state["status"] = "running"
        self.state.pop("error_type", None)
        self.state.pop("failure", None)
        self._save()
        try:
            if self.state.get("phase") == "perception":
                return self._finish(self.state["stop_reason"])
            while self.state["round"] < self.search.max_rounds:
                current = Harness.from_dict(self.state["harness"])
                rejected = {entry["candidate"] for entry in self.state["history"]
                            if not entry["validation"]["accepted"]}
                available = [item for item in self.catalog.values()
                             if item.id not in rejected and item.available(current, self.p_view)]
                if not available:
                    return self._finish("catalog_exhausted")
                path = f"recommendations/{self.state['round']:03d}.json"
                recorded = self.store.read(path)
                if recorded is None:
                    diagnoses, by_stack = self._recommendations(current, available)
                    recorded = {"harness_id": current.id, "diagnoses": diagnoses,
                                "diagnoses_by_stack": by_stack,
                                "catalog": [item.to_dict() for item in available],
                                "aggregation": AGGREGATION_RULE}
                    self.store.write(path, recorded, immutable=True)
                if recorded["harness_id"] != current.id:
                    raise ValueError("shared round recommendations belong to a different harness")
                diagnoses = recorded["diagnoses"]
                if not diagnoses:
                    return self._finish("no_calibration_failures")
                self.state["profile"] = failure_profile(diagnoses)
                self.store.write(f"profiles/{self.state['round']:03d}.json", self.state["profile"], immutable=True)
                selection = rank_candidates(diagnoses, available)
                selection["aggregation"] = AGGREGATION_RULE
                slot = f"selections/{self.state['round']:03d}-{self.state['attempt']:02d}.json"
                self.store.write(slot, selection, immutable=True)
                candidate_id = selection["candidate_id"]
                if candidate_id is None:
                    if self._advance_round(False):
                        return self._finish("converged")
                    continue
                candidate = self.catalog[candidate_id].apply(current)
                result = self._validate(current, candidate)
                self.state["history"].append({"round": self.state["round"], "candidate": candidate_id,
                    "from": current.id, "to": candidate.id, "validation": result})
                self.state["attempt"] += 1
                if result["accepted"]:
                    self.state["harness"] = candidate.to_dict()
                if result["accepted"] or self.state["attempt"] >= self.search.candidates_per_round:
                    if self._advance_round(result["accepted"]):
                        return self._finish("converged")
                self._save()
            return self._finish("round_budget")
        except Exception as exc:
            self.state["status"] = "error"
            self.state["error_type"] = type(exc).__name__
            self.state["failure"] = classify_failure(exc)
            self._save()
            raise

    def _finish(self, reason: str) -> dict:
        if self.state["status"] != "completed":
            if self.state.get("phase") != "perception":
                self.state.update(phase="perception", structural_harness=self.state["harness"], stop_reason=reason)
                self._save()
            base = Harness.from_dict(self.state["structural_harness"])
            selection = calibrate_shared_perception(self, base)
            self.state.update(harness=selection["selected_harness"],
                              perception_calibration=selection, phase="completed")
        self.state.update(status="completed", stop_reason=reason)
        self._save()
        self.store.write("result.json", self.state, immutable=True)
        harness = Harness.from_dict(self.state["harness"])
        self.store.write("frozen_harness.json", {
            "schema": "moha_shared_frozen_v1",
            "harness": harness.to_dict(),
            "harness_id": harness.id,
            "run_identity": self.store.identity_hash,
            "stack_ids": list(self.calibrators),
            "aggregation": AGGREGATION_RULE,
        }, immutable=True)
        return self.state
