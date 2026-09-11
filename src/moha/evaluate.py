"""Complete paired validation with clustered uncertainty and explicit cost."""
from __future__ import annotations
import random
from collections import defaultdict
from .models import Episode, Harness, Sample, ValidationPolicy
from .records import numeric


def checked(results: list[Episode], samples: list[Sample], harness: Harness, repeat: int) -> list[Episode]:
    by_id = {e.sample_id: e for e in results}
    if len(by_id) != len(results) or set(by_id) != {s.sample_id for s in samples}:
        raise ValueError("incomplete, duplicated or unexpected validation samples")
    ordered = [by_id[s.sample_id] for s in samples]
    for sample, result in zip(samples, ordered):
        if (result.sample_hash != sample.id or result.expected_answer != sample.expected_answer
                or result.video_key != list(sample.video_key) or result.harness_id != harness.id
                or result.repeat != repeat):
            raise ValueError("episode identity does not match validation request")
        if result.status not in {"completed", "budget_exhausted", "abstained", "invalid_final_answer"}:
            raise ValueError("infrastructure/error episode cannot enter a promotion comparison")
    return ordered


def compare(old_runs: list[list[Episode]], new_runs: list[list[Episode]], samples: list[Sample],
            old: Harness, new: Harness, policy: ValidationPolicy) -> dict:
    if not samples or len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("validation samples must be nonempty and unique")
    if len(old_runs) != policy.repeats or len(new_runs) != policy.repeats:
        raise ValueError("incorrect repeat count")
    clusters = defaultdict(list)
    for i, sample in enumerate(samples):
        clusters[sample.video_key].append(i)
    groups = list(clusters.values())
    gains, old_costs, new_costs = [], [], []
    counts = {"wrong_to_correct": 0, "correct_to_wrong": 0, "correct_to_correct": 0, "wrong_to_wrong": 0}
    old_correct = new_correct = 0
    missing_cost = False
    for repeat in range(policy.repeats):
        left = checked(old_runs[repeat], samples, old, repeat)
        right = checked(new_runs[repeat], samples, new, repeat)
        row = []
        for a, b in zip(left, right):
            old_correct += a.correct
            new_correct += b.correct
            counts[f"{'correct' if a.correct else 'wrong'}_to_{'correct' if b.correct else 'wrong'}"] += 1
            ca, cb = numeric(a.usage.get(policy.cost_metric)), numeric(b.usage.get(policy.cost_metric))
            if ca is None or cb is None:
                missing_cost = True
            else:
                old_costs.append(ca)
                new_costs.append(cb)
            penalty = 0 if ca is None or cb is None else policy.cost_penalty * (cb - ca) / policy.cost_scale
            row.append(int(b.correct) - int(a.correct) - penalty)
        gains.append(row)
    n = len(samples) * policy.repeats
    base = {"old_accuracy": old_correct / n, "new_accuracy": new_correct / n,
            "accuracy_delta": (new_correct - old_correct) / n, "paired": counts,
            "sample_count": len(samples), "video_count": len(groups), "repeats": policy.repeats,
            "cost_metric": policy.cost_metric}
    if missing_cost:
        return {**base, "accepted": False, "reason": "missing_measured_cost"}
    old_cost, new_cost = sum(old_costs) / n, sum(new_costs) / n
    # Resample whole repeat blocks and whole videos, preserving within-video
    # question dependence and the run-wide variation shared by one repeat.
    rng = random.Random(0)
    bootstrap = []
    for _ in range(policy.bootstrap_samples):
        indices = [i for _ in groups for i in groups[rng.randrange(len(groups))]]
        repeat_ids = [rng.randrange(policy.repeats) for _ in range(policy.repeats)]
        bootstrap.append(sum(gains[r][i] for r in repeat_ids for i in indices) / (len(indices) * policy.repeats))
    bootstrap.sort()
    tail = (1 - policy.confidence) / 2
    low = bootstrap[int(tail * (len(bootstrap) - 1))]
    high = bootstrap[int((1 - tail) * (len(bootstrap) - 1))]
    repeat_gains = [sum(row) / len(row) for row in gains]
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
    return {**base, "accepted": reason == "accepted", "reason": reason, "utility_delta": gain,
            "interval": [low, high], "confidence": policy.confidence,
            "interval_method": "percentile_bootstrap_repeat_and_video_blocks_v1",
            "repeat_gains": repeat_gains, "old_cost": old_cost, "new_cost": new_cost}
