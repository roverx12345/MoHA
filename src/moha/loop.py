"""The complete MOHA state machine. Checkpoint after each candidate decision."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from .catalog import catalog
from .evaluate import checked, compare
from .models import Episode, Harness, Sample, ValidationPolicy, check_splits, digest, positive_int
from .evidence import diagnosis_view, selection_trace
from .roles import failure_profile
from .store import RunStore


@dataclass(frozen=True)
class SearchPolicy:
    max_rounds: int = 6
    candidates_per_round: int = 2
    patience: int = 2
    min_diagnosis_coverage: float = 0.5
    max_diagnosis_error_rate: float = 0.2
    failure_stability: float = 0.1

    def __post_init__(self):
        for key in ("max_rounds", "candidates_per_round", "patience"):
            positive_int(getattr(self, key), key)
        for key in ("min_diagnosis_coverage", "max_diagnosis_error_rate", "failure_stability"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"invalid {key}")


class Calibrator:
    def __init__(self, *, runner, judge, selector, store: RunStore,
                 calibration: list[Sample], validation: list[Sample],
                 initial: Harness = Harness(), search: SearchPolicy = SearchPolicy(),
                 validation_policy: ValidationPolicy = ValidationPolicy(), p_view: int | None = None,
                 resolver=None, allowed_ids: list[str] | None = None):
        check_splits(calibration, validation)
        self.runner, self.judge, self.selector, self.store = runner, judge, selector, store
        self.calibration, self.validation = calibration, validation
        self.search, self.validation_policy, self.p_view = search, validation_policy, p_view
        all_items = catalog()
        if allowed_ids is not None and set(allowed_ids) - set(all_items):
            raise ValueError("unknown allowed intervention")
        self.catalog = {k: v for k, v in all_items.items() if allowed_ids is None or k in allowed_ids}
        self.resolver = resolver
        identity = {"initial": initial.to_dict(), "search": asdict(search), "validation": asdict(validation_policy),
                    "calibration": [s.id for s in calibration], "heldout": [s.id for s in validation],
                    "catalog": [x.to_dict() for x in self.catalog.values()], "p_view": p_view,
                    "observer_probes": resolver is not None}
        self.store.write("experiment.json", identity, immutable=True)
        self.state = self.store.read("checkpoint.json") or {
            "round": 0, "attempt": 0, "harness": initial.to_dict(), "history": [],
            "no_progress_rounds": 0, "previous_profile": None, "profile": None,
            "status": "running", "stop_reason": None}

    def batch(self, harness: Harness, samples: list[Sample], repeat: int, split: str) -> list[Episode]:
        results = []
        for sample in samples:
            key = digest({"harness": harness.id, "sample": sample.id, "repeat": repeat, "split": split})
            path = f"episodes/{key}.json"
            saved = self.store.read(path)
            if saved is not None:
                episode = Episode.from_dict(saved)
            else:
                episode = self.runner.run(harness, sample, repeat)
                if episode.status == "error":
                    self.store.record_error("episode", episode.to_dict())
                    raise RuntimeError("episode execution failed; error artifact saved")
                checked([episode], [sample], harness, repeat)
                self.store.write(path, episode.to_dict(), immutable=True)
            results.append(episode)
            self.store.write("progress.json", {"phase": split, "completed": len(results), "total": len(samples),
                "harness_id": harness.id, "sample_id": sample.sample_id, "repeat": repeat,
                "cached": saved is not None, "status": episode.status, "correct": episode.correct})
        return checked(results, samples, harness, repeat)

    def diagnose(self, harness: Harness, episodes: list[Episode]) -> list[dict]:
        diagnoses = []
        for sample, episode in zip(self.calibration, episodes):
            if episode.correct:
                continue
            key = digest({"harness": harness.id, "sample": sample.id, "episode": episode.to_dict()})
            path = f"diagnoses/{key}.json"
            result = self.store.read(path)
            if result is None:
                result = self.judge.diagnose(diagnosis_view(episode, sample, harness))
                if result.get("status") == "valid" and result.get("failure") == "observer":
                    if self.resolver is None:
                        result["observer_resolution"] = {"status": "unavailable"}
                    else:
                        result["observer_resolution"] = self.resolver.resolve(sample, harness, episode, result)
                        if result["observer_resolution"].get("status") == "error":
                            result["status"] = "error"
                result["sample_id"] = sample.sample_id
                if result.get("status") == "valid":
                    result["trace_evidence"] = selection_trace(episode, sample, harness, result)
                if result.get("status") == "error":
                    self.store.record_error("diagnosis", result)
                else:
                    self.store.write(path, result, immutable=True)
            diagnoses.append(result)
        if diagnoses:
            valid = sum(x.get("status") == "valid" and x.get("failure") != "unresolved" for x in diagnoses)
            errors = sum(x.get("status") == "error" for x in diagnoses)
            health = {"failed": len(diagnoses), "valid": valid, "errors": errors,
                      "coverage": valid / len(diagnoses), "error_rate": errors / len(diagnoses)}
            self.store.write(f"health/{harness.id}.json", health)
            if health["error_rate"] > self.search.max_diagnosis_error_rate or health["coverage"] < self.search.min_diagnosis_coverage:
                raise RuntimeError("diagnosis health gate failed; inspect health artifacts before validation")
        return diagnoses

    def _save(self):
        self.store.write("checkpoint.json", self.state)

    def _advance_round(self, promoted):
        current, previous = self.state["profile"], self.state["previous_profile"]
        distance = (sum(abs(current.get(k, 0) - previous.get(k, 0)) for k in set(current) | set(previous))
                    if previous is not None else None)
        self.state.update(round=self.state["round"] + 1, attempt=0, previous_profile=current,
                          profile_distance=distance,
                          no_progress_rounds=0 if promoted else self.state["no_progress_rounds"] + 1)
        self._save()
        return (self.state["no_progress_rounds"] >= self.search.patience
                and distance is not None and distance < self.search.failure_stability)

    def run(self) -> dict:
        if self.state["status"] == "completed":
            return self._finish(self.state["stop_reason"])
        self.state["status"] = "running"
        self.state.pop("error_type", None)
        self._save()
        try:
            while self.state["round"] < self.search.max_rounds:
                current = Harness.from_dict(self.state["harness"])
                diagnoses = self.diagnose(current, self.batch(current, self.calibration, self.state["round"], "calibration"))
                if not diagnoses:
                    return self._finish("no_calibration_failures")
                self.state["profile"] = failure_profile(diagnoses)
                self.store.write(f"profiles/{self.state['round']:03d}.json", self.state["profile"], immutable=True)
                rejected = {h["candidate"] for h in self.state["history"]
                            if not h["validation"]["accepted"]}
                available = [c for c in self.catalog.values() if c.id not in rejected and c.available(current, self.p_view)]
                available = [c for c in available if c.coordinate != "specialists" or any(
                    d.get("failed_capability") == c.value
                    and d.get("observer_resolution", {}).get("status") == "no_rescue"
                    for d in diagnoses)]
                if not available:
                    return self._finish("catalog_exhausted")
                slot = f"selections/{self.state['round']:03d}-{self.state['attempt']:02d}.json"
                selection = self.store.read(slot)
                if selection is None:
                    selection = self.selector.select(diagnoses, current, available, self.state["history"])
                    if selection.get("status") == "error":
                        self.store.record_error("selection", selection)
                    else:
                        self.store.write(slot, selection, immutable=True)
                if selection.get("status") == "error":
                    raise RuntimeError("selector failed; inspect selection artifact")
                candidate_id = selection.get("candidate_id")
                if candidate_id is None:
                    if self._advance_round(False):
                        return self._finish("converged")
                    continue
                if candidate_id not in {x.id for x in available}:
                    raise ValueError("selector proposed an unavailable candidate")
                candidate = self.catalog[candidate_id].apply(current)
                left, right = [], []
                for repeat in range(self.validation_policy.repeats):
                    # Alternate acquisition order across repeats to reduce fixed
                    # baseline-first time drift; cached repeats keep their identity.
                    order = [(current, left), (candidate, right)]
                    for config, target in (order if repeat % 2 == 0 else reversed(order)):
                        target.append(self.batch(config, self.validation, repeat, "validation"))
                result = compare(left, right, self.validation, current, candidate, self.validation_policy)
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
            self._save()
            raise

    def _finish(self, reason: str) -> dict:
        self.state.update(status="completed", stop_reason=reason)
        self._save()
        self.store.write("result.json", self.state, immutable=True)
        harness = Harness.from_dict(self.state["harness"])
        self.store.write("frozen_harness.json", {"schema": "moha_frozen_v1", "harness": harness.to_dict(),
                         "harness_id": harness.id, "run_identity": self.store.identity_hash}, immutable=True)
        return self.state
