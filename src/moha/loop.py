"""The complete MOHA state machine. Checkpoint after each candidate decision."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from threading import Lock
from .catalog import catalog
from .evaluate import checked, compare
from .models import Episode, Harness, Sample, ValidationPolicy, check_splits, digest, positive_int
from .evidence import diagnosis_view
from .roles import failure_profile, rank_candidates
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
    def __init__(self, *, runners, judges, store: RunStore,
                 calibration: list[Sample], validation: list[Sample],
                 initial: Harness = Harness(), search: SearchPolicy = SearchPolicy(),
                 validation_policy: ValidationPolicy = ValidationPolicy(), p_view: int | None = None,
                 resolvers=None, allowed_ids: list[str] | None = None):
        check_splits(calibration, validation)
        self.runners = tuple(runners)
        if not self.runners or len({id(r) for r in self.runners}) != len(self.runners):
            raise ValueError("each execution lane requires an independent runner")
        self.resolvers = tuple(resolvers) if resolvers is not None else (None,) * len(self.runners)
        if len(self.resolvers) != len(self.runners):
            raise ValueError("probe resolvers must match execution lanes")
        self.judges = tuple(judges)
        if not self.judges or len({id(j) for j in self.judges}) != len(self.judges):
            raise ValueError("each diagnosis worker requires an independent judge")
        # A probe keeps its original episode service and verdict client. Only
        # one diagnosis may use either at a time, even with more Judge workers.
        self.probe_locks = [Lock() for _ in self.runners]
        self.store = store
        self.calibration, self.validation = calibration, validation
        self.search, self.validation_policy, self.p_view = search, validation_policy, p_view
        all_items = catalog()
        if allowed_ids is not None and set(allowed_ids) - set(all_items):
            raise ValueError("unknown allowed intervention")
        self.catalog = {k: v for k, v in all_items.items() if allowed_ids is None or k in allowed_ids}
        identity = {"initial": initial.to_dict(), "search": asdict(search), "validation": asdict(validation_policy),
                    "calibration": [s.id for s in calibration], "heldout": [s.id for s in validation],
                    "catalog": [x.to_dict() for x in self.catalog.values()], "p_view": p_view,
                    "observer_probes": any(r is not None for r in self.resolvers), "adaptation": "judge_uniform_vote_v1",
                    "episode_schedule": {"rule": "sample_index_mod_lanes_v1", "lanes": len(self.runners)},
                    "diagnosis_schedule": {"rule": "bounded_trace_workers_v1", "workers": len(self.judges)}}
        self.store.write("experiment.json", identity, immutable=True)
        self.state = self.store.read("checkpoint.json") or {
            "round": 0, "attempt": 0, "harness": initial.to_dict(), "history": [],
            "no_progress_rounds": 0, "previous_profile": None, "profile": None,
            "status": "running", "stop_reason": None}

    def _episode(self, lane, harness, sample, repeat, split):
        state = {"lane": lane, "phase": split, "sample_id": sample.sample_id,
                 "harness_id": harness.id, "repeat": repeat, "status": "running",
                 "started_at": datetime.now(timezone.utc).isoformat()}
        self.store.write(f"workers/{lane}.json", state)
        try:
            key = digest({"harness": harness.id, "sample": sample.id, "repeat": repeat, "split": split})
            path = f"episodes/{key}.json"
            saved = self.store.read(path)
            if saved is not None:
                episode = Episode.from_dict(saved)
            else:
                episode = self.runners[lane].run(harness, sample, repeat)
                if episode.status == "error":
                    self.store.record_error("episode", episode.to_dict())
                    raise RuntimeError("episode execution failed; error artifact saved")
            checked([episode], [sample], harness, repeat)
            if saved is None:
                self.store.write(path, episode.to_dict(), immutable=True)
            state.update(status="completed", cached=saved is not None, episode_status=episode.status)
            return episode, saved is not None
        except BaseException as exc:
            state.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.store.write(f"workers/{lane}.json", state)

    def batch(self, harness: Harness, samples: list[Sample], repeat: int, split: str) -> list[Episode]:
        if len({s.sample_id for s in samples}) != len(samples):
            raise ValueError("duplicate batch samples")
        results, pending = [], {}
        # Stable lanes preserve endpoint assignment across H0, candidates and
        # resumes. Each lane has at most one episode in flight; no backlog can
        # keep starting model calls after another lane reports an error.
        lanes = [iter(samples[i::len(self.runners)]) for i in range(len(self.runners))]
        with ThreadPoolExecutor(max_workers=len(self.runners)) as pool:
            def launch(lane):
                sample = next(lanes[lane], None)
                if sample is not None:
                    pending[pool.submit(self._episode, lane, harness, sample, repeat, split)] = lane
            for lane in range(len(self.runners)):
                launch(lane)
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                free = []
                for future in done:
                    lane = pending.pop(future)
                    episode, cached = future.result()
                    results.append(episode)
                    self.store.write("progress.json", {"phase": split, "completed": len(results), "total": len(samples),
                        "harness_id": harness.id, "sample_id": episode.sample_id, "repeat": repeat,
                        "cached": cached, "status": episode.status, "correct": episode.correct,
                        "lane": lane, "parallel_lanes": len(self.runners)})
                    free.append(lane)
                for lane in free:
                    launch(lane)
        return checked(results, samples, harness, repeat)

    def _diagnosis(self, worker, index, harness, sample, episode, available):
        lane = index % len(self.runners)
        judge = self.judges[worker]
        state = {"worker": worker, "probe_lane": lane, "sample_id": sample.sample_id,
                 "harness_id": harness.id, "round": self.state["round"], "status": "running",
                 "started_at": datetime.now(timezone.utc).isoformat()}
        self.store.write(f"diagnosis_workers/{worker}.json", state)
        try:
            key = digest({"harness": harness.id, "sample": sample.id, "episode": episode.to_dict(),
                          "round": self.state["round"], "catalog": [c.to_dict() for c in available]})
            path = f"diagnoses/{key}.json"
            result = self.store.read(path)
            state["cached"] = result is not None
            if result is None:
                payload = diagnosis_view(episode, sample, harness)
                result = judge.diagnose(payload, available)
                if result.get("status") == "valid" and result.get("failure") == "observer":
                    resolver = self.resolvers[lane]
                    if resolver is None:
                        result["observer_resolution"] = {"status": "unavailable"}
                    else:
                        with self.probe_locks[lane]:
                            result["observer_resolution"] = resolver.resolve(sample, harness, episode, result)
                        if result["observer_resolution"].get("status") == "error":
                            result["status"] = "error"
                    if result.get("status") == "valid":
                        proposal = judge.recommend(payload, result, available)
                        result["observer_proposal"] = proposal
                        if proposal["status"] == "error":
                            result["status"] = "error"
                        else:
                            result.update(candidate_id=proposal["candidate_id"], proposal_reason=proposal["proposal_reason"])
                result["sample_id"] = sample.sample_id
                if result.get("status") == "error":
                    self.store.record_error("diagnosis", result)
                else:
                    self.store.write(path, result, immutable=True)
            state.update(status="completed", diagnosis_status=result.get("status"))
            return result
        except BaseException as exc:
            state.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.store.write(f"diagnosis_workers/{worker}.json", state)

    def diagnose(self, harness: Harness, episodes: list[Episode], available) -> list[dict]:
        failures = [(i, s, e) for i, (s, e) in enumerate(zip(self.calibration, episodes)) if not e.correct]
        remaining, pending, results = iter(failures), {}, {}
        # One active trace per independent Judge; no queued backlog. Unexpected
        # exceptions stop submissions while other in-flight traces save caches.
        with ThreadPoolExecutor(max_workers=len(self.judges)) as pool:
            def launch(worker):
                item = next(remaining, None)
                if item is not None:
                    index, sample, episode = item
                    pending[pool.submit(self._diagnosis, worker, index, harness, sample, episode, available)] = (worker, index)
            for worker in range(len(self.judges)):
                launch(worker)
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                free = []
                for future in done:
                    worker, index = pending.pop(future)
                    results[index] = future.result()
                    self.store.write("progress.json", {"phase": "diagnosis", "completed": len(results),
                        "total": len(failures), "harness_id": harness.id, "round": self.state["round"],
                        "sample_id": results[index]["sample_id"], "parallel_workers": len(self.judges),
                        "status": results[index].get("status")})
                    free.append(worker)
                for worker in free:
                    launch(worker)
        # Completion order must not affect frozen votes, ranking or health gates.
        diagnoses = [results[i] for i, _, _ in failures]
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
                rejected = {h["candidate"] for h in self.state["history"] if not h["validation"]["accepted"]}
                available = [c for c in self.catalog.values() if c.id not in rejected and c.available(current, self.p_view)]
                if not available:
                    return self._finish("catalog_exhausted")
                # Freeze all trace votes once per round. A rejection removes a
                # candidate from the ranking; it never elicits replacement votes.
                round_path = f"recommendations/{self.state['round']:03d}.json"
                recorded = self.store.read(round_path)
                if recorded is None:
                    episodes = self.batch(current, self.calibration, self.state["round"], "calibration")
                    diagnoses = self.diagnose(current, episodes, available)
                    recorded = {"harness_id": current.id, "diagnoses": diagnoses,
                                "catalog": [c.to_dict() for c in available]}
                    self.store.write(round_path, recorded, immutable=True)
                if recorded["harness_id"] != current.id:
                    raise ValueError("round recommendations belong to a different harness")
                diagnoses = recorded["diagnoses"]
                if not diagnoses:
                    return self._finish("no_calibration_failures")
                self.state["profile"] = failure_profile(diagnoses)
                self.store.write(f"profiles/{self.state['round']:03d}.json", self.state["profile"], immutable=True)
                selection = rank_candidates(diagnoses, available)
                slot = f"selections/{self.state['round']:03d}-{self.state['attempt']:02d}.json"
                self.store.write(slot, selection, immutable=True)
                candidate_id = selection["candidate_id"]
                if candidate_id is None:
                    if self._advance_round(False):
                        return self._finish("converged")
                    continue
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
