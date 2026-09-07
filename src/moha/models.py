"""JSON-safe contracts shared by execution, storage and calibration."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any
from .execution import ExecutionPolicy, GOALS, CHOICES


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Harness:
    max_steps: int = 16
    history_turns: int = 8
    history_tokens: int = 18000
    overview: bool = False
    memory: bool = False
    verification: bool = False
    retrieval_guard: bool = False
    # Sorted pairs make hashes stable and prevent mutation of a frozen config.
    execution: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = (("default", ()),)
    specialists: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("max_steps", "history_turns", "history_tokens"):
            positive_int(getattr(self, name), name)
        for name in ("overview", "memory", "verification", "retrieval_guard"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        pairs = []
        for goal, settings in self.execution:
            if goal not in GOALS or isinstance(settings, str):
                raise ValueError("execution requires source-relative settings; legacy presets need their frozen source")
            values = dict(settings)
            if len(values) != len(settings) or set(values) - set(CHOICES):
                raise ValueError("unknown or duplicate execution field")
            normalized = ExecutionPolicy(**values).to_dict()
            pairs.append((goal, tuple(sorted((k, normalized[k]) for k in values))))
        pairs = tuple(sorted(pairs))
        if len(dict(pairs)) != len(pairs) or "default" not in dict(pairs):
            raise ValueError("execution needs one default and unique goal keys")
        if len(set(self.specialists)) != len(self.specialists) or set(self.specialists) - {"ocr", "asr"}:
            raise ValueError("specialists must be unique ocr/asr capabilities")
        object.__setattr__(self, "execution", pairs)
        object.__setattr__(self, "specialists", tuple(sorted(self.specialists)))

    def execution_for_goal(self, goal: str) -> ExecutionPolicy:
        policies = dict(self.execution)
        return ExecutionPolicy(**{**dict(policies["default"]), **dict(policies.get(goal, ()))})

    def to_dict(self) -> dict:
        return {**asdict(self), "execution": {k: dict(v) for k, v in self.execution}, "specialists": list(self.specialists)}

    @classmethod
    def from_dict(cls, data: dict) -> Harness:
        value = dict(data)
        if "execution" in value:
            value["execution"] = tuple(value["execution"].items())
        if "specialists" in value:
            value["specialists"] = tuple(value["specialists"])
        return cls(**value)

    @property
    def id(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class Sample:
    sample_id: str
    video_id: str
    dataset: str
    asset_id: str
    task: dict
    expected_answer: str

    def __post_init__(self) -> None:
        for key in ("sample_id", "video_id", "dataset", "asset_id", "expected_answer"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"sample requires {key}")
        if not isinstance(self.task, dict) or not isinstance(self.task.get("question"), str):
            raise ValueError("sample requires task.question")
        if self.expected_answer not in self.task.get("options", {}):
            raise ValueError("expected_answer must be a task option")
        object.__setattr__(self, "task", copy.deepcopy(self.task))
        canonical(asdict(self))

    @property
    def id(self) -> str:
        return digest(asdict(self))

    @property
    def video_key(self) -> tuple[str, str]:
        return self.dataset, self.video_id


def check_splits(calibration: list[Sample], validation: list[Sample]) -> None:
    for name, samples in (("calibration", calibration), ("validation", validation)):
        if not samples or len({s.sample_id for s in samples}) != len(samples):
            raise ValueError(f"{name} must be nonempty with unique sample IDs")
    if {s.sample_id for s in calibration} & {s.sample_id for s in validation}:
        raise ValueError("sample overlap across splits")
    if {s.video_key for s in calibration} & {s.video_key for s in validation}:
        raise ValueError("video overlap across splits")


@dataclass
class Episode:
    sample_id: str
    sample_hash: str
    harness_id: str
    repeat: int
    video_key: list[str]
    expected_answer: str
    answer: str | None
    status: str
    events: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.status == "completed" and self.answer == self.expected_answer

    def to_dict(self) -> dict:
        return {"schema": "moha_episode_v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict) -> Episode:
        data = dict(value)
        if data.pop("schema", None) != "moha_episode_v1":
            raise ValueError("unsupported episode schema; explicit migration is required")
        return cls(**data)


@dataclass(frozen=True)
class ValidationPolicy:
    repeats: int = 1
    min_gain: float = 0.0
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    cost_metric: str = "video_tokens"
    max_cost_ratio: float | None = None
    cost_penalty: float = 0.0
    cost_scale: float = 65536.0
    require_positive_interval: bool = False
    require_each_repeat_nonnegative: bool = False

    def __post_init__(self) -> None:
        for k in ("repeats", "bootstrap_samples"):
            positive_int(getattr(self, k), k)
        for k in ("min_gain", "confidence", "cost_penalty", "cost_scale"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{k} must be finite numeric")
        if not 0 < self.confidence < 1 or self.min_gain < 0:
            raise ValueError("invalid validation thresholds")
        if self.max_cost_ratio is not None and (isinstance(self.max_cost_ratio, bool)
                or not isinstance(self.max_cost_ratio, (int, float))
                or not math.isfinite(self.max_cost_ratio) or self.max_cost_ratio <= 0):
            raise ValueError("max_cost_ratio must be null or positive finite numeric")
        if self.cost_penalty < 0 or self.cost_scale <= 0:
            raise ValueError("invalid cost normalization")
        if self.cost_metric not in {"video_tokens", "sampled_frames", "observer_calls", "audio_seconds"}:
            raise ValueError("unsupported measured cost")
        for key in ("require_each_repeat_nonnegative", "require_positive_interval"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean")
