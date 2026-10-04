"""Source-relative observer controls and deterministic budget allocation."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass

EXECUTION_POLICY = "moha_sampling_density_v2"
FRAME_CAP = 128
SAMPLING_RATES = (0.5, 1.0, 2.0)
MIN_DETAIL_FRACTION = 0.25
GOALS = ("default", "general", "presence", "attribute", "count", "text", "speech", "relation", "sequence")
CHOICES = {"frames": ("auto", 32, 64, 128), "source_scale": (0.5, 0.75, 1.0),
           "priority": ("temporal", "spatial", "balanced"), "target_fps": SAMPLING_RATES}


@dataclass(frozen=True)
class ExecutionPolicy:
    frames: str | int = "auto"
    source_scale: float = 1.0
    priority: str = "balanced"
    target_fps: float | None = None

    def __post_init__(self):
        for key, choices in CHOICES.items():
            value = getattr(self, key)
            if key == "target_fps" and value is None:
                continue
            if isinstance(value, bool) or value not in choices:
                raise ValueError(f"unsupported execution {key}: {value!r}")
        if self.frames != "auto" and type(self.frames) is not int:
            raise ValueError("frames must be auto or a catalog integer")
        object.__setattr__(self, "source_scale", float(self.source_scale))
        if self.target_fps is not None:
            if self.frames != "auto":
                raise ValueError("choose target_fps or fixed frames, not both")
            object.__setattr__(self, "target_fps", float(self.target_fps))

    @property
    def sampling_rate(self):
        # Fixed-frame primitives remain available for explicit renderer work.
        return (self.target_fps if self.target_fps is not None else 1.0) if self.frames == "auto" else None

    @property
    def max_window_seconds(self):
        """Maximum source-time window that can honor a rate-based frame cap."""
        if self.frames != "auto":
            return None
        return FRAME_CAP / self.sampling_rate

    def to_dict(self):
        values = asdict(self)
        if self.target_fps is None:
            values.pop("target_fps")
        return values

    def sampling_request(self, duration):
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("window duration must be positive finite seconds")
        requested = math.ceil(duration * self.sampling_rate) if self.frames == "auto" else self.frames
        return {"window_duration": duration, "target_fps": self.sampling_rate,
                "requested_frames": requested, "target_frames": min(requested, FRAME_CAP),
                "frame_cap": FRAME_CAP, "frame_cap_hit": requested > FRAME_CAP}


def allocate(renderer, media, window, policy):
    """Choose (frames, source edge ratio) before the pinned renderer packs it.

    Search at most 128 frame counts and a declared 10% descending pixel-scale
    ladder. All priorities share that feasible set and the same hard budgets.
    Episode totals retain the pinned runtime's advisory accounting semantics.
    """
    from flat.core.errors import BudgetExceeded
    start, end = window
    if not 0 <= start < end <= media.duration_seconds:
        raise ValueError("invalid fixed observation window")
    duration = end - start
    request = policy.sampling_request(duration)
    target = min(request["target_frames"], renderer.budget.f_view,
                 max(1, math.ceil(duration * media.frame_rate)))
    if renderer.token_profile is None:
        raise ValueError("source-relative allocation requires a calibrated token profile")
    # Share a meaningful quality floor across priorities. Use the best shape
    # allowed by this source/scale and the hard single-frame pixel envelope,
    # so high-resolution sources remain usable under a smaller p_view budget.
    ceiling = renderer._output_resolution(media.width, media.height, frames=1,
        allow_upscale=False, pixel_cap_override=max(4, math.floor(
            media.width * media.height * policy.source_scale ** 2)))
    minimum = tuple(max(2, int(edge * MIN_DETAIL_FRACTION) // 2 * 2) for edge in ceiling)
    caps = []
    for step in range(128):
        scale = policy.source_scale * 0.9 ** step
        cap = max(4, math.floor(media.width * media.height * scale ** 2))
        if not caps or cap != caps[-1]:
            caps.append(cap)
        if cap == 4:
            break
    best = None
    seen = set()
    for frames in range(1, target + 1):
        for cap in caps:
            try:
                resolution = renderer._output_resolution(media.width, media.height,
                    frames=frames, allow_upscale=False, pixel_cap_override=cap)
            except BudgetExceeded:
                continue
            if any(actual < floor for actual, floor in zip(resolution, minimum)):
                continue
            if (frames, resolution) in seen:
                continue
            seen.add((frames, resolution))
            tokens = renderer.token_profile.estimate_adaptive_video(duration,
                frames=frames, pixels_per_frame=resolution[0]*resolution[1], include_audio=False,
                frame_width=resolution[0], frame_height=resolution[1])
            if tokens > renderer.budget.b_video:
                continue
            spatial = min(resolution[0]/media.width, resolution[1]/media.height) / policy.source_scale
            temporal = frames / target
            if policy.priority == "temporal":
                rank = (temporal, spatial)
            elif policy.priority == "spatial":
                rank = (spatial, temporal)
            else:
                rank = (min(temporal, spatial), temporal + spatial, spatial, temporal)
            if best is None or rank > best[0]:
                best = (rank, frames, resolution, cap, tokens)
    if best is None:
        raise BudgetExceeded("no source-relative visual packet fits the shared budget")
    _, frames, resolution, cap, tokens = best
    return {"policy": EXECUTION_POLICY, "requested": policy.to_dict(), **request,
            "source_resolution": [media.width, media.height], "allocation_frame_limit": target,
            "minimum_resolution": list(minimum), "minimum_detail_fraction": MIN_DETAIL_FRACTION,
            "frames": frames, "resolution": list(resolution), "video_tokens": tokens,
            "source_scale_realized": min(resolution[0]/media.width, resolution[1]/media.height),
            "window": [start, end], "source_sha256": media.source_sha256,
            "experiment_render": {"requested_frames": frames, "pixel_cap_override": cap}}


def plan_signature(plan):
    from .models import digest
    return digest({k: plan[k] for k in ("source_sha256", "window", "frames", "resolution")})
