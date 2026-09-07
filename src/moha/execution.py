"""Source-relative observer controls and deterministic budget allocation."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass

EXECUTION_POLICY = "moha_source_relative_v1"
GOALS = ("default", "general", "presence", "attribute", "count", "text", "speech", "relation", "sequence")
CHOICES = {"frames": ("auto", 32, 64, 128), "source_scale": (0.5, 0.75, 1.0),
           "priority": ("temporal", "spatial", "balanced")}


@dataclass(frozen=True)
class ExecutionPolicy:
    frames: str | int = "auto"
    source_scale: float = 1.0
    priority: str = "balanced"

    def __post_init__(self):
        for key, choices in CHOICES.items():
            value = getattr(self, key)
            if isinstance(value, bool) or value not in choices:
                raise ValueError(f"unsupported execution {key}: {value!r}")
        if self.frames != "auto" and type(self.frames) is not int:
            raise ValueError("frames must be auto or a catalog integer")
        object.__setattr__(self, "source_scale", float(self.source_scale))

    def to_dict(self):
        return asdict(self)


def allocate(renderer, media, window, policy):
    """Choose (frames, source edge ratio) before the pinned renderer packs it.

    Search at most 128 frame counts and a declared 10% descending pixel-scale
    ladder. All priorities share that feasible set and the same hard budgets.
    Episode totals retain the pinned runtime's advisory accounting semantics.
    """
    from video_os.core.errors import BudgetExceeded
    start, end = window
    if not 0 <= start < end <= media.duration_seconds:
        raise ValueError("invalid fixed observation window")
    duration = end - start
    target = math.ceil(duration) if policy.frames == "auto" else policy.frames
    target = min(target, 128, renderer.budget.f_view, max(1, math.ceil(duration * media.frame_rate)))
    if renderer.token_profile is None:
        raise ValueError("source-relative allocation requires a calibrated token profile")
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
    return {"policy": EXECUTION_POLICY, "requested": policy.to_dict(),
            "source_resolution": [media.width, media.height], "target_frames": target,
            "frames": frames, "resolution": list(resolution), "video_tokens": tokens,
            "source_scale_realized": min(resolution[0]/media.width, resolution[1]/media.height),
            "window": [start, end], "source_sha256": media.source_sha256,
            "experiment_render": {"requested_frames": frames, "pixel_cap_override": cap}}


def plan_signature(plan):
    from .models import digest
    return digest({k: plan[k] for k in ("source_sha256", "window", "frames", "resolution")})
