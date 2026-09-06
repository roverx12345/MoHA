"""One executable intervention catalog; no symbolic shadow state."""
from dataclasses import dataclass, replace
from .models import Harness


PRESETS = {
    "default": (1.0, 384),
    "dense_temporal": (2.0, 384),
    "high_resolution": (1.0, 768),
}
GOAL_PRESETS = {
    "general": ("dense_temporal", "high_resolution"),
    "presence": ("dense_temporal", "high_resolution"),
    "attribute": ("high_resolution",),
    "count": ("dense_temporal", "high_resolution"),
    "text": ("high_resolution",),
    "relation": ("high_resolution", "dense_temporal"),
    "sequence": ("dense_temporal",),
}


@dataclass(frozen=True)
class Intervention:
    id: str
    coordinate: str
    value: str | bool
    description: str

    def apply(self, base: Harness) -> Harness:
        if self.coordinate.startswith("execution."):
            goal = self.coordinate.split(".", 1)[1]
            policy = dict(base.execution)
            policy[goal] = self.value
            return replace(base, execution=tuple(policy.items()))
        if self.coordinate == "specialists":
            return replace(base, specialists=tuple(sorted(set(base.specialists) | {self.value})))
        return replace(base, **{self.coordinate: self.value})

    def available(self, base: Harness, p_view: int | None = None) -> bool:
        proposed = self.apply(base)
        if proposed == base:
            return False
        if self.coordinate.startswith("execution."):
            goal = self.coordinate.split(".", 1)[1]
            old = dict(base.execution)
            old_fps, old_res = PRESETS[old.get(goal, old["default"])]
            new_fps, new_res = PRESETS[str(self.value)]
            if (old_fps, old_res) == (new_fps, new_res):
                return False
            # Proof valid for every aspect ratio >= 1 and every frame count:
            # both requested pixel caps exceed the same hard per-frame cap.
            if p_view is not None and old_fps == new_fps and min(old_res, new_res) ** 2 >= p_view:
                return False
        return True

    def to_dict(self) -> dict:
        return {"id": self.id, "coordinate": self.coordinate, "value": self.value,
                "description": self.description}


def catalog() -> dict[str, Intervention]:
    modules = {
        "overview": ("overview", "Prefetch a navigation-only overview before planning; it consumes perception budget and supplies coarse regions, not answer evidence."),
        "memory_basic": ("memory", "Keep a compact source-linked evidence bank across bounded history truncation; it preserves acquired claims but does not acquire or verify new evidence."),
        "verification_basic": ("verification", "Expose the advisory evidence ledger and coverage/conflict feedback as planner context; never force a verification tool call or block answers."),
        "retrieval_basic": ("retrieval_guard", "Enable the existing retrieval-stagnation intervention."),
    }
    items = [Intervention(f"planner.module.{key}", field, True, text)
             for key, (field, text) in modules.items()]
    for goal, presets in {"default": tuple(PRESETS), **GOAL_PRESETS}.items():
        for preset in presets:
            items.append(Intervention(f"observer.execution.{goal}.{preset}", f"execution.{goal}",
                                      preset, f"Use {preset} for {goal} observations."))
    items.extend(Intervention(f"observer.specialist.{x}", "specialists", x,
                             f"Use the configured {x.upper()} specialist for typed {x.upper()} goals.")
                 for x in ("ocr", "asr"))
    return {item.id: item for item in items}
