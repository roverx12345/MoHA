"""One executable intervention catalog; no symbolic shadow state."""
from dataclasses import dataclass, replace
from .models import Harness
from .execution import GOALS, CHOICES


@dataclass(frozen=True)
class Intervention:
    id: str
    coordinate: str
    value: str | bool | int | float
    description: str

    def apply(self, base: Harness) -> Harness:
        if self.coordinate.startswith("execution."):
            _, goal, field = self.coordinate.split(".")
            policy = dict(base.execution)
            settings = dict(policy.get(goal, ()))
            settings[field] = self.value
            policy[goal] = tuple(settings.items())
            return replace(base, execution=tuple(policy.items()))
        if self.coordinate == "specialists":
            return replace(base, specialists=tuple(sorted(set(base.specialists) | {self.value})))
        return replace(base, **{self.coordinate: self.value})

    def available(self, base: Harness, p_view: int | None = None) -> bool:
        proposed = self.apply(base)
        if proposed == base:
            return False
        if self.coordinate.startswith("execution."):
            _, goal, field = self.coordinate.split(".")
            if getattr(base.execution_for_goal(goal), field) == self.value:
                return False
            # Source size, duration and token-profile rounding determine media
            # no-ops. Those are checked locally by the fixed-window probe planner.
        return True

    def to_dict(self) -> dict:
        return {"id": self.id, "coordinate": self.coordinate, "value": self.value,
                "description": self.description}


def catalog() -> dict[str, Intervention]:
    modules = {
        "overview": ("overview", "Prefetch a navigation-only overview before planning; it consumes perception budget and supplies coarse regions, not answer evidence."),
        "memory_basic": ("memory", "Expose original result and working-note ledgers through memory_read and memory_note. Preserve complete scoped observations across history truncation without summarizing, ranking or merging. Notes are supplied by the planner."),
        "verification_basic": ("verification", "Expose verify_fresh: one independent prompt-only diagnosis of original text evidence. It uses the shared model-call budget and returns natural text without rule-based coverage or enforced verdicts. Without memory, only currently visible observations are available."),
        "retrieval_basic": ("retrieval_guard", "Enable the existing retrieval-stagnation intervention."),
    }
    items = [Intervention(f"planner.module.{key}", field, True, text)
             for key, (field, text) in modules.items()]
    for goal in GOALS[1:]:
        for field, values in CHOICES.items():
            for value in values:
                items.append(Intervention(f"observer.execution.{goal}.{field}.{value}",
                    f"execution.{goal}.{field}", value,
                    f"Set {field}={value} for {goal}; retain the other execution controls and shared budgets."))
    items.extend(Intervention(f"observer.specialist.{x}", "specialists", x,
                             f"Use the configured {x.upper()} specialist for typed {x.upper()} goals.")
                 for x in ("ocr", "asr"))
    return {item.id: item for item in items}
