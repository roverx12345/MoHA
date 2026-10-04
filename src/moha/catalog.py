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
        try:
            proposed = self.apply(base)
        except ValueError:
            return False  # A rate edit cannot overwrite an explicit frame primitive.
        if proposed == base:
            return False
        if self.coordinate.startswith("execution."):
            _, goal, field = self.coordinate.split(".")
            policy = base.execution_for_goal(goal)
            current = policy.sampling_rate if field == "target_fps" else getattr(policy, field)
            if current == self.value:
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
        "memory_basic": ("memory", "Automatically restore original perceptual evidence omitted from bounded history, before recent interaction. The bounded history is expanded to the planner step limit (16 by default). No memory tools or planner notes. Use at most 6000 tokens within the shared history allowance; restore the complete missing set or explicitly report capacity limits. Preserve scoped claims and conflicts, deduplicating only exact copies in the view."),
        "verification_basic": ("verification", "Enable one evidence-grounded, option-wise outcome audit as an extra call outside the planner step budget. It runs once when a candidate is submitted, compares every complete option against observations visible in the actual planner context, and may replace a contradicted or insufficient candidate with the audit's supported option. verify_fresh may use the same single audit allowance early. No extra planner loop, direct memory access or corrective perception."),
        "retrieval_basic": ("retrieval_guard", "Enable the existing retrieval-stagnation intervention."),
    }
    items = [Intervention(f"planner.module.{key}", field, True, text)
             for key, (field, text) in modules.items()]
    for goal in GOALS[1:]:
        for field, values in CHOICES.items():
            if field == "frames":
                continue  # Renderer primitives are not semantic sampling policies.
            for value in values:
                items.append(Intervention(f"observer.execution.{goal}.{field}.{value}",
                    f"execution.{goal}.{field}", value,
                    f"Set {field}={value} for {goal}; retain the other execution controls and shared budgets."))
    items.extend(Intervention(f"observer.specialist.{x}", "specialists", x,
                             f"Use the configured {x.upper()} specialist for typed {x.upper()} goals.")
                 for x in ("ocr", "asr"))
    return {item.id: item for item in items}
