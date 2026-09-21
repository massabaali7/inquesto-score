"""The Inquesto score: one number for the leaderboard, with the formula on the tin.

    inquesto_score = 100 x task_success x turn_taking x speed x thrift

- task_success and turn_taking are the evaluator values (0..1).
- speed is 1.0 when median response latency is at or under the target, 0.0 at
  or over the zero point, linear in between. Defaults: 500 ms -> 2000 ms.
- thrift does the same for cost per conversation, and only if a cost budget is
  given; without one, cost is shown beside the score but does not enter it.

Multiplicative on purpose: an agent that resolves every call but talks over
the caller a third of the time is a 67, not a 92. An agent that is slow loses
proportionally. Nothing is hidden in a weight table.

A score is only computed when every component was measured. A runtime that
cannot measure turn-taking (no audio) gets "n/a" and a reason, not a number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoreSpec:
    """Budgets that turn latency and cost into 0..1 factors."""

    latency_target_ms: float = 500.0
    latency_zero_ms: float = 2000.0
    cost_target_usd: float | None = None
    cost_zero_usd: float | None = None

    @classmethod
    def parse(cls, text: str | None) -> ScoreSpec:
        """Parse "latency=500:2000,cost=0.05:0.20"; either part may be omitted."""
        if not text:
            return cls()
        kw: dict[str, float] = {}
        for part in text.split(","):
            name, sep, rng = part.strip().partition("=")
            lo, sep2, hi = rng.partition(":")
            if not sep or not sep2 or name not in ("latency", "cost"):
                raise ValueError(
                    f"bad score spec {part!r}; expected latency=TARGET:ZERO or cost=TARGET:ZERO"
                )
            try:
                target, zero = float(lo), float(hi)
            except ValueError as e:
                raise ValueError(f"bad score spec {part!r}: numbers expected") from e
            if not zero > target >= 0:
                raise ValueError(f"bad score spec {part!r}: need 0 <= target < zero")
            unit = "ms" if name == "latency" else "usd"
            kw[f"{name}_target_{unit}"] = target
            kw[f"{name}_zero_{unit}"] = zero
        return cls(**kw)

    @property
    def scores_cost(self) -> bool:
        return self.cost_target_usd is not None and self.cost_zero_usd is not None

    def describe(self) -> str:
        s = f"speed: full credit <= {self.latency_target_ms:g}ms, none >= {self.latency_zero_ms:g}ms"
        if self.scores_cost:
            s += f"; thrift: full credit <= ${self.cost_target_usd:g}, none >= ${self.cost_zero_usd:g}"
        else:
            s += "; cost not scored (no budget given)"
        return s


def ramp(x: float, target: float, zero: float) -> float:
    """1.0 at or under target, 0.0 at or over zero, linear between."""
    if x <= target:
        return 1.0
    if x >= zero:
        return 0.0
    return 1.0 - (x - target) / (zero - target)


@dataclass
class InquestoScore:
    value: float | None  # 0..100, or None when a component is missing
    components: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    spec: ScoreSpec = field(default_factory=ScoreSpec)

    @property
    def formula(self) -> str:
        parts = ["100", "task_success", "turn_taking", "speed"]
        if self.spec.scores_cost:
            parts.append("thrift")
        return " x ".join(parts)

    def explain(self) -> str:
        if self.value is None:
            return "n/a: " + ", ".join(f"{m} not measured" for m in self.missing)
        c = self.components
        s = f"{self.value:.1f} = 100 x {c['task_success']:.2f} x {c['turn_taking']:.2f} x {c['speed']:.2f}"
        if "thrift" in c:
            s += f" x {c['thrift']:.2f}"
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "components": dict(self.components),
            "missing": list(self.missing),
            "formula": self.formula,
            "spec": asdict(self.spec),
        }


def inquesto_score(metrics: dict[str, float], spec: ScoreSpec | None = None) -> InquestoScore:
    """Compute the score from an evaluation's metrics, or say what is missing."""
    spec = spec or ScoreSpec()
    needed = ["task_success", "turn_taking", "latency"] + (["cost"] if spec.scores_cost else [])
    missing = [m for m in needed if m not in metrics]
    if missing:
        return InquestoScore(value=None, missing=missing, spec=spec)
    comp = {
        "task_success": float(metrics["task_success"]),
        "turn_taking": float(metrics["turn_taking"]),
        "speed": ramp(float(metrics["latency"]), spec.latency_target_ms, spec.latency_zero_ms),
    }
    if spec.scores_cost:
        comp["thrift"] = ramp(float(metrics["cost"]), spec.cost_target_usd, spec.cost_zero_usd)
    value = 100.0
    for v in comp.values():
        value *= v
    return InquestoScore(value=round(value, 2), components=comp, spec=spec)


__all__ = ["InquestoScore", "ScoreSpec", "inquesto_score", "ramp"]
