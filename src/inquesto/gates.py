"""Regression gates: block the release when the conversation got worse.

This is the piece teams run forever. `inquesto gate` compares a fresh evaluation
against a committed baseline artifact and exits non-zero when a guarded metric
moves the wrong way by more than its tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runners import EvalResult

# Metrics where a *lower* number is the better one.
LOWER_IS_BETTER = {"latency", "cost"}

DEFAULT_TOLERANCES = {
    "task_success": 0.01, "turn_taking": 0.02, "latency": 0.10, "cost": 0.15, "inquesto_score": 2.0,
}


@dataclass
class GateVerdict:
    passed: bool
    violations: list[str] = field(default_factory=list)
    checked: dict[str, tuple[float, float]] = field(default_factory=dict)

    def report(self) -> str:
        lines = []
        for m, (before, after) in self.checked.items():
            arrow = "->"
            flag = "FAIL" if any(m in v for v in self.violations) else "ok"
            lines.append(f"  [{flag:4}] {m}: {before:.4g} {arrow} {after:.4g}")
        head = "PASS: conversation quality held" if self.passed else "BLOCKED: quality regressed"
        return head + "\n" + "\n".join(lines)


def save_baseline(result: EvalResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_dict(), indent=2))
    return p


def load_baseline(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def check(
    current: EvalResult,
    baseline: dict[str, Any] | EvalResult,
    tolerances: dict[str, float] | None = None,
) -> GateVerdict:
    base_metrics = (
        baseline.metrics if isinstance(baseline, EvalResult) else baseline.get("metrics", {})
    )
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}

    violations: list[str] = []
    checked: dict[str, tuple[float, float]] = {}

    for name, before in base_metrics.items():
        if name not in current.metrics:
            continue
        after = current.metrics[name]
        checked[name] = (before, after)
        t = tol.get(name, 0.05)
        # Direction comes from the evaluator that produced the number; the
        # LOWER_IS_BETTER set is the fallback for metrics the runner didn't tag.
        lower = not current.higher_is_better.get(name, name not in LOWER_IS_BETTER)
        if lower:
            if after > before * (1 + t):
                violations.append(f"{name} rose {before:.4g} -> {after:.4g} (tol {t:.0%})")
        else:
            if after < before - t:
                violations.append(f"{name} fell {before:.4g} -> {after:.4g} (tol {t:.0%})")

    return GateVerdict(passed=not violations, violations=violations, checked=checked)


__all__ = ["DEFAULT_TOLERANCES", "GateVerdict", "check", "load_baseline", "save_baseline"]
