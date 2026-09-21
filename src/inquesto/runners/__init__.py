"""The evaluation runner: conversations in, measured evidence out."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..adapters import get as get_runtime
from ..evaluators import Evaluator, Score, default_suite
from ..program import Conversation, VoiceProgram
from ..score import ScoreSpec, inquesto_score
from ..testsets import Testset


@dataclass
class EvalResult:
    program: str
    testset: str
    config_fingerprint: str
    metrics: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    higher_is_better: dict[str, bool] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)
    # metric aggregates per caller style: {"hesitant": {"task_success": 0.6, ...}, ...}
    breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    # Evaluators the runtime could not measure (e.g. turn_taking without audio).
    # They are dropped from `metrics` rather than reported as a fake number.
    unmeasured: list[str] = field(default_factory=list)
    # The Inquesto score (value, components, missing, formula); value is also in
    # metrics["inquesto_score"] when every component was measured.
    score: dict[str, Any] = field(default_factory=dict)
    n_scenarios: int = 0
    seed: int = 0
    seeds: int = 1  # evaluations averaged over seed .. seed+seeds-1
    config: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> float:
        if name not in self.metrics:
            raise KeyError(f"no metric {name!r}; have {sorted(self.metrics)}")
        return self.metrics[name]

    def failure_summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.failures:
            out[f["reason"]] = out.get(f["reason"], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def failures_by_evaluator(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.failures:
            out[f["evaluator"]] = out.get(f["evaluator"], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def worst_scenarios(self, n: int = 3) -> list[tuple[str, int]]:
        """Scenarios with the most failed checks, worst first."""
        out: dict[str, int] = {}
        for f in self.failures:
            out[f["scenario"]] = out.get(f["scenario"], 0) + 1
        ranked = sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:n]

    def worst_style(self, metric: str = "task_success") -> str | None:
        """The caller style where `metric` is weakest, or None if unknown."""
        rows = {s: m[metric] for s, m in self.breakdown.items() if metric in m}
        if not rows:
            return None
        better = self.higher_is_better.get(metric, True)
        return min(rows, key=rows.get) if better else max(rows, key=rows.get)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    program: VoiceProgram,
    testset: Testset,
    evaluators: list[Evaluator] | None = None,
    runtime: str = "mock",
    seed: int = 0,
    on_conversation: Callable[[Conversation], None] | None = None,
    seeds: int = 1,
    score_spec: ScoreSpec | None = None,
) -> EvalResult:
    """Run every scenario, score every conversation, aggregate.

    `on_conversation` is called with each Conversation as soon as it finishes,
    so transcripts from a slow real runtime can be saved or inspected before
    the whole suite is done. `seeds > 1` runs the whole suite once per seed
    (seed, seed+1, ...) and aggregates over all of them, which is how you stop
    an optimizer from chasing single-seed noise.
    """
    evs = evaluators or default_suite()
    rt = get_runtime(runtime)
    if seeds < 1:
        raise ValueError("seeds must be >= 1")

    suite = list(testset)
    scenarios = suite * seeds
    convs: list[Conversation] = []
    for i, s in enumerate(scenarios):
        conv = rt.run(program, s, seed + i // len(suite))
        if on_conversation:
            on_conversation(conv)
        convs.append(conv)
    per_eval: dict[str, list[Score]] = {e.name: [] for e in evs}
    per_style: dict[str, dict[str, list[Score]]] = {}
    failures: list[dict[str, str]] = []

    skipped: set[str] = set()
    for scenario, conv in zip(scenarios, convs):
        style = scenario.caller_style
        bucket = per_style.setdefault(style, {e.name: [] for e in evs})
        unmeasured_here = set(conv.metadata.get("unmeasured", ()))
        for e in evs:
            if e.name in unmeasured_here:
                skipped.add(e.name)
                continue
            sc = e.score_one(conv)
            per_eval[e.name].append(sc)
            bucket[e.name].append(sc)
            if sc.failure:
                failures.append(
                    {
                        "scenario": conv.scenario_id,
                        "evaluator": e.name,
                        "reason": sc.failure,
                        "caller_style": style,
                    }
                )

    measured = [e for e in evs if per_eval[e.name]]
    metrics = {e.name: e.aggregate(per_eval[e.name]) for e in measured}
    units = {e.name: e.unit for e in measured}
    higher = {e.name: e.higher_is_better for e in measured}
    breakdown = {
        style: {e.name: e.aggregate(scores[e.name]) for e in measured if scores[e.name]}
        for style, scores in sorted(per_style.items())
    }

    # The Inquesto score is a derived metric: present only when its parts are.
    sc = inquesto_score(metrics, score_spec)
    if sc.value is not None:
        metrics["inquesto_score"] = sc.value
        units["inquesto_score"] = "pts"
        higher["inquesto_score"] = True
        for style, m in breakdown.items():
            style_score = inquesto_score(m, score_spec)
            if style_score.value is not None:
                m["inquesto_score"] = style_score.value

    return EvalResult(
        program=program.name,
        testset=testset.name,
        config_fingerprint=program.config.fingerprint(),
        metrics=metrics,
        units=units,
        higher_is_better=higher,
        failures=failures,
        breakdown=breakdown,
        unmeasured=sorted(skipped),
        score=sc.to_dict(),
        n_scenarios=len(suite),
        seed=seed,
        seeds=seeds,
        config=asdict(program.config),
    )


__all__ = ["EvalResult", "evaluate"]
