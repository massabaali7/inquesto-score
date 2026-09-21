"""Optimizers propose better configurations and prove the improvement.

Optimization is multi-objective by default in spirit: "make task success go up"
is only interesting if latency, cost and turn-taking did not get worse. Pass
`constraints` to say what "worse" means, and the optimizer only returns a
configuration that respects them. Or optimize `inquesto_score` directly, which
already folds turn-taking and latency in.

Three searches, all seeded and deterministic:

- `random`      seeded draws. The floor.
- `coordinate`  greedy one-knob-at-a-time from the current config. Cheap and
                hard to beat on small spaces with independent knobs.
- `tpe`         Tree-structured Parzen Estimator: learns which values show up
                in the best trials and samples from them. Handles knobs that
                only pay off together, and larger spaces. No dependencies.

None of them evaluates the same configuration twice, and `seeds > 1` averages
every evaluation over several seeds so the search does not chase noise.
"""

from __future__ import annotations

import itertools
import math
import operator
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..evaluators import Evaluator
from ..program import VoiceProgram
from ..runners import EvalResult, evaluate
from ..score import ScoreSpec
from ..testsets import Testset

# The v0.1 search space. Deliberately small: these are the knobs that move
# conversational quality most, per the turn-taking literature.
SEARCH_SPACE: dict[str, list[Any]] = {
    "model": ["gpt-4o-mini", "gpt-4o", "claude-sonnet"],
    "endpointing_ms": [400, 700, 900, 1100],
    "interrupt_sensitivity": [0.2, 0.5, 0.8],
    "max_context_turns": [8, 12],
    "temperature": [0.2, 0.7],
}

_OPS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
}


@dataclass(frozen=True)
class Constraint:
    """A bound on a metric, e.g. Constraint("latency", "<=", 500)."""

    metric: str
    op: str
    value: float

    @classmethod
    def parse(cls, metric: str, spec: str) -> Constraint:
        """Parse "<= 500" (with or without the space) into a Constraint."""
        s = spec.strip()
        for op in ("<=", ">=", "<", ">"):  # two-char ops first
            if s.startswith(op):
                try:
                    return cls(metric, op, float(s[len(op):].strip()))
                except ValueError:
                    break
        raise ValueError(
            f"bad constraint for {metric!r}: {spec!r}; expected e.g. '<= 500' or '>= 0.9'"
        )

    @classmethod
    def parse_expr(cls, expr: str) -> Constraint:
        """Parse "latency<=500" or "turn_taking >= 0.9" (CLI form)."""
        for op in ("<=", ">=", "<", ">"):
            if op in expr:
                metric, _, rest = expr.partition(op)
                return cls.parse(metric.strip(), op + rest)
        raise ValueError(f"bad constraint {expr!r}; expected e.g. 'latency<=500'")

    def holds(self, metrics: dict[str, float]) -> bool:
        if self.metric not in metrics:
            raise KeyError(f"constraint on unknown metric {self.metric!r}; have {sorted(metrics)}")
        return _OPS[self.op](metrics[self.metric], self.value)

    def __str__(self) -> str:
        return f"{self.metric} {self.op} {self.value:g}"


def parse_constraints(
    spec: dict[str, str] | list[str | Constraint] | None,
) -> list[Constraint]:
    """Accept {"latency": "<= 500"}, ["latency<=500"], or Constraint objects."""
    if not spec:
        return []
    if isinstance(spec, dict):
        return [Constraint.parse(k, v) for k, v in spec.items()]
    return [c if isinstance(c, Constraint) else Constraint.parse_expr(c) for c in spec]


def violations(constraints: list[Constraint], metrics: dict[str, float]) -> list[str]:
    return [f"{c} (got {metrics[c.metric]:.4g})" for c in constraints if not c.holds(metrics)]


@dataclass
class Trial:
    step: int
    overrides: dict[str, Any]
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    violations: list[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    metric: str
    baseline: EvalResult
    best: EvalResult
    best_overrides: dict[str, Any]
    trials: list[Trial] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    # False when no evaluated configuration (baseline included) met the constraints.
    feasible: bool = True
    optimizer: str = ""
    seeds: int = 1

    @property
    def delta(self) -> float:
        return self.best.metric(self.metric) - self.baseline.metric(self.metric)

    @property
    def improved(self) -> bool:
        return bool(self.best_overrides)

    def changes(self) -> dict[str, tuple[Any, Any]]:
        return {
            k: (self.baseline.config[k], v)
            for k, v in self.best_overrides.items()
            if self.baseline.config[k] != v
        }

    def regressions(self, guard: dict[str, float] | None = None) -> list[str]:
        """Guarded metrics that got worse in the winning config.

        `guard` maps metric -> relative tolerance. Direction comes from the
        evaluator: latency going up is a regression, turn_taking going down is.
        """
        out = []
        for name, tol in (guard or {}).items():
            before, after = self.baseline.metrics.get(name), self.best.metrics.get(name)
            if before is None or after is None:
                continue
            higher = self.best.higher_is_better.get(name, True)
            worse = after < before * (1 - tol) if higher else after > before * (1 + tol)
            if worse:
                out.append(f"{name}: {before:.3f} -> {after:.3f}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "optimizer": self.optimizer,
            "seeds": self.seeds,
            "constraints": list(self.constraints),
            "feasible": self.feasible,
            "baseline": self.baseline.to_dict(),
            "best": self.best.to_dict(),
            "changes": dict(self.best_overrides),
            "delta": self.delta,
            "trials": [t.__dict__ for t in self.trials],
        }


def _key(overrides: dict[str, Any]) -> tuple:
    return tuple(sorted(overrides.items()))


class _Search:
    """Bookkeeping shared by every optimizer: evaluate, record, keep the best.

    The incumbent is the baseline only when the baseline meets the constraints;
    otherwise the first feasible candidate wins regardless of score. A
    configuration is never evaluated twice: `try_` returns the earlier trial.
    """

    def __init__(self, program, testset, metric, evaluators, runtime, seed, constraints,
                 on_trial, seeds=1, score_spec=None):
        self.program, self.testset, self.metric = program, testset, metric
        self.evaluators, self.runtime, self.seed = evaluators, runtime, seed
        self.seeds, self.score_spec = seeds, score_spec
        self.cons = parse_constraints(constraints)
        self.on_trial = on_trial
        self.baseline = self._evaluate(program)
        if metric not in self.baseline.metrics:
            raise KeyError(
                f"cannot optimize {metric!r}: not in {sorted(self.baseline.metrics)}"
                + (" (inquesto_score needs every component measured)" if metric == "inquesto_score" else "")
            )
        higher = self.baseline.higher_is_better.get(metric, True)
        self.better = operator.gt if higher else operator.lt
        self.sign = 1.0 if higher else -1.0
        baseline_ok = not violations(self.cons, self.baseline.metrics)
        self.best, self.best_overrides = self.baseline, {}
        self.best_score = self.baseline.metric(metric) if baseline_ok else None
        self.trials: list[Trial] = []
        self.baseline_point = {k: getattr(program.config, k) for k in SEARCH_SPACE}
        self.seen: dict[tuple, Trial] = {}

    def _evaluate(self, program: VoiceProgram) -> EvalResult:
        return evaluate(program, self.testset, self.evaluators, self.runtime, self.seed,
                        seeds=self.seeds, score_spec=self.score_spec)

    def try_(self, overrides: dict[str, Any]) -> Trial:
        """Evaluate one candidate; adopt it if feasible and better than the incumbent."""
        key = _key(overrides)
        if key in self.seen:
            return self.seen[key]
        candidate = self.program.with_config(**overrides)
        res = self._evaluate(candidate)
        score = res.metric(self.metric)
        viol = violations(self.cons, res.metrics)
        trial = Trial(
            step=len(self.trials) + 1, overrides=dict(overrides), score=score,
            metrics=dict(res.metrics), feasible=not viol, violations=viol,
        )
        self.trials.append(trial)
        self.seen[key] = trial
        if self.on_trial:
            self.on_trial(trial)
        if not viol and (self.best_score is None or self.better(score, self.best_score)):
            self.best_score, self.best, self.best_overrides = score, res, dict(overrides)
        return trial

    def is_seen(self, overrides: dict[str, Any]) -> bool:
        """True for an evaluated candidate, or for the baseline's own configuration."""
        if _key(overrides) in self.seen:
            return True
        return all(getattr(self.program.config, k) == v for k, v in overrides.items())

    def result(self, optimizer: str) -> OptimizationResult:
        return OptimizationResult(
            metric=self.metric,
            baseline=self.baseline,
            best=self.best,
            best_overrides=self.best_overrides,
            trials=self.trials,
            constraints=[str(c) for c in self.cons],
            feasible=self.best_score is not None,
            optimizer=optimizer,
            seeds=self.seeds,
        )


def _random_unseen(s: _Search, space: dict[str, list[Any]], rng: random.Random) -> dict | None:
    """A random configuration not evaluated yet, or None if the space is exhausted."""
    for _ in range(200):
        c = {k: rng.choice(v) for k, v in space.items()}
        if not s.is_seen(c):
            return c
    keys = list(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        c = dict(zip(keys, combo))
        if not s.is_seen(c):
            return c
    return None


def _random_neighbour(s: _Search, space: dict[str, list[Any]], rng: random.Random) -> dict | None:
    """An unseen configuration one knob away from the baseline, or None."""
    base = {k: getattr(s.program.config, k) for k in space}
    moves = [(k, v) for k, vals in space.items() for v in vals if v != base[k]]
    rng.shuffle(moves)
    for k, v in moves:
        c = {**base, k: v}
        if not s.is_seen(c):
            return c
    return None


class Optimizer:
    name = "optimizer"

    def optimize(
        self,
        program: VoiceProgram,
        testset: Testset,
        metric: str = "task_success",
        steps: int = 12,
        evaluators: list[Evaluator] | None = None,
        runtime: str = "mock",
        seed: int = 0,
        space: dict[str, list[Any]] | None = None,
        constraints: dict[str, str] | list[str | Constraint] | None = None,
        on_trial: Callable[[Trial], None] | None = None,
        seeds: int = 1,
        score_spec: ScoreSpec | None = None,
    ) -> OptimizationResult:
        search = _Search(program, testset, metric, evaluators, runtime, seed, constraints,
                         on_trial, seeds=seeds, score_spec=score_spec)
        self.search(search, space or SEARCH_SPACE, steps, random.Random(seed))
        return search.result(self.name)

    def search(self, s: _Search, space: dict[str, list[Any]], steps: int,
               rng: random.Random) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class RandomSearch(Optimizer):
    """Seeded random search over SEARCH_SPACE.

    Not clever. It is a floor, not a ceiling. Every candidate is evaluated with
    the same seed as the baseline, so the comparison is paired: the same
    simulated callers, different configuration.
    """

    name = "random"

    def search(self, s, space, steps, rng):
        for _ in range(steps):
            c = _random_unseen(s, space, rng)
            if c is None:
                break
            s.try_(c)


class CoordinateSearch(Optimizer):
    """Greedy one-knob-at-a-time search starting from the current config.

    For each knob in turn, try every other value while holding the rest fixed;
    keep the change if it helps and stays inside the constraints. Sweep again
    until a full pass changes nothing or the step budget runs out. No
    randomness at all, and it starts from a known-good point, so it respects a
    budget far better than random draws. It is greedy: it can miss a gain that
    needs two knobs to move together.
    """

    name = "coordinate"

    def search(self, s, space, steps, rng):
        current = {k: getattr(s.program.config, k) for k in space}
        budget = steps
        while budget > 0:
            improved = False
            for knob, values in space.items():
                for v in values:
                    cand = {**current, knob: v}
                    if v == current[knob] or budget == 0 or s.is_seen(cand):
                        continue
                    trial = s.try_(cand)
                    budget -= 1
                    if s.best_overrides and s.best_overrides == trial.overrides:
                        current[knob] = v
                        improved = True
            if not improved:
                break


class TPESearch(Optimizer):
    """Tree-structured Parzen Estimator over a discrete space, no dependencies.

    After a few random start-up trials, every evaluated configuration is split
    into "good" (the top `gamma` fraction by the objective, feasible only) and
    "bad" (the rest, plus everything infeasible). For each knob, the frequency
    of each value in the good set versus the bad set gives a preference; the
    next candidate is the unseen configuration, out of `n_candidates` draws
    from the good distribution, with the highest good/bad likelihood ratio.

    That is the whole trick. It is what Optuna's default sampler does, reduced
    to categorical knobs, and it finds knobs that only pay off together because
    the good set keeps the combinations, not just the marginals' argmax.

    Defaults were picked against an exhaustive sweep of the 144-config mock
    space over six seeds (tests/test_optimizers.py pins the results): with
    24 evaluations it reaches the true optimum on every seed for a cost-
    constrained objective, a three-constraint objective, and the Inquesto score,
    where greedy coordinate search stalls on 2 of 6 seeds at any budget.
    """

    name = "tpe"

    def __init__(self, n_startup: int = 3, gamma: float = 0.2, n_candidates: int = 96,
                 alpha: float = 1.0) -> None:
        self.n_startup, self.gamma, self.n_candidates, self.alpha = (
            n_startup, gamma, n_candidates, alpha,
        )

    def search(self, s, space, steps, rng):
        # The baseline counts as an observation too.
        points: list[tuple[dict[str, Any], float, bool]] = [
            (s.baseline_point, s.baseline.metric(s.metric),
             not violations(s.cons, s.baseline.metrics)),
        ]
        for _ in range(steps):
            if len(s.trials) < self.n_startup:
                # Start-up trials are one-knob neighbours of the baseline, not
                # uniform draws: under tight constraints most of the space is
                # infeasible and the baseline's neighbourhood is where the
                # feasible configurations are.
                cand = _random_neighbour(s, space, rng) or _random_unseen(s, space, rng)
            else:
                cand = self._propose(s, space, rng, points) or _random_unseen(s, space, rng)
            if cand is None:
                break
            t = s.try_(cand)
            points.append((t.overrides, t.score, t.feasible))

    def _propose(self, s, space, rng, points):
        feasible = sorted((p for p in points if p[2]), key=lambda p: s.sign * p[1], reverse=True)
        n_good = max(1, math.ceil(self.gamma * len(points)))
        good = feasible[:n_good]
        bad = feasible[n_good:] + [p for p in points if not p[2]]
        if not good:
            return None
        weights: dict[str, dict[Any, float]] = {}
        ratio: dict[str, dict[Any, float]] = {}
        for knob, values in space.items():
            wg = {v: self.alpha for v in values}
            wb = {v: self.alpha for v in values}
            for cfg, _, _ in good:
                if cfg.get(knob) in wg:
                    wg[cfg[knob]] += 1
            for cfg, _, _ in bad:
                if cfg.get(knob) in wb:
                    wb[cfg[knob]] += 1
            zg, zb = sum(wg.values()), sum(wb.values())
            weights[knob] = {v: wg[v] / zg for v in values}
            ratio[knob] = {v: math.log(wg[v] / zg) - math.log(wb[v] / zb) for v in values}
        best, best_lr = None, -math.inf
        for _ in range(self.n_candidates):
            c = {k: rng.choices(vals, weights=[weights[k][v] for v in vals])[0]
                 for k, vals in space.items()}
            if s.is_seen(c):
                continue
            lr = sum(ratio[k][c[k]] for k in space)
            if lr > best_lr:
                best, best_lr = c, lr
        return best


OPTIMIZERS: dict[str, type[Optimizer]] = {
    "coordinate": CoordinateSearch,
    "random": RandomSearch,
    "tpe": TPESearch,
}


def get(name: str) -> Optimizer:
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer {name!r}; available: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name]()


__all__ = [
    "OPTIMIZERS",
    "SEARCH_SPACE",
    "Constraint",
    "CoordinateSearch",
    "OptimizationResult",
    "Optimizer",
    "RandomSearch",
    "TPESearch",
    "Trial",
    "get",
    "parse_constraints",
    "violations",
]
