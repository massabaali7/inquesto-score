"""Optimizer quality, pinned against an exhaustive sweep of the search space.

The mock runtime is deterministic and the space has 144 configurations, so the
true optimum under any constraint is known. A "strong" optimizer must find it
inside a small budget, never waste an evaluation on a repeat, and beat random
search at equal budget across seeds.
"""

import functools
import itertools
import statistics
import sys
from pathlib import Path

import pytest

from inquesto import evaluate, testsets
from inquesto.optimizers import (
    SEARCH_SPACE,
    CoordinateSearch,
    RandomSearch,
    TPESearch,
    parse_constraints,
    violations,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "support_agent"))
from agent import SupportAgent

TS = testsets.load("support-v1")
COST_BUDGET = {"cost": "<= 0.05"}
TIGHT = {"cost": "<= 0.05", "latency": "<= 950", "turn_taking": ">= 0.94"}


@functools.cache
def sweep():
    """Every configuration's metrics, computed once."""
    keys = list(SEARCH_SPACE)
    out = {}
    for combo in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        ov = dict(zip(keys, combo))
        out[tuple(sorted(ov.items()))] = evaluate(SupportAgent().with_config(**ov), TS).metrics
    return out


def optimum(metric, constraints):
    cons = parse_constraints(constraints)
    feasible = [m[metric] for m in sweep().values() if not violations(cons, m)]
    return max(feasible)


def test_sweep_sanity():
    assert len(sweep()) == 144
    assert optimum("task_success", {}) == pytest.approx(0.95)
    assert optimum("task_success", COST_BUDGET) == pytest.approx(0.90)


@pytest.mark.parametrize("opt,steps", [(CoordinateSearch, 12), (TPESearch, 24)])
def test_finds_the_budget_optimum(opt, steps):
    out = opt().optimize(SupportAgent(), TS, steps=steps, constraints=COST_BUDGET)
    assert out.feasible
    assert out.best.metric("task_success") == pytest.approx(optimum("task_success", COST_BUDGET))


@pytest.mark.parametrize("opt", [CoordinateSearch, TPESearch])
def test_finds_the_optimum_under_three_constraints(opt):
    out = opt().optimize(SupportAgent(), TS, steps=30, constraints=TIGHT)
    assert out.best.metric("task_success") == pytest.approx(optimum("task_success", TIGHT))
    for c in parse_constraints(TIGHT):
        assert c.holds(out.best.metrics)


def test_tpe_optimizes_the_inquesto_score_directly():
    out = TPESearch().optimize(SupportAgent(), TS, metric="inquesto_score", steps=30)
    assert out.metric == "inquesto_score"
    assert out.best.metric("inquesto_score") >= out.baseline.metric("inquesto_score")
    assert out.best.metric("inquesto_score") == pytest.approx(optimum("inquesto_score", {}))


@pytest.mark.parametrize("opt", [RandomSearch, CoordinateSearch, TPESearch])
def test_never_evaluates_the_same_config_twice(opt):
    out = opt().optimize(SupportAgent(), TS, steps=40)
    keys = [tuple(sorted(t.overrides.items())) for t in out.trials]
    assert len(keys) == len(set(keys))
    assert all(t.step == i + 1 for i, t in enumerate(out.trials))


@pytest.mark.parametrize("opt", [RandomSearch, CoordinateSearch, TPESearch])
def test_is_deterministic(opt):
    a = opt().optimize(SupportAgent(), TS, steps=10, constraints=COST_BUDGET, seed=3)
    b = opt().optimize(SupportAgent(), TS, steps=10, constraints=COST_BUDGET, seed=3)
    assert [t.overrides for t in a.trials] == [t.overrides for t in b.trials]
    assert a.best_overrides == b.best_overrides


def test_tpe_beats_random_at_equal_budget_across_seeds():
    """Mean best objective over seeds 0..5, 16 evaluations each, cost budget."""
    def best_for(opt, seed):
        return opt().optimize(SupportAgent(), TS, steps=16, constraints=COST_BUDGET,
                              seed=seed).best.metric("task_success")

    tpe = [best_for(TPESearch, s) for s in range(6)]
    rnd = [best_for(RandomSearch, s) for s in range(6)]
    assert statistics.mean(tpe) > statistics.mean(rnd), (tpe, rnd)
    assert min(tpe) >= min(rnd)


def test_search_stops_when_the_space_is_exhausted():
    tiny = {"temperature": [0.2, 0.7], "max_context_turns": [8, 12]}
    for opt in (RandomSearch, TPESearch, CoordinateSearch):
        out = opt().optimize(SupportAgent(), TS, steps=50, space=tiny)
        assert len(out.trials) <= 3, opt  # 4 configs minus the baseline itself


def test_optimizing_an_unmeasured_metric_is_a_clear_error():
    with pytest.raises(KeyError, match="cannot optimize"):
        TPESearch().optimize(SupportAgent(), TS, metric="does_not_exist", steps=1)


# --- seeds ----------------------------------------------------------------------


def test_multi_seed_evaluation_averages_over_seeds():
    singles = [evaluate(SupportAgent(), TS, seed=s).metrics["task_success"] for s in range(3)]
    r = evaluate(SupportAgent(), TS, seed=0, seeds=3)
    assert r.seeds == 3 and r.n_scenarios == 40
    assert r.metrics["task_success"] == pytest.approx(statistics.mean(singles))
    assert min(singles) <= r.metrics["task_success"] <= max(singles)


def test_optimizer_with_seeds_reports_averaged_metrics():
    out = TPESearch().optimize(SupportAgent(), TS, steps=6, seeds=2, constraints=COST_BUDGET)
    assert out.seeds == 2 and out.baseline.seeds == 2 and out.best.seeds == 2
    two = evaluate(SupportAgent(), TS, seeds=2).metrics["task_success"]
    assert out.baseline.metric("task_success") == pytest.approx(two)


def test_result_records_the_optimizer_name():
    out = CoordinateSearch().optimize(SupportAgent(), TS, steps=2)
    assert out.optimizer == "coordinate"
    assert out.to_dict()["optimizer"] == "coordinate" and out.to_dict()["seeds"] == 1


@functools.cache
def sweep_seed(seed):
    keys = list(SEARCH_SPACE)
    return [evaluate(SupportAgent().with_config(**dict(zip(keys, combo))), TS, seed=seed).metrics
            for combo in itertools.product(*(SEARCH_SPACE[k] for k in keys))]


@pytest.mark.parametrize("metric,constraints", [
    ("task_success", COST_BUDGET),
    ("task_success", TIGHT),
    ("inquesto_score", {}),
])
def test_tpe_hits_the_true_optimum_on_every_seed_with_24_evaluations(metric, constraints):
    """The benchmark behind the default TPE settings, pinned."""
    cons = parse_constraints(constraints)
    for seed in range(6):
        optimum = max(m[metric] for m in sweep_seed(seed) if not violations(cons, m))
        out = TPESearch().optimize(SupportAgent(), TS, metric=metric, steps=24, seed=seed,
                                   constraints=constraints or None)
        assert out.best.metric(metric) == pytest.approx(optimum), f"seed {seed}"
