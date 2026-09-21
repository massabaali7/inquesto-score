"""The Inquesto score: formula, bounds, budgets, and honesty about missing parts."""

import sys
from pathlib import Path

import pytest

from inquesto import evaluate, testsets
from inquesto.score import ScoreSpec, inquesto_score, ramp

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "support_agent"))
from agent import SupportAgent


def test_ramp_boundaries_and_slope():
    assert ramp(100, 500, 2000) == 1.0
    assert ramp(500, 500, 2000) == 1.0
    assert ramp(2000, 500, 2000) == 0.0
    assert ramp(9999, 500, 2000) == 0.0
    assert ramp(1250, 500, 2000) == pytest.approx(0.5)


def test_formula_is_the_documented_product():
    m = {"task_success": 0.9, "turn_taking": 0.95, "latency": 1250.0, "cost": 0.04}
    s = inquesto_score(m)
    assert s.value == pytest.approx(100 * 0.9 * 0.95 * 0.5, abs=0.01)
    assert s.components == {"task_success": 0.9, "turn_taking": 0.95, "speed": 0.5}
    assert s.formula == "100 x task_success x turn_taking x speed"
    assert s.explain().startswith("42.8 = 100 x 0.90 x 0.95 x 0.50")


def test_cost_enters_only_with_a_budget():
    m = {"task_success": 1.0, "turn_taking": 1.0, "latency": 100.0, "cost": 0.10}
    assert inquesto_score(m).value == 100.0
    spec = ScoreSpec(cost_target_usd=0.05, cost_zero_usd=0.15)
    s = inquesto_score(m, spec)
    assert s.components["thrift"] == pytest.approx(0.5)
    assert s.value == pytest.approx(50.0)
    assert "thrift" in s.formula


def test_missing_component_means_no_number():
    s = inquesto_score({"task_success": 1.0, "latency": 100.0})
    assert s.value is None
    assert s.missing == ["turn_taking"]
    assert s.explain() == "n/a: turn_taking not measured"
    spec = ScoreSpec(cost_target_usd=0.05, cost_zero_usd=0.15)
    assert inquesto_score({"task_success": 1, "turn_taking": 1, "latency": 1}, spec).missing == ["cost"]


def test_score_is_bounded_and_monotone():
    base = {"task_success": 0.8, "turn_taking": 0.9, "latency": 700.0}
    s0 = inquesto_score(base).value
    assert 0 <= s0 <= 100
    assert inquesto_score({**base, "task_success": 0.9}).value > s0
    assert inquesto_score({**base, "turn_taking": 0.95}).value > s0
    assert inquesto_score({**base, "latency": 400.0}).value > s0
    assert inquesto_score({**base, "task_success": 0.0}).value == 0.0


def test_spec_parsing():
    spec = ScoreSpec.parse("latency=300:1500,cost=0.05:0.20")
    assert (spec.latency_target_ms, spec.latency_zero_ms) == (300, 1500)
    assert (spec.cost_target_usd, spec.cost_zero_usd) == (0.05, 0.20)
    assert ScoreSpec.parse("latency=400:1000") == ScoreSpec(400, 1000)
    assert ScoreSpec.parse(None) == ScoreSpec()
    for bad in ("latency=500", "speed=1:2", "latency=2000:500", "cost=a:b"):
        with pytest.raises(ValueError):
            ScoreSpec.parse(bad)
    assert "cost not scored" in ScoreSpec().describe()


def test_evaluate_reports_the_score_as_a_metric():
    ts = testsets.load("support-v1")
    r = evaluate(SupportAgent(), ts)
    assert "inquesto_score" in r.metrics
    assert r.units["inquesto_score"] == "pts"
    assert r.higher_is_better["inquesto_score"] is True
    assert r.score["value"] == r.metrics["inquesto_score"]
    expected = inquesto_score({k: r.metrics[k] for k in ("task_success", "turn_taking", "latency")})
    assert r.metrics["inquesto_score"] == expected.value
    # per caller style too, for the leaderboard breakdown
    assert all("inquesto_score" in m for m in r.breakdown.values())


def test_evaluate_omits_the_score_when_a_component_is_unmeasured():
    from inquesto import adapters
    from inquesto.program import Conversation

    class NoAudio:
        name = "no-audio"

        def run(self, program, scenario, seed):
            return Conversation(scenario_id=scenario.id, task_completed=True,
                                response_latencies_ms=[300], metadata={"unmeasured": ["turn_taking"]})

    adapters._REGISTRY["no-audio"] = NoAudio
    try:
        r = evaluate(SupportAgent(), testsets.load("support-v1"), runtime="no-audio")
    finally:
        del adapters._REGISTRY["no-audio"]
    assert "inquesto_score" not in r.metrics
    assert r.score["value"] is None and r.score["missing"] == ["turn_taking"]
