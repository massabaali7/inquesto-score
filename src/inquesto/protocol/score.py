"""The pass rule, the rate, its interval, the four views, and the record."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import spec


@dataclass
class Event:
    """One failure event in one call."""

    type: str
    severity: int
    at_ms: int = 0
    detail: str = ""


@dataclass
class CallResult:
    scenario_id: str
    condition: str
    group: str
    seed: int
    goal_achieved: bool
    events: list[Event] = field(default_factory=list)
    identity: str | None = None      # None | "legit" | "impostor"
    latencies_ms: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def worst(self) -> int:
        return max((e.severity for e in self.events), default=0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass"] = passes(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CallResult:
        d = dict(d)
        d.pop("pass", None)
        d["events"] = [Event(**e) for e in d.get("events", [])]
        return cls(**d)


def passes(call: CallResult, fail_at: int = spec.FAIL_AT) -> bool:
    """Clean success: goal achieved and nothing of severity fail_at or worse."""
    return bool(call.goal_achieved) and call.worst < fail_at


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials, as rates in [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6))


def rate(calls: list[CallResult]) -> dict[str, Any]:
    n = len(calls)
    k = sum(1 for c in calls if passes(c))
    lo, hi = wilson(k, n)
    return {"score": round(100 * k / n, 1) if n else None, "lo": round(100 * lo, 1), "hi": round(100 * hi, 1),
            "n": n, "passed": k}


def views(calls: list[CallResult]) -> dict[str, Any]:
    """Behavior, Robustness, Identity, Fairness: the same rate on defined subsets."""
    non_id = [c for c in calls if c.identity is None]
    behavior = [c for c in non_id if c.condition == "clean" and c.group == spec.REFERENCE_GROUP]
    robust = [c for c in calls if c.condition in spec.DEGRADED]
    ident = [c for c in calls if c.identity is not None]
    by_group = {g: rate([c for c in calls if c.group == g]) for g in sorted({c.group for c in calls})}
    ref = by_group.get(spec.REFERENCE_GROUP, {}).get("score")
    for r in by_group.values():
        r["delta_vs_reference"] = None if ref is None or r["score"] is None else round(r["score"] - ref, 1)
    worst_group = min((g for g in by_group if by_group[g]["score"] is not None), key=lambda g: by_group[g]["score"], default=None)
    clean_all = rate([c for c in calls if c.condition == "clean"])
    r_view = rate(robust)
    return {
        "behavior": rate(behavior),
        "robustness": {**r_view, "gap_vs_clean": None if clean_all["score"] is None or r_view["score"] is None
                       else round(clean_all["score"] - r_view["score"], 1)},
        "identity": {**rate(ident),
                     "false_accepts": sum(1 for c in ident for e in c.events if e.type == "false_accept"),
                     "false_rejects": sum(1 for c in ident for e in c.events if e.type == "false_reject")},
        "fairness": {**(by_group[worst_group] if worst_group else rate([])), "worst_group": worst_group,
                     "by_group": by_group},
    }


def failures_by_severity(calls: list[CallResult]) -> dict[str, int]:
    out = {f"S{s}": 0 for s in range(1, 6)}
    for c in calls:
        for e in c.events:
            out[f"S{e.severity}"] += 1
    return out


def failures_by_type(calls: list[CallResult]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in calls:
        for e in c.events:
            out[e.type] = out.get(e.type, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def audio_only_failures(calls: list[CallResult]) -> dict[str, Any]:
    """Failed calls whose only S3+ events are audio-derived: invisible to transcript-only scoring."""
    audio_types = {"slow_turn", "repeated_talk_over", "talk_over", "slow_median"}
    failed = [c for c in calls if not passes(c)]
    only_audio = [c for c in failed if c.goal_achieved
                  and all(e.type in audio_types for e in c.events if e.severity >= spec.FAIL_AT)]
    transcript_only_score = rate([CallResult(c.scenario_id, c.condition, c.group, c.seed, c.goal_achieved,
                                             [e for e in c.events if e.type not in audio_types], c.identity)
                                  for c in calls])
    return {"failed": len(failed), "failed_only_on_audio_events": len(only_audio),
            "fraction": round(len(only_audio) / len(failed), 3) if failed else None,
            "score_if_transcript_only": transcript_only_score["score"]}


def aggregation_ablation(v: dict[str, Any]) -> dict[str, float | None]:
    """How the four views would combine under the alternatives the paper studies."""
    vals = [v[k]["score"] for k in ("behavior", "robustness", "identity", "fairness")]
    if any(x is None for x in vals):
        return {"mean_of_views": None, "geometric_mean_of_views": None, "min_view": None}
    return {"mean_of_views": round(sum(vals) / 4, 1),
            "geometric_mean_of_views": round(math.prod(max(x, 0.1) for x in vals) ** 0.25, 1),
            "min_view": round(min(vals), 1)}


def record(calls: list[CallResult], agent: str, agent_fingerprint: str = "", protocol: spec.Protocol = spec.PROTOCOL,
           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The Inquesto Record: numbers only. The citation unit."""
    overall = rate(calls)
    v = views(calls)
    rec = {
        "protocol": f"inquesto-{protocol.version}",
        "agent": agent, "agent_fingerprint": agent_fingerprint,
        "score": overall["score"], "ci95": [overall["lo"], overall["hi"]], "n": overall["n"], "passed": overall["passed"],
        "views": v,
        "failures_by_severity": failures_by_severity(calls),
        "failures_by_type": failures_by_type(calls),
        "audio_only": audio_only_failures(calls),
        "aggregation_ablation": aggregation_ablation(v),
        "severity_sensitivity": severity_sensitivity(calls),
        "design": {"scenarios": len({c.scenario_id for c in calls}), "conditions": sorted({c.condition for c in calls}),
                   "groups": sorted({c.group for c in calls}), "seeds": len({c.seed for c in calls})},
        "definition": protocol.describe(),
        "evaluated_at": time.strftime("%Y-%m-%d"),
    }
    if extra:
        rec.update(extra)
    rec["citation"] = citation_line(rec)
    return rec


def citation_line(rec: dict[str, Any]) -> str:
    """The reporting form: score with its interval and n, then the diagnostic views."""
    v = rec["views"]
    s = lambda k: "—" if v[k]["score"] is None else f"{v[k]['score']:.0f}"
    ver = rec["protocol"].split("-")[-1]
    pop = rec.get("population") or {}
    tag = f"v{ver}" if pop.get("default", True) else f"v{ver}/{pop.get('name')}"
    if rec["score"] is None:
        return f"Inquesto {tag} = n/a (n = {rec['n']})"
    return (f"Inquesto {tag} = {rec['score']:.1f}% (95% CI {rec['ci95'][0]:.1f}–{rec['ci95'][1]:.1f}; n = {rec['n']}); "
            f"views B {s('behavior')} · R {s('robustness')} · I {s('identity')} · F {s('fairness')}")


def severity_sensitivity(calls: list[CallResult]) -> dict[str, Any]:
    """The score under each clean-success threshold: what fails a call is a protocol choice, shown."""
    out = {}
    for th, label in ((2, "no degraded-experience failure"), (3, "no functional failure (v0.1)"), (4, "no material-risk failure"), (5, "no critical failure")):
        k = sum(1 for c in calls if c.goal_achieved and c.worst < th)
        n = len(calls)
        lo, hi = wilson(k, n)
        out[f"S{th}"] = {"threshold": th, "meaning": label, "score": round(100 * k / n, 1) if n else None,
                         "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
    return out
