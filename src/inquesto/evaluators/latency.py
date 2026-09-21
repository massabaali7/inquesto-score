"""Median response latency, in milliseconds. Lower is better."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import median

from ..program import Conversation
from . import Evaluator, Score, register


@register("latency")
class Latency(Evaluator):
    name = "latency"
    unit = "ms"
    higher_is_better = False

    def score_one(self, conv: Conversation) -> Score:
        lat = median(conv.response_latencies_ms) if conv.response_latencies_ms else 0.0
        return Score(
            name=self.name,
            value=float(lat),
            unit=self.unit,
            higher_is_better=False,
            failure=None if lat <= 1200 else f"median latency {lat:.0f}ms over budget",
        )

    def aggregate(self, scores: Iterable[Score]) -> float:
        vals = [s.value for s in scores]
        return float(median(vals)) if vals else 0.0
