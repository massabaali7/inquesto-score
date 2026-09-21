"""Cost per conversation, in USD. Lower is better."""

from __future__ import annotations

from ..program import Conversation
from . import Evaluator, Score, register


@register("cost")
class Cost(Evaluator):
    name = "cost"
    unit = "USD"
    higher_is_better = False

    def score_one(self, conv: Conversation) -> Score:
        return Score(name=self.name, value=conv.cost_usd, unit=self.unit, higher_is_better=False)
