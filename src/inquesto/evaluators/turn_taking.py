"""Turn-taking: does the agent talk over the caller?

The flagship speech-specific evaluator. A conversation passes when the agent
never barges in on a caller who is still speaking.
"""

from __future__ import annotations

from ..program import Conversation
from . import Evaluator, Score, register


@register("turn_taking")
class TurnTaking(Evaluator):
    name = "turn_taking"
    unit = "%"
    higher_is_better = True

    def score_one(self, conv: Conversation) -> Score:
        n = conv.interruptions
        agent_turns = max(1, sum(1 for t in conv.turns if t.speaker == "agent"))
        clean = 1.0 - min(1.0, n / agent_turns)
        return Score(
            name=self.name,
            value=clean,
            unit=self.unit,
            failure=None if n == 0 else f"agent interrupted caller {n}x",
        )
