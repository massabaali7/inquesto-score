"""Did the caller's task actually get done?"""

from __future__ import annotations

from ..program import Conversation
from . import Evaluator, Score, register


@register("task_success")
class TaskSuccess(Evaluator):
    name = "task_success"
    unit = "%"
    higher_is_better = True

    def score_one(self, conv: Conversation) -> Score:
        ok = bool(conv.task_completed)
        return Score(
            name=self.name,
            value=1.0 if ok else 0.0,
            unit=self.unit,
            failure=None if ok else "task not completed",
        )
