"""Inquesto — make voice agents measurably better.

    from inquesto import VoiceProgram, evaluate, testsets

    class SupportAgent(VoiceProgram):
        task = "Resolve billing questions in under three minutes."
        constraints = ["never interrupt the caller", "never invent policy"]
        tools = ["lookup_account", "issue_refund"]

    result = evaluate(SupportAgent(), testsets.load("support-v1"))
    print(result.metrics["task_success"])
"""

from . import adapters, evaluators, gates, optimizers, testsets
from .gates import check as gate_check
from .optimizers import CoordinateSearch, RandomSearch, TPESearch
from .program import Config, Conversation, Turn, VoiceProgram
from .runners import EvalResult, evaluate
from .score import ScoreSpec, inquesto_score

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Conversation",
    "CoordinateSearch",
    "EvalResult",
    "RandomSearch",
    "ScoreSpec",
    "TPESearch",
    "Turn",
    "VoiceProgram",
    "__version__",
    "adapters",
    "evaluate",
    "evaluators",
    "gate_check",
    "gates",
    "inquesto_score",
    "optimize",
    "optimizers",
    "testsets",
]


def optimize(program, testset, metric: str = "task_success", constraints=None, **kw):
    """Shorthand for the default optimizer (coordinate search).

        inquesto.optimize(agent, suite, metric="task_success",
                       constraints={"latency": "<= 900", "cost": "<= 0.05"})
    """
    return CoordinateSearch().optimize(
        program, testset, metric=metric, constraints=constraints, **kw
    )
