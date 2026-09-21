"""Evaluators measure the conversation, not the components."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..program import Conversation


@dataclass
class Score:
    """One evaluator's verdict on one conversation."""

    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True
    failure: str | None = None  # set when this conversation failed the check


class Evaluator:
    """Base class. Implement `score_one` and you're done."""

    name: str = "evaluator"
    unit: str = ""
    higher_is_better: bool = True

    def score_one(self, conv: Conversation) -> Score:  # pragma: no cover - abstract
        raise NotImplementedError

    def aggregate(self, scores: Iterable[Score]) -> float:
        vals = [s.value for s in scores]
        return sum(vals) / len(vals) if vals else 0.0


_REGISTRY: dict[str, Callable[[], Evaluator]] = {}


def register(name: str) -> Callable[[type[Evaluator]], type[Evaluator]]:
    def deco(cls: type[Evaluator]) -> type[Evaluator]:
        _REGISTRY[name] = cls
        return cls

    return deco


def get(name: str) -> Evaluator:
    if name not in _REGISTRY:
        raise KeyError(f"unknown evaluator {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)


def default_suite() -> list[Evaluator]:
    return [get(n) for n in ("task_success", "turn_taking", "latency", "cost")]


from . import cost, latency, task_success, turn_taking  # noqa: F401  (registers)

__all__ = ["Evaluator", "Score", "available", "default_suite", "get", "register"]
