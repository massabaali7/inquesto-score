"""Adapters connect Inquesto to whatever actually runs your agent.

Inquesto is not a runtime. An adapter takes a VoiceProgram plus a Scenario and
returns a Conversation. LiveKit, Pipecat, or your own stack — Inquesto only needs
the transcript, the timings and the outcome.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..program import Conversation, VoiceProgram
from ..testsets import Scenario


@runtime_checkable
class Runtime(Protocol):
    """Implement this to plug in your stack."""

    name: str

    def run(self, program: VoiceProgram, scenario: Scenario, seed: int) -> Conversation: ...


_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def get(name: str) -> Runtime:
    if name not in _REGISTRY:
        raise KeyError(f"unknown runtime {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)


from . import local, mock, pipecat  # noqa: F401  (registers)

__all__ = ["Runtime", "available", "get", "register"]
