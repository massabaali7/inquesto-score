"""The VoiceProgram abstraction: declare what your agent should do, not how it runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class Config:
    """A tunable agent configuration. Optimizers search over these fields."""

    model: str = "gpt-4o-mini"
    stt: str = "whisper-1"
    tts: str = "openai-tts"
    system_prompt: str = ""
    endpointing_ms: int = 700
    interrupt_sensitivity: float = 0.5
    max_context_turns: int = 8
    temperature: float = 0.7

    def merged(self, **overrides: Any) -> Config:
        data = asdict(self)
        data.update(overrides)
        return Config(**data)

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def diff(self, other: Config) -> dict[str, tuple[Any, Any]]:
        a, b = asdict(self), asdict(other)
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}


@dataclass
class Turn:
    """One utterance in a conversation."""

    speaker: str  # "user" | "agent"
    text: str
    start_ms: int = 0
    end_ms: int = 0
    barge_in: bool = False  # agent spoke while user was still speaking


@dataclass
class Conversation:
    """The output of running one scenario against an agent."""

    scenario_id: str
    turns: list[Turn] = field(default_factory=list)
    task_completed: bool = False
    tool_calls: list[str] = field(default_factory=list)
    response_latencies_ms: list[int] = field(default_factory=list)
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def interruptions(self) -> int:
        return sum(1 for t in self.turns if t.barge_in)


class VoiceProgram:
    """Subclass this to declare an agent.

    Everything here is declarative: the task, the behavioral constraints, the
    tools. Your runtime (LiveKit, Pipecat, your own stack) supplies the
    conversation; Inquesto supplies the evaluation, experimentation and
    optimization loop around it.

        class SupportAgent(VoiceProgram):
            task = "Resolve billing questions in under 3 minutes."
            constraints = ["never invent policy", "never interrupt the caller"]
            tools = ["lookup_account", "issue_refund"]
    """

    task: ClassVar[str] = ""
    constraints: ClassVar[list[str]] = []
    tools: ClassVar[list[str]] = []
    config: Config = Config()

    def __init__(self, config: Config | None = None) -> None:
        if config is not None:
            self.config = config
        elif not self.config.system_prompt:
            self.config = self.config.merged(system_prompt=self.render_prompt())

    @property
    def name(self) -> str:
        return type(self).__name__

    def render_prompt(self) -> str:
        """Default prompt rendering. Override for full control."""
        parts = [self.task.strip()]
        if self.constraints:
            parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in self.constraints))
        if self.tools:
            parts.append("Tools available: " + ", ".join(self.tools))
        return "\n\n".join(p for p in parts if p)

    def with_config(self, **overrides: Any) -> VoiceProgram:
        return type(self)(config=self.config.merged(**overrides))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "constraints": list(self.constraints),
            "tools": list(self.tools),
            "config": asdict(self.config),
            "fingerprint": self.config.fingerprint(),
        }
