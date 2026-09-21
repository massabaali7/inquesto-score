"""Testsets are reproducible conversational scenarios.

A scenario is a seeded description of a caller: what they want, how they speak,
how hard they are. The same testset + the same config always produces the same
conversations, which is what makes a before/after claim trustworthy.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BUILTIN_DIR = Path(__file__).parent / "data"

# Scenarios harder than this need a human. No configuration should solve them,
# and the correct agent behaviour is a handoff. Shared by every runtime.
NEEDS_HUMAN_ABOVE = 0.78


@dataclass
class Scenario:
    id: str
    goal: str
    caller_style: str = "neutral"  # neutral | fast | hesitant | accented | noisy
    difficulty: float = 0.5  # 0 easy .. 1 hard
    turns_expected: int = 6
    requires_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_human(self) -> bool:
        return self.difficulty > NEEDS_HUMAN_ABOVE


@dataclass
class Testset:
    name: str
    scenarios: list[Scenario] = field(default_factory=list)
    description: str = ""

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self.scenarios)

    def __len__(self) -> int:
        return len(self.scenarios)

    @classmethod
    def from_json(cls, path: str | Path) -> Testset:
        raw = json.loads(Path(path).read_text())
        return cls(
            name=raw["name"],
            description=raw.get("description", ""),
            scenarios=[Scenario(**s) for s in raw["scenarios"]],
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "name": self.name,
                    "description": self.description,
                    "scenarios": [asdict(s) for s in self.scenarios],
                },
                indent=2,
            )
        )


def load(name_or_path: str) -> Testset:
    """Load a builtin testset by name, or a Testset JSON file by path."""
    p = Path(name_or_path)
    if p.exists():
        return Testset.from_json(p)
    builtin = BUILTIN_DIR / f"{name_or_path}.json"
    if builtin.exists():
        return Testset.from_json(builtin)
    raise FileNotFoundError(
        f"no testset {name_or_path!r}; builtins: {[f.stem for f in BUILTIN_DIR.glob('*.json')]}"
    )


def available() -> list[str]:
    return sorted(f.stem for f in BUILTIN_DIR.glob("*.json"))


__all__ = ["NEEDS_HUMAN_ABOVE", "Scenario", "Testset", "available", "load"]
