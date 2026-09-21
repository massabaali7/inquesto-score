"""A deterministic simulated runtime.

Why this exists: `pip install inquesto && inquesto evaluate` has to produce a real
number on the first try, with no API keys and no phone line. The mock runtime
models the failure modes we care about — endpointing that clips the caller,
context truncation, weak models on hard calls — so the loop is honest even
though the audio isn't. Swap in a real adapter and every other layer is
unchanged.
"""

from __future__ import annotations

import math
import random

from ..program import Config, Conversation, Turn, VoiceProgram
from ..testsets import Scenario
from . import register

# Rough capability / price of each model, used by the simulator.
_MODELS = {
    "gpt-4o-mini": (0.58, 0.002),
    "gpt-4o": (0.72, 0.010),
    "claude-sonnet": (0.73, 0.009),
    "llama-3-8b": (0.45, 0.001),
}

# How much each caller style punishes a slow / insensitive endpointer.
BASE_COMPETENCE = 0.64
SHARPNESS = 4.5

_STYLE_PRESSURE = {
    "neutral": 0.35,
    "fast": 0.75,
    "hesitant": 0.95,
    "accented": 0.60,
    "noisy": 0.70,
}


def _endpoint_penalty(cfg: Config, style: str) -> float:
    """Hesitant callers get cut off by short endpointing; fast callers by long."""
    pressure = _STYLE_PRESSURE.get(style, 0.5)
    if style in ("hesitant", "accented", "noisy"):
        # needs patience: short windows clip them
        gap = max(0, 900 - cfg.endpointing_ms) / 900
    else:
        # needs responsiveness: long windows feel sluggish but rarely clip
        gap = max(0, cfg.endpointing_ms - 900) / 1500
    return pressure * gap * (0.4 + 0.6 * cfg.interrupt_sensitivity)


@register("mock")
class MockRuntime:
    """Deterministic simulator. Same program + scenario + seed => same conversation."""

    name = "mock"

    def run(self, program: VoiceProgram, scenario: Scenario, seed: int) -> Conversation:
        cfg = program.config
        rng = random.Random(f"{scenario.id}:{seed}")

        capability, price = _MODELS.get(cfg.model, (0.55, 0.004))
        clip = _endpoint_penalty(cfg, scenario.caller_style)

        # Does the agent keep enough context for a call of this length?
        context_gap = max(0, scenario.turns_expected - cfg.max_context_turns) * 0.05

        # Prompt discipline: explicit constraints help, rambling temperature hurts.
        prompt_bonus = 0.06 if "never interrupt" in cfg.system_prompt.lower() else 0.0
        temp_penalty = max(0.0, cfg.temperature - 0.4) * 0.12

        score = (
            capability
            + prompt_bonus
            + BASE_COMPETENCE
            - scenario.difficulty * 0.60
            - clip * 0.55
            - context_gap
            - temp_penalty
        )
        # Graded, not cliff-edged: marginal calls fail some of the time, which is
        # what real testsets look like.
        p_success = 1.0 / (1.0 + math.exp(-SHARPNESS * (score - 0.5)))
        # Irreducible difficulty: the hardest calls need a human. No amount of
        # configuration search should drive failures to zero, and an optimizer
        # that claims 100% is lying to you.
        if scenario.needs_human:
            p_success = 0.0
        completed = rng.random() < p_success

        n_turns = scenario.turns_expected
        turns: list[Turn] = []
        latencies: list[int] = []
        t = 0
        interrupts = 0
        for i in range(n_turns):
            dur = rng.randint(1200, 2600)
            turns.append(Turn("user", f"[{scenario.caller_style}] turn {i}", t, t + dur))
            t += dur
            # An interruption happens when endpointing clips this caller style.
            barge = rng.random() < clip
            interrupts += int(barge)
            lat = int(cfg.endpointing_ms * 0.6 + 240 + rng.randint(0, 260))
            latencies.append(lat)
            turns.append(Turn("agent", f"response {i}", t, t + 900, barge_in=barge))
            t += 900

        cost = round(price * n_turns * (cfg.max_context_turns / 8) + 0.004 * n_turns, 4)

        return Conversation(
            scenario_id=scenario.id,
            turns=turns,
            task_completed=completed,
            tool_calls=list(scenario.requires_tools) if completed else [],
            response_latencies_ms=latencies,
            cost_usd=cost,
            metadata={"runtime": self.name, "caller_style": scenario.caller_style,
                      "quality_score": round(score, 3)},
        )
