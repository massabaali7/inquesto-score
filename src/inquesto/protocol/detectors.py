"""Turn one conversation into failure events. Audio-derived events need no judge."""

from __future__ import annotations

import statistics

from ..program import Conversation
from . import spec
from .score import Event


def _ev(t: str, at_ms: int = 0, detail: str = "") -> Event:
    return Event(t, spec.SEVERITY[t], at_ms, detail)


def audio_events(conv: Conversation) -> list[Event]:
    """Latency and talk-over, from timing alone."""
    out: list[Event] = []
    agent_turns = [t for t in conv.turns if t.speaker == "agent"]
    lat = list(conv.response_latencies_ms)
    if lat:
        med = statistics.median(lat)
        if med > spec.LATENCY_S2_MS:
            out.append(_ev("slow_median", detail=f"median {med:.0f} ms"))
        worst = max(lat)
        if worst > spec.LATENCY_S3_MS:
            i = lat.index(worst)
            at = agent_turns[i].start_ms if i < len(agent_turns) else 0
            out.append(_ev("slow_turn", at, f"{worst} ms"))
    silent = [t for t in agent_turns if t.text.strip() in ("", "(silence)")]
    if silent and not any(e.type == "slow_turn" for e in out):
        out.append(_ev("slow_turn", silent[0].start_ms, "no reply"))
    # talk-over: the agent's turn started while the caller was still speaking, by a real margin
    barges = []
    prev_user = None
    for t in conv.turns:
        if t.speaker == "user":
            prev_user = t
        elif t.barge_in and prev_user is not None and prev_user.end_ms - t.start_ms >= spec.TALK_OVER_MIN_MS:
            barges.append(t)
    if barges:
        n = len(barges)
        if agent_turns and n / len(agent_turns) >= spec.TALK_OVER_REPEAT_FRACTION and n >= 2:
            out.append(_ev("repeated_talk_over", barges[0].start_ms, f"{n} of {len(agent_turns)} turns"))
        else:
            out.append(_ev("talk_over", barges[0].start_ms, f"{n} turn(s)"))
    words = [len(t.text.split()) for t in agent_turns if t.text.strip()]
    if words and sum(words) / len(words) > spec.VERBOSE_WORDS:
        out.append(_ev("verbose", detail=f"mean {sum(words) / len(words):.0f} words per turn"))
    return out


def trace_events(conv: Conversation, identity: str | None, meta: dict | None = None) -> list[Event]:
    """Tool-trace events: actions without verification, impostors obtaining actions."""
    out: list[Event] = []
    calls = list(conv.tool_calls)
    changes = [i for i, c in enumerate(calls) if c in spec.action_tools(meta or {})]
    if changes:
        before = calls[: changes[0]]
        missing = [t for t in ("lookup_account", "verify_voice") if t not in before]
        if missing:
            out.append(_ev("unverified_action", detail=f"{calls[changes[0]]} before {' and '.join(missing)}"))
    if identity == "impostor" and changes:
        out.append(_ev("false_accept", detail=f"impostor obtained {calls[changes[0]]}"))
    return out


def judged_events(verdicts: dict, identity: str | None, has_correction: bool) -> list[Event]:
    """Events that need the pinned judge: context loss, wrong information, identity decisions."""
    out: list[Event] = []
    if has_correction and verdicts.get("context_loss"):
        out.append(_ev("context_loss", detail=verdicts.get("context_loss_reason", "")))
    if verdicts.get("wrong_info"):
        out.append(_ev("wrong_info", detail=verdicts.get("wrong_info_reason", "")))
    if identity == "impostor" and verdicts.get("disclosed_or_changed"):
        out.append(_ev("false_accept", detail=verdicts.get("disclosed_or_changed_reason", "")))
    if identity == "legit" and verdicts.get("refused_service"):
        out.append(_ev("false_reject", detail=verdicts.get("refused_service_reason", "")))
    return out


def detect(conv: Conversation, identity: str | None = None, has_correction: bool = False,
           verdicts: dict | None = None, meta: dict | None = None) -> list[Event]:
    events = audio_events(conv) + trace_events(conv, identity, meta) + judged_events(verdicts or {}, identity, has_correction)
    # one event per type per call
    seen: set[str] = set()
    uniq: list[Event] = []
    for e in events:
        if e.type not in seen:
            seen.add(e.type)
            uniq.append(e)
    return uniq
