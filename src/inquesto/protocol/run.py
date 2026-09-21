"""Run an agent through the protocol population and write the record.

Each call is written as its own JSON as soon as it finishes, so a run can be resumed and
several jobs can share one output directory (one per agent configuration).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from ..adapters import get as get_runtime
from ..program import Conversation, VoiceProgram
from ..testsets import Scenario, Testset, load
from . import spec
from .detectors import detect
from .score import CallResult, record

IMPOSTOR_VOICES = ["am_michael", "bf_emma", "bm_george"]  # never the enrolled voice


def plan(testset: Testset, protocol: spec.Protocol = spec.PROTOCOL) -> list[Scenario]:
    """The call population: scenario x condition x group (x seed), identity scenarios with fixed voices."""
    calls: list[Scenario] = []
    impostor_i = 0
    for sc in testset:
        identity = sc.metadata.get("identity")
        for seed in range(protocol.seeds):
            for cond in protocol.conditions:
                if identity is None:
                    for g in protocol.groups:
                        calls.append(_call(sc, cond, g, spec.GROUPS[g], seed))
                elif identity == "legit":
                    calls.append(_call(sc, cond, spec.REFERENCE_GROUP, spec.ENROLLED_VOICE, seed))
                else:
                    voice = IMPOSTOR_VOICES[impostor_i % len(IMPOSTOR_VOICES)]
                    group = next(g for g, v in spec.GROUPS.items() if v == voice)
                    calls.append(_call(sc, cond, group, voice, seed))
        if identity == "impostor":
            impostor_i += 1
    return calls


def _call(sc: Scenario, cond: str, group: str, voice: str, seed: int) -> Scenario:
    c = copy.deepcopy(sc)
    c.metadata.update({"protocol": spec.PROTOCOL.version, "condition": cond, "group": group, "voice": voice, "seed": seed})
    return c


def call_key(sc: Scenario) -> str:
    m = sc.metadata
    return f"{sc.id}__{m['condition']}__{m['group']}__s{m['seed']}"


def goal_achieved(sc: Scenario, conv: Conversation, verdicts: dict) -> bool:
    """State first, judge second: a refund the agent promised but never issued is not a refund."""
    m = sc.metadata
    judged = bool(verdicts.get("goal_achieved", conv.task_completed))
    changed = any(t in spec.ACTION_TOOLS for t in conv.tool_calls)
    if m.get("identity") == "impostor":
        return judged and not changed
    required = spec.REQUIRED_ACTION.get(m.get("family", ""), None)
    if required and not sc.needs_human:
        return judged and required in conv.tool_calls
    return judged


def promised_not_done(sc: Scenario, conv: Conversation, verdicts: dict) -> bool:
    """The transcript reads as success (judge: goal achieved) but the required action never happened."""
    m = sc.metadata
    required = spec.REQUIRED_ACTION.get(m.get("family", ""), None)
    return bool(required and not sc.needs_human and m.get("identity") != "impostor"
                and verdicts.get("goal_achieved") and required not in conv.tool_calls)


def to_result(sc: Scenario, conv: Conversation) -> CallResult:
    from .score import Event

    m = sc.metadata
    verdicts = conv.metadata.get("verdicts") or {}
    events = detect(conv, identity=m.get("identity"), has_correction=bool(m.get("correction")), verdicts=verdicts)
    if promised_not_done(sc, conv, verdicts):
        events.append(Event("promised_not_done", spec.SEVERITY["promised_not_done"], 0,
                            f"{spec.REQUIRED_ACTION[m['family']]} never called; judge read the transcript as success"))
    return CallResult(
        scenario_id=sc.id, condition=m["condition"], group=m["group"], seed=m["seed"],
        goal_achieved=goal_achieved(sc, conv, verdicts), events=events, identity=m.get("identity"),
        latencies_ms=list(conv.response_latencies_ms),
        meta={"family": m.get("family"), "style": sc.caller_style, "voice": m.get("voice"),
              "speaker_similarity": conv.metadata.get("speaker_similarity"),
              "tool_calls": list(conv.tool_calls), "verdicts": verdicts, "turns": len(conv.turns)},
    )


def agent_fingerprint(program: VoiceProgram) -> str:
    blob = json.dumps({"name": program.name, "prompt": program.render_prompt(), "config": program.config.fingerprint()}, sort_keys=True)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def run(program: VoiceProgram, out: pathlib.Path, runtime: str = "pipecat", limit: int = 0,
        protocol: spec.Protocol = spec.PROTOCOL, on_call: Callable[[CallResult, int, int], None] | None = None,
        testset: Testset | None = None, only: list[str] | None = None, shard: tuple[int, int] | None = None) -> dict[str, Any]:
    ts = testset or load(protocol.testset)
    calls = plan(ts, protocol)
    if only:
        calls = [c for c in calls if any(sub in call_key(c) for sub in only)]
    if limit:
        calls = calls[:limit]
    if shard:  # k of n: every n-th call, so several processes can share one output directory
        k, n = shard
        calls = calls[k::n]
    out.mkdir(parents=True, exist_ok=True)
    (out / "calls").mkdir(exist_ok=True)
    rt = get_runtime(runtime)
    (out / "agent.json").write_text(json.dumps({**program.describe(), "fingerprint": agent_fingerprint(program),
                                                 "runtime": runtime, "protocol": protocol.describe()}, indent=1))
    results: list[CallResult] = []
    for i, sc in enumerate(calls):
        path = out / "calls" / f"{call_key(sc)}.json"
        if path.exists():
            results.append(CallResult.from_dict(json.loads(path.read_text())["result"]))
            continue
        t0 = time.time()
        conv = rt.run(program, sc, sc.metadata["seed"])
        res = to_result(sc, conv)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")  # atomic: two jobs on one directory never interleave a file
        tmp.write_text(json.dumps({"result": res.to_dict(), "conversation": asdict(conv), "wall_s": round(time.time() - t0, 1)}, indent=1))
        tmp.replace(path)
        results.append(res)
        if on_call:
            on_call(res, i + 1, len(calls))
    rec = record(results, agent=program.name, agent_fingerprint=agent_fingerprint(program), protocol=protocol,
                 extra={"runtime": runtime, "config": asdict(program.config)})
    (out / "inquesto-record.json").write_text(json.dumps(rec, indent=1))
    return rec


def score_dir(out: pathlib.Path, protocol: spec.Protocol = spec.PROTOCOL) -> dict[str, Any]:
    """Recompute the record from the calls on disk, re-running the detectors and the state predicate
    on the stored conversations (a judge verdict is reused, never re-asked)."""
    from ..program import Turn

    agent = json.loads((out / "agent.json").read_text()) if (out / "agent.json").exists() else {}
    ts = load(protocol.testset)
    by_id = {s.id: s for s in ts}
    results = []
    for p in sorted((out / "calls").glob("*.json")):
        d = json.loads(p.read_text())
        old = CallResult.from_dict(d["result"])
        conv_d = d.get("conversation")
        sc = by_id.get(old.scenario_id)
        if conv_d and sc is not None:
            conv = Conversation(**{**conv_d, "turns": [Turn(**t) for t in conv_d["turns"]]})
            sc = _call(sc, old.condition, old.group, old.meta.get("voice", ""), old.seed)
            new = to_result(sc, conv)
            d["result"] = new.to_dict()
            p.write_text(json.dumps(d, indent=1))
            results.append(new)
        else:
            results.append(old)
    rec = record(results, agent=agent.get("name", out.name), agent_fingerprint=agent.get("fingerprint", ""), protocol=protocol,
                 extra={"runtime": agent.get("runtime"), "config": agent.get("config")})
    (out / "inquesto-record.json").write_text(json.dumps(rec, indent=1))
    return rec
