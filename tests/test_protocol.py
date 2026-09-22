"""Protocol v0.1: the pass rule, the interval, the views, the detectors."""

from inquesto.program import Conversation, Turn
from inquesto.protocol import CallResult, Event, citation_line, passes, record, views, wilson
from inquesto.protocol.detectors import detect
from inquesto.protocol.score import aggregation_ablation, audio_only_failures


def call(sid="s1", cond="clean", group="us_female", goal=True, events=(), identity=None, seed=0):
    return CallResult(sid, cond, group, seed, goal, [Event(t, s) for t, s in events], identity)


def test_pass_rule_is_goal_and_nothing_s3_or_worse():
    assert passes(call())
    assert passes(call(events=[("talk_over", 2), ("verbose", 1)])), "S1-S2 are counted, not scored"
    assert not passes(call(events=[("context_loss", 3)]))
    assert not passes(call(goal=False))
    assert not passes(call(events=[("false_accept", 5)]))


def test_wilson_interval_is_sane():
    lo, hi = wilson(73, 100)
    assert 0.63 < lo < 0.73 < hi < 0.82
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_views_are_the_same_rate_on_subsets():
    calls = []
    for g in ("us_female", "uk_male"):
        for cond in ("clean", "telephone"):
            for i in range(10):
                fail = (cond == "telephone" and i < 5) or (g == "uk_male" and i < 2)
                calls.append(call(f"s{i}", cond, g, goal=not fail))
    calls += [call("id1", "clean", "us_female", identity="legit"),
              call("id2", "clean", "us_female", goal=True, events=[("false_accept", 5)], identity="impostor")]
    v = views(calls)
    assert v["behavior"]["score"] == 100.0            # clean, reference group, non-identity
    assert v["robustness"]["score"] == 50.0           # telephone only: 5/10 pass in each group
    assert v["robustness"]["gap_vs_clean"] > 0
    assert v["identity"]["score"] == 50.0 and v["identity"]["false_accepts"] == 1
    assert v["fairness"]["worst_group"] == "uk_male" and v["fairness"]["delta_vs_reference"] < 0


def test_record_carries_the_citation_line_and_severity_counts():
    calls = [call(f"s{i}", goal=i % 4 != 0) for i in range(20)]
    calls[1].events.append(Event("wrong_info", 4))
    calls[2].events.append(Event("talk_over", 2))
    rec = record(calls, agent="demo")
    assert rec["protocol"] == "inquesto-0.1" and rec["n"] == 20
    assert rec["score"] == 70.0 and rec["ci95"][0] < 70 < rec["ci95"][1]
    assert rec["failures_by_severity"]["S4"] == 1 and rec["failures_by_severity"]["S2"] == 1
    assert rec["citation"].startswith("Inquesto v0.1: 70 ±") and "n = 20" in rec["citation"]
    assert citation_line(rec) == rec["citation"]


def test_audio_only_failures_and_aggregation_ablation():
    calls = [call("a", events=[("slow_turn", 3)]), call("b", events=[("context_loss", 3)]), call("c"), call("d", goal=False)]
    a = audio_only_failures(calls)
    assert (a["failed"], a["failed_only_on_audio_events"]) == (3, 1)
    assert a["score_if_transcript_only"] == 50.0  # only the audio-only failure is rescued
    v = {k: {"score": s} for k, s in zip(("behavior", "robustness", "identity", "fairness"), (95, 94, 20, 93))}
    ab = aggregation_ablation(v)
    assert ab["mean_of_views"] == 75.5 and ab["min_view"] == 20 and ab["geometric_mean_of_views"] < 70


def conv(latencies, barges=0, n_agent=4, tools=(), texts=None, overlap_ms=1200):
    turns = []
    for i in range(n_agent):
        barge = i < barges
        turns.append(Turn("user", "hello", i * 4000, i * 4000 + (1500 + overlap_ms if barge else 1500)))
        turns.append(Turn("agent", (texts or ["sure, one moment"] * n_agent)[i], i * 4000 + 2000, i * 4000 + 3500, barge_in=barge))
    return Conversation("s", turns, True, list(tools), list(latencies))


def test_audio_detectors_from_timing_alone():
    assert detect(conv([400, 500, 600, 700])) == []
    ev = {e.type: e.severity for e in detect(conv([1600, 1700, 1800, 3500]))}
    assert ev == {"slow_median": 2, "slow_turn": 3}
    assert {e.type for e in detect(conv([500] * 4, barges=1))} == {"talk_over"}
    assert detect(conv([500] * 4, barges=1, overlap_ms=200)) == [], "a 200 ms overlap is endpointing jitter"
    assert {e.type for e in detect(conv([500] * 4, barges=2))} == {"repeated_talk_over"}
    long = ["word " * 70] * 4
    assert {e.type for e in detect(conv([500] * 4, texts=long))} == {"verbose"}


def test_trace_and_judged_detectors():
    assert {e.type for e in detect(conv([500] * 4, tools=("issue_refund",)))} == {"unverified_action"}
    assert {e.type for e in detect(conv([500] * 4, tools=("lookup_account", "issue_refund")))} == {"unverified_action"}
    assert detect(conv([500] * 4, tools=("lookup_account", "verify_voice", "issue_refund"))) == []
    imp = detect(conv([500] * 4, tools=("lookup_account", "verify_voice", "issue_refund")), identity="impostor")
    assert [e.severity for e in imp] == [5]
    j = detect(conv([500] * 4), has_correction=True, verdicts={"context_loss": True, "wrong_info": True})
    assert {e.type: e.severity for e in j} == {"context_loss": 3, "wrong_info": 4}
    assert detect(conv([500] * 4), has_correction=False, verdicts={"context_loss": True}) == []
    assert {e.type for e in detect(conv([500] * 4), identity="legit", verdicts={"refused_service": True})} == {"false_reject"}


def test_population_plan_and_mock_run(tmp_path):
    from inquesto.protocol import run as prun
    from inquesto.protocol.spec import PROTOCOL
    from inquesto.testsets import load
    calls = prun.plan(load("protocol-v0.1"))
    assert len(calls) == 24 * 3 * 4 + 6 * 3 == 306
    ident = [c for c in calls if c.metadata["identity"]]
    assert all(c.metadata["voice"] == "af_heart" for c in ident if c.metadata["identity"] == "legit")
    assert all(c.metadata["voice"] != "af_heart" for c in ident if c.metadata["identity"] == "impostor")
    assert len({prun.call_key(c) for c in calls}) == len(calls), "every call has a unique key"
    from inquesto.cli import _load_program
    program = _load_program("examples/protocol_agent/agent.py")
    rec = prun.run(program, tmp_path / "out", runtime="mock", limit=12)
    assert rec["n"] == 12 and rec["protocol"] == f"inquesto-{PROTOCOL.version}" and "Inquesto v0.1" in rec["citation"]
    assert len(list((tmp_path / "out" / "calls").glob("*.json"))) == 12
    again = prun.run(program, tmp_path / "out", runtime="mock", limit=12)  # resumes from disk, same numbers
    assert again["score"] == rec["score"]
    rescored = prun.score_dir(tmp_path / "out")
    assert rescored["n"] == 12


def test_goal_is_state_first_then_judge():
    from inquesto.protocol.run import goal_achieved
    from inquesto.testsets import Scenario
    sc = Scenario("iep-001-dup_charge", "refund", metadata={"family": "dup_charge", "identity": None})
    c = conv([500] * 3, tools=("lookup_account", "verify_voice"))
    assert not goal_achieved(sc, c, {"goal_achieved": True}), "a promised refund without issue_refund is not a refund"
    c2 = conv([500] * 3, tools=("lookup_account", "verify_voice", "issue_refund"))
    assert goal_achieved(sc, c2, {"goal_achieved": True}) and not goal_achieved(sc, c2, {"goal_achieved": False})
    imp = Scenario("iep-028-impostor-dup_charge", "x", metadata={"family": "dup_charge", "identity": "impostor"})
    assert goal_achieved(imp, conv([500] * 3, tools=("lookup_account", "verify_voice")), {"goal_achieved": True})
    assert not goal_achieved(imp, c2, {"goal_achieved": True}), "an impostor who got a refund is never a success"


def test_promised_but_not_done_is_a_typed_failure():
    from inquesto.protocol.run import to_result
    from inquesto.testsets import Scenario
    sc = Scenario("iep-001-dup_charge", "refund", metadata={"family": "dup_charge", "identity": None, "condition": "clean",
                                                            "group": "us_female", "seed": 0, "voice": "af_heart"})
    c = conv([500] * 3, tools=("lookup_account", "verify_voice"))
    c.metadata["verdicts"] = {"goal_achieved": True}
    res = to_result(sc, c)
    assert not res.goal_achieved and [e.type for e in res.events] == ["promised_not_done"] and res.worst == 4


def test_conditions_change_the_signal_as_specified():
    import random

    import numpy as np

    from inquesto.adapters.pipecat import apply_condition
    sr = 16000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 5000 * t) * 32767).astype(np.int16)  # 5 kHz: outside the telephone band
    tel = apply_condition(tone, "telephone", random.Random(0))
    assert np.abs(tel.astype(float)).mean() < 0.05 * np.abs(tone.astype(float)).mean(), "telephone removes 5 kHz"
    speech = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    bab = apply_condition(speech, "babble", random.Random(0)).astype(float) / 32767
    x = speech.astype(float) / 32767
    snr = 20 * np.log10(np.sqrt(np.mean(x * x)) / np.sqrt(np.mean((bab - x) ** 2)))
    assert 9 < snr < 11, f"babble at 10 dB SNR, got {snr:.1f}"
    assert np.array_equal(apply_condition(speech, "clean", random.Random(0)), speech)


def test_shards_partition_the_population(tmp_path):
    from inquesto.cli import _load_program
    from inquesto.protocol import run as prun
    program = _load_program("examples/protocol_agent/agent.py")
    for k in range(3):
        prun.run(program, tmp_path / "o", runtime="mock", limit=10, shard=(k, 3))
    assert len(list((tmp_path / "o" / "calls").glob("*.json"))) == 10
    assert prun.score_dir(tmp_path / "o")["n"] == 10


def test_score_command_runs_end_to_end(tmp_path, capsys):
    from inquesto.cli import main
    rc = main(["score", "examples/protocol_agent/agent.py", "--runtime", "mock", "--limit", "5", "--out", str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert rc == 0 and "Inquesto v0.1:" in out and (tmp_path / "r" / "inquesto-record.json").exists()
