"""The inquesto CLI. Stdlib only — it has to run the moment pip finishes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__, adapters, evaluators, gates, optimizers, testsets
from .optimizers import parse_constraints
from .program import VoiceProgram
from .runners import evaluate
from .score import ScoreSpec

G = "\033[32m"
B = "\033[1m"
D = "\033[2m"
R = "\033[31m"
X = "\033[0m"


def _fmt(name: str, value: float, unit: str) -> str:
    if unit == "%":
        return f"{value * 100:.0f}%"
    if unit == "ms":
        return f"{value:.0f}ms"
    if unit == "USD":
        return f"${value:.3f}"
    return f"{value:.3f}"


def _apply_overrides(program: VoiceProgram, sets: list[str]) -> VoiceProgram:
    """Apply --set key=value pairs, coercing to the Config field's current type."""
    if not sets:
        return program
    overrides = {}
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep or not hasattr(program.config, key):
            sys.exit(f"{R}error{X}: --set expects key=value with a Config field; got {item!r}")
        current = getattr(program.config, key)
        try:
            overrides[key] = type(current)(raw) if not isinstance(current, bool) else raw == "true"
        except ValueError:
            sys.exit(f"{R}error{X}: --set {key} expects {type(current).__name__}, got {raw!r}")
    return program.with_config(**overrides)


def _score_spec(args) -> ScoreSpec:
    try:
        return ScoreSpec.parse(args.score_spec)
    except ValueError as e:
        sys.exit(f"{R}error{X}: {e}")


def _load_testset(args):
    ts = testsets.load(args.suite)
    if args.limit:
        ts = testsets.Testset(name=ts.name, description=ts.description,
                              scenarios=ts.scenarios[: args.limit])
    return ts


def _load_program(spec: str) -> VoiceProgram:
    """Load `path/to/file.py:ClassName`, or the only VoiceProgram in the file."""
    path_str, _, cls_name = spec.partition(":")
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"{R}error{X}: no such file {path}")
    mod_spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(mod_spec)  # type: ignore[arg-type]
    mod_spec.loader.exec_module(module)  # type: ignore[union-attr]

    if cls_name:
        obj = getattr(module, cls_name, None)
        if obj is None:
            sys.exit(f"{R}error{X}: {path} has no {cls_name}")
    else:
        found = [
            v
            for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, VoiceProgram) and v is not VoiceProgram
        ]
        if len(found) != 1:
            sys.exit(
                f"{R}error{X}: found {len(found)} VoiceProgram classes in {path}; "
                "name one explicitly as file.py:ClassName"
            )
        obj = found[0]
    return obj()


def _print_result(res, title: str) -> None:
    print(f"\n{B}{title}{X}  {D}{res.program} · {res.testset} · {res.n_scenarios} scenarios · "
          f"config {res.config_fingerprint}{X}")
    for name, value in res.metrics.items():
        if name == "inquesto_score":
            continue
        print(f"  {G}{name:<14}{X} {_fmt(name, value, res.units.get(name, ''))}")
    for name in res.unmeasured:
        print(f"  {D}{name:<14} not measured by this runtime{X}")
    if res.score:
        if res.score.get("value") is not None:
            c = res.score["components"]
            why = " x ".join(f"{v:.2f}" for v in c.values())
            print(f"  {B}{G}inquesto_score   {res.score['value']:.1f}{X}  {D}= 100 x {why}{X}")
        else:
            print(f"  {D}inquesto_score    n/a: "
                  + ", ".join(f"{m} not measured" for m in res.score.get("missing", [])) + X)
    if res.seeds > 1:
        print(f"  {D}averaged over {res.seeds} seeds{X}")
    summary = res.failure_summary()
    if summary:
        print(f"  {D}top failures:{X}")
        for reason, count in list(summary.items())[:3]:
            print(f"    {count:>3}x  {reason}")


def _print_failure_report(res, metric: str = "task_success") -> None:
    """Where does the agent fail? By evaluator, by caller style, by scenario."""
    if not res.failures:
        print(f"\n{B}failures{X}  {D}none{X}")
        return
    print(f"\n{B}failures{X}")
    for ev, n in res.failures_by_evaluator().items():
        print(f"  {ev:<14} {n:>3}")

    if res.breakdown and metric in res.metrics:
        unit = res.units.get(metric, "")
        worst = res.worst_style(metric)
        print(f"\n{B}{metric} by caller style{X}")
        for style, m in res.breakdown.items():
            mark = f"  {R}<- weakest{X}" if style == worst else ""
            print(f"  {style:<14} {_fmt(metric, m[metric], unit):>6}{mark}")

    worst_scn = res.worst_scenarios(3)
    if worst_scn:
        print(f"\n{B}worst scenarios{X}")
        for sid, n in worst_scn:
            print(f"  {sid:<14} {n} failed checks")


def cmd_evaluate(args) -> int:
    program = _apply_overrides(_load_program(args.program), args.set)
    ts = _load_testset(args)
    evs = [evaluators.get(n) for n in args.metrics] if args.metrics else None
    on_conv = None
    if args.transcripts:
        out_dir = Path(args.transcripts)
        out_dir.mkdir(parents=True, exist_ok=True)

        def on_conv(conv):
            (out_dir / f"{conv.scenario_id}.json").write_text(
                json.dumps(asdict(conv), indent=2)
            )
            ok = "ok " if conv.task_completed else "FAIL"
            print(f"  {D}[{ok}] {conv.scenario_id}  {len(conv.turns)} turns{X}")

    res = evaluate(program, ts, evs, runtime=args.runtime, seed=args.seed,
                   on_conversation=on_conv, seeds=args.seeds, score_spec=_score_spec(args))
    _print_result(res, "evaluation")
    if args.failures:
        _print_failure_report(res)
    if args.save_baseline:
        p = gates.save_baseline(res, args.save_baseline)
        print(f"\n  {D}baseline written to {p}{X}")
    if args.json:
        Path(args.json).write_text(json.dumps(res.to_dict(), indent=2))
        print(f"  {D}result written to {args.json}{X}")
    print()
    return 0


def cmd_optimize(args) -> int:
    program = _apply_overrides(_load_program(args.program), args.set)
    ts = _load_testset(args)
    try:
        opt = optimizers.get(args.optimizer)
        cons = parse_constraints(args.constraint)
    except (KeyError, ValueError) as e:
        sys.exit(f"{R}error{X}: {e}")

    def on_trial(t):
        if args.verbose:
            note = f"  {R}x {'; '.join(t.violations)}{X}" if not t.feasible else ""
            print(f"  {D}trial {t.step:>2}  {args.metric}={t.score:.3f}{X}{note}")

    if cons:
        print(f"\n{B}constraints{X}  " + ", ".join(str(c) for c in cons))
    print(f"{D}{opt.name} search, up to {args.steps} configurations...{X}")

    try:
        out = opt.optimize(
            program, ts, metric=args.metric, steps=args.steps,
            runtime=args.runtime, seed=args.seed, constraints=cons, on_trial=on_trial,
            seeds=args.seeds, score_spec=_score_spec(args),
        )
    except KeyError as e:
        sys.exit(f"{R}error{X}: {e.args[0]}")
    _print_result(out.baseline, "baseline")
    _print_result(out.best, "optimized")

    unit = out.best.units.get(args.metric, "")
    before = _fmt(args.metric, out.baseline.metric(args.metric), unit)
    after = _fmt(args.metric, out.best.metric(args.metric), unit)
    delta = f"{out.delta * (100 if unit == '%' else 1):+.1f}{'pt' if unit == '%' else ''}"
    print(f"\n{B}{G}{args.metric}: {before} -> {after}{X}  ({delta})")

    if out.improved:
        print(f"\n{B}changes{X}")
        for k, (old, new) in out.changes().items():
            print(f"  {k}: {D}{old}{X} -> {G}{new}{X}")
    elif not out.feasible:
        print(f"\n  {R}no configuration met the constraints in {args.steps} trials{X}"
              f" {D}(baseline returned unchanged){X}")
    else:
        print(f"\n  {D}no configuration beat the baseline in {args.steps} trials{X}")

    # Any metric we didn't constrain explicitly still gets a sanity check.
    guard = {m: t for m, t in {"latency": 0.10, "cost": 0.15, "turn_taking": 0.02}.items()
             if m not in {c.metric for c in cons}}
    regs = out.regressions(guard)
    if regs:
        print(f"\n{R}watch out{X}: " + "; ".join(regs)
              + f"\n  {D}add --constraint to keep the search inside a budget{X}")

    if args.json:
        Path(args.json).write_text(json.dumps(out.to_dict(), indent=2))
        print(f"\n  {D}experiment written to {args.json}{X}")
    print()
    return 0 if out.feasible else 2


def cmd_gate(args) -> int:
    program = _apply_overrides(_load_program(args.program), args.set)
    ts = _load_testset(args)
    res = evaluate(program, ts, runtime=args.runtime, seed=args.seed, seeds=args.seeds,
                   score_spec=_score_spec(args))
    base = gates.load_baseline(args.baseline)
    verdict = gates.check(res, base)
    print()
    print(verdict.report())
    if not verdict.passed:
        print(f"\n{R}" + "\n".join("  " + v for v in verdict.violations) + X)
        print(f"\n{R}release blocked{X}\n")
        return 1
    print(f"\n{G}release clear{X}\n")
    return 0


def cmd_protocol(args) -> int:
    """inquesto protocol run|score: the Inquesto Protocol v0.1 population, record and citation line."""
    from .protocol import run as prun

    out = Path(args.out)
    if args.action == "score":
        rec = prun.score_dir(out)
    else:
        program = _apply_overrides(_load_program(args.agent), args.set)
        t_start = time.time()

        def on_call(res, i, n):
            ok = res.to_dict()["pass"]
            ev = ", ".join(f"{e.type}(S{e.severity})" for e in res.events) or ("clean" if ok else "")
            if not res.goal_achieved:
                ev = ("goal not achieved" + (", " + ev if ev else ""))
            mark = f"{G}pass{X}" if ok else f"{R}FAIL{X}"
            el = time.time() - t_start
            eta = el / i * (n - i)
            print(f"  {D}[{i}/{n}]{X} {res.scenario_id} {res.condition}/{res.group} {mark}  {ev}  {D}eta {eta / 60:.0f} min{X}", flush=True)

        only = [x for x in (args.only or "").split(",") if x]
        shard = None
        if args.shard:
            k, n = args.shard.split("/")
            shard = (int(k), int(n))
        rec = prun.run(program, out, runtime=args.runtime, limit=args.limit, on_call=on_call, only=only or None, shard=shard)
    v = rec["views"]
    print(f"\n{B}{rec['citation']}{X}")
    for k in ("behavior", "robustness", "identity", "fairness"):
        r = v[k]
        extra = f"  worst group {r['worst_group']}" if k == "fairness" and r.get("worst_group") else ""
        extra = f"  gap vs clean {r['gap_vs_clean']}" if k == "robustness" and r.get("gap_vs_clean") is not None else extra
        print(f"  {k:<11} {r['score'] if r['score'] is not None else '—':>5}  [{r['lo']}, {r['hi']}]  n={r['n']}{extra}")
    print(f"  failures by severity: {rec['failures_by_severity']}")
    print(f"  failed only on audio events: {rec['audio_only']['failed_only_on_audio_events']} of {rec['audio_only']['failed']}"
          f"  (transcript-only score would be {rec['audio_only']['score_if_transcript_only']})")
    print(f"  {D}record: {out / 'inquesto-record.json'}{X}")
    return 0


def cmd_list(args) -> int:
    print(f"\n{B}evaluators{X}  " + ", ".join(evaluators.available()))
    print(f"{B}testsets{X}    " + ", ".join(testsets.available()))
    print(f"{B}runtimes{X}    " + ", ".join(adapters.available()))
    print(f"{B}optimizers{X}  " + ", ".join(sorted(optimizers.OPTIMIZERS)) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="inquesto", description="Make voice agents measurably better.")
    p.add_argument("--version", action="version", version=f"inquesto {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("program", help="path/to/agent.py[:ClassName]")
        sp.add_argument("--suite", default="support-v1", help="testset name or path")
        sp.add_argument("--runtime", default="mock", help="runtime adapter")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a Config field, e.g. --set model=qwen2.5:7b (repeatable)")
        sp.add_argument("--limit", type=int, default=0, metavar="N",
                        help="only run the first N scenarios (smoke tests on real runtimes)")
        sp.add_argument("--seeds", type=int, default=1, metavar="N",
                        help="average every evaluation over N seeds (default 1)")
        sp.add_argument("--score-spec", default=None, metavar="SPEC",
                        help="Inquesto score budgets, e.g. latency=500:2000,cost=0.05:0.20")

    e = sub.add_parser("evaluate", help="measure an agent")
    common(e)
    e.add_argument("--metrics", nargs="*", help="evaluator names (default: all four)")
    e.add_argument("--failures", action="store_true",
                   help="break failures down by evaluator, caller style and scenario")
    e.add_argument("--save-baseline", metavar="PATH", help="write result as a gate baseline")
    e.add_argument("--json", metavar="PATH", help="write full result JSON")
    e.add_argument("--transcripts", metavar="DIR",
                   help="write each conversation as DIR/<scenario>.json as it finishes")
    e.set_defaults(func=cmd_evaluate)

    o = sub.add_parser("optimize", help="search for a better configuration")
    common(o)
    o.add_argument("--metric", default="task_success",
                   help="objective; any metric, including inquesto_score")
    o.add_argument("--steps", type=int, default=12, help="evaluation budget")
    o.add_argument("--optimizer", default="tpe", choices=sorted(optimizers.OPTIMIZERS),
                   help="tpe: Parzen-estimator search, learns from the best trials (default); "
                        "coordinate: greedy one-knob-at-a-time; random: seeded draws")
    o.add_argument("--constraint", action="append", default=[], metavar="EXPR",
                   help="only accept configs where EXPR holds, e.g. latency<=900 "
                        "or cost<=0.05 (repeatable)")
    o.add_argument("--json", metavar="PATH", help="write the experiment artifact")
    o.add_argument("-v", "--verbose", action="store_true")
    o.set_defaults(func=cmd_optimize)

    g = sub.add_parser("gate", help="block a release that regressed")
    common(g)
    g.add_argument("--baseline", required=True, help="path to a saved baseline JSON")
    g.set_defaults(func=cmd_gate)

    pr = sub.add_parser("protocol", help="Inquesto Protocol v0.1: run the call population, print the citation line")
    pr.add_argument("action", choices=["run", "score"], help="run the population, or re-score a finished directory")
    pr.add_argument("agent", nargs="?", default="examples/protocol_agent/agent.py", help="path/to/agent.py[:Class]")
    pr.add_argument("--out", required=True, metavar="DIR", help="output directory (one per agent configuration; resumable)")
    pr.add_argument("--runtime", default="pipecat", help="runtime adapter (pipecat = audio; local = transcript only)")
    pr.add_argument("--limit", type=int, default=0, metavar="N", help="only the first N calls of the population")
    pr.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override, e.g. model=qwen2.5:7b")
    pr.add_argument("--only", default="", metavar="SUBSTR[,SUBSTR]", help="only calls whose key contains one of these (smoke tests)")
    pr.add_argument("--shard", default="", metavar="K/N", help="run every N-th call starting at K; shards may share --out")
    pr.set_defaults(func=cmd_protocol)

    ls = sub.add_parser("list", help="show available evaluators, testsets, runtimes")
    ls.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
