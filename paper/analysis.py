"""Tables and numbers for the paper, straight from the records. Never type a number by hand.

    python paper/analysis.py runs/qwen7b-ep700 runs/qwen3b-ep400 ... --out paper/generated
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def load(run: pathlib.Path) -> dict:
    return json.loads((run / "inquesto-record.json").read_text())


def label(rec: dict, run: pathlib.Path) -> str:
    cfg = rec.get("config") or {}
    raw = str(cfg.get("model", "?"))
    hosted = {"gpt-4.1-mini": "GPT-4.1-mini$^h$", "gpt-5.4-mini": "GPT-5.4-mini$^h$", "gemini/gemini-2.5-flash": "Gemini-2.5-Flash$^h$",
              "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude-Haiku-4.5$^h$", "us.anthropic.claude-sonnet-4-6": "Claude-Sonnet-4.6$^h$",
              "us.meta.llama3-1-8b-instruct-v1:0": "Llama-3.1-8B$^h$"}
    if raw in hosted:
        return f"{hosted[raw]} / {cfg.get('endpointing_ms', '?')} ms"
    m = raw.replace("qwen2.5:", "Qwen2.5-").replace("llama3.1:", "Llama-3.1-").replace("gemma2:", "Gemma-2-")
    m = m[:-1] + "B" if m.endswith("b") else m
    return f"{m} / {cfg.get('endpointing_ms', '?')} ms"


ROW_END = " \\\\"


def fmt(x, nd=0):
    return "--" if x is None else f"{x:.{nd}f}"


def main_table(recs: list[tuple[str, dict]]) -> str:
    rows = []
    for name, r in recs:
        v = r["views"]
        sev = r["failures_by_severity"]
        au = r["audio_only"]
        bg = v["fairness"]["by_group"]
        groups = " & ".join(fmt(bg[g]["score"]) if g in bg else "--" for g in ("us_female", "us_male", "uk_female", "uk_male"))
        rows.append(f"{name} & {fmt(r['score'])} & [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] & "
                    f"{fmt(v['behavior']['score'])} & {fmt(v['robustness']['score'])} & {fmt(v['identity']['score'])} & {fmt(v['fairness']['score'])} & "
                    f"{groups} & {sev['S3']}/{sev['S4']}/{sev['S5']} & {fmt(au['score_if_transcript_only'])} & {au['failed_only_on_audio_events']}" + ROW_END)
    return "\n".join(rows)


def ablation_table(recs: list[tuple[str, dict]]) -> str:
    rows = []
    for name, r in recs:
        a = r["aggregation_ablation"]
        rows.append(f"{name} & {fmt(r['score'])} & {fmt(a['mean_of_views'])} & {fmt(a['geometric_mean_of_views'])} & {fmt(a['min_view'])}" + ROW_END)
    return "\n".join(rows)


def audio_table(recs: list[tuple[str, dict]]) -> str:
    rows = []
    for name, r in recs:
        a = r["audio_only"]
        rows.append(f"{name} & {fmt(r['score'])} & {fmt(a['score_if_transcript_only'])} & {a['failed_only_on_audio_events']}/{a['failed']}" + ROW_END)
    return "\n".join(rows)


def fairness_table(recs: list[tuple[str, dict]]) -> str:
    rows = []
    for name, r in recs:
        bg = r["views"]["fairness"]["by_group"]
        cells = " & ".join(fmt(bg[g]["score"]) if g in bg else "--" for g in ("us_female", "us_male", "uk_female", "uk_male"))
        rows.append(f"{name} & {cells} & {fmt(r['views']['fairness'].get('delta_vs_reference'), 1)}" + ROW_END)
    return "\n".join(rows)


def sensitivity_table(recs: list[tuple[str, dict]]) -> str:
    rows = []
    for name, r in recs:
        ss = r.get("severity_sensitivity") or {}
        cells = " & ".join(fmt(ss.get(f"S{t}", {}).get("score")) for t in (2, 3, 4, 5))
        rows.append(f"{name} & {cells}" + ROW_END)
    return "\n".join(rows)


def failure_types(recs: list[tuple[str, dict]]) -> str:
    total: dict[str, int] = {}
    for _, r in recs:
        for k, n in r["failures_by_type"].items():
            total[k] = total.get(k, 0) + n
    return "\n".join(f"{k.replace('_', ' ')} & {n}" + ROW_END for k, n in sorted(total.items(), key=lambda kv: -kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("paper/generated"))
    ns = ap.parse_args()
    ns.out.mkdir(parents=True, exist_ok=True)
    recs = [(label(load(r), r), load(r)) for r in ns.runs if (r / "inquesto-record.json").exists()]
    recs.sort(key=lambda x: -(x[1]["score"] or 0))
    def tab(spec_: str, head: str, body: str) -> str:
        return f"\\begin{{tabular}}{{{spec_}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    (ns.out / "main_table.tex").write_text(tab("@{}lcc cccc cccc c cc@{}", "Agent (LLM / endpointing) & IS & 95\\,\\% CI & B & R & I & F & US-F & US-M & UK-F & UK-M & S3/4/5 & tx-only & audio-only", main_table(recs)))
    (ns.out / "ablation_table.tex").write_text(tab("@{}lcccc@{}", "Agent & IS (rate) & mean of views & geometric & min view", ablation_table(recs)))
    (ns.out / "audio_table.tex").write_text(tab("@{}lccc@{}", "Agent & IS & transcript-only & audio-only failures", audio_table(recs)))
    (ns.out / "fairness_table.tex").write_text(tab("@{}lcccc c@{}", "Agent & US-F & US-M & UK-F & UK-M & $\\Delta$ worst", fairness_table(recs)))
    (ns.out / "failure_types.tex").write_text(tab("@{}lc@{}", "Event & count", failure_types(recs)))
    (ns.out / "sensitivity_table.tex").write_text(tab("@{}lcccc@{}", "Agent & S2+ & \\textbf{S3+ (v0.1)} & S4+ & S5", sensitivity_table(recs)))
    n_calls = sum(r["n"] for _, r in recs)
    macros = [f"\\newcommand{{\\nagents}}{{{len(recs)}}}", f"\\newcommand{{\\ncalls}}{{{n_calls}}}"]
    if recs:
        best, worst = recs[0], recs[-1]
        macros += [f"\\newcommand{{\\bestscore}}{{{fmt(best[1]['score'])}}}", f"\\newcommand{{\\worstscore}}{{{fmt(worst[1]['score'])}}}",
                   f"\\newcommand{{\\bestname}}{{{best[0]}}}", f"\\newcommand{{\\worstname}}{{{worst[0]}}}"]
        audio_frac = [r["audio_only"]["fraction"] for _, r in recs if r["audio_only"]["fraction"] is not None]
        macros.append(f"\\newcommand{{\\audiofrac}}{{{fmt(100 * sum(audio_frac) / len(audio_frac)) if audio_frac else '--'}}}")
        deltas = [r["audio_only"]["score_if_transcript_only"] - r["score"] for _, r in recs if r["score"] is not None]
        macros.append(f"\\newcommand{{\\audiodelta}}{{{fmt(sum(deltas) / len(deltas)) if deltas else '--'}}}")
        failed = sum(r["audio_only"]["failed"] for _, r in recs)
        pnd = sum(r["failures_by_type"].get("promised_not_done", 0) for _, r in recs)
        macros.append(f"\\newcommand{{\\pndshare}}{{{fmt(100 * pnd / failed) if failed else '--'}}}")
        macros.append(f"\\newcommand{{\\pndcount}}{{{pnd}}}")
        by_cfg = {(str((r.get('config') or {}).get('model')), (r.get('config') or {}).get('endpointing_ms')): r["score"] for _, r in recs}
        for ep, name in ((400, "epfour"), (700, "epseven"), (1100, "epeleven")):
            macros.append(f"\\newcommand{{\\{name}}}{{{fmt(by_cfg.get(('qwen2.5:7b', ep)))}}}")
        def find(model):
            return next((r for _, r in recs if str((r.get("config") or {}).get("model")) == model), None)
        ll, lh = find("llama3.1:8b"), find("us.meta.llama3-1-8b-instruct-v1:0")
        macros.append(f"\\newcommand{{\\llamalocal}}{{{fmt(ll['score']) if ll else '--'}}}")
        macros.append(f"\\newcommand{{\\llamahosted}}{{{fmt(lh['score']) if lh else '--'}}}")
        macros.append(f"\\newcommand{{\\llamalocaltx}}{{{fmt(ll['audio_only']['score_if_transcript_only']) if ll else '--'}}}")
        macros.append(f"\\newcommand{{\\llamahostedtx}}{{{fmt(lh['audio_only']['score_if_transcript_only']) if lh else '--'}}}")
        ab = [r["aggregation_ablation"] for _, r in recs if r["aggregation_ablation"]["mean_of_views"] is not None]
        if ab:
            for key, name in (("mean_of_views", "aggmean"), ("geometric_mean_of_views", "agggeo"), ("min_view", "aggmin")):
                macros.append(f"\\newcommand{{\\{name}delta}}{{{fmt(sum(x[key] - r['score'] for x, (_, r) in zip(ab, [(n, rr) for n, rr in recs if rr['aggregation_ablation']['mean_of_views'] is not None])) / len(ab), 1)}}}")
        def lat(run: pathlib.Path):
            import statistics
            if not (run / "calls").exists():
                return None, None
            ttfb, resp = [], []
            for f in (run / "calls").glob("*.json"):
                d = json.loads(f.read_text())["conversation"]
                ttfb += [t["llm_ttfb_ms"] for t in d["metadata"].get("turns_detail", [])]
                resp += list(d["response_latencies_ms"])
            return (statistics.median(ttfb) if ttfb else None, statistics.median(resp) if resp else None)
        for tag, run in (("local", "llama8b-ep700"), ("hosted", "us-meta-llama3-1-8b-instruct-v1-0-ep700")):
            t_, r_ = lat(pathlib.Path("runs") / run)
            macros.append(f"\\newcommand{{\\ttfb{tag}}}{{{fmt(t_ / 1000, 2) if t_ else '--'}}}")
            macros.append(f"\\newcommand{{\\lat{tag}}}{{{fmt(r_ / 1000, 1) if r_ else '--'}}}")
        hk = find("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        macros.append(f"\\newcommand{{\\haikuis}}{{{fmt(hk['score']) if hk else '--'}}}")
        macros.append(f"\\newcommand{{\\haikutx}}{{{fmt(hk['audio_only']['score_if_transcript_only']) if hk else '--'}}}")
        besttx = max(recs, key=lambda x: x[1]["audio_only"]["score_if_transcript_only"] or 0) if recs else None
        macros.append(f"\\newcommand{{\\besttxname}}{{{besttx[0] if besttx else '--'}}}")
        macros.append(f"\\newcommand{{\\besttx}}{{{fmt(besttx[1]['audio_only']['score_if_transcript_only']) if besttx else '--'}}}")
        macros.append(f"\\newcommand{{\\besttxis}}{{{fmt(besttx[1]['score']) if besttx else '--'}}}")
        macros.append(f"\\newcommand{{\\nfalseaccept}}{{{sum(r['failures_by_type'].get('false_accept', 0) for _, r in recs)}}}")
        macros.append(f"\\newcommand{{\\nfalsereject}}{{{sum(r['failures_by_type'].get('false_reject', 0) for _, r in recs)}}}")
        ids = [r["views"]["identity"]["score"] for _, r in recs if r["views"]["identity"]["score"] is not None]
        macros.append(f"\\newcommand{{\\idmin}}{{{fmt(min(ids)) if ids else '--'}}}")
        macros.append(f"\\newcommand{{\\idmax}}{{{fmt(max(ids)) if ids else '--'}}}")
        macros.append(f"\\newcommand{{\\nunverified}}{{{sum(r['failures_by_type'].get('unverified_action', 0) for _, r in recs)}}}")
        v0 = recs[0][1]["views"]
        for k, name in (("behavior", "nB"), ("robustness", "nR"), ("identity", "nI")):
            macros.append(f"\\newcommand{{\\{name}}}{{{v0[k]['n']}}}")
        gs = [g["n"] for g in v0["fairness"]["by_group"].values()]
        macros.append(f"\\newcommand{{\\nG}}{{{min(gs) if gs else '--'}--{max(gs) if gs else '--'}}}")
        best = recs[0][1]
        macros.append(f"\\newcommand{{\\bestcite}}{{{best['citation'].split('; views')[0].replace('%', chr(92) + '%')}}}")
        ss = [r.get("severity_sensitivity") for _, r in recs if r.get("severity_sensitivity")]
        if ss:
            for t in (2, 4, 5):
                macros.append(f"\\newcommand{{\\sensS{'two' if t == 2 else 'four' if t == 4 else 'five'}}}{{{fmt(sum(x[f'S{t}']['score'] - x['S3']['score'] for x in ss) / len(ss), 1)}}}")
        # agreement between the pinned judge and the previous judge model, over every stored call
        agree, tot = {}, {}
        for run in ns.runs:
            for f in (run / "calls").glob("*.json"):
                m = json.loads(f.read_text())["conversation"]["metadata"]
                new_v, prevs = m.get("verdicts") or {}, m.get("verdicts_prev") or []
                if not prevs:
                    continue
                prev = prevs[-1]
                for k in ("goal_achieved", "wrong_info", "context_loss"):
                    if k in prev and k in new_v:
                        tot[k] = tot.get(k, 0) + 1
                        agree[k] = agree.get(k, 0) + int(bool(prev[k]) == bool(new_v[k]))
        for k, name in (("goal_achieved", "jjgoal"), ("wrong_info", "jjwrong"), ("context_loss", "jjcontext")):
            macros.append(f"\\newcommand{{\\{name}}}{{{fmt(100 * agree[k] / tot[k]) if tot.get(k) else '--'}}}")
        # judge-free variant: state predicates plus audio and trace events only; how much does the ranking depend on the judge?
        from inquesto.protocol import spec as _spec
        JUDGED = {"context_loss", "wrong_info", "false_reject", "promised_not_done", "false_accept"}
        pairs = []
        for run in ns.runs:
            full = free = n = 0
            for f in (run / "calls").glob("*.json"):
                c = json.loads(f.read_text()); r = c["result"]; tools = c["conversation"]["tool_calls"]
                req = _spec.REQUIRED_ACTION.get(r["meta"].get("family"))
                if r.get("identity") == "impostor":
                    g = not any(t in _spec.ACTION_TOOLS for t in tools)
                elif req:
                    g = req in tools
                else:
                    g = True
                worst = max([e["severity"] for e in r["events"] if e["type"] not in JUDGED] or [0])
                full += int(r["pass"]); free += int(g and worst < 3); n += 1
            if n:
                pairs.append((100 * full / n, 100 * free / n))
        if len(pairs) > 2:
            def rk(v):
                o = sorted(range(len(v)), key=lambda i: -v[i]); out = [0] * len(v)
                for i, j in enumerate(o): out[j] = i
                return out
            ra, rb = rk([p[0] for p in pairs]), rk([p[1] for p in pairs]); m = len(pairs)
            rho = 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (m * (m * m - 1))
            macros.append(f"\\newcommand{{\\jfrho}}{{{rho:.2f}}}")
            macros.append(f"\\newcommand{{\\jfshift}}{{{fmt(sum(b - a_ for a_, b in pairs) / m)}}}")
        hosted = [r for _, r in recs if "$^h$" in _]
        macros.append(f"\\newcommand{{\\nhosted}}{{{len(hosted)}}}")
        worst_gap = min((r["views"]["fairness"].get("delta_vs_reference") for _, r in recs if r["views"]["fairness"].get("delta_vs_reference") is not None), default=None)
        macros.append(f"\\newcommand{{\\worstgap}}{{{fmt(worst_gap, 1)}}}")
    (ns.out / "macros.tex").write_text("\n".join(macros) + "\n")
    for name, r in recs:
        print(f"{name:<28} {r['citation']}")
    print(f"written to {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
