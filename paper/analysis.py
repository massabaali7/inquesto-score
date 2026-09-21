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
    m = str(cfg.get("model", "?")).replace("qwen2.5:", "Qwen2.5-").replace("llama3.1:", "Llama-3.1-").replace("gemma2:", "Gemma-2-")
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
        rows.append(f"{name} & {fmt(r['score'])} & [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] & "
                    f"{fmt(v['behavior']['score'])} & {fmt(v['robustness']['score'])} & {fmt(v['identity']['score'])} & {fmt(v['fairness']['score'])} & "
                    f"{sev['S3']}/{sev['S4']}/{sev['S5']}" + ROW_END)
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
    (ns.out / "main_table.tex").write_text(tab("@{}lcc cccc c@{}", "Agent (LLM / endpointing) & IS & 95\\,\\% CI & B & R & I & F & S3/4/5", main_table(recs)))
    (ns.out / "ablation_table.tex").write_text(tab("@{}lcccc@{}", "Agent & IS (rate) & mean of views & geometric & min view", ablation_table(recs)))
    (ns.out / "audio_table.tex").write_text(tab("@{}lccc@{}", "Agent & IS & transcript-only & audio-only failures", audio_table(recs)))
    (ns.out / "fairness_table.tex").write_text(tab("@{}lcccc c@{}", "Agent & US-F & US-M & UK-F & UK-M & $\\Delta$ worst", fairness_table(recs)))
    (ns.out / "failure_types.tex").write_text(tab("@{}lc@{}", "Event & count", failure_types(recs)))
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
        worst_gap = min((r["views"]["fairness"].get("delta_vs_reference") for _, r in recs if r["views"]["fairness"].get("delta_vs_reference") is not None), default=None)
        macros.append(f"\\newcommand{{\\worstgap}}{{{fmt(worst_gap, 1)}}}")
    (ns.out / "macros.tex").write_text("\n".join(macros) + "\n")
    for name, r in recs:
        print(f"{name:<28} {r['citation']}")
    print(f"written to {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
