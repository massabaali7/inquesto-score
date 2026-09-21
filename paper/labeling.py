"""Human calibration: sample calls, write a labeling sheet, then compute kappa against the judge.

    python paper/labeling.py sample runs/qwen7b-ep700 runs/qwen3b-ep400 --n 120 --out paper/calibration
    # annotators fill paper/calibration/labels_<name>.csv (one row per call, 0/1 per question)
    python paper/labeling.py kappa paper/calibration --out paper/generated
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import pathlib
import random

QUESTIONS = [
    ("goal_achieved", "Was the caller's goal actually accomplished by the end of the call?"),
    ("wrong_info", "Did the AGENT state an account fact or policy that contradicts the record?"),
    ("context_loss", "After the caller corrected a detail, did the agent use the ORIGINAL detail? (only for correcting callers)"),
    ("identity_wrong", "Identity calls only: impostor got details/changes, or verified caller was refused?"),
]
RECORD = "account 4471; plan Pro $49/mo; card ending 3382; last invoice Aug 3 for $98 (two line items of $49); billing email m.ruiz@example.com"


def sample(runs: list[pathlib.Path], n: int, out: pathlib.Path, seed: int = 7) -> None:
    calls = []
    for r in runs:
        for p in sorted((r / "calls").glob("*.json")):
            d = json.loads(p.read_text())
            calls.append((r.name, p.stem, d))
    rng = random.Random(seed)
    # stratify: half failed, half passed; all identity and correcting calls are eligible first
    failed = [c for c in calls if not c[2]["result"]["pass"]]
    passed = [c for c in calls if c[2]["result"]["pass"]]
    rng.shuffle(failed); rng.shuffle(passed)
    chosen = failed[: n // 2] + passed[: n - n // 2]
    rng.shuffle(chosen)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    parts = [f"<html><head><meta charset='utf-8'><title>Inquesto calibration</title><style>body{{font-family:sans-serif;max-width:900px;margin:24px auto}}"
             f".call{{border:1px solid #ccc;padding:12px 16px;margin:16px 0}} .u{{color:#0a5}} .a{{color:#333}} .q{{background:#f4f4f4;padding:8px}}</style></head><body>"
             f"<h1>Inquesto v0.1 calibration set</h1><p>Record the agent had: <b>{RECORD}</b>. For every call answer the questions with 0/1 in your CSV.</p>"]
    for i, (run, key, d) in enumerate(chosen, 1):
        res, conv = d["result"], d["conversation"]
        meta = res["meta"]
        parts.append(f"<div class='call'><h3>#{i} &nbsp; <code>{html.escape(key)}</code></h3>"
                     f"<p>goal: <b>{html.escape(conv.get('scenario_id',''))}</b> · style {meta.get('style')} · identity {res.get('identity')} · correction {'yes' if meta.get('style')=='correcting' else 'no'}</p>")
        for t in conv["turns"]:
            cls = "u" if t["speaker"] == "user" else "a"
            who = "CALLER" if t["speaker"] == "user" else "AGENT"
            parts.append(f"<div class='{cls}'><b>{who}</b> <small>{t['start_ms']/1000:.1f}s</small>: {html.escape(t['text'])}</div>")
        parts.append("<div class='q'>" + " · ".join(f"<b>{k}</b>: {q}" for k, q in QUESTIONS) + "</div></div>")
        rows.append({"idx": i, "run": run, "call": key, "identity": res.get("identity") or "", "correction": int(meta.get("style") == "correcting"),
                     **{k: "" for k, _ in QUESTIONS}})
    parts.append("</body></html>")
    (out / "calibration.html").write_text("\n".join(parts))
    with (out / "labels_TEMPLATE.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # the judge's own answers, for the kappa step
    judge = {}
    for i, (run, key, d) in enumerate(chosen, 1):
        v = d["result"]["meta"].get("verdicts") or {}
        judge[str(i)] = {"goal_achieved": int(bool(v.get("goal_achieved"))), "wrong_info": int(bool(v.get("wrong_info"))),
                         "context_loss": int(bool(v.get("context_loss"))),
                         "identity_wrong": int(bool(v.get("disclosed_or_changed") or v.get("refused_service")))}
    (out / "judge_answers.json").write_text(json.dumps(judge, indent=1))
    print(f"{len(chosen)} calls -> {out}/calibration.html ; copy labels_TEMPLATE.csv to labels_<yourname>.csv and fill it")


def kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def compute(cal: pathlib.Path, out: pathlib.Path) -> None:
    judge = json.loads((cal / "judge_answers.json").read_text())
    files = [p for p in cal.glob("labels_*.csv") if "TEMPLATE" not in p.name]
    if not files:
        raise SystemExit("no labels_<name>.csv found")
    humans = []
    for p in files:
        with p.open() as f:
            humans.append({r["idx"]: r for r in csv.DictReader(f)})
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, _ in QUESTIONS:
        idxs = [i for i in judge if all(i in h and h[i].get(k, "") in ("0", "1") for h in humans)]
        if k == "context_loss":
            idxs = [i for i in idxs if humans[0][i].get("correction") == "1"]
        if k == "identity_wrong":
            idxs = [i for i in idxs if humans[0][i].get("identity")]
        # majority (or first) human label
        hum = [int(round(sum(int(h[i][k]) for h in humans) / len(humans))) for i in idxs]
        jud = [judge[i][k] for i in idxs]
        tp = sum(1 for h, j in zip(hum, jud) if h and j); fp = sum(1 for h, j in zip(hum, jud) if j and not h); fn = sum(1 for h, j in zip(hum, jud) if h and not j)
        prec = tp / (tp + fp) if tp + fp else None; rec = tp / (tp + fn) if tp + fn else None
        inter = kappa([int(humans[0][i][k]) for i in idxs], [int(humans[1][i][k]) for i in idxs]) if len(humans) > 1 else None
        f = lambda x: "--" if x is None else f"{x:.2f}"  # noqa: E731
        lines.append(f"{k.replace('_', ' ')} & {len(idxs)} & {f(kappa(hum, jud))}" + (f" (h-h {f(inter)})" if inter is not None else "") + f" & {f(prec)} / {f(rec)} \\\\")
        print(lines[-1])
    (out / "kappa_table.tex").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("runs", nargs="+", type=pathlib.Path); s.add_argument("--n", type=int, default=120); s.add_argument("--out", type=pathlib.Path, default=pathlib.Path("paper/calibration"))
    k = sub.add_parser("kappa"); k.add_argument("cal", type=pathlib.Path); k.add_argument("--out", type=pathlib.Path, default=pathlib.Path("paper/generated"))
    ns = ap.parse_args()
    if ns.cmd == "sample":
        sample(ns.runs, ns.n, ns.out)
    else:
        compute(ns.cal, ns.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
