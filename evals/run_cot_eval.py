"""Module A -- chain-of-thought evaluation, evaluated honestly.

Compares three Tier-2 adjudication strategies on the same cases and retrieval:

    direct            single verdict, no explicit reasoning (the original)
    cot               one chain-of-thought path
    self_consistency  k CoT paths, majority label (Wang et al., 2022)

It reports, per strategy, not just accuracy but two model-agnostic
*faithfulness* metrics (rag/reasoning.py), because a rationale that reads well
but does not reflect the actual decision is a governance liability, not an
asset:

    citation_grounded          every cited clause appears in the reasoning
    reasoning_label_agreement  the reasoning's stated decision == final_label

and, on the labelled probe set only, controlling_clause_cited -- whether the
model cited the clause the case actually turns on.

Offline (default) it runs on evals/cot_probe.py, a masked diagnostic set, with
the deterministic MockLLM; the numbers characterise the *mechanism* (how the
strategies differ), not a real model -- exactly like the smoke mode of the
behavioural suite. For real-model numbers, run with a real LLM and the real
HateCheck-XR hard slice:

    LLM_BASE_URL=... LLM_MODEL=... \
      python -m evals.run_cot_eval --csv datasets/hatecheck-xr/hatecheck-xr.csv --hard-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evals.cot_probe import load_probe_cases  # noqa: E402
from rag.llm import get_llm  # noqa: E402
from rag.policy_store import PolicyStore  # noqa: E402
from rag.reasoning import faithfulness, make_adjudicator  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from serving.config import ServingConfig  # noqa: E402

LABEL_MAP = {"hateful": "Hateful", "offensive": "Offensive", "clean": "Clean"}
# HateCheck-XR functionalities that are context-dependent (the hard slice).
REAL_HARD_FUNCS = {
    "counter_quote_clean", "counter_ref_clean", "negate_neg_clean", "slur_homonym_clean",
    "slur_reclaimed_clean", "slur_reclaimed_offensive", "ident_neutral_clean", "ident_pos_clean",
    "derog_impl_h", "phrase_question_h", "phrase_opinion_h",
}


def load_real_slice(csv_path: Path, hard_only: bool, limit: int | None) -> list[dict]:
    cases = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            gold = row.get("label_gold", "").strip().lower()
            func = row["functionality"].strip()
            if gold not in LABEL_MAP:
                continue
            if hard_only and func not in REAL_HARD_FUNCS:
                continue
            cases.append({"functionality": func, "text": row["test_case"], "gold": LABEL_MAP[gold],
                          "controlling_clause": None, "hard": func in REAL_HARD_FUNCS})
            if limit and len(cases) >= limit:
                break
    if not cases:
        raise ValueError(f"No usable rows in {csv_path}")
    return cases


def _rate(items: list) -> float | None:
    vals = [x for x in items if x is not None]
    return (sum(1 for x in vals if x) / len(vals)) if vals else None


def evaluate_strategy(cases, store, cfg, strategy: str, samples: int, top_k: int) -> dict:
    adj = make_adjudicator(Retriever(store, top_k=top_k), get_llm(), strategy=strategy, samples=samples)
    per_func = defaultdict(lambda: {"n": 0, "correct": 0})
    correct = hard_correct = hard_n = easy_correct = easy_n = 0
    grounded, agreement, controlling = [], [], []
    for c in cases:
        v = adj.adjudicate(c["text"])
        ok = v["final_label"] == c["gold"]
        correct += ok
        pf = per_func[c["functionality"]]
        pf["n"] += 1
        pf["correct"] += ok
        if c["hard"]:
            hard_n += 1
            hard_correct += ok
        else:
            easy_n += 1
            easy_correct += ok
        ff = faithfulness(v)
        grounded.append(ff["citation_grounded"])
        agreement.append(ff["reasoning_label_agreement"])
        if c.get("controlling_clause"):
            controlling.append(c["controlling_clause"] in v["cited_clauses"])
    return {
        "strategy": strategy,
        "samples": samples if strategy == "self_consistency" else 1,
        "overall_accuracy": correct / len(cases),
        "hard_accuracy": (hard_correct / hard_n) if hard_n else None,
        "easy_accuracy": (easy_correct / easy_n) if easy_n else None,
        "citation_grounded_rate": _rate(grounded),
        "reasoning_label_agreement_rate": _rate(agreement),
        "controlling_clause_cited_rate": _rate(controlling),
        "per_functionality": {
            k: {"n": b["n"], "accuracy": b["correct"] / b["n"]} for k, b in sorted(per_func.items())
        },
    }


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def write_reports(
    rows: list[dict], cases: list[dict], out_dir: Path, smoke: bool, llm_name: str, source: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke (deterministic MockLLM)" if smoke else f"real ({llm_name})",
        "source": source,
        "n_cases": len(cases),
        "n_hard": sum(c["hard"] for c in cases),
    }
    with open(out_dir / "cot_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "strategies": rows}, f, indent=2)

    lines = ["# Chain-of-thought adjudication -- strategy comparison (Module A)", ""]
    if smoke:
        lines += [
            "> **SMOKE MODE (deterministic MockLLM).** These numbers characterise how the",
            "> strategies differ *mechanically* on a masked probe set; they are NOT a real",
            "> model's accuracy. Re-run with `LLM_BASE_URL` set and `--csv` on the real",
            "> HateCheck-XR hard slice for model-specific numbers.",
            "",
        ]
    lines += [
        f"- Mode: `{meta['mode']}` | Source: `{source}` | Cases: {meta['n_cases']} "
        f"({meta['n_hard']} hard / {meta['n_cases'] - meta['n_hard']} easy)",
        "",
        "| Strategy | Overall | Hard | Easy | Grounded | Reason=label | Ctrl-clause |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = f"{r['strategy']}" + (f" (k={r['samples']})" if r["strategy"] == "self_consistency" else "")
        lines.append(
            f"| {name} | {_fmt(r['overall_accuracy'])} | {_fmt(r['hard_accuracy'])} "
            f"| {_fmt(r['easy_accuracy'])} | {_fmt(r['citation_grounded_rate'])} "
            f"| {_fmt(r['reasoning_label_agreement_rate'])} | {_fmt(r['controlling_clause_cited_rate'])} |"
        )
    lines += [
        "",
        "Reading this: CoT and self-consistency should help most on the **hard slice** "
        "(context-dependent counter-speech / negation / homonym / reclaimed cases), where a "
        "single surface reading fails. `citation_grounded` and `reasoning=label` are "
        "faithfulness checks -- a rationale that scores low is decorative, not load-bearing, "
        "and should not be trusted in an appeal or audit.",
        "",
        "## Hard-slice accuracy by functionality",
        "",
        "| Functionality | " + " | ".join(
            r["strategy"] + (f"(k={r['samples']})" if r["strategy"] == "self_consistency" else "")
            for r in rows
        ) + " |",
        "|---|" + "---|" * len(rows),
    ]
    hard_funcs = sorted({c["functionality"] for c in cases if c["hard"]})
    for func in hard_funcs:
        cells = []
        for r in rows:
            pf = r["per_functionality"].get(func)
            cells.append(_fmt(pf["accuracy"]) if pf else "n/a")
        lines.append(f"| {func} | " + " | ".join(cells) + " |")
    (out_dir / "cot_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="Real HateCheck-XR CSV (use with a real LLM).")
    ap.add_argument("--hard-only", action="store_true", help="With --csv, keep context-dependent funcs only.")
    ap.add_argument("--samples", type=int, default=5, help="Self-consistency paths (default 5).")
    ap.add_argument("--top-k", type=int, default=6, help="Clauses retrieved per case.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="evals/reports")
    args = ap.parse_args()

    smoke = not (os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL"))
    cfg = ServingConfig.load()
    store = PolicyStore(cfg.policy_path, cfg.chroma_dir, cfg.collection_name)
    llm_name = get_llm().name

    if args.csv:
        cases = load_real_slice(Path(args.csv), args.hard_only, args.limit)
        source = f"{Path(args.csv).name}" + (" (hard slice)" if args.hard_only else "")
    else:
        cases = load_probe_cases()
        if args.limit:
            cases = cases[: args.limit]
        source = "evals/cot_probe.py (masked probe)"

    plan = [("direct", 1), ("cot", 1), ("self_consistency", args.samples)]
    rows = [evaluate_strategy(cases, store, cfg, s, k, args.top_k) for s, k in plan]
    write_reports(rows, cases, Path(args.out_dir), smoke, llm_name, source)

    print(f"[cot-eval] mode={'smoke' if smoke else 'real'} source='{source}' cases={len(cases)}")
    for r in rows:
        print(f"  {r['strategy']:<16} overall={_fmt(r['overall_accuracy'])} "
              f"hard={_fmt(r['hard_accuracy'])} grounded={_fmt(r['citation_grounded_rate'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
