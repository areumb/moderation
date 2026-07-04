"""Behavioral test gate over HateCheck-XR.

Runs the Tier-1 classifier over the HateCheck-XR challenge set (the
re-annotated ternary extension of HateCheck produced for the thesis), computes
per-functionality accuracy plus the full ternary confusion matrix, and writes
a JSON + Markdown report.

Two directional error rates are reported (and gated via ceilings in
thresholds.json) as first-class metrics, because the thesis found the
Hate<->Clean boundary to be where the model fails out of distribution:
- gold-Hateful->Clean ("hateful_as_clean_max"): the dominant OOD error —
  hateful cases misread as Clean when overt lexical cues are absent. This is
  the error the serving router cannot catch with probability thresholds,
  which is why it audit-samples the auto-approved bucket (serving/router.py).
- gold-Clean->Hateful ("clean_as_hateful_max"): the over-moderation
  direction — counterspeech, quotation, negation, and slur homonyms misread
  as Hateful. Sizable on HateCheck-XR per the thesis; in the service this
  direction self-escalates (non-Clean labels always reach Tier 2), where the
  adjudicator can overturn it against the allowed-content clauses.

Modes:
- Real mode (MODEL_DIR set): loads the fine-tuned model and FAILS (exit 1) if
  overall accuracy or any per-functionality accuracy drops below the
  thresholds in evals/thresholds.json. This is the CI model-quality gate.
- Smoke mode (MODEL_DIR unset): runs the StubClassifier just to verify the
  pipeline executes end-to-end and produces reports. No thresholds are
  enforced, and the numbers are meaningless by construction (the stub is not
  a model). Reports are clearly marked as smoke output.

Usage:
    python -m evals.run_behavioral_suite [--csv PATH] [--limit N] [--out-dir DIR]
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

from serving.config import LABELS  # noqa: E402
from serving.predictor import get_classifier  # noqa: E402

# Same base label mapping as hs_generalization/test.py.
LABEL_MAP = {"hateful": "Hateful", "offensive": "Offensive", "clean": "Clean"}
DEFAULT_CSV = REPO_ROOT / "datasets" / "hatecheck-xr" / "hatecheck-xr.csv"


def load_cases(csv_path: Path, limit: int | None = None) -> list[dict]:
    cases = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            gold = row.get("label_gold", "").strip().lower()
            if gold not in LABEL_MAP:
                continue
            cases.append(
                {
                    "functionality": row["functionality"].strip(),
                    "text": row["test_case"],
                    "gold": LABEL_MAP[gold],
                }
            )
            if limit and len(cases) >= limit:
                break
    if not cases:
        raise ValueError(f"No usable rows in {csv_path}")
    return cases


def evaluate(cases: list[dict], classifier) -> dict:
    per_func: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    overall = {"n": 0, "correct": 0}
    confusion: dict[str, dict[str, int]] = {g: {p: 0 for p in LABELS} for g in LABELS}
    for case in cases:
        pred = classifier.predict(case["text"])["label"]
        confusion[case["gold"]][pred] += 1
        bucket = per_func[case["functionality"]]
        bucket["n"] += 1
        overall["n"] += 1
        if pred == case["gold"]:
            bucket["correct"] += 1
            overall["correct"] += 1

    n_hateful = sum(confusion["Hateful"].values())
    n_hateful_as_clean = confusion["Hateful"]["Clean"]
    n_clean = sum(confusion["Clean"].values())
    n_clean_as_hateful = confusion["Clean"]["Hateful"]
    return {
        "overall_accuracy": overall["correct"] / overall["n"],
        "n_cases": overall["n"],
        "confusion": confusion,
        # The two OOD error directions the thesis found at the Hate<->Clean
        # boundary, tracked as first-class metrics (see module docstring).
        "hateful_as_clean": {
            "n_hateful": n_hateful,
            "n_predicted_clean": n_hateful_as_clean,
            "rate": (n_hateful_as_clean / n_hateful) if n_hateful else None,
        },
        "clean_as_hateful": {
            "n_clean": n_clean,
            "n_predicted_hateful": n_clean_as_hateful,
            "rate": (n_clean_as_hateful / n_clean) if n_clean else None,
        },
        "per_functionality": {
            name: {"n": b["n"], "accuracy": b["correct"] / b["n"]} for name, b in sorted(per_func.items())
        },
    }


def check_thresholds(results: dict, thresholds: dict) -> list[str]:
    failures = []
    if results["overall_accuracy"] < thresholds.get("overall", 0.0):
        failures.append(
            f"overall accuracy {results['overall_accuracy']:.3f} < {thresholds['overall']}"
        )
    default_floor = thresholds.get("default", 0.0)
    for name, res in results["per_functionality"].items():
        floor = thresholds.get(name, default_floor)
        if res["accuracy"] < floor:
            failures.append(f"{name}: accuracy {res['accuracy']:.3f} < {floor}")

    # Ceilings (not floors) on the two directional Hate<->Clean error rates.
    for key, results_key, desc in (
        ("hateful_as_clean_max", "hateful_as_clean", "gold-Hateful predicted Clean"),
        ("clean_as_hateful_max", "clean_as_hateful", "gold-Clean predicted Hateful"),
    ):
        ceiling = thresholds.get(key)
        rate = results[results_key]["rate"]
        if ceiling is not None and rate is not None and rate > ceiling:
            failures.append(f"{desc}: rate {rate:.3f} > ceiling {ceiling}")
    return failures


def write_reports(results: dict, out_dir: Path, smoke: bool, engine: str, failures: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if smoke else "real",
        "classifier_engine": engine,
        "thresholds_enforced": not smoke,
        "failures": failures,
    }
    with open(out_dir / "behavioral_results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **results}, f, indent=2)

    lines = ["# HateCheck-XR behavioral report", ""]
    if smoke:
        lines += [
            "> **SMOKE MODE** — run with the deterministic StubClassifier (no trained",
            "> model). Numbers below only prove the pipeline executes; they say nothing",
            "> about model quality.",
            "",
        ]
    hac = results["hateful_as_clean"]
    hac_line = (
        f"- Gold-Hateful predicted Clean: **{hac['rate']:.3f}** "
        f"({hac['n_predicted_clean']}/{hac['n_hateful']}) — the dominant OOD error mode found in the thesis"
        if hac["rate"] is not None
        else "- Gold-Hateful predicted Clean: n/a (no gold-Hateful cases in this run)"
    )
    cah = results["clean_as_hateful"]
    cah_line = (
        f"- Gold-Clean predicted Hateful: **{cah['rate']:.3f}** "
        f"({cah['n_predicted_hateful']}/{cah['n_clean']}) — over-moderation of counterspeech/quotation/"
        "negation/homonyms"
        if cah["rate"] is not None
        else "- Gold-Clean predicted Hateful: n/a (no gold-Clean cases in this run)"
    )
    lines += [
        f"- Mode: `{meta['mode']}` | Engine: `{engine}` | Cases: {results['n_cases']}",
        f"- Overall accuracy: **{results['overall_accuracy']:.3f}**",
        hac_line,
        cah_line,
        "",
        "## Confusion matrix (gold × predicted)",
        "",
        "| gold \\ pred | " + " | ".join(LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]
    for gold in LABELS:
        row = " | ".join(str(results["confusion"][gold][p]) for p in LABELS)
        lines.append(f"| {gold} | {row} |")
    lines += [
        "",
        "## Per-functionality accuracy",
        "",
        "| Functionality | n | Accuracy |",
        "|---|---|---|",
    ]
    for name, res in results["per_functionality"].items():
        lines.append(f"| {name} | {res['n']} | {res['accuracy']:.3f} |")
    if failures:
        lines += ["", "## Threshold failures", ""] + [f"- {f}" for f in failures]
    (out_dir / "behavioral_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases.")
    parser.add_argument("--out-dir", default="evals/reports")
    parser.add_argument("--thresholds", default=str(REPO_ROOT / "evals" / "thresholds.json"))
    args = parser.parse_args()

    smoke = not os.environ.get("MODEL_DIR")
    classifier = get_classifier()
    cases = load_cases(Path(args.csv), limit=args.limit)
    results = evaluate(cases, classifier)

    failures: list[str] = []
    if not smoke:
        with open(args.thresholds, encoding="utf-8") as f:
            thresholds = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        failures = check_thresholds(results, thresholds)

    write_reports(results, Path(args.out_dir), smoke, classifier.name, failures)

    print(f"[behavioral-suite] mode={'smoke' if smoke else 'real'} "
          f"cases={results['n_cases']} overall_acc={results['overall_accuracy']:.3f}")
    if failures:
        print("[behavioral-suite] THRESHOLD FAILURES:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
