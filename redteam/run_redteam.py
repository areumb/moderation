"""Run the red-team harness and write the ASR before/after report (Module B).

    python -m redteam.run_redteam [--top-k N] [--out-dir DIR] [--gate]

Offline (default) it targets the deterministic MockLLM + StubClassifier, so the
before/after numbers are reproducible in CI. With a real LLM (LLM_BASE_URL) and
real model (MODEL_DIR) the identical harness characterises those components. The
report is written as JSON + Markdown and, with --gate, the AFTER-defense ASR is
checked against redteam/thresholds.json (build fails on a regression).

RESPONSIBLE DISCLOSURE: payloads are masked and generic; aggregate rates and the
taxonomy are published, operational specifics are not (docs/safety_case.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rag.policy_store import PolicyStore  # noqa: E402
from redteam.harness import run  # noqa: E402
from serving.config import ServingConfig  # noqa: E402

_GATE_KEYS = {
    "adjudicator_injection": "adjudicator_injection_after_max",
    "adjudicator_jailbreak": "adjudicator_jailbreak_after_max",
    "tier1_evasion": "tier1_evasion_after_max",
}


def check_gate(res: dict, thresholds: dict) -> list[str]:
    failures = []
    for surface, key in _GATE_KEYS.items():
        ceiling = thresholds.get(key)
        after = res["surfaces"][surface]["after"]["asr"]
        if ceiling is not None and after > ceiling:
            failures.append(f"{surface}: after-defense ASR {after:.3f} > ceiling {ceiling}")
    return failures


def write_reports(res: dict, out_dir: Path, smoke: bool, failures: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "smoke (mock LLM + stub)" if smoke else f"real (llm={res['llm']}, clf={res['classifier']})"
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "gate_failures": failures,
    }
    with open(out_dir / "redteam_results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **res}, f, indent=2)

    h = res["headline"]
    lines = ["# Red-team results -- attack success rate before vs after defense (Module B)", ""]
    if smoke:
        lines += [
            "> **SMOKE MODE (deterministic MockLLM + StubClassifier).** The MockLLM is",
            "> scripted to obey exactly the injected imperatives the defenses detect, so",
            "> these numbers are a *wiring and regression check* — they verify the defense",
            "> layers are applied end-to-end and the fail-closed integrity check holds,",
            "> largely by construction. They are NOT an effectiveness measurement for any",
            "> real model; for that, re-run with `LLM_BASE_URL` (and `MODEL_DIR`) set.",
            "",
        ]
    lines += [
        "> **Responsible disclosure.** All payloads are masked and use generic, well-known",
        "> technique templates. The taxonomy and aggregate rates are published; operational",
        "> specifics are withheld. See `docs/safety_case.md`.",
        "",
        f"**Headline: overall attack success rate {h['before_asr']:.3f} -> {h['after_asr']:.3f} "
        f"after defenses.**",
        "",
        "| Surface | n | ASR before | ASR after |",
        "|---|---|---|---|",
    ]
    for surface, d in res["surfaces"].items():
        lines.append(
            f"| {surface} | {d['before']['n']} | {d['before']['asr']:.3f} | {d['after']['asr']:.3f} |"
        )

    lines += [
        "", "## By technique", "",
        "| Surface | Technique | ASR before | ASR after |", "|---|---|---|---|",
    ]
    residuals = []
    for surface, d in res["surfaces"].items():
        for cat, cd in d["after"]["by_category"].items():
            before = d["before"]["by_category"][cat]["asr"]
            lines.append(f"| {surface} | {cat} | {before:.3f} | {cd['asr']:.3f} |")
            if cd["asr"] > 0:
                residuals.append(f"{surface}/{cat} (ASR {cd['asr']:.3f})")

    lines += ["", "## Residual risk (non-zero after defense)", ""]
    lines += ([f"- {r}" for r in residuals] if residuals else ["- None on this run."])
    lines += [
        "",
        "These residuals are expected and explained in `docs/safety_case.md`: the "
        "counter-speech exception can be abused to masquerade hate as quotation, and "
        "character transposition defeats lexical normalisation (it needs model-level "
        "robustness, which is why non-Clean labels still escalate to Tier 2).",
    ]
    if failures:
        lines += ["", "## Gate failures", ""] + [f"- {f}" for f in failures]
    (out_dir / "redteam_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--out-dir", default="redteam/reports")
    ap.add_argument("--gate", action="store_true", help="Fail if after-defense ASR exceeds thresholds.")
    ap.add_argument("--thresholds", default=str(REPO_ROOT / "redteam" / "thresholds.json"))
    args = ap.parse_args()

    smoke = not (os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL"))
    cfg = ServingConfig.load()
    store = PolicyStore(cfg.policy_path, cfg.chroma_dir, cfg.collection_name)
    res = run(store, top_k=args.top_k)

    failures = []
    if args.gate:
        with open(args.thresholds, encoding="utf-8") as f:
            thresholds = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        failures = check_gate(res, thresholds)

    write_reports(res, Path(args.out_dir), smoke, failures)
    h = res["headline"]
    print(f"[red-team] mode={'smoke' if smoke else 'real'} ASR {h['before_asr']:.3f} -> {h['after_asr']:.3f}")
    for surface, d in res["surfaces"].items():
        print(f"  {surface:<24} before={d['before']['asr']:.3f} after={d['after']['asr']:.3f}")
    if failures:
        print("[red-team] GATE FAILURES:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
