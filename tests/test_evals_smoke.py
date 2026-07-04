"""Smoke test for the behavioral suite: pipeline runs end-to-end offline and
produces both reports. Uses a small case limit to stay fast."""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_behavioral_suite_smoke(tmp_path, offline_env):
    out_dir = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable, "-m", "evals.run_behavioral_suite",
            "--limit", "200",
            "--out-dir", str(out_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    results_json = json.loads((out_dir / "behavioral_results.json").read_text(encoding="utf-8"))
    assert results_json["meta"]["mode"] == "smoke"
    assert results_json["meta"]["thresholds_enforced"] is False
    assert results_json["n_cases"] == 200
    assert results_json["per_functionality"]
    assert set(results_json["confusion"]) == {"Hateful", "Offensive", "Clean"}
    hac = results_json["hateful_as_clean"]
    assert set(hac) == {"n_hateful", "n_predicted_clean", "rate"}
    cah = results_json["clean_as_hateful"]
    assert set(cah) == {"n_clean", "n_predicted_hateful", "rate"}
    total_confusion = sum(sum(row.values()) for row in results_json["confusion"].values())
    assert total_confusion == 200

    report = (out_dir / "behavioral_report.md").read_text(encoding="utf-8")
    assert "SMOKE MODE" in report
