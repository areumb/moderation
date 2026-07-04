import glob
import subprocess
import sys
from typing import List
from pathlib import Path 
import click
REPO_ROOT = Path(__file__).resolve().parents[1]
#Edit as needed
@click.command(context_settings=dict(show_default=True))
@click.option("-c", "--config", "config_path", required=True, type=str)
@click.option("--dataset", type=click.Choice(["davidson", "hatecheck_xr"]), default="davidson")
@click.option("--eval-mode", required=True, type=click.Choice(["3class", "hate_nonhate", "nonclean_clean", "hate_clean"]))
@click.option("--train-mode", required=True, type=click.Choice(["3class", "hate_nonhate", "nonclean_clean", "hate_clean"]))
@click.option("--seeds", multiple=True, required=True, type=int,
              help="One or more seeds. Example: --seeds 5 11 42 100 2021")
@click.option("--ckpt-pattern", required=True,
              help="Glob with {seed} placeholder, e.g. outputs/davidson/RoBERTa-base/hate-clean/best/seed{seed}_*.pt") #assuming you are choosing the best checkpoints.
@click.option("--hatecheck-csv", default="dataset/extended_hatecheck/reannotation.csv",
              help="Used only if --dataset hatecheck_xr")

def main(config_path: str, dataset: str, eval_mode: str, train_mode: str,
         seeds: List[int], ckpt_pattern: str, hatecheck_csv: str):
    """
    Runs hs_generalization.test across seeds and matching checkpoints without manual editing.
    """
    for seed in seeds:
        files = glob.glob(ckpt_pattern.format(seed=seed))
        if not files:
            click.echo(f"[seed={seed}] No checkpoints found. Pattern: {ckpt_pattern}")
            sys.exit(1)

        for ckpt in files:
            click.echo(f"[seed={seed}] Evaluating {ckpt}")
            args = [
                sys.executable, "-m", "hs_generalization.test",   # was hs_generalization.evaluate
                "-c", config_path,
                "--dataset", dataset,
                "--eval-mode", eval_mode,
                "--train-mode", train_mode,
                "--seed", str(seed),
                "--checkpoint", ckpt,
            ]
            if dataset == "hatecheck_xr":
                args += ["--hatecheck-csv", hatecheck_csv]
            subprocess.run(args, check=True)

if __name__ == "__main__":
    main()
