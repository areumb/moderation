from __future__ import annotations
import functools
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import click
import evaluate
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, AutoModelForSequenceClassification, AutoTokenizer

from hs_generalization.train import get_dataloader, combine_compute
from hs_generalization.modes import Mode, num_labels_for, apply_eval_scheme, projection_for
from hs_generalization.utils import get_dataset, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")


def evaluate_loop(
    model: Any,
    dataloader: DataLoader,
    metric: Any,
    device: str,
    project_logits_to_eval,  # Callable: logits -> predicted eval labels
    compute_loss: bool = True,
) -> Tuple[float, Dict, Any, Any, Any, Any]:
    model.eval()

    preds = torch.tensor([], dtype=torch.long)
    refs = torch.tensor([], dtype=torch.long)
    confs = torch.tensor([])

    losses = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # The loss is only meaningful (and only safe to compute) when the reference labels
            # live in the same label space the model was trained on.
            labels = batch["labels"].to(device) if compute_loss else None
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            y_hat_eval = project_logits_to_eval(outputs.logits)

            preds = torch.cat([preds, y_hat_eval.to("cpu")])
            refs = torch.cat([refs, batch["labels"].to("cpu")])
            confs = torch.cat([confs, outputs.logits.softmax(dim=-1).to("cpu")])

            if compute_loss:
                losses.append(outputs.loss.detach().cpu().numpy())

    eval_loss = float(np.mean(losses)) if losses else float("nan")
    score_micro = metric.compute(predictions=preds, references=refs, average="micro")
    score_macro = metric.compute(predictions=preds, references=refs, average="macro")
    metrics_micro = {f"eval_micro_{k}": v for k, v in score_micro.items()}
    metrics_macro = {f"eval_macro_{k}": v for k, v in score_macro.items()}
    metrics = metrics_micro | metrics_macro
    cm = confusion_matrix(refs, preds)

    return eval_loss, metrics, preds.int().tolist(), cm, confs.tolist(), refs.int().tolist()


@click.command(context_settings=dict(show_default=True))
@click.option("-c", "--config-path", required=True, type=str, help="Path to test/eval config JSON.")
@click.option("--dataset", type=click.Choice(["davidson", "hatecheck_xr"]), default="davidson",
              help="Which test set to evaluate on.")
@click.option("--eval-mode", type=click.Choice([m.value for m in Mode]),
              default=Mode.three_class.value,
              help="How to score the ground-truth labels.")
@click.option("--train-mode", type=click.Choice([m.value for m in Mode]),
              default=Mode.three_class.value,
              help="Label space the checkpoint was trained on (used to project logits).")
@click.option("--seed", type=int, default=None, help="Seed; expands any {seed} placeholders in config.")
@click.option("--checkpoint", type=str, default=None, help="Override checkpoint path.")
@click.option("--hatecheck-csv", type=str, default="datasets/hatecheck-xr/hatecheck-xr.csv",
              help="Only used with --dataset hatecheck_xr.")
def main(config_path: str, dataset: str, eval_mode: str, train_mode: str,
         seed: int | None, checkpoint: str | None, hatecheck_csv: str):
    """
    Unified evaluator for Davidson dataset (test split) and HateCheck-XR (CSV).
    """
    config = load_config(config_path)

    # Seed + path placeholders (only formats keys that are present in the config).
    if seed is not None:
        for top, key in [
            ("pipeline", "seed"), ("task", "checkpoint"),
            ("pipeline", "output_predictions"), ("wandb", "run_name"),
        ]:
            if top in config and key in config[top]:
                config[top][key] = str(config[top][key]).format(seed=seed)
        config["pipeline"]["seed"] = int(config["pipeline"].get("seed", seed))

    if checkpoint is not None:
        config["task"]["checkpoint"] = checkpoint

    if "{seed}" in str(config["pipeline"].get("seed", "")) or "{seed}" in str(config["task"].get("checkpoint", "")):
        raise click.UsageError("The config contains '{seed}' placeholders; pass --seed to expand them.")

    # Init reproducibility and logging
    config["pipeline"]["seed"] = int(config["pipeline"]["seed"])
    set_seed(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True
    wandb.init(config=config, project=config["wandb"]["project_name"], name=config["wandb"]["run_name"],
               mode=config["wandb"].get("mode", "disabled"))

    accelerator = Accelerator()
    device = config["pipeline"].get("device", accelerator.device)

    model_name = config["task"]["model_name"]
    dataset_name = config["task"]["dataset_name"]
    dataset_directory = config["task"].get("dataset_directory")
    padding = config["processing"]["padding"]

    # ---------- Load test data ----------
    if dataset == "davidson":

        ds, tokenizer = get_dataset(
            dataset_name,
            model_name,
            padding=padding,
            tokenize=True,
            batched=True,
            return_tokenizer=True,
            dataset_directory=dataset_directory,
        )
        ds = ds["test"]
    else:
        # for HateCheck-XR
        raw = load_dataset(
            "csv",
            data_files={"test": [hatecheck_csv]},
            delimiter=";",
            encoding="utf-8-sig",
            download_mode="force_redownload",
        )["test"]

        label_map = {"hateful": 0, "offensive": 1, "clean": 2}  # base encoding
        def unify(ex):
            return {"text": ex["test_case"], "labels": label_map[ex["label_gold"]]}
        ds = raw.map(unify, remove_columns=raw.column_names)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ds = ds.map(lambda x: tokenizer(x["text"].lower(),
                                        truncation=True,
                                        max_length=config["processing"].get("max_seq_length", 512),
                                        padding=padding),
                    batched=False)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Apply the evaluation label scheme to references.
    ds = apply_eval_scheme(ds, Mode(eval_mode))

    # ---------- Dataloader ----------
    batch_size = config["pipeline"]["batch_size"]
    dataloader = get_dataloader(ds, tokenizer, batch_size, padding)

    # ---------- Metric & model ----------
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    metric.compute = functools.partial(combine_compute, metric)

    # IMPORTANT: model head size must match how the model was trained, not eval space.
    n_train_labels = num_labels_for(Mode(train_mode))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_train_labels)

    ckpt = torch.load(config["task"]["checkpoint"], map_location=torch.device(device))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # ---------- Evaluation ----------
    project_logits = projection_for(Mode(train_mode), Mode(eval_mode))
    logger.info(f"Device: {device}. Starting evaluation on '{dataset}' with eval_mode={eval_mode}, train_mode={train_mode}")
    eval_loss, metrics, predictions, cm, confidences, references = evaluate_loop(
        model, dataloader, metric, device, project_logits, compute_loss=(train_mode == eval_mode)
    )
    logger.info(f"Average Loss: {eval_loss}, Metrics: {metrics}")

    save_dict = {
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "average_loss": float(eval_loss),
        "results": metrics,
        "confidences": confidences,
        "references": references,
    }

    if "output_predictions" in config["pipeline"]:
        p = Path(config["pipeline"]["output_predictions"]).parent
        p.mkdir(exist_ok=True, parents=True)
        with open(config["pipeline"]["output_predictions"], "w") as f:
            json.dump(save_dict, f)

    wandb.log(save_dict)
    wandb.finish()

if __name__ == "__main__":
    main()
