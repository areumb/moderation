from __future__ import annotations
from enum import Enum
from typing import Callable
import torch
from datasets import Dataset

# Davidson dataset base encoding assumed throughout: 0=hate, 1=offensive, 2=clean
class Mode(str, Enum):
    three_class = "3class"
    hate_nonhate = "hate_nonhate"
    nonclean_clean = "nonclean_clean"
    hate_clean = "hate_clean"

def num_labels_for(mode: Mode) -> int:
    return 3 if mode == Mode.three_class else 2

# ---------- Dataset transforms (references/ground truth) ----------
def apply_train_scheme(ds: Dataset, mode: Mode) -> Dataset:
    """Transform labels for TRAINING."""
    if mode == Mode.three_class:
        return ds

    if mode == Mode.hate_nonhate:
        def _map(ex):
            ex["labels"] = 0 if ex["labels"] == 0 else 1
            return ex
        return ds.map(_map)

    if mode == Mode.nonclean_clean:
        def _map(ex):
            # 2 -> 1 (clean), {0,1} -> 0 (non-clean)
            ex["labels"] = 1 if ex["labels"] == 2 else 0
            return ex
        return ds.map(_map)

    if mode == Mode.hate_clean:
        ds = ds.filter(lambda ex: ex["labels"] != 1)  # drop offensive
        def _map(ex):
            # hate(0)->0, clean(2)->1
            ex["labels"] = 0 if ex["labels"] == 0 else 1
            return ex
        return ds.map(_map)

    raise ValueError(f"Unknown mode: {mode}")

def apply_eval_scheme(ds: Dataset, mode: Mode) -> Dataset:
    """Transform labels for EVALUATION."""
    # Same transforms as training, but kept separate for clarity.
    return apply_train_scheme(ds, mode)

# ---------- Logit projections (predictions) ----------
def projection_for(train_mode: Mode, eval_mode: Mode) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a function that projects raw model logits (shape [B, K]) from the model's
    training label space into the requested evaluation label space, producing hard
    predictions in {0,1} or {0,1,2} as required.
    """
    if train_mode == eval_mode:
        return lambda logits: logits.argmax(dim=-1)

    def _merge_probs(logits: torch.Tensor, idx_groups: list[list[int]]) -> torch.Tensor:
        # Convert to probabilities, merge classes by summing probs, then argmax.
        probs = torch.softmax(logits, dim=-1)
        merged = [probs[:, idxs].sum(dim=-1) for idxs in idx_groups]
        return torch.stack(merged, dim=-1).argmax(dim=-1)

    # 3-class model evaluated as 2-class tasks
    if train_mode == Mode.three_class and eval_mode == Mode.hate_nonhate:
        # [hate] vs [offensive+clean]
        return lambda logits: _merge_probs(logits, [[0], [1, 2]])
    if train_mode == Mode.three_class and eval_mode == Mode.nonclean_clean:
        # [hate+offensive] vs [clean]
        return lambda logits: _merge_probs(logits, [[0, 1], [2]])
    if train_mode == Mode.three_class and eval_mode == Mode.hate_clean:
        # [hate] vs [clean] (ignore offensive by comparing hate vs clean)
        return lambda logits: _merge_probs(logits, [[0], [2]])

    # hate/clean model evaluated on other 2-class tasks
    if train_mode == Mode.hate_clean and eval_mode == Mode.hate_nonhate:
        # hate->0, clean->1 (non-hate)
        return lambda logits: torch.where(logits.argmax(dim=-1) == 0,
                                          torch.zeros_like(logits[:, 0], dtype=torch.long),
                                          torch.ones_like(logits[:, 0], dtype=torch.long))
    if train_mode == Mode.hate_clean and eval_mode == Mode.nonclean_clean:
        # hate->0 (non-clean), clean->1 (clean) is already aligned
        return lambda logits: logits.argmax(dim=-1)

    # Other cross-projections (e.g. a binary model evaluated on the 3-class task) are not
    # well-defined: the model cannot produce labels outside its training space. Raising here
    # prevents silently reporting invalid results.
    raise ValueError(
        f"Projection from train_mode='{train_mode.value}' to eval_mode='{eval_mode.value}' is not supported. "
        f"Supported: identical modes, 3class->any binary mode, hate_clean->hate_nonhate, "
        f"and hate_clean->nonclean_clean."
    )
