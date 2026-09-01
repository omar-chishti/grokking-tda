"""What condition a run belongs to, and whether it re-runs one the programme already covers.

Both questions were answered twice — once in ``analysis/bank.py`` for the condition table and
once by string surgery in ``aggregate.early_window_table`` for the cross-validation groups —
and the two disagreed, which is how forty-two duplicate optimisation paths came to straddle
folds. They are answered here, once.
"""

from __future__ import annotations

from math import factorial

# the fields a condition is defined by; anything varying within one of them is a seed
CONDITION_FIELDS = (
    "model",
    "operation",
    "modulus",
    "train_fraction",
    "label_permutation",
    "loss",
    "optimizer",
    "lr",
    "weight_decay",
)


def task_modulus(data: dict) -> int:
    """Recomputed rather than trusted: an early S_5 batch predates the guard and carries 97."""
    if data.get("task") == "permutation_group":
        return factorial(int(data["n_symbols"]))
    return int(data["modulus"])


def config_fields(config: dict) -> dict:
    return {
        "model": config["model"]["name"],
        "operation": config["data"]["operation"],
        "modulus": task_modulus(config["data"]),
        "train_fraction": config["data"]["train_fraction"],
        "label_permutation": config["data"]["label_permutation"],
        "loss": config["train"]["loss"],
        "optimizer": config["train"]["optimizer"]["name"],
        "lr": config["train"]["optimizer"]["lr"],
        "weight_decay": config["train"]["optimizer"]["weight_decay"],
    }


def condition_key(config: dict) -> str:
    """A grouping key for cross-validation, from the configuration rather than the run name.

    Deriving it from the name splits a condition whenever the naming convention changes, and
    the splitter is then free to train on one half of a configuration and test on the other.
    """
    return "|".join(f"{key}={value}" for key, value in config_fields(config).items())


def is_replicate(run_name: str, config: dict) -> bool:
    """Is this run outside the main programme's condition table?

    Three kinds are: a dense re-run and a trajectory re-run, which repeat a condition on a
    different snapshot schedule and would double-count it, and the R14 recipe sweep, whose cells
    differ only in architecture and batching --- fields ``config_fields`` does not carry, so all
    fourteen would otherwise collapse into the reference regime's condition and drag a 200k
    budget into its bootstrap interval and the null band. The sweep has its own reduction in
    ``analysis/recipe.py``.

    The substring test is load-bearing, and ``trajectory_dim > 0`` is not the structural fix it
    looks like: ``R9c-s5-final.runs`` sets it on the five S_5 runs, which are main programme and
    are the non-cyclic condition of thesis §5.3.
    """
    return (
        "_dense_" in run_name
        or "_traj_" in run_name
        or "_recipe-" in run_name
        or config["train"].get("dense_to", 0) > 0
    )
