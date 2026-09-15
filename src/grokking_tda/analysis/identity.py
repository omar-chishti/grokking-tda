"""What condition a run belongs to, and whether it re-runs one the programme already covers.

Answered once, here, for both the condition table and the cross-validation groups.
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


# A run name carrying one of these repeats a condition already covered: dense, trajectory and
# stride-one re-runs share their twin's optimisation path, and recipe cells differ in fields
# `config_fields` omits. A new re-run programme adds its tag here before it launches.
REPLICATE_TAGS = ("_dense_", "_traj_", "_stride1", "_recipe-", "_sq-")


def is_replicate(run_name: str, config: dict) -> bool:
    """Is this run outside the main programme's condition table?

    By name rather than by ``trajectory_dim > 0``, which ``R9c-s5-final.runs`` also sets on the
    five main-programme S_5 runs of thesis §5.2.
    """
    return any(tag in run_name for tag in REPLICATE_TAGS) or config["train"].get("dense_to", 0) > 0
