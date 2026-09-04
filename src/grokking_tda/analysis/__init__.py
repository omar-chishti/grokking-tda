"""Offline analysis. Importing this package registers the built-in observables."""

from grokking_tda.analysis.aggregate import aggregate_runs
from grokking_tda.analysis.context import ObservationContext, run_observables
from grokking_tda.observable import OBSERVABLES, ensure_builtins, register_observable

ensure_builtins()  # importers of this package expect a full registry; `observable` owns the list

__all__ = [
    "OBSERVABLES",
    "ObservationContext",
    "aggregate_runs",
    "register_observable",
    "run_observables",
]

