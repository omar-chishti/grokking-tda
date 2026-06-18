"""Offline analysis: observables over a run's snapshots.

Importing this package registers the built-in observables (TDA + baselines) so that
``run_observables`` can resolve any name listed in an ``AnalysisCfg``.
"""

# Side-effect imports: populate the observable registry.
import grokking_tda.baselines  # noqa: E402,F401
import grokking_tda.tda.observables  # noqa: E402,F401
from grokking_tda.analysis.aggregate import aggregate_runs
from grokking_tda.analysis.observable import (
    OBSERVABLES,
    ObservationContext,
    register_observable,
    run_observables,
)

__all__ = [
    "OBSERVABLES",
    "ObservationContext",
    "aggregate_runs",
    "register_observable",
    "run_observables",
]

