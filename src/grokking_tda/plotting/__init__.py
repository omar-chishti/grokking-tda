"""Vector-first figures rendered from run artifacts."""

from grokking_tda.plotting.curves import plot_training_curves
from grokking_tda.plotting.persistence import (
    plot_observables_over_time,
    plot_persistence_diagram,
)
from grokking_tda.plotting.style import use_vector_style
from grokking_tda.plotting.trajectory import plot_crocker

__all__ = [
    "plot_crocker",
    "plot_observables_over_time",
    "plot_persistence_diagram",
    "plot_training_curves",
    "use_vector_style",
]

