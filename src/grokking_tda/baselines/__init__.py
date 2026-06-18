"""Non-topological baseline observables.

Importing this module registers the baselines so topology is never reported in
isolation: Fourier concentration (the redundancy threat), weight norm (Omnigrok's
LU mechanism), and local intrinsic dimension (the geometric baseline Tang et al.
themselves compare against).
"""

from grokking_tda.baselines import fourier, lid, weight_norm  # noqa: F401

__all__ = ["fourier", "lid", "weight_norm"]
