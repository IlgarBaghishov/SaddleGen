"""
v7-2a1a — Auxiliary eigenmode-prediction head.

Trained alongside the velocity head from the SAME trunk features (after
delta-injection, force-injection, time-FiLM, and the head's intermediate
SO3_Linear+Gate layers). Output is a per-atom (N, 3) vector representing the
saddle's unstable-mode shape (the eigenvector of the lowest Hessian eigenvalue).
At inference this head is silently dropped — its only job is to regularize the
trunk during training to encode saddle-specific geometric structure that the
velocity head can also exploit.

Sign-invariant loss is computed in `FlowMatchingLoss.forward` (eigenvectors
are defined only up to overall sign, so we use cos² instead of cos).

Zero-initialised so at init the aux loss term is exactly zero — it cannot
perturb early-training dynamics until the trunk has begun encoding useful
features.
"""

import torch
import torch.nn as nn
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear


class EigenmodeHead(nn.Module):
    """Project the velocity head's pre-final trunk features to a per-atom
    3-vector eigenmode prediction.

    Reads the same `(N, num_sph, sphere_channels)` tensor that the velocity
    head's last `SO3_Linear` operates on (exposed via `velocity_head.forward(
    ..., return_features=True)`), keeps only l=0,1, and projects to a single
    output channel reshaped to (N, 3).
    """

    def __init__(self, sphere_channels: int, input_lmax: int = 1):
        super().__init__()
        assert input_lmax >= 1, "EigenmodeHead needs l=1 channels to emit (N, 3)"
        self.sphere_channels = sphere_channels
        self.input_lmax = input_lmax
        self.proj = SO3_Linear(sphere_channels, 1, lmax=input_lmax)
        # IMPORTANT: do NOT zero-init this projection. With cos² loss
        #   loss = 1 - inner_sys² / (np_sys · nt_sys + ε)
        # the gradient of `loss` with respect to `pred` is identically zero
        # when `pred = 0` (both numerator and its derivative are zero
        # everywhere `inner_sys = 0`). Zero-init would therefore be a
        # permanent fixed point — the aux head never trains. Leave SO3_Linear's
        # default Kaiming-style init in place so the very first forward yields
        # a non-zero (random) eigenmode prediction and gradients flow.

    def forward(self, h_l01: torch.Tensor) -> torch.Tensor:
        """h_l01: (N, 4, sphere_channels) — l=0,1 trunk features from
        the velocity head's pre-final tensor.

        Returns: (N, 3) per-atom eigenmode prediction (un-normalised; the loss
        is normalisation-invariant, and frozen-atom rows are masked out
        downstream).
        """
        out = self.proj(h_l01)  # (N, 4, 1)
        return out[:, 1:4, :].reshape(-1, 3)
