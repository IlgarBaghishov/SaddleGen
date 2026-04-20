"""
Velocity head: projects per-atom irreps to per-atom velocity vectors.

`depth=1` (default) mirrors fairchem's `Linear_Force_Head` exactly — a single
`SO3_Linear(C, 1, lmax=1)` whose l=1 output is reshaped to (N, 3). `depth ≥ 2`
stacks `SO3_Linear(C, C, lmax=1) → UMAGate → ...` with equivariant Gate-style
activations before the final projection.

Time conditioning (FiLM, AdaLN-zero style). A sinusoidal embedding of flow-time
t ∈ [0, 1] is passed through a small MLP that outputs 2C features, split into:
    - a per-channel scalar bias added to the l=0 channels, and
    - a per-channel scalar factor (1 + γ) multiplying the l≥1 channels.
Both operations are SO(3)-equivariant because `t` is a scalar (invariant),
so every derived per-atom per-channel factor is SO(3)-invariant. Multiplying
an invariant scalar by an l=ℓ feature yields an l=ℓ feature — see the proof
in CLAUDE.md §"Why time FiLM is equivariant". The MLP's last layer is
zero-initialized, so at init (bias=0, γ=0) the head is numerically identical
to fairchem's `Linear_Force_Head`; time-dependence is learned during training.

A plain additive l=0 bias alone would NOT propagate `t` to the output because
`SO3_Linear` keeps l-channels decoupled — the l=0 signal never reaches the
l=1 projection. The multiplicative gate on l≥1 is what lets `t` actually matter.

The architecture consumes only the l=0,1 slice of the input — higher-l features
are dropped, matching UMA's own `Linear_Force_Head` (l>1 is not needed to
produce a 3-vector output and dropping it halves the parameter count).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear
from fairchem.core.models.uma.outputs import get_l_component_range


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding for a scalar flow-time.

    Args:
        t: scalar, 0-d tensor, or (B,) tensor of flow times.
        dim: embedding dimension; must be even.
    Returns:
        (B, dim) where B = `t.numel()`.
    """
    assert dim % 2 == 0
    t = t.reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=t.dtype, device=t.device) / half
    )
    args = t.unsqueeze(-1) * freqs  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class UMAGate(nn.Module):
    """Equivariant gated activation on UMA's `(N, num_sph, C)` layout.

    - l=0 (scalar) channels: pass through SiLU.
    - l≥1 channels: multiplied per-channel by `sigmoid(W · x_l0)`.

    Scalar × scalar is equivariant; scalar × equivariant is equivariant.
    Channel count is preserved.
    """

    def __init__(self, sphere_channels: int, lmax: int):
        super().__init__()
        self.lmax = lmax
        self.gate_proj = nn.Linear(sphere_channels, sphere_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l0 = x[:, 0, :]
        gate = torch.sigmoid(self.gate_proj(x_l0)).unsqueeze(1)  # (N, 1, C)
        out_l0 = F.silu(x_l0).unsqueeze(1)  # (N, 1, C)
        if self.lmax >= 1:
            out_lge1 = x[:, 1:, :] * gate  # (N, num_sph-1, C)
            return torch.cat([out_l0, out_lge1], dim=1)
        return out_l0


class VelocityHead(nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        input_lmax: int = 2,
        depth: int = 1,
        time_embed_dim: int = 64,
        time_mlp_hidden: int = 128,
    ):
        super().__init__()
        assert depth >= 1
        assert input_lmax >= 1, "VelocityHead needs l=1 channels to emit (N, 3)"
        self.sphere_channels = sphere_channels
        self.input_lmax = input_lmax
        self.depth = depth
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_mlp_hidden),
            nn.SiLU(),
            nn.Linear(time_mlp_hidden, 2 * sphere_channels),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

        self.layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.layers.append(SO3_Linear(sphere_channels, sphere_channels, lmax=1))
            self.layers.append(UMAGate(sphere_channels, lmax=1))
        self.final = SO3_Linear(sphere_channels, 1, lmax=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, (input_lmax+1)², sphere_channels) — output of GlobalAttn.
            t: scalar / 0-d / (B,) flow time in [0, 1]. For batched input supply one
                `t` per system and a `batch_idx` of length N.
            batch_idx: (N,) long mapping each atom to its system (0..B-1).
                May be None iff all atoms share a single scalar `t`.
        Returns:
            (N, 3) velocity vectors (un-masked, un-CoM-projected — downstream
            flow code applies `v[fixed] = 0` and CoM subtraction on mobile atoms).
        """
        x_l01 = get_l_component_range(x, l_min=0, l_max=1)  # (N, 4, C)

        t_emb = sinusoidal_time_embedding(t, self.time_embed_dim)  # (B, time_embed_dim)
        t_params = self.time_mlp(t_emb)  # (B, 2C)
        t_bias, t_scale = t_params.chunk(2, dim=-1)  # each (B, C)
        t_gate = 1.0 + t_scale  # zero-init → 1 at init → head ≡ Linear_Force_Head at init

        if batch_idx is not None:
            t_bias_per_atom = t_bias[batch_idx]  # (N, C)
            t_gate_per_atom = t_gate[batch_idx]  # (N, C)
        else:
            assert t_params.shape[0] == 1, (
                "When batch_idx is None, t must be a single scalar (got batch size "
                f"{t_params.shape[0]})"
            )
            n = x_l01.shape[0]
            t_bias_per_atom = t_bias.expand(n, -1)
            t_gate_per_atom = t_gate.expand(n, -1)

        h_l0 = (x_l01[:, 0, :] + t_bias_per_atom).unsqueeze(1)  # (N, 1, C)
        h_l1 = x_l01[:, 1:, :] * t_gate_per_atom.unsqueeze(1)   # (N, 3, C)
        h = torch.cat([h_l0, h_l1], dim=1)                       # (N, 4, C)

        for layer in self.layers:
            h = layer(h)
        out = self.final(h)  # (N, 4, 1)
        return out[:, 1:4, :].reshape(-1, 3)
