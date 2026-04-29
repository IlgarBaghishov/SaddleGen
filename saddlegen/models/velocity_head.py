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


def sinusoidal_time_embedding(t: torch.Tensor, dim: int,
                                max_period: float = 1.0) -> torch.Tensor:
    """Sinusoidal embedding for a scalar flow-time `t ∈ [0, max_period]`.

    **Bug fix (2026-04-21).** The previous implementation used the stock
    Transformer positional-encoding base of 10000, which is calibrated for
    token positions up to ~10k (or DDPM-style discretized time T=1000). On
    continuous flow-time in `[0, 1]`, that makes the highest-frequency dim
    `sin(10000^{-31/32} · t) ≈ sin(1.3e-4 · t) ≈ 0` for the whole interval,
    so 31 of 32 embedding dimensions carried essentially zero signal and
    the time-FiLM effectively saw ~4 useful dims instead of 64.

    Now we spread frequencies geometrically from 1 cycle up to `half` cycles
    on `[0, max_period]` — so every dim varies meaningfully across the
    interval without aliasing at realistic integration step counts (K~50).

    Args:
        t: scalar, 0-d tensor, or (B,) tensor of flow times.
        dim: embedding dimension; must be even.
        max_period: the flow-time span. Default 1.0.
    Returns:
        (B, dim) where B = `t.numel()`.
    """
    assert dim % 2 == 0
    t = t.reshape(-1)
    half = dim // 2
    max_freq_cycles = float(half)
    ks = torch.arange(half, dtype=t.dtype, device=t.device) / max(half - 1, 1)
    freqs = (2.0 * math.pi / max_period) * torch.pow(
        torch.tensor(max_freq_cycles, dtype=t.dtype, device=t.device), ks
    )  # geometric 2π·1 → 2π·half (rad per unit flow-time)
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
    """Equivariant velocity head; optionally Mode-1 product-conditional.

    **Mode 0 (delta_endpoint_channels=0).** Backward-compatible default. The
    head sees only UMA features and time `t`; no partner conditioning. Used by
    the existing ice-cream-cone training recipe.

    **Mode 1 (delta_endpoint_channels > 0).** A per-atom partner-displacement
    vector `Δ_partner = partner − x_t` (MIC-shortest image) is injected into
    the head before the time-FiLM. Equivariance construction:

      • Build a single-channel irrep tensor with l=0 = ‖Δ‖, l=1 = Δ, l≥2 = 0.
      • `delta_proj` (an `SO3_Linear(1, C_d, lmax)`) expands it to `C_d` channels.
      • Concat with UMA's `(N, num_sph, sphere_channels)` features along the
        channel axis → `(N, num_sph, sphere_channels + C_d)`.
      • `delta_fuse` (an `SO3_Linear(sphere_channels + C_d, sphere_channels, lmax)`)
        mixes UMA channels and Δ channels per-l (l=0↔l=0 only, l=1↔l=1 only —
        SO3_Linear keeps paths decoupled, which is required for equivariance).
      • Then proceed exactly as Mode 0 (time-FiLM on l=0,1 slice; final SO3_Linear).

    `delta_proj` is zero-initialised (weight and bias), so at init Δ has zero
    contribution and the Mode-1 head is numerically identical to the Mode-0
    head. The "partner direction" signal is learned from Mode-1 training data,
    starting from the validated Mode-0 baseline architecturally.

    Why not e3nn / direct concat without `delta_proj`? `SO3_Linear` requires
    a fixed channel count to mix; `delta_proj`'s sole job is to expand a
    1-channel input to `C_d` channels so the fuse layer has enough room to
    learn meaningful UMA↔Δ mixings. `C_d = 32` (default) gives the head plenty
    of representational headroom (4 DOF input → 32 channels mirrors the time
    embedding's 1 DOF → 64 channels expansion).
    """

    def __init__(
        self,
        sphere_channels: int,
        input_lmax: int = 2,
        depth: int = 1,
        time_embed_dim: int = 64,
        time_mlp_hidden: int = 128,
        delta_endpoint_channels: int = 0,
    ):
        super().__init__()
        assert depth >= 1
        assert input_lmax >= 1, "VelocityHead needs l=1 channels to emit (N, 3)"
        assert delta_endpoint_channels >= 0
        self.sphere_channels = sphere_channels
        self.input_lmax = input_lmax
        self.depth = depth
        self.time_embed_dim = time_embed_dim
        self.delta_endpoint_channels = delta_endpoint_channels

        if delta_endpoint_channels > 0:
            self.delta_proj = SO3_Linear(1, delta_endpoint_channels, lmax=input_lmax)
            self.delta_fuse = SO3_Linear(
                sphere_channels + delta_endpoint_channels, sphere_channels, lmax=input_lmax,
            )
            # Zero-init so at init Mode 1 ≡ Mode 0 (Δ contributes nothing).
            # Training learns useful weights from the partner signal.
            nn.init.zeros_(self.delta_proj.weight)
            nn.init.zeros_(self.delta_proj.bias)

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

    def _inject_delta(self, x: torch.Tensor, delta_endpoint: torch.Tensor) -> torch.Tensor:
        """Build the per-atom Δ irrep tensor and fuse into UMA features.

        delta_endpoint: (N, 3) per-atom partner displacement.
        Returns: (N, num_sph, sphere_channels) — same shape as input x.
        """
        N, num_sph, _ = x.shape
        irreps = torch.zeros(N, num_sph, 1, device=x.device, dtype=x.dtype)
        # l=0 slot at index 0: invariant magnitude.
        irreps[:, 0, 0] = torch.linalg.norm(delta_endpoint, dim=-1)
        # l=1 slots at indices 1..3: equivariant 3-vector.
        irreps[:, 1:4, 0] = delta_endpoint
        delta_feats = self.delta_proj(irreps)  # (N, num_sph, C_d)
        fused = torch.cat([x, delta_feats], dim=-1)  # (N, num_sph, C+C_d)
        return self.delta_fuse(fused)  # (N, num_sph, C)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
        delta_endpoint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, (input_lmax+1)², sphere_channels) — output of GlobalAttn.
            t: scalar / 0-d / (B,) flow time in [0, 1]. For batched input supply one
                `t` per system and a `batch_idx` of length N.
            batch_idx: (N,) long mapping each atom to its system (0..B-1).
                May be None iff all atoms share a single scalar `t`.
            delta_endpoint: (N, 3) per-atom partner displacement (Mode 1). When
                `delta_endpoint_channels == 0` this must be None; otherwise it
                must be provided.
        Returns:
            (N, 3) velocity vectors (un-masked, un-CoM-projected — downstream
            flow code applies `v[fixed] = 0` and CoM subtraction on mobile atoms).
        """
        if (delta_endpoint is None) != (self.delta_endpoint_channels == 0):
            raise ValueError(
                "delta_endpoint must be provided iff delta_endpoint_channels > 0; "
                f"got delta_endpoint={'None' if delta_endpoint is None else 'tensor'}, "
                f"delta_endpoint_channels={self.delta_endpoint_channels}"
            )
        if delta_endpoint is not None:
            x = self._inject_delta(x, delta_endpoint)

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
