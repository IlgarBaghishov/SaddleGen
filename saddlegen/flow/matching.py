"""
Flow-matching training loss for SaddleGen.

Two straight-line OT objectives, selected per-sample by a weighted draw over
`loss_weights = (w_1, w_2)`:

    obj 1 — Ice-cream-cone around the r_R→r_S axis. x_0 is drawn uniformly from
            the union of balls B(c, R(c)·|c−r_R|/|Δ|) for c on [r_R, r_S], with
            R_TS = min(alpha·|Δ|, R_max_abs). This is the default (w_1=1, w_2=0)
            and the recipe that produced the flower field in LiC / LiC_simpler.

    obj 2 — Reactant + Gaussian. x_0 = r_R + ε, with ε ~ N(0, σ_rs_pert²·I_{3M})
            on mobile atoms. Useful as a multimodality breaker / regularizer;
            disabled by default (w_2 = 0) because in practice obj 1 on its own
            already produces the correct curving field.

See CLAUDE.md §"Flow formulation" for derivation and §"What was tried" for the
history of why obj 1 (ice-cream-cone) replaced the previous three-objective
scheme entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from ..data.transforms import gaussian_perturbation, wrap_positions


@dataclass
class FlowMatchingConfig:
    """Training-time sampling hyperparameters.

    **obj 1 (ice-cream-cone)**:
        `alpha`      cone half-angle is `arcsin(alpha)`; default 0.5 → 30°
        `R_max_abs`  absolute cap on `R_TS` in Å; default 1.0
        `R_TS = min(alpha·|Δ|, R_max_abs)` is computed per-sample.

    **obj 2 (reactant Gaussian)**:
        `sigma_rs_pert`  Gaussian std in Å for ε at r_R; default 0.036
                        (LiC rule-of-thumb ≈ 0.05·⟨‖Δ‖⟩ / √(3M))

    **Mixing**:
        `loss_weights = (w_1, w_2)`. Per-sample objective is drawn as
        `k ~ Categorical(w_1, w_2)`. Default (1.0, 0.0) — pure obj 1.
    """

    # obj 1 — ice-cream-cone
    alpha: float = 0.5
    R_max_abs: float = 1.0            # Å
    max_rejection_tries: int = 100

    # obj 2 — reactant Gaussian
    sigma_rs_pert: float = 0.036      # Å

    # mixing
    loss_weights: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self):
        assert 0.0 < self.alpha <= 1.0, f"alpha must be in (0, 1], got {self.alpha}"
        assert self.R_max_abs > 0, f"R_max_abs must be positive, got {self.R_max_abs}"
        assert self.sigma_rs_pert >= 0, f"sigma_rs_pert must be >= 0, got {self.sigma_rs_pert}"
        assert len(self.loss_weights) == 2, f"loss_weights must be (w_1, w_2), got {self.loss_weights}"
        assert sum(self.loss_weights) > 0, "at least one loss weight must be positive"


def _draw_objective(loss_weights: tuple[float, float], n: int,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Return (n,) long tensor of objective indices in {1, 2}."""
    probs = torch.tensor(loss_weights, dtype=torch.float32)
    probs = probs / probs.sum()
    return torch.multinomial(probs, n, replacement=True, generator=generator) + 1


def sample_icecream_cone(
    r_R_mob: torch.Tensor,
    r_S_mob: torch.Tensor,
    R_TS: float,
    max_tries: int,
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Sample one 3D point uniformly from the ice-cream-cone region.

    Region: union over c ∈ [r_R, r_S] of B(c, R_TS · |c − r_R| / |Δ|).
    Method: rejection sampling on a bounding cylinder of radius R_TS and
    axial length |Δ| + R_TS, with acceptance check using
        cone region:  r_perp² · (|Δ|² − R_TS²) < a² · R_TS²   for a ≤ L_cone
        cap region:   (a − |Δ|)² + r_perp² < R_TS²            for a > L_cone
    where a is the axial coordinate measured from r_R and L_cone is the
    tangent-circle position |Δ| − R_TS²/|Δ|.
    """
    delta = r_S_mob - r_R_mob
    mag_delta = torch.linalg.norm(delta)
    axis = delta / mag_delta

    # Build an orthonormal frame (axis, e1, e2).
    e_seed = torch.zeros(3, dtype=dtype, device=device)
    imin = int(torch.argmin(torch.abs(axis)).item())
    e_seed[imin] = 1.0
    e1 = e_seed - (e_seed @ axis) * axis
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.cross(axis, e1, dim=0)

    mag_delta_sq = mag_delta * mag_delta
    R_TS_sq = R_TS * R_TS
    L_cone = mag_delta - R_TS_sq / mag_delta
    denom_cone = mag_delta_sq - R_TS_sq
    a_max = mag_delta + R_TS

    two_pi = torch.tensor(2.0 * math.pi, dtype=dtype, device=device)
    last_a = torch.tensor(0.0, dtype=dtype, device=device)
    last_r = torch.tensor(0.0, dtype=dtype, device=device)
    last_theta = torch.tensor(0.0, dtype=dtype, device=device)

    for _ in range(max_tries):
        a = torch.rand((), generator=generator, dtype=dtype, device=device) * a_max
        V = torch.rand((), generator=generator, dtype=dtype, device=device)
        r_perp = R_TS * torch.sqrt(V)
        theta = two_pi * torch.rand((), generator=generator, dtype=dtype, device=device)

        last_a, last_r, last_theta = a, r_perp, theta
        a_val = a.item()
        r_val = r_perp.item()
        if a_val <= L_cone.item():
            if r_val * r_val * denom_cone.item() < a_val * a_val * R_TS_sq:
                break
        else:
            da = a_val - mag_delta.item()
            if da * da + r_val * r_val < R_TS_sq:
                break

    x0 = r_R_mob + last_a * axis + last_r * (torch.cos(last_theta) * e1 + torch.sin(last_theta) * e2)
    return x0


def sample_endpoints(
    sample: dict,
    objective: int,
    config: FlowMatchingConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Build one training example `(x_0, x_1, t, mobile_mask)` for the given objective.

    obj 1 — ice-cream-cone sampling of x_0 (requires single mobile atom).
    obj 2 — x_0 = r_R + ε, ε ~ Gaussian on mobile atoms (any mobile count).
    """
    r_start = sample["start_pos"]
    r_saddle = sample["saddle_un_pos"]
    mobile = ~sample["fixed"]

    if objective == 1:
        # Ice-cream-cone sampling (single-mobile-atom case only for now).
        M = int(mobile.sum().item())
        if M != 1:
            raise NotImplementedError(
                f"ice-cream-cone (obj 1) currently requires exactly one mobile atom; "
                f"this sample has {M}. Use obj 2 (Gaussian) for multi-mobile systems, "
                f"or extend sample_icecream_cone to handle multi-mobile cones."
            )
        r_R_mob = r_start[mobile][0]
        r_S_mob = r_saddle[mobile][0]
        dtype = r_saddle.dtype
        device = r_saddle.device
        mag_delta = float(torch.linalg.norm(r_S_mob - r_R_mob).item())
        R_TS = min(config.alpha * mag_delta, config.R_max_abs)

        x0 = r_saddle.clone()
        if R_TS <= 1e-6 or mag_delta <= 1e-6:
            x0[mobile] = r_R_mob.unsqueeze(0)
        else:
            x0_mob = sample_icecream_cone(
                r_R_mob, r_S_mob, R_TS,
                config.max_rejection_tries, generator, dtype, device,
            )
            x0[mobile] = x0_mob.unsqueeze(0)
        x1 = r_saddle

    elif objective == 2:
        # Reactant + Gaussian perturbation. Works for any mobile count.
        eps = gaussian_perturbation(
            mobile, config.sigma_rs_pert, generator=generator, dtype=r_start.dtype,
        )
        x0 = r_start + eps
        x1 = r_saddle

    else:
        raise ValueError(f"objective must be in {{1, 2}}, got {objective}")

    t = torch.rand((), generator=generator).item()
    return x0, x1, t, mobile


def build_atomic_data(
    positions: torch.Tensor,
    Z: torch.Tensor,
    cell: torch.Tensor,
    task_name: str,
    charge: int,
    spin: int,
    fixed: torch.Tensor,
) -> AtomicData:
    """Package a single-system snapshot as an `AtomicData` for UMA forward."""
    atoms = Atoms(
        positions=positions.detach().float().cpu().numpy(),
        numbers=Z.detach().cpu().numpy(),
        cell=cell.detach().float().cpu().numpy(),
        pbc=True,
    )
    fixed_idx = torch.where(fixed)[0].cpu().tolist()
    if fixed_idx:
        atoms.set_constraint(FixAtoms(indices=fixed_idx))
    data = AtomicData.from_ase(atoms, task_name=task_name)
    data.charge = torch.tensor([charge], dtype=torch.long)
    data.spin = torch.tensor([spin], dtype=torch.long)
    return data


def _com_projection_batched(
    v: torch.Tensor, mobile: torch.Tensor, batch_idx: torch.Tensor, num_systems: int,
) -> torch.Tensor:
    """Subtract per-system mean over mobile atoms; frozen atoms pass through unchanged.

    Systems that contain **any** frozen atom are skipped — the frozen atoms
    already pin the cell's reference frame, so subtracting the mean over the
    remaining mobile atoms would remove legitimate motion (fatal when there is
    a single mobile atom, like Li-on-C: mean = v[Li] ⇒ v[Li] -= v[Li] = 0).
    """
    w = mobile.to(v.dtype).unsqueeze(-1)
    sums = torch.zeros(num_systems, 3, dtype=v.dtype, device=v.device)
    counts = torch.zeros(num_systems, dtype=v.dtype, device=v.device)
    sums.index_add_(0, batch_idx, v * w)
    counts.index_add_(0, batch_idx, mobile.to(v.dtype))
    means = sums / counts.clamp(min=1).unsqueeze(-1)

    frozen_counts = torch.zeros(num_systems, dtype=v.dtype, device=v.device)
    frozen_counts.index_add_(0, batch_idx, (~mobile).to(v.dtype))
    has_frozen = (frozen_counts > 0).to(v.dtype).unsqueeze(-1)
    means = means * (1.0 - has_frozen)

    return v - means[batch_idx] * w


def apply_output_projections(
    v: torch.Tensor, fixed: torch.Tensor, batch_idx: torch.Tensor, num_systems: int,
) -> torch.Tensor:
    """Flow output projections: (i) hard-mask `v[fixed] = 0`, (ii) CoM subtraction
    over mobile atoms — but only for systems where no atoms are frozen."""
    v = v.masked_fill(fixed.unsqueeze(-1), 0.0)
    return _com_projection_batched(v, ~fixed, batch_idx, num_systems)


class FlowMatchingLoss(nn.Module):
    """Wraps `backbone + global_attn + velocity_head` with the two-objective loss.

    **Gotcha — backbone is kept in eval mode always.** UMA-S-1.2 has non-zero
    dropout / composition-dropout / mole-dropout (p=0.05–0.10) that would be
    active under `self.training=True`. Without explicit suppression, calling
    `loss_module.train()` recursively activates those dropouts — and because
    the backbone is frozen (no grad), we'd be training the head against a
    noisy, stochastic feature field that does not match what the head sees at
    inference (deterministic features). We override `.train()` below to always
    put the backbone back into eval mode.
    """

    def __init__(
        self,
        config: FlowMatchingConfig,
        backbone: nn.Module,
        global_attn: nn.Module,
        velocity_head: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.global_attn = global_attn
        self.velocity_head = velocity_head

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.velocity_head.parameters()).device

    def forward(
        self,
        batch: list[dict],
        generator: torch.Generator | None = None,
    ) -> dict:
        """Compute the flow-matching loss over a list of sample dicts.

        Returns a dict:
            loss:         scalar (MSE averaged over mobile atoms)
            per_obj:      (2,) long — count of samples assigned to obj 1 / 2
            per_obj_loss: (2,) float — mean per-atom loss for each obj (NaN where count=0)
            n_mobile:     int — total mobile-atom count contributing to the mean
        """
        device = self.device
        B = len(batch)
        if B == 0:
            raise ValueError("empty batch")

        objectives = _draw_objective(self.config.loss_weights, B, generator=generator)

        data_list: list[AtomicData] = []
        v_targets: list[torch.Tensor] = []
        t_values: list[float] = []
        fixed_list: list[torch.Tensor] = []

        for sample, k in zip(batch, objectives.tolist()):
            x0, x1, t, _ = sample_endpoints(sample, k, self.config, generator=generator)
            v_target = x1 - x0
            x_t = wrap_positions((1.0 - t) * x0 + t * x1, sample["cell"])
            data = build_atomic_data(
                x_t, sample["Z"], sample["cell"],
                sample["task_name"], sample["charge"], sample["spin"],
                sample["fixed"],
            )
            data_list.append(data)
            v_targets.append(v_target)
            t_values.append(t)
            fixed_list.append(sample["fixed"])

        batch_data = data_list_collater(data_list, otf_graph=True).to(device)
        v_target = torch.cat(v_targets, dim=0).to(device)
        fixed_all = torch.cat(fixed_list, dim=0).to(device)
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device)
        batch_idx = batch_data.batch

        # Backbone is frozen — run it in no-grad to free activation memory.
        with torch.no_grad():
            feat = self.backbone(batch_data)
        x = feat["node_embedding"]
        x = self.global_attn(x, batch_idx)
        v = self.velocity_head(x, t_tensor, batch_idx)
        v = apply_output_projections(v, fixed_all, batch_idx, num_systems=B)

        sq_err = (v - v_target).pow(2).sum(dim=-1)  # (N_total,)
        mobile = ~fixed_all
        n_mobile = int(mobile.sum().item())

        if n_mobile > 0:
            loss = sq_err[mobile].mean()
        else:
            loss = sq_err.sum() * 0.0

        # Per-objective diagnostics (CPU-side, for logging only).
        per_obj = torch.zeros(2, dtype=torch.long)
        per_obj_loss = torch.full((2,), float("nan"))
        atom_batch = batch_idx.detach().cpu()
        sq_err_cpu = sq_err.detach().cpu()
        mobile_cpu = mobile.detach().cpu()
        for k in (1, 2):
            sample_mask = (objectives == k)
            per_obj[k - 1] = int(sample_mask.sum())
            if sample_mask.any():
                atom_sel = torch.isin(
                    atom_batch, torch.nonzero(sample_mask, as_tuple=False).squeeze(-1),
                )
                m = atom_sel & mobile_cpu
                if m.any():
                    per_obj_loss[k - 1] = sq_err_cpu[m].mean()

        return {
            "loss": loss,
            "per_obj": per_obj,
            "per_obj_loss": per_obj_loss,
            "n_mobile": n_mobile,
        }
