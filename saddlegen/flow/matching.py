"""
Flow-matching training loss for SaddleGen.

Three straight-line OT objectives per CLAUDE.md §"Training objectives":
    (1) TS + Gaussian → TS         (TS-denoise; default weight 0)
    (2) (R + Gaussian) → TS        (multimodality breaker)
    (3) R → TS (clean, t > ε)      (anchor; highest weight by default)

Objective selection is Monte Carlo: each sample draws `k ~ Categorical(w_1, w_2, w_3)`
and evaluates only that objective. This is an unbiased estimator of the weight-mixed
loss with 1/N-th the compute of evaluating all three per sample; matters when some
weights are zero (obj 1 default) because sampling skips those entirely.

Design choices adopted from FlowMM's implementation:
    - uniform `t ~ U(0, 1)` (obj 1, 2) or `U(ε, 1)` (obj 3); no logit-normal.
    - plain MSE, no time-dependent reweighting.
    - per-atom averaging over mobile atoms (frozen contribute zero by construction).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from ..data.transforms import gaussian_perturbation, wrap_positions


@dataclass
class FlowMatchingConfig:
    sigma_rs_pert: float
    sigma_ts_pert: float | None = None  # None → use sigma_rs_pert
    epsilon: float = 0.01
    loss_weights: tuple[float, float, float] = (0.0, 1.0, 2.0)

    def __post_init__(self):
        assert 0.0 < self.epsilon < 1.0, f"epsilon must be in (0, 1), got {self.epsilon}"
        assert sum(self.loss_weights) > 0, "at least one loss weight must be positive"
        if self.sigma_ts_pert is None:
            self.sigma_ts_pert = self.sigma_rs_pert


def _draw_objective(loss_weights: tuple[float, float, float], n: int,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Return (n,) long tensor of objective indices in {1, 2, 3}."""
    probs = torch.tensor(loss_weights, dtype=torch.float32)
    probs = probs / probs.sum()
    return torch.multinomial(probs, n, replacement=True, generator=generator) + 1


def sample_endpoints(
    sample: dict,
    objective: int,
    config: FlowMatchingConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Build one training example: (x_0, x_1, t, mobile_mask) for the given objective.

    x_0, x_1 are in unwrapped space; v_target = x_1 - x_0 is constant along the trajectory.
    """
    r_start = sample["start_pos"]
    r_saddle = sample["saddle_un_pos"]
    mobile = ~sample["fixed"]

    if objective == 3:
        x0, x1 = r_start, r_saddle
        u = torch.rand((), generator=generator).item()
        t = config.epsilon + u * (1.0 - config.epsilon)
    elif objective == 2:
        eps = gaussian_perturbation(mobile, config.sigma_rs_pert, generator=generator,
                                    dtype=r_start.dtype)
        x0, x1 = r_start + eps, r_saddle
        t = torch.rand((), generator=generator).item()
    elif objective == 1:
        eps = gaussian_perturbation(mobile, config.sigma_ts_pert, generator=generator,
                                    dtype=r_saddle.dtype)
        x0, x1 = r_saddle + eps, r_saddle
        t = torch.rand((), generator=generator).item()
    else:
        raise ValueError(f"objective must be in {{1, 2, 3}}, got {objective}")
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
        positions=positions.detach().cpu().numpy(),
        numbers=Z.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
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

    v: (N, 3); mobile: (N,) bool; batch_idx: (N,) long in [0, num_systems).
    """
    w = mobile.to(v.dtype).unsqueeze(-1)                                           # (N, 1)
    sums = torch.zeros(num_systems, 3, dtype=v.dtype, device=v.device)
    counts = torch.zeros(num_systems, dtype=v.dtype, device=v.device)
    sums.index_add_(0, batch_idx, v * w)
    counts.index_add_(0, batch_idx, mobile.to(v.dtype))
    means = sums / counts.clamp(min=1).unsqueeze(-1)                               # (num_systems, 3)

    # Per-system: does this system contain any frozen atom? If yes, skip projection.
    frozen_counts = torch.zeros(num_systems, dtype=v.dtype, device=v.device)
    frozen_counts.index_add_(0, batch_idx, (~mobile).to(v.dtype))
    has_frozen = (frozen_counts > 0).to(v.dtype).unsqueeze(-1)                     # (num_systems, 1)
    means = means * (1.0 - has_frozen)                                             # zero out means for systems with frozen atoms

    return v - means[batch_idx] * w


def apply_output_projections(
    v: torch.Tensor, fixed: torch.Tensor, batch_idx: torch.Tensor, num_systems: int,
) -> torch.Tensor:
    """Flow output projections: (i) hard-mask `v[fixed] = 0`, (ii) CoM subtraction
    over mobile atoms — but only for systems where no atoms are frozen (see
    `_com_projection_batched` docstring for why)."""
    v = v.masked_fill(fixed.unsqueeze(-1), 0.0)
    return _com_projection_batched(v, ~fixed, batch_idx, num_systems)


class FlowMatchingLoss(nn.Module):
    """Wraps `backbone + global_attn + velocity_head` with the three-objective loss."""

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

    @property
    def device(self) -> torch.device:
        return next(self.velocity_head.parameters()).device

    def forward(
        self,
        batch: list[dict],
        generator: torch.Generator | None = None,
    ) -> dict:
        """Compute the flow-matching loss over a list of sample dicts.

        Returns a dict with:
            loss:        scalar (mean squared velocity error, averaged over mobile atoms)
            per_obj:     (3,) long — count of samples assigned to each of objs {1,2,3}
            per_obj_loss: (3,) float — mean loss on each objective (NaN where count=0)
            n_mobile:    int — total mobile-atom count contributing to the mean
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

        feat = self.backbone(batch_data)
        x = feat["node_embedding"]
        x = self.global_attn(x, batch_idx)
        v = self.velocity_head(x, t_tensor, batch_idx)
        v = apply_output_projections(v, fixed_all, batch_idx, num_systems=B)

        sq_err = (v - v_target).pow(2).sum(dim=-1)  # (N_total,)
        mobile = ~fixed_all
        n_mobile = int(mobile.sum().item())
        loss = sq_err[mobile].mean() if n_mobile > 0 else sq_err.sum() * 0.0

        per_obj = torch.zeros(3, dtype=torch.long)
        per_obj_loss = torch.full((3,), float("nan"))
        atom_batch = batch_idx.detach().cpu()
        objs = objectives
        sq_err_cpu = sq_err.detach().cpu()
        mobile_cpu = mobile.detach().cpu()
        for k in (1, 2, 3):
            sample_mask = (objs == k)
            per_obj[k - 1] = int(sample_mask.sum())
            if sample_mask.any():
                atom_sel = torch.isin(atom_batch, torch.nonzero(sample_mask, as_tuple=False).squeeze(-1))
                m = atom_sel & mobile_cpu
                if m.any():
                    per_obj_loss[k - 1] = sq_err_cpu[m].mean()

        return {
            "loss": loss,
            "per_obj": per_obj,
            "per_obj_loss": per_obj_loss,
            "n_mobile": n_mobile,
        }
