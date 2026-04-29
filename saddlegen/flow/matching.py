"""
Flow-matching training loss for SaddleGen.

Each `mode` of `FlowMatchingConfig` defines exactly one straight-line OT
objective; modes are mutually exclusive (no per-sample mixing):

    Mode 0 — Ice-cream-cone around the r_R→r_S axis. x_0 is drawn uniformly
             from the union of balls B(c, R(c)·|c−r_R|/|Δ|) for c on [r_R, r_S],
             with R_TS = min(alpha·|Δ|, R_max_abs). Single mobile atom only;
             the legacy recipe that produced the flower field on LiC / LiC_simpler.

    Mode 1 — Product-conditional (no noise). x_0 = start exactly; the head
             receives a per-atom partner-displacement Δ_partner = MIC(partner − x_t)
             at every flow step. The R/P doubling in the dataset gives a 50/50
             R-side/P-side split per epoch automatically.

    Mode 2 — Dimer-trajectory (placeholder, not yet wired into the loss).
             x_0 will be sampled uniformly from the saddle's Dimer + minimization
             trajectory; the dataset reader is in place, the loss is not.

See CLAUDE.md §"Flow formulation" for derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from ..data.transforms import mic_displacement, wrap_positions
from ..models.time_filmed_backbone import TimeFiLMBackbone


@dataclass
class FlowMatchingConfig:
    """Training-time sampling hyperparameters.

    **mode** selects the recipe:
        0 — Mode 0 (ice-cream-cone). Uses `alpha` and `R_max_abs`.
        1 — Mode 1 (product-conditional). No knobs of its own here — the
            partner-displacement injection is configured on the head
            (`VelocityHead.delta_endpoint_channels > 0`).
        2 — Mode 2 (trajectory). Raises until the loss is wired up.
    """

    mode: int = 0

    # Mode 0 — ice-cream-cone
    alpha: float = 0.5
    R_max_abs: float = 1.0            # Å
    max_rejection_tries: int = 100

    # Mode 1 v5 — Gaussian perturbation on x_t before backbone forward.
    # Default 0.0 disables. Applied to mobile atoms only. Velocity target
    # stays `saddle − x_0` (unchanged). Forces the model to encounter
    # off-line samples where force at x_t is informative.
    xt_perturb_sigma: float = 0.0

    def __post_init__(self):
        assert self.mode in (0, 1, 2), f"mode must be 0, 1, or 2, got {self.mode}"
        assert 0.0 < self.alpha <= 1.0, f"alpha must be in (0, 1], got {self.alpha}"
        assert self.R_max_abs > 0, f"R_max_abs must be positive, got {self.R_max_abs}"
        if self.mode == 2:
            raise NotImplementedError(
                "Mode 2 (Dimer-trajectory) loss is not implemented yet — only the "
                "dataset scaffolding has landed. Use mode=0 or mode=1."
            )


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
    config: FlowMatchingConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Build one training example `(x_0, x_1, t, mobile_mask)` per the config's mode.

    Mode 0 — ice-cream-cone sampling of x_0 (requires single mobile atom).
    Mode 1 — x_0 = start exactly (no noise). The partner-displacement vector
             is computed inside `FlowMatchingLoss.forward` because it depends
             on `x_t`, not just `(x_0, x_1)`.
    """
    r_start = sample["start_pos"]
    r_saddle = sample["saddle_un_pos"]
    mobile = ~sample["fixed"]

    if config.mode == 1:
        x0 = r_start.clone()
        x1 = r_saddle
        t = torch.rand((), generator=generator).item()
        return x0, x1, t, mobile

    if config.mode == 0:
        M = int(mobile.sum().item())
        if M != 1:
            raise NotImplementedError(
                f"Mode 0 (ice-cream-cone) currently requires exactly one mobile atom; "
                f"this sample has {M}. Switch to Mode 1 (product-conditional) for "
                f"multi-mobile systems, or extend sample_icecream_cone."
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
        t = torch.rand((), generator=generator).item()
        return x0, x1, t, mobile

    raise ValueError(f"unsupported mode {config.mode} in sample_endpoints")


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
    """Wraps `backbone + global_attn + velocity_head` with the per-mode flow-matching loss.

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
        force_head: nn.Module | None = None,
        force_tasks: dict | None = None,
    ):
        """Args:
            force_head, force_tasks: provided iff the velocity head was built
                with `force_field_channels > 0` (Mode 1 v2+). The wrapper from
                `saddlegen.utils.forces.load_uma_force_head` plus its tasks
                dict; we use them at every forward to compute UMA-quality
                forces at x_t and feed them to the head as an l=1 input.
        """
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.global_attn = global_attn
        self.velocity_head = velocity_head
        self.force_head = force_head
        # `tasks` is a dict of non-Module objects — store as a plain attribute,
        # not a submodule. nn.Module.__setattr__ will correctly skip non-Module
        # values, so this works.
        self.force_tasks = force_tasks
        if (force_head is None) != (getattr(velocity_head, "force_field_channels", 0) == 0):
            raise ValueError(
                "force_head must be provided iff velocity_head.force_field_channels > 0; "
                f"got force_head={'set' if force_head is not None else 'None'}, "
                f"force_field_channels={getattr(velocity_head, 'force_field_channels', 0)}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if self.force_head is not None:
            self.force_head.eval()
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
            loss:     scalar (MSE averaged over mobile atoms)
            mode:     int — the active config mode (echoed for logging)
            n_batch:  int — number of samples in the batch
            n_mobile: int — total mobile-atom count contributing to the mean
        """
        device = self.device
        B = len(batch)
        if B == 0:
            raise ValueError("empty batch")

        data_list: list[AtomicData] = []
        v_targets: list[torch.Tensor] = []
        t_values: list[float] = []
        fixed_list: list[torch.Tensor] = []
        delta_partner_list: list[torch.Tensor] = []  # only used in Mode 1

        for sample in batch:
            x0, x1, t, _ = sample_endpoints(sample, self.config, generator=generator)
            v_target = x1 - x0
            x_t_unwrapped = (1.0 - t) * x0 + t * x1
            # Mode 1 v5 — perturb x_t with Gaussian noise on mobile atoms only.
            # Target velocity stays `saddle − x_0`, so the model has to learn
            # to predict that even from off-line points (where force is
            # informative).
            if self.config.xt_perturb_sigma > 0.0:
                from ..data.transforms import gaussian_perturbation
                mobile = ~sample["fixed"]
                eps = gaussian_perturbation(
                    mobile, self.config.xt_perturb_sigma,
                    generator=generator, dtype=x_t_unwrapped.dtype,
                )
                x_t_unwrapped = x_t_unwrapped + eps
            x_t = wrap_positions(x_t_unwrapped, sample["cell"])
            data = build_atomic_data(
                x_t, sample["Z"], sample["cell"],
                sample["task_name"], sample["charge"], sample["spin"],
                sample["fixed"],
            )
            data_list.append(data)
            v_targets.append(v_target)
            t_values.append(t)
            fixed_list.append(sample["fixed"])

            if self.config.mode == 1:
                # Per-atom MIC-shortest displacement from x_t (wrapped) to the
                # partner. Frozen atoms have partner == start so their delta is
                # ~0 for the duration of the flow, which is the correct signal
                # (frozen atoms have no partner-direction).
                partner = sample["partner_un_pos"]
                delta_partner_list.append(mic_displacement(partner, x_t, sample["cell"]))

        batch_data = data_list_collater(data_list, otf_graph=True).to(device)
        v_target = torch.cat(v_targets, dim=0).to(device)
        fixed_all = torch.cat(fixed_list, dim=0).to(device)
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device)
        batch_idx = batch_data.batch

        delta_partner_all: torch.Tensor | None = None
        if self.config.mode == 1:
            delta_partner_all = torch.cat(delta_partner_list, dim=0).to(device)

        # Backbone forward needs grad-enabled mode if EITHER the backbone has
        # trainable params (v1+ unfreezes blocks[-1]) OR we need to compute
        # forces via autograd through the energy head (v2+).
        any_backbone_trainable = any(
            p.requires_grad for p in self.backbone.parameters()
        )
        need_grad = any_backbone_trainable or (self.force_head is not None)
        if self.force_head is not None:
            # CRITICAL: positions must require grad BEFORE backbone forward,
            # so the autograd graph captures pos → energy.
            batch_data["pos"].requires_grad_(True)

        is_v6_force_film = (
            isinstance(self.backbone, TimeFiLMBackbone)
            and getattr(self.backbone, "inject_force", False)
        )

        # First forward: no force-FiLM (force=None) so we can derive force.
        if isinstance(self.backbone, TimeFiLMBackbone):
            backbone_call = lambda f=None: self.backbone(batch_data, t_tensor, batch_idx, force=f)
        else:
            backbone_call = lambda f=None: self.backbone(batch_data)

        if need_grad:
            feat = backbone_call()
        else:
            with torch.no_grad():
                feat = backbone_call()

        # Compute forces (Mode 1 v2+) BEFORE running the velocity head, so the
        # autograd graph for forces is built first. Detach forces — we use
        # them as features only, not as a path that should propagate gradients
        # back into UMA.
        force_field_all: torch.Tensor | None = None
        if self.force_head is not None:
            from ..utils.forces import compute_uma_forces
            forces = compute_uma_forces(
                batch_data, feat, self.force_head, self.force_tasks,
                create_graph=False, task_name=batch[0]["task_name"],
            )
            force_field_all = forces.detach()

            # v6: re-run the backbone WITH force-FiLM. This is the second pass
            # of the two-pass scheme. Approx 2× backbone compute but the only
            # way to feed force into UMA's blocks (we couldn't pre-compute
            # force without first running backbone forward).
            if is_v6_force_film:
                # Need to re-prepare data["pos"].requires_grad for autograd
                # consistency on the head's gradient path.
                if need_grad:
                    feat = self.backbone(batch_data, t_tensor, batch_idx, force=force_field_all)
                else:
                    with torch.no_grad():
                        feat = self.backbone(batch_data, t_tensor, batch_idx, force=force_field_all)

        x = feat["node_embedding"]
        x = self.global_attn(x, batch_idx)
        v = self.velocity_head(
            x, t_tensor, batch_idx,
            delta_endpoint=delta_partner_all,
            force_field=force_field_all,
        )
        v = apply_output_projections(v, fixed_all, batch_idx, num_systems=B)

        sq_err = (v - v_target).pow(2).sum(dim=-1)  # (N_total,)
        mobile = ~fixed_all
        n_mobile = int(mobile.sum().item())

        if n_mobile > 0:
            loss = sq_err[mobile].mean()
        else:
            loss = sq_err.sum() * 0.0

        return {
            "loss": loss,
            "mode": self.config.mode,
            "n_batch": B,
            "n_mobile": n_mobile,
        }
