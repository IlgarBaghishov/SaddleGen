"""
Forward-Euler integrator for flow-matching inference.

Given a reactant structure, draws `n_perturbations` independent Gaussian starts
(`x_0 = r_R + ε_rs_pert`), then integrates each trajectory forward under the
velocity field `v_θ(x_t, t)` for `K` uniform Euler steps. Output is a
`(n_perturbations, N, 3)` tensor of candidate saddles. Clustering / Hungarian
matching against reference TSs is done downstream in `utils/eval.py`.

All `n_perturbations` trajectories are batched into a single AtomicData at each
Euler step (one UMA forward per step, not per trajectory). This keeps cost
O(K · one_batched_UMA_forward), independent of how many perturbations you draw.
"""

import torch
import torch.nn as nn
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from ..data.transforms import gaussian_perturbation, mic_displacement, wrap_positions
from ..models.time_filmed_backbone import TimeFiLMBackbone
from .matching import apply_output_projections, build_atomic_data


@torch.no_grad()
def sample_saddles(
    reactant: dict,
    backbone: nn.Module,
    global_attn: nn.Module,
    velocity_head: nn.Module,
    sigma_inf: float,
    n_perturbations: int = 32,
    K: int = 50,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    partner_pos: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Euler-integrate `n_perturbations` trajectories from perturbed reactant to t=1.

    **Mode 0 (default).** Pure flow-matching from `r_R + ε` to a saddle, no
    partner conditioning. `velocity_head` must be a Mode-0 head
    (`delta_endpoint_channels == 0`). Caller does NOT pass `partner_pos`.

    **Mode 1 (`partner_pos` provided).** Product-conditional flow. The head is
    Mode 1 (`delta_endpoint_channels > 0`) and at each Euler step we feed it
    the per-atom MIC-shortest displacement from the current x_t to the partner
    (the OTHER local minimum — P when starting from R, R when starting from P).
    Mode 1 is deterministic given `partner_pos`; you typically pass `sigma_inf=0`
    and `n_perturbations=1` to get a single deterministic prediction. To still
    get a small ensemble (for clustering / robustness), pass a small `sigma_inf`
    and >1 perturbations — the partner conditioning will keep all trajectories
    aimed at the same saddle.

    Args:
        reactant: a sample dict with the same keys used by `FlowMatchingLoss`
            (`start_pos`, `Z`, `cell`, `fixed`, `task_name`, `charge`, `spin`).
            `start_pos` is the reactant (or product) structure in Å.
        backbone / global_attn / velocity_head: the UMA backbone and the two
            SaddleGen modules. The caller is responsible for having set `.eval()`
            on them; this routine does NOT toggle training mode.
        sigma_inf: Gaussian std (Å) for the inference-time perturbation that
            spreads initial Li positions around r_R before Euler integration.
            Decoupled from training; a value of ~0.15 is typical for LiC Mode 0.
            For Mode 1 a small value (or 0) is the natural choice.
        n_perturbations: number of independent trajectories. Default 32.
        K: number of Euler steps. Default 50 per CLAUDE.md.
        device: override the device (default = `velocity_head`'s device).
        generator: torch.Generator for reproducibility. If provided on CPU while
            the model is on CUDA, perturbations are drawn on CPU and moved —
            identical distribution, just less efficient.
        return_trajectory: if True, also return the full `(K+1, n_perturbations, N, 3)`
            tensor of intermediate positions (including the initial perturbed starts).
        partner_pos: (N, 3) MIC-unwrapped partner positions (R or P). Required
            for Mode 1, must be omitted for Mode 0. Same atom ordering as
            `reactant['start_pos']`.

    Returns:
        `(n_perturbations, N, 3)` candidate saddle positions, wrapped into the unit cell.
        If `return_trajectory` is True, returns a tuple `(final, trajectory)`.
    """
    if device is None:
        device = next(velocity_head.parameters()).device
    else:
        device = torch.device(device)

    r_R = reactant["start_pos"].to(device)
    Z = reactant["Z"].to(device)
    cell = reactant["cell"].to(device)
    fixed = reactant["fixed"].to(device)
    task_name = reactant["task_name"]
    charge = int(reactant["charge"])
    spin = int(reactant["spin"])
    mobile = ~fixed
    N = r_R.shape[0]

    # Mode-0 vs Mode-1 input validation against the head's configuration.
    head_mode_1 = getattr(velocity_head, "delta_endpoint_channels", 0) > 0
    if head_mode_1 and partner_pos is None:
        raise ValueError(
            "velocity_head is configured for Mode 1 (delta_endpoint_channels>0) "
            "but partner_pos was not provided to sample_saddles."
        )
    if (not head_mode_1) and partner_pos is not None:
        raise ValueError(
            "partner_pos was provided but velocity_head is configured for Mode 0 "
            "(delta_endpoint_channels==0). Either omit partner_pos or rebuild the "
            "head with delta_endpoint_channels > 0."
        )
    partner = partner_pos.to(device) if partner_pos is not None else None

    # Draw perturbations. Use CPU-side `mobile` if the generator is CPU, since
    # torch.randn requires generator.device == target device.
    if generator is not None and generator.device.type != device.type:
        mobile_gen = mobile.cpu()
    else:
        mobile_gen = mobile
    eps_stack = torch.stack(
        [gaussian_perturbation(mobile_gen, sigma_inf, generator=generator)
         for _ in range(n_perturbations)],
        dim=0,
    ).to(device)  # (n_perturbations, N, 3)

    x = wrap_positions(r_R.unsqueeze(0) + eps_stack, cell)  # (n_perturbations, N, 3)

    if return_trajectory:
        traj = torch.empty(K + 1, n_perturbations, N, 3, device=device, dtype=x.dtype)
        traj[0] = x

    fixed_all = fixed.repeat(n_perturbations)  # (n_perturbations * N,)
    dt = 1.0 / K

    for step in range(K):
        t_scalar = step / K
        t_tensor = torch.full(
            (n_perturbations,), t_scalar, dtype=torch.float32, device=device,
        )
        data_list = [
            build_atomic_data(x[i], Z, cell, task_name, charge, spin, fixed)
            for i in range(n_perturbations)
        ]
        batch_data = data_list_collater(data_list, otf_graph=True).to(device)
        batch_idx = batch_data.batch

        delta_partner_all: torch.Tensor | None = None
        if partner is not None:
            # Per-atom MIC displacement from each trajectory's current x[i] to
            # the (shared) partner. (n_perturbations, N, 3) -> flat (P*N, 3).
            delta_each = torch.stack(
                [mic_displacement(partner, x[i], cell) for i in range(n_perturbations)],
                dim=0,
            )  # (n_perturbations, N, 3)
            delta_partner_all = delta_each.reshape(-1, 3)

        if isinstance(backbone, TimeFiLMBackbone):
            feat = backbone(batch_data, t_tensor, batch_idx)
        else:
            feat = backbone(batch_data)
        h = global_attn(feat["node_embedding"], batch_idx)
        v = velocity_head(h, t_tensor, batch_idx, delta_endpoint=delta_partner_all)
        v = apply_output_projections(v, fixed_all, batch_idx, num_systems=n_perturbations)
        v = v.view(n_perturbations, N, 3)

        x = wrap_positions(x + dt * v, cell)

        if return_trajectory:
            traj[step + 1] = x

    if return_trajectory:
        return x, traj
    return x
