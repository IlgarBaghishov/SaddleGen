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
        # v7-2a: start integration from the midpoint between R (start) and P
        # (partner_un_pos), instead of from R. partner_un_pos is already
        # MIC-unwrapped relative to start, so the arithmetic mean is the
        # PBC-correct geodesic midpoint. Re-parameterizes the L2-Bayes-optimal
        # predictor from `E[saddle] - start ≈ midpoint - start` (large) to
        # `E[saddle] - midpoint ≈ 0`, forcing the head to use its features to
        # predict the *residual* deviation of the saddle from the midpoint
        # instead of rediscovering the midpoint itself.
        partner = sample["partner_un_pos"]
        x0 = 0.5 * (r_start + partner)
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
        eigenmode_head: nn.Module | None = None,
        eigenmode_loss_weight: float = 0.0,
        frozen_force_backbone: nn.Module | None = None,
        use_dimer_residual: bool = False,
        dimer_residual_alpha_init: float = 0.0,
    ):
        """Args:
            backbone: TRAINABLE UMA backbone (typically `TimeFiLMBackbone`-
                wrapped, with blocks[-1]/blocks[-2] unfrozen). Produces the
                features the velocity / eigenmode heads consume.
            force_head, force_tasks: UMA's pretrained energy / force head and
                its tasks dict, used for autograd-derived per-atom forces.
            frozen_force_backbone: v7-4-redesign — a SECOND, fully-frozen UMA
                backbone (no FiLM wrapping, no params trainable). When set,
                forces are computed by autograd through THIS frozen backbone,
                not the trainable one. Decouples force quality from training
                drift in the trainable backbone — the forces fed to the
                velocity head and used for F_dimer are guaranteed to be UMA's
                pretrained predictions throughout training. Adds one extra
                UMA forward per training step (frozen path, autograd-traced
                only through positions, not params).
            eigenmode_head: predicts the saddle's eigenmode from the
                trainable backbone's post-global-attn features (NOT from a
                pre-final velocity-head trunk). Used both for the cos² aux
                loss AND for F_dimer construction.
            eigenmode_loss_weight: scalar multiplier on the eigenmode aux
                loss term. v7-3 default is 0.1.
            use_dimer_residual: v7-4-redesign — when True, the model output
                gets a Dimer-style nudge `v_actual = v_pred + α · F_dimer`
                where α is the learnable scalar `dimer_residual_alpha`. This
                replaces the v7-3 "F_dimer as feature input" path: the
                eigenmode signal influences velocity through an explicit
                additive climb-direction term rather than through learned
                feature fusion. Requires both `eigenmode_head` and either
                `frozen_force_backbone` or a working `force_head` so that
                F_dimer can be computed.
            dimer_residual_alpha_init: initial value of α. Default 0.0 — at
                init the nudge contributes nothing and the model is identical
                to v7-3-without-F_dimer-feature; α grows during training as
                the model learns whether the climb signal helps.
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
        self.eigenmode_head = eigenmode_head
        self.eigenmode_loss_weight = float(eigenmode_loss_weight)
        # v7-4-redesign: separate frozen UMA copy for force computation.
        # Hidden inside a plain Python list so nn.Module.__setattr__ does NOT
        # register it as a submodule. This means:
        #   - its 291M params are NOT in `loss_module.parameters()` →
        #     not picked up by the optimizer or by accelerate's DDP grad-sync
        #     (which is what we want — they're frozen forever)
        #   - they're NOT saved to model.safetensors via accelerator.save_state
        #     → checkpoint stays the original ~1.2 GB instead of ballooning to
        #     ~2.4 GB × 60 epochs = +72 GB of duplicated UMA pretrained weights
        #   - .to(device) on the parent won't propagate, so the caller must
        #     pre-place the frozen backbone on the correct device
        # Each rank in DDP creates its own frozen copy independently in train.py
        # before accelerate.prepare; weights are deterministic (UMA's pretrained
        # snapshot), so all ranks have identical frozen backbones without any
        # sync. Access via the `frozen_force_backbone` property below.
        self._frozen_force_backbone_holder = (
            [frozen_force_backbone] if frozen_force_backbone is not None else []
        )
        if frozen_force_backbone is not None:
            for p in frozen_force_backbone.parameters():
                p.requires_grad_(False)
            frozen_force_backbone.eval()

        if (eigenmode_head is None) and self.eigenmode_loss_weight > 0:
            raise ValueError(
                "eigenmode_loss_weight > 0 requires an eigenmode_head; got None."
            )
        # v7-3 legacy F_dimer-as-feature path; we may still keep it active if
        # the head was built with dimer_force_channels > 0, but the v7-4
        # default is to disable it (channels=0) and use the output-side nudge.
        dimer_channels = int(getattr(velocity_head, "dimer_force_channels", 0))
        if dimer_channels > 0 and eigenmode_head is None:
            raise ValueError(
                "velocity_head has dimer_force_channels>0 but no eigenmode_head "
                "was provided — F_dimer cannot be computed without ê."
            )
        self._compute_dimer_force_feature = dimer_channels > 0

        # v7-4-redesign: output-side Dimer nudge with a learnable scalar α.
        # Place α on the same device as the velocity head's params (the head
        # was passed in already-on-device by the caller). Without this, α
        # lands on CPU and the EMA shadow built later cannot align with the
        # post-`accelerator.prepare` GPU-resident parameter — the in-place
        # `s.mul_().add_(p)` fails with a device mismatch.
        self.use_dimer_residual = bool(use_dimer_residual)
        if self.use_dimer_residual:
            _alpha_device = next(velocity_head.parameters()).device
            self.dimer_residual_alpha = nn.Parameter(
                torch.tensor(float(dimer_residual_alpha_init), device=_alpha_device)
            )
        else:
            self.dimer_residual_alpha = None
        if self.use_dimer_residual:
            if eigenmode_head is None:
                raise ValueError(
                    "use_dimer_residual=True requires eigenmode_head (need ê for F_dimer)."
                )
            if force_head is None and frozen_force_backbone is None:
                raise ValueError(
                    "use_dimer_residual=True requires either force_head or "
                    "frozen_force_backbone — F_dimer needs F."
                )
        # Whether ANY F_dimer pathway is active (feature OR output residual).
        self._compute_dimer_force = self._compute_dimer_force_feature or self.use_dimer_residual

        if (force_head is None) != (getattr(velocity_head, "force_field_channels", 0) == 0):
            raise ValueError(
                "force_head must be provided iff velocity_head.force_field_channels > 0; "
                f"got force_head={'set' if force_head is not None else 'None'}, "
                f"force_field_channels={getattr(velocity_head, 'force_field_channels', 0)}"
            )

    @property
    def frozen_force_backbone(self):
        """The fully-frozen UMA copy used only for force computation.
        Hidden in `_frozen_force_backbone_holder` to keep it out of the
        registered-submodule tree (and thus out of state_dict / DDP / etc.).
        Returns None if no frozen backbone was supplied."""
        return self._frozen_force_backbone_holder[0] if self._frozen_force_backbone_holder else None

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if self.force_head is not None:
            self.force_head.eval()
        if self.frozen_force_backbone is not None:
            self.frozen_force_backbone.eval()
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
        eigenmode_targets: list[torch.Tensor] = []   # v7-3: only when aux loss > 0
        # v7-2b: collect R and P AtomicData per sample so we can run UMA on the
        # endpoints in static mode (no time-FiLM, no force-FiLM) and feed the
        # resulting per-atom features to the velocity head.
        want_endpoint_features = (
            self.config.mode == 1
            and getattr(self.velocity_head, "endpoint_features_enabled", False)
        )
        data_list_R: list[AtomicData] = []
        data_list_P: list[AtomicData] = []

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
                # v7-2a: pass BOTH (R - x_t) and (P - x_t) as per-atom MIC
                # displacements. Starting from the midpoint, the head needs to
                # know where both endpoints sit relative to the current point;
                # passing only the partner (as the v6 head did) loses the
                # symmetric R-side information. Stacked as (N, 2, 3); the head
                # now expects a 2-endpoint delta signal.
                start_pos = sample["start_pos"]
                partner = sample["partner_un_pos"]
                delta_R = mic_displacement(start_pos, x_t, sample["cell"])
                delta_P = mic_displacement(partner, x_t, sample["cell"])
                delta_partner_list.append(torch.stack([delta_R, delta_P], dim=1))

            if want_endpoint_features:
                # v7-2b: build R and P AtomicData (wrapped into the unit cell so
                # the OTF graph constructor sees a clean periodic copy).
                # partner_un_pos is MIC-unwrapped to start; wrap it back for UMA.
                R_pos = wrap_positions(sample["start_pos"], sample["cell"])
                P_pos = wrap_positions(sample["partner_un_pos"], sample["cell"])
                data_list_R.append(build_atomic_data(
                    R_pos, sample["Z"], sample["cell"],
                    sample["task_name"], sample["charge"], sample["spin"],
                    sample["fixed"],
                ))
                data_list_P.append(build_atomic_data(
                    P_pos, sample["Z"], sample["cell"],
                    sample["task_name"], sample["charge"], sample["spin"],
                    sample["fixed"],
                ))

            # v7-3: collect ground-truth eigenmode targets when aux loss is on.
            # `eigenmode` is (N, 3); frozen-atom rows are 0 by construction
            # (DFT eigenmodes vanish on FixAtoms by definition).
            if self.eigenmode_loss_weight > 0:
                eig = sample.get("eigenmode")
                if eig is None:
                    raise KeyError(
                        "eigenmode aux loss enabled but sample dict has no "
                        "'eigenmode' key — check the dataset adapter."
                    )
                eigenmode_targets.append(eig)

        batch_data = data_list_collater(data_list, otf_graph=True).to(device)
        v_target = torch.cat(v_targets, dim=0).to(device)
        fixed_all = torch.cat(fixed_list, dim=0).to(device)
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device)
        batch_idx = batch_data.batch

        delta_partner_all: torch.Tensor | None = None
        if self.config.mode == 1:
            # v7-2a: shape is (N_total, 2, 3) — [delta_R, delta_P] per atom.
            delta_partner_all = torch.cat(delta_partner_list, dim=0).to(device)

        # v7-2b: featurize R and P through UMA in static mode FIRST, before the
        # x_t backbone forward. UMA's MoLE caches per-graph chunk-dispatch
        # state on each forward; gradient checkpointing on the x_t forward
        # later re-runs it during backward and asserts that the cached state
        # matches the current input. If we let R/P forwards run after x_t (but
        # before backward), MoLE's state is overwritten to R/P's graph and the
        # x_t backward recomputation explodes with a shape-mismatch assert.
        # By running R and P first, the LAST forward before backward is x_t
        # (and v6's second pass is also on x_t), so MoLE's state is consistent
        # with what backward expects.
        endpoint_features_all: torch.Tensor | None = None
        if want_endpoint_features:
            if not isinstance(self.backbone, TimeFiLMBackbone):
                raise RuntimeError(
                    "v7-2b endpoint features require a TimeFiLMBackbone wrapper "
                    "(it provides forward_static)."
                )
            R_batch = data_list_collater(data_list_R, otf_graph=True).to(device)
            P_batch = data_list_collater(data_list_P, otf_graph=True).to(device)
            with torch.no_grad():
                R_feat = self.backbone.forward_static(R_batch)["node_embedding"]
                P_feat = self.backbone.forward_static(P_batch)["node_embedding"]
            endpoint_features_all = torch.cat([R_feat, P_feat], dim=-1).detach()

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

        # ============================================================
        # Step A — REAL forces from FROZEN UMA at x_t (v7-4-redesign).
        # ============================================================
        # If a `frozen_force_backbone` is provided, compute forces by
        # autograd through THAT (not the trainable one). This guarantees
        # forces are UMA's pretrained predictions regardless of how far the
        # trainable backbone has drifted. Done BEFORE the trainable forward
        # so its features (used by force-FiLM 2-pass) see real forces too.
        # No autograd backward into the frozen backbone (its params are
        # requires_grad=False), only through positions.
        force_field_all: torch.Tensor | None = None
        if self.force_head is not None and self.frozen_force_backbone is not None:
            from ..utils.forces import compute_uma_forces
            # Vanilla UMA forward — no time-FiLM, no force-FiLM. The frozen
            # backbone is plain UMA, not TimeFiLMBackbone.
            feat_frozen = self.frozen_force_backbone(batch_data)
            forces_frozen = compute_uma_forces(
                batch_data, feat_frozen, self.force_head, self.force_tasks,
                create_graph=False, task_name=batch[0]["task_name"],
            )
            force_field_all = forces_frozen.detach()

        # ============================================================
        # Step B — Trainable backbone forward(s).
        # ============================================================
        if isinstance(self.backbone, TimeFiLMBackbone):
            backbone_call = lambda f=None: self.backbone(batch_data, t_tensor, batch_idx, force=f)
        else:
            backbone_call = lambda f=None: self.backbone(batch_data)

        # If we have real forces from Step A, the trainable backbone's
        # FIRST pass can already use them via force-FiLM (single forward
        # is enough — no need for the v6 two-pass scheme that derives
        # forces from the trainable backbone's own features). If forces
        # come from the trainable backbone (legacy path, no frozen UMA),
        # we still need the two-pass scheme below.
        if need_grad:
            feat = backbone_call(force_field_all if is_v6_force_film else None)
        else:
            with torch.no_grad():
                feat = backbone_call(force_field_all if is_v6_force_film else None)

        # Legacy path: if we DON'T have a frozen force backbone but DO have
        # a force_head, derive forces from the trainable backbone's features
        # (this matches the v7-3 behaviour). Then run the v6 second pass.
        if self.force_head is not None and self.frozen_force_backbone is None:
            from ..utils.forces import compute_uma_forces
            forces = compute_uma_forces(
                batch_data, feat, self.force_head, self.force_tasks,
                create_graph=False, task_name=batch[0]["task_name"],
            )
            force_field_all = forces.detach()
            if is_v6_force_film:
                if need_grad:
                    feat = self.backbone(batch_data, t_tensor, batch_idx, force=force_field_all)
                else:
                    with torch.no_grad():
                        feat = self.backbone(batch_data, t_tensor, batch_idx, force=force_field_all)

        # endpoint_features_all was computed above (BEFORE the x_t forward) so
        # MoLE state remains consistent with the x_t backward recompute.
        x = feat["node_embedding"]
        x = self.global_attn(x, batch_idx)

        # v7-3 / v7-4: predict the saddle's eigenmode. v7-4-redesign: the
        # eigenmode head is a VelocityHead subclass, so we feed it the SAME
        # conditioning the velocity head sees (delta_R/delta_P, real UMA
        # force, UMA(R)/UMA(P) features, time-FiLM). `dimer_force=None`
        # because F_dimer requires ê to compute — it can't be an input here.
        # Must run BEFORE F_dimer construction (which uses ê).
        eig_pred: torch.Tensor | None = None
        dimer_force_all: torch.Tensor | None = None
        if self.eigenmode_head is not None:
            eig_pred = self.eigenmode_head(
                x, t_tensor, batch_idx,
                delta_endpoint=delta_partner_all,
                force_field=force_field_all,
                endpoint_features=endpoint_features_all,
                dimer_force=None,
            )                                                          # (N_total, 3)
            if self._compute_dimer_force:
                # F_dimer = F − 2 (F · v) v / ‖v‖²,  per-system (the dot product
                # and norm are over the full 3N_b vector, NOT per atom).
                # Implementation: per-atom inner = F[i]·v[i]; per-atom norm² =
                # v[i]·v[i]; sum each per-system; combine into the per-atom
                # subtraction.
                F_at = force_field_all
                v_at = eig_pred
                inner_atom = (F_at * v_at).sum(dim=-1)                # (N_total,)
                norm_sq_atom = (v_at * v_at).sum(dim=-1)               # (N_total,)
                inner_sys = torch.zeros(B, device=device, dtype=F_at.dtype)
                norm_sq_sys = torch.zeros(B, device=device, dtype=F_at.dtype)
                inner_sys.index_add_(0, batch_idx, inner_atom)
                norm_sq_sys.index_add_(0, batch_idx, norm_sq_atom)
                # Per-system scalar coefficient `2 (F·v) / ‖v‖²`. Detach so the
                # F_dimer signal does not back-propagate gradients into the
                # eigenmode head from the velocity loss; the eigenmode head is
                # trained ONLY by its own cos² loss. (Without this detach, the
                # velocity loss would push the eigenmode head to predict
                # whatever rotation of F is locally convenient, which is not
                # what we want supervision-wise.)
                scale_sys = (2.0 * inner_sys / (norm_sq_sys + 1e-12)).detach()
                scale_per_atom = scale_sys[batch_idx].unsqueeze(-1)    # (N_total, 1)
                dimer_force_all = (F_at - scale_per_atom * v_at.detach())  # (N_total, 3)

        # v7-4-redesign: only pass `dimer_force` as a feature input if the
        # head was built for it (legacy v7-3 path). The new v7-4 default uses
        # the output-side residual nudge instead — see below.
        v = self.velocity_head(
            x, t_tensor, batch_idx,
            delta_endpoint=delta_partner_all,
            force_field=force_field_all,
            endpoint_features=endpoint_features_all,
            dimer_force=(dimer_force_all if self._compute_dimer_force_feature else None),
        )

        # v7-4-redesign: output-side Dimer nudge.  v_actual = v_pred + α · F_dimer.
        # Applied BEFORE `apply_output_projections` so the CoM-subtraction etc.
        # operate on the final velocity. α is a learnable scalar (init 0).
        if self.use_dimer_residual and dimer_force_all is not None:
            v = v + self.dimer_residual_alpha * dimer_force_all

        v = apply_output_projections(v, fixed_all, batch_idx, num_systems=B)

        sq_err = (v - v_target).pow(2).sum(dim=-1)  # (N_total,)
        mobile = ~fixed_all
        n_mobile = int(mobile.sum().item())

        if n_mobile > 0:
            velocity_loss = sq_err[mobile].mean()
        else:
            velocity_loss = sq_err.sum() * 0.0

        # v7-3: sign-invariant cos² eigenmode loss (per system, over the full
        # 3N_b vector, mobile atoms only).
        eigenmode_loss = torch.tensor(0.0, device=device, dtype=velocity_loss.dtype)
        if self.eigenmode_loss_weight > 0:
            eig_target = torch.cat(eigenmode_targets, dim=0).to(device).to(velocity_loss.dtype)
            mob_f = mobile.to(velocity_loss.dtype).unsqueeze(-1)        # (N_total, 1)
            ep = eig_pred * mob_f
            et = eig_target * mob_f
            inner_atom_e = (ep * et).sum(dim=-1)                        # (N_total,)
            np_atom = (ep * ep).sum(dim=-1)
            nt_atom = (et * et).sum(dim=-1)

            inner_sys_e = torch.zeros(B, device=device, dtype=velocity_loss.dtype)
            np_sys = torch.zeros(B, device=device, dtype=velocity_loss.dtype)
            nt_sys = torch.zeros(B, device=device, dtype=velocity_loss.dtype)
            inner_sys_e.index_add_(0, batch_idx, inner_atom_e)
            np_sys.index_add_(0, batch_idx, np_atom)
            nt_sys.index_add_(0, batch_idx, nt_atom)

            cos_sq = (inner_sys_e ** 2) / (np_sys * nt_sys + 1e-12)
            valid = nt_sys > 1e-8
            if valid.any():
                eigenmode_loss = (1.0 - cos_sq[valid]).mean()

        loss = velocity_loss + self.eigenmode_loss_weight * eigenmode_loss

        return {
            "loss": loss,
            "velocity_loss": velocity_loss.detach(),
            "eigenmode_loss": eigenmode_loss.detach(),
            "mode": self.config.mode,
            "n_batch": B,
            "n_mobile": n_mobile,
        }
