# CLAUDE.md

## What is SaddleGen

SaddleGen is a PyTorch library for generating transition-state (saddle-point) structures for periodic materials, given only the reactant (initial) state. The target use case is **reaction discovery**: propose plausible saddles from a reactant alone, without needing a guessed product. An optional future mode will also accept a product state to condition toward a specific reaction; the reactant-only mode is the main research contribution.

The ML method is **flow matching** (extendable to MeanFlow). The velocity field is a three-stage module: Meta FAIR's pretrained **UMA** (`uma-s-1p2`) as the local-equivariant backbone, a light **invariant-weighted equivariant global self-attention** layer (`GlobalAttn`) that allows distant atoms to coordinate which one reacts, and a small `VelocityHead` that projects the globally-aware per-atom features to per-atom velocity vectors.

## Audience / style anchor

Code targets a computational chemist who reads PyTorch fluently and knows ASE and pymatgen. Style reference: `../GenFlows` (Ilgar's previous project) — pure PyTorch, minimal abstractions, clean "methods are pure math, models handle conditioning" separation. Target total size: ~1–2k lines excluding docs and the fairchem dependency. Use ASE and fairchem wherever possible to avoid reinventing; avoid heavy config / framework layers (Hydra, PyTorch Lightning, TorchTNT) unless they buy real value.

## Architecture

### Backbone — fairchem UMA (`uma-s-1p2`)

Pretrained SO(3)-equivariant eSCN-MD architecture. Loaded via `fairchem.core.calculate.pretrained_mlip.get_predict_unit("uma-s-1p2", ...)`; the returned `MLIPPredictUnit` wraps an `AveragedModel(HydraModel)` whose `.module.backbone` we consume directly (and whose `.module.output_heads` we discard — UMA's energy/force heads are replaced by `GlobalAttn → VelocityHead`).

UMA-S-1.2 is the small variant — backbone config `K4L2` (`sphere_channels=128`, `lmax=2`, `num_layers=4`, cutoff `6.0 Å`, `max_neighbors=30` non-strict). The backbone returns `{"node_embedding": (N, (lmax+1)², sphere_channels) = (N, 9, 128)}` per forward pass. Trained in bf16; we match. Param count: 6.6M active / 290M total (all 32 MoE experts merged).

UMA's mixture-of-experts is routed by `data.dataset` (string per graph; `task_name` is exposed as a property alias). Supported routing values: `{omat, oc20, oc22, oc25, omol, odac, omc}` — full set used for our triplets. Routing happens inside the backbone via `csd_embedding(charge, spin, dataset)`, which produces per-expert coefficients consumed by the MOLE layers — so `charge` and `spin` are mandatory inputs (default `0` and `0`; `spin` is only physically consumed by the `omol` head). For training efficiency, batches should be homogeneous in `task_name` where possible to amortize the MoE coefficient-set step.

**Frozen by default** on the first training run; a `--finetune` flag enables end-to-end backprop through UMA. Frozen baseline is cheaper and simpler; fine-tuning is the second-iteration knob for accuracy.

### Global attention — `GlobalAttn`

Sits between the UMA backbone and the head. Solves a critical failure mode of pure local GNNs: when two independent reaction sites sit in the same cell at distances beyond UMA's cutoff (~6 Å), UMA cannot mediate between them and would predict both reactions happening simultaneously — which is not a first-order saddle (it would be a second-order saddle at best) and is physically almost never traversed. Global attention lets distant atoms "negotiate" which site reacts.

**Architecture — invariant-weighted equivariant attention** (Option B from design discussion):

- Split UMA's per-atom output into invariant channels (`l = 0` scalars) and equivariant channels (`l ≥ 1` vectors / higher-rank tensors).
- Compute multi-head self-attention **weights** from invariant channels only:
  `Q = W_Q · x_l0`, `K = W_K · x_l0`, `attn = softmax(Q · K^T / √d)`.
  Because `x_l0` is scalar under SO(3), `attn` is rotation-invariant.
- Values `V = W_V · x_full_irreps` — equivariant linear projection (via UMA's `SO3_Linear`) preserving the full irrep structure. Originally specified as e3nn's `o3.Linear`, but switched to `SO3_Linear` during implementation to avoid `(N, (lmax+1)², C) ↔ (N, Irreps.dim)` layout-conversion at every module boundary; the two operations are mathematically equivalent (independent per-l weights, bias only on `l=0`).
- Output per atom: `out_i = Σ_j attn_{ij} · V_j`. Because `attn_{ij}` is a scalar and each `V_j` is equivariant, the output is equivariant.
- Default: 1 layer, configurable (1–4) for ablation.

**Equivariance proof (sketch):** under a global rotation `R`, `x_l0` is unchanged so `attn` is unchanged; `V_j` transforms like the underlying irreps; scalar-weighted sum of equivariant vectors is equivariant. ✓

**Scaling:** O(N²) per forward pass. Fine for the Li/C test (N ≈ 60). For 200-atom systems in the 30M run, manageable on A100 with bf16. Distance-sparse variant (attend only within 2 × UMA cutoff) reserved as an optimization if pure attention becomes a bottleneck.

**Code budget:** ~100 lines using UMA's `SO3_Linear` for `W_V`/`W_out` and standard `nn.Linear` for `W_Q`, `W_K`. Residual connection is additive (`x = x + attn(x)`); no normalization layer yet — we'll add an SO3-aware RMSNorm (UMA's `rms_norm_sh`) if training is unstable.

**Batched graphs:** attention is masked so atoms in different systems cannot attend to each other. Uses the PyG-style `batch_idx: (N,)` passed in from the dataloader; `logits.masked_fill(~(batch_idx_i == batch_idx_j), -inf)` before softmax.

### Head — `VelocityHead`

Custom, ~140 lines. Mirrors fairchem's `Linear_Force_Head` (a single equivariant `SO3_Linear(sphere_channels=128, 1, lmax=1)` projecting the `l=0,1` slice and keeping the `l=1` output reshaped to `(N, 3)`), but operates on the **globally-aware features from `GlobalAttn`** instead of directly on UMA's output. Implementation lives in `saddlegen/models/velocity_head.py`.

**Implementation note: uses UMA's `SO3_Linear`, not e3nn's `o3.Linear`.** UMA stores irreps as `(N, (lmax+1)², C)` while e3nn flattens them to `(N, Irreps.dim)` — using e3nn would require ~50 lines of layout-conversion plumbing at every module boundary. `SO3_Linear` is the equivalent equivariant operation (independent per-l weights, bias only on `l=0`) and keeps us in UMA's native layout. The `UMAGate` nonlinearity (for `depth ≥ 2`) is a small hand-rolled equivalent of `e3nn.nn.Gate` that also works in UMA's layout — SiLU on `l=0` channels, sigmoid-of-linear(l=0) as a multiplicative gate on `l≥1` channels.

**Time conditioning — FiLM / AdaLN-zero style.** A sinusoidal embedding of `t ∈ [0, 1]` is passed through a small MLP that outputs `2C` features, split into (i) a per-channel scalar bias added to the `l=0` channels, and (ii) a per-channel scalar factor `(1 + γ)` multiplying the `l≥1` channels. The final MLP layer is **zero-initialized**, so at init `bias=0` and `γ=0` ⇒ `(1+γ)=1`, and the head is numerically identical to `Linear_Force_Head`. Time-dependence is learned during training.

**Why time FiLM is equivariant.** A plain additive bias on `l=0` channels alone is NOT sufficient: `SO3_Linear` keeps l-channel paths independent, so an `l=0` bias never propagates to the `l=1` output — time would have zero effect on the predicted velocity. The multiplicative gate on `l≥1` fixes this. Equivariance is preserved because:

- Both `t_bias` and `t_gate` are derived from `sinusoidal(t)` — a function of a scalar flow-time `t` and a system-membership index `batch_idx`. Neither depends on atomic coordinates, so under any `g ∈ SO(3)` both are **invariant** (`g · t_bias = t_bias`, `g · t_gate = t_gate`).
- Adding an invariant to `l=0` (which is itself invariant) gives an invariant. ✓
- Multiplying an `l=ℓ` feature by an invariant scalar gives an `l=ℓ` feature: under rotation,
    `(g·h)_l1 = (g·x)_l1 · (g·t_gate) = (D¹(g)·x_l1) · t_gate = D¹(g)·(x_l1·t_gate) = D¹(g)·h_l1`. ✓
- The form `x_l1·(1+γ) = x_l1 + x_l1·γ` is a sum of two `l=1` features (the second via the invariant×vector = vector rule), which is `l=1`. ✓

The general rule: **"scalar" in the equivariant sense means SO(3)-invariant, not spatially constant**. Per-atom, per-channel invariant factors (like `t_bias[n, c]`) are safe to add to/multiply with equivariant features, because the equivariance condition only asks about rotations of 3D-space, and a function of `t` and `system_id` has no 3D-space input to rotate. This is the same trick e3nn's `Gate` and UMA's own `csd_embedding`/MOLE gating rely on.

Empirically verified in-tree: equivariance error `≤ 5e-7` (noise floor) both at init and after 20 Adam steps; at init `v(t=0.1) == v(t=0.9)` exactly; after training, `‖v(0.9)−v(0.1)‖ ≈ 0.37` — time flows through as intended.

**Capacity knob — `depth: int = 1`** (default, head ≡ `Linear_Force_Head` + time FiLM). For `depth ≥ 2`, inserts `(depth − 1)` `SO3_Linear → UMAGate` blocks before the final projection. Reasoning: UMA itself uses linear because the 4-layer eSCN-MD backbone provides enough nonlinearity, but with UMA frozen those layers can't adapt to our velocity target, so head capacity may matter. Raise as an ablation if Li/C recall plateaus.

Hyperparameters pinned: `time_embed_dim=64`, `time_mlp_hidden=128`, zero-init on the final `time_mlp` layer. Subject to ablation.

### Output projections (applied after the head, in order)

1. **Hard mask for frozen atoms:** `v[fixed] = 0`. ASE's `FixAtoms` constraint is enforced, not learned.
2. **Per-system Center-of-mass projection on mobile atoms — conditional:** `v[mobile] −= v[mobile].mean(dim=0, keepdim=True)`, applied **only for systems with zero frozen atoms**. Skipped for any system with at least one frozen atom. Implemented in `saddlegen/flow/matching.py::_com_projection_batched`.

**Why conditional — critical gotcha, don't remove this check.** The motivation for CoM projection is to remove the translational-symmetry degree of freedom so the network can't drift the whole system through space. That argument only applies to fully-mobile systems. **When frozen atoms are present they already pin the reference frame**, so there is no translational symmetry left for the network to abuse.

Worse, unconditional projection is **actively destructive** when the mobile-atom count is small. Concrete failure mode (the Li/C test case):

- `N_mobile = 1` (a single Li on a frozen C-sheet).
- `v[mobile].mean(dim=0)` is then literally `v[Li]`.
- `v[Li] -= v[Li]` yields `v[Li] = 0`.
- The model's output for the only atom that's allowed to move is zeroed on every forward pass.
- `v_target = r_S_un − r_R` for Li is ~1.25 Å along some direction; `(0 − 1.25)² ≈ 1.56` becomes a constant loss with zero gradient to every time-FiLM parameter.
- Training silently fails — loss plateaus, weights don't update in any meaningful way.

This was a latent bug in the initial implementation (CLAUDE.md spec previously mandated unconditional projection, tests passed on fully-mobile toy inputs, only the first integrated run on real Li/C would have exposed it — with zero gradient, not a visible error). Fixed by the per-system conditional above. The unit test in `_com_projection_batched`'s docstring covers the three cases: single-mobile-atom frozen system, all-mobile system, batched mix.

**For fully-mobile systems (no `FixAtoms`)** the projection still runs as before. **For any system with `FixAtoms`** the projection is skipped; the frozen atoms are the pin. This is the right rule for surface chemistry, slabs, adsorbates on fixed substrates, and any constrained reaction — i.e., essentially all of the 30M-triplet training set.

## Flow formulation

### Space and endpoints
- **Flow state:** actual Cartesian atomic coordinates `r(t) ∈ ℝ^(N×3)` in Å. UMA is always fed physical atomic positions, never displacement vectors.
- **Endpoints:** `r(0) = r_reactant + ε_rs_pert` (perturbed reactant; Gaussian perturbation with std `σ_rs_pert` on mobile atoms), `r(1) = r_saddle_unwrapped` (MIC-unwrapped saddle position).
- **Path:** straight-line Optimal Transport, `r(t) = (1−t) · r(0) + t · r(1)`, wrapped back into the unit cell before each UMA forward pass.
- **Target velocity:** `v_target = r(1) − r(0)`, constant in `t` for each training sample.

### Minimum-image unwrap (one-time, at dataset conversion)

```
Δ_raw = r_saddle_raw − r_reactant
Δs    = inv(cell) @ Δ_raw
Δs   -= round(Δs)                 # wrap fractional components into [−½, ½]
Δ     = cell @ Δs
r_saddle_unwrapped = r_reactant + Δ
```

`r_saddle_unwrapped` may lie outside the unit cell; that's fine. Interpolation is done in unwrapped space. Positions are wrapped back into the cell only immediately before being passed to UMA (for clean neighbor-list generation and canonical coordinates).

### Perturbation geometry — Gaussian on mobile atoms

Perturbations `ε_rs_pert` (reactant-state, used by obj (2) and inference) and `ε_ts_pert` (TS-state, used by optional obj (1)) are drawn from an isotropic Gaussian **only over mobile (non-frozen) atoms**. Frozen atoms receive zero perturbation. This choice is deliberate:

- **Sparsity matches physics.** Real TS displacement vectors are sparse — a few atoms participate in the bond-forming / breaking event, most atoms barely move. Gaussian's per-atom magnitude distribution (χ₃ tail) naturally produces "one-atom-moves-a-lot" samples, which a fixed-norm shell cannot — a shell concentrates per-atom magnitude tightly around `R/√M` and suppresses single-atom-dominant perturbations.
- **Same distribution at training and inference** — no train/inference asymmetry.

**Sampling:**
```
ε_rs_pert_raw       ~ N(0, σ_rs_pert² · I_{3M})     # reactant-state perturbation, 3 components per mobile atom
ε_rs_pert[mobile]    = ε_rs_pert_raw.reshape(M, 3)
ε_rs_pert[frozen]    = 0
```

- `σ_rs_pert` in Å (absolute, not dataset-relative). Primary hyperparameter. Rule-of-thumb default: `σ_rs_pert ≈ 0.05 · ⟨‖Δ‖⟩ / √(3M)`, where `⟨‖Δ‖⟩` is the dataset-mean displacement norm (computed at LMDB conversion time).
- An analogous TS-state perturbation `ε_ts_pert` with std `σ_ts_pert` is used by objective (1) (TS-denoising) with identical structure — Gaussian on mobile atoms, zero on frozen.
- `GlobalAttn` + multiple distant-atom supervision handles the "two atoms moving simultaneously" case that pure-local perturbation alone cannot.

**Notation note.** The symbol `ε` is used in two different senses throughout this document and in the algorithms below:
- **`ε` (no subscript)** — scalar flow-time cutoff for objective (3), dimensionless, ∈ (0, 1).
- **`ε_rs_pert`, `ε_ts_pert` (subscripted)** — spatial perturbation vectors in Å, `(N, 3)` tensors. Always subscripted when referring to perturbation.

**Mode selection at inference** (the central research bet, H2) still holds: different random Gaussian draws at `t = 0` produce different input positions, which the trained velocity field routes to different saddles via local environments and global attention.

### Training objectives

Three objectives, selected per-sample by weighted draw. Weights `(w_1, w_2, w_3)` default to `(0, 1, 2)` — obj (1) disabled initially; obj (3)'s clean-ray supervision gets double weight because its target signal is the cleanest.

**Objective (3) — Clean reactant → TS ray [anchor, high weight].** Pure straight-line flow from `r_reactant` to `r_saddle_unwrapped`, with target velocity `Δ`. Time sampled from `Uniform(ε, 1)` — the `t > ε` cutoff is essential because at `t = 0` all rays to different saddles of the same reactant share the same input `r_reactant`, causing mode-averaging of targets. For `t > ε`, the intermediate position `(1-t) · r_R + t · r_S_un` differs between rays and the targets are unambiguous. This provides the highest-quality, noise-free supervision on the physically meaningful paths.

**Objective (2) — (reactant + Gaussian) → TS [multimodality breaker].** Gaussian perturbation `ε_rs_pert ~ N(0, σ_rs_pert² · I_{3M})` on mobile atoms. Straight-line flow from `r_reactant + ε_rs_pert` to `r_saddle_unwrapped`, target velocity `Δ − ε_rs_pert`, `t ~ Uniform(0, 1)`. Provides training coverage at the perturbed start points where inference begins, and the implicit-latent-via-perturbation mechanism that routes different initial conditions to different saddles.

**Objective (1) — TS + Gaussian → TS [optional regularizer].** Gaussian noise `ε_ts_pert ~ N(0, σ_ts_pert² · I_{3M})` on mobile atoms. Straight-line flow from `r_saddle_unwrapped + ε_ts_pert` back to `r_saddle_unwrapped`. Teaches the network to denoise toward a saddle from nearby points — a Dimer-like contraction behavior around each saddle. Default weight 0; enable for ablation.

**On H3 (objective-overlap conflicts)** — substantially mitigated under this scheme. Obj (3) samples points on the clean ray `x_t^(3) = (1-t)r_R + t·r_S_un`; obj (2) samples points `x_t^(2) = (1-t)(r_R + ε_rs_pert) + t·r_S_un` with Gaussian `ε_rs_pert`. Their training loci coincide only when `ε_rs_pert = 0`, a measure-zero event. For interior values of `ε_rs_pert`, obj (2) supervises different `(x_t, t)` than obj (3), so there is no direct gradient conflict. See the updated H3 in the risks section.

### Training algorithm

```
HYPERPARAMETERS
    σ_rs_pert       Gaussian std for obj (2) reactant-state perturbation, in Å  (primary HP)
    σ_ts_pert       Gaussian std for obj (1) TS-state perturbation,       in Å  (typically ≈ σ_rs_pert)
    ε               flow-time cutoff for obj (3) clean-ray sampling, dimensionless ∈ (0, 1)
    w_1, w_2, w_3   loss weights (defaults: 0, 1, 2)

# Notation reminder: ε (unsubscripted) = flow-time cutoff;
# ε_rs_pert, ε_ts_pert (subscripted) = spatial perturbation vectors.
# Always distinguished by subscript.

PER TRAINING SAMPLE
    Load triplet from LMDB:
        r_R, r_S_un, r_P_un, Z, cell, fixed, task_name, charge, spin
    # r_S_un = r_R + MIC(r_saddle_raw − r_R), precomputed at LMDB conversion time
    Δ       = r_S_un − r_R                    # precomputed MIC displacement
    mobile  = (fixed == False); M = mobile.sum()

    Draw k ~ Categorical(w_1, w_2, w_3)

    # Every objective is a straight-line OT flow between two endpoints defined in
    # UNWRAPPED space. v_target = x_1 − x_0 universally for straight-line OT.
    # Wrap is applied only at the final x_t step, immediately before the UMA forward.

    if k == 3:                                # clean reactant → TS ray
        x_0 = r_R                             # unwrapped endpoint at flow-time 0
        x_1 = r_S_un                          # unwrapped endpoint at flow-time 1
        t   ~ Uniform(ε, 1)

    if k == 2:                                # (reactant + Gaussian) → TS
        ε_rs_pert_raw     ~ N(0, σ_rs_pert² · I_{3M})
        ε_rs_pert         = zeros(N, 3)
        ε_rs_pert[mobile] = ε_rs_pert_raw.reshape(M, 3)
        x_0 = r_R + ε_rs_pert                 # unwrapped
        x_1 = r_S_un                          # unwrapped
        t   ~ Uniform(0, 1)

    if k == 1:                                # TS + Gaussian → TS (optional)
        ε_ts_pert_raw     ~ N(0, σ_ts_pert² · I_{3M})
        ε_ts_pert         = zeros(N, 3)
        ε_ts_pert[mobile] = ε_ts_pert_raw.reshape(M, 3)
        x_0 = r_S_un + ε_ts_pert              # unwrapped
        x_1 = r_S_un                          # unwrapped
        t   ~ Uniform(0, 1)

    v_target = x_1 − x_0                      # constant in t for straight-line OT
    x_t      = wrap((1 − t) · x_0 + t · x_1)  # interpolate in unwrapped space, then wrap for UMA

    # Shared forward pass (objective-agnostic)
    atoms        = Atoms(positions=x_t, numbers=Z, cell=cell, pbc=True)
    data         = AtomicData.from_ase(atoms, task_name=task_name)
    data.charge  = charge; data.spin = spin                                 # required inputs to UMA's csd_embedding
    local_feat   = UMA_backbone(data)["node_embedding"]                     # (N, (lmax+1)², sphere_channels) = (N, 9, 128) for UMA-S-1.2
    global_feat  = GlobalAttn(local_feat)                                   # globally-aware equivariant per-atom irreps
    v_pred       = VelocityHead(global_feat, sinusoidal_time_embed(t))      # (N, 3)
    v_pred[fixed]   = 0                                                     # hard mask frozen atoms
    if fixed.sum() == 0:                                                    # ONLY if the system is fully mobile
        v_pred[mobile] -= v_pred[mobile].mean(dim=0, keepdim=True)          # CoM projection — otherwise frozen atoms pin the cell
    # See §"Output projections" for why this is conditional and not unconditional.

    loss_sample = mean_squared_error(v_pred, v_target)

# Per-batch: average loss, backprop, optimizer step, EMA update.
```

### Inference / sampling algorithm

```
GIVEN: r_R, Z, cell, fixed, task_name, charge, spin, n_perturbations, K (integration steps)

For i in 1..n_perturbations:
    ε_rs_pert_raw_i       ~ N(0, σ_rs_pert² · I_{3M})     # same distribution as obj (2) training
    ε_rs_pert[mobile]      = ε_rs_pert_raw_i.reshape(M, 3)
    ε_rs_pert[frozen]      = 0
For each ε_rs_pert (batched in parallel):
    x = wrap(r_R + ε_rs_pert)                             # wrap at start (r_R canonical, ε small)
    for step in range(K):
        t            = step / K
        atoms        = Atoms(positions=x, numbers=Z, cell=cell, pbc=True)
        data         = AtomicData.from_ase(atoms, task_name=task_name)
        data.charge  = charge; data.spin = spin
        local_feat   = UMA_backbone(data)["node_embedding"]
        global_feat  = GlobalAttn(local_feat)
        v            = VelocityHead(global_feat, sinusoidal_time_embed(t))
        v[fixed]     = 0
        if fixed.sum() == 0:                              # CoM projection only if system is fully mobile
            v[mobile] -= v[mobile].mean(dim=0, keepdim=True)
        x            = wrap(x + (1/K) · v)                # wrap after each Euler step
    candidate_saddles.append(x)

Cluster candidates by pairwise RMSD under PBC (agglomerative, cutoff ≈ 0.1 Å).
Take cluster centroids as final candidate TSs.
Optional: Dimer-refine each centroid using fairchem's ASE Calculator (UMA potential).
```

**Default integrator:** forward Euler with `K = 50`. User-configurable. Swap in `torchdiffeq` RK45 later if needed — one-line change in the sampler.

## Training data

- **~30 million (reactant, saddle, product) triplets**, from Henkelman group searches (NEB + Dimer) over Materials Project, ICSD, Alexandria, OC20, and OC22.
- Same atom count, same constant cell, aligned atom indices across each triplet. Optional `FixAtoms` constraints preserved.
- **Source format:** ASE `.traj` files with per-frame metadata in `atoms.info` (includes the `task_name` per sample for UMA's MoE routing).
- Currently all 3D PBC; slabs and molecules are a future-compatible add-on (UMA supports all three via `task_name`).
- **Split:** by *reactant*, not by triplet — all saddles of a given reactant go to one split. Tests cross-reactant generalization (the actual research claim), not cross-saddle memorization.

### Data format — dual backend: raw `.traj` or ASE-DB

**Source** is always ASE `.traj` files containing flat `[R₁, S₁, P₁, R₂, S₂, P₂, …]` sequences (one frame each for reactant/saddle/product; `atoms.info['side'] ∈ {-1, 0, 1}` labels them). We deliberately support two downstream read paths, chosen by scale:

- **Backend A — read `.traj` directly** (`saddlegen.data.TrajTripletDataset`). No conversion step. Used for the Li/C test and any moderate-size problem where iterating fast matters more than throughput. `ase.io.Trajectory` has O(1) random-access via its frame index.
- **Backend B — one-time conversion to ASE-DB** (`saddlegen.data.convert_to_db` → `AseDbSaddleDataset`). Used for the 30M run and anywhere we want queryable metadata or fast random access across many files. Output format follows the file extension:
  - `*.db` / `*.sqlite` — SQLite (human-inspectable; OK up to a few million rows).
  - `*.aselmdb` — LMDB-backed ASE-DB via the `ase-db-backends` package (fairchem's production format; the right choice at 30M scale).

Both backends expose the same per-sample dict so `matching.py`, `sampler.py`, and the trainer are backend-agnostic:

```
# One sample = one (start, saddle) pair. Each triplet produces TWO samples.
{
    start_pos     : (N, 3) float32  (Å; either R or P positions, raw — not unwrapped)
    saddle_un_pos : (N, 3) float32  (Å; saddle MIC-unwrapped relative to start_pos)
    Z             : (N,)   long     (atomic numbers)
    cell          : (3, 3) float32  (lattice vectors; constant across the triplet)
    fixed         : (N,)   bool     (FixAtoms mask; constant across the triplet)
    task_name     : str             ({omat, oc20, oc22, oc25, omol, odac, omc} — full UMA-S-1.2 set)
    charge        : int             (UMA csd_embedding input; default 0; pulled from atoms.info if present)
    spin          : int             (UMA csd_embedding input; default 0; pulled from atoms.info if present — UMA's omat head does not use this)
    delta_norm    : float32         (‖saddle_un_pos − start_pos‖; used for dataset-level scaling)
    role          : str             ("R2S" or "P2S"; diagnostic only, flow code treats both identically)
    triplet_id    : int             (preserves provenance back to the source triplet)
    metadata      : dict            (sanitized atoms.info dicts from all three frames of the triplet)
}
```

**R→S and P→S doubling.** By microscopic reversibility, both `(start=R, saddle=S)` and `(start=P, saddle=S)` are valid training pairs for the same saddle. `triplet_to_pair_records` emits both; MIC-unwrap is recomputed for each because the periodic-image choice depends on the anchor. Effective dataset size is `2 × num_triplets`.

**Dataset-level `⟨‖Δ‖⟩`** is computed at conversion time (Backend B, written to `<db_path>.stats.json`) or at first-access time (Backend A, cached via `stats_cache`). This scalar sets the default `σ_rs_pert` per the rule-of-thumb in §Perturbation geometry.

**No precomputed neighbor lists.** Graphs are rebuilt on the fly by `AtomicData.from_ase` at every forward pass — required anyway because `r(t)` changes with flow time and the graph can genuinely change (bond-breaking reactions cross UMA's cutoff radius). Neighbor-list cost is ~1–10 ms CPU per system and is dominated by the UMA forward pass.

### First test case: Li on defective C-sheet

- Frozen C-sheet with two C-vacancy defects (`FixAtoms` on all carbon). `FixAtoms` indices 0–125; **Li is atom index 126**, the only mobile atom.
- Cell diag `[17.09, 19.56, -15.00]` (negative `z` is the vacuum direction — left-handed cell, but MIC math / fairchem neighbor-list handle it fine).
- `atoms.info` carries `task_name='omat'`, `charge=0`, `spin=0`, plus Henkelman-pipeline keys (`band_converged`, `frozen_set`, `image_idx`, etc.). **No `side` key** — `validate_triplet` falls back to positional ordering `[R, S, P, R, S, P, …]`.
- Sizes: `train_set.traj` — 12 triplets; `test_set.traj` — 171 triplets.

**Site-level structure (R↔P-symmetric count).** Every triplet contributes **two** local-minimum Li sites (R and P; both are "reactants" by microscopic reversibility), so counting saddles per site must include both endpoints. The numbers that result match the hexagonal-lattice physics:

| set | triplets | unique sites (R∪P) | mean saddles / site | max | histogram |
|---|---|---|---|---|---|
| train | 12 | 16 | 1.5 | 3 | most sites have 1 saddle — sparse |
| test | 171 | 63 | **5.43** | 7 | 41 sites with exactly 6 (hex bulk), 21 sites with 3–5 (defect-adjacent), 1 with 7 |

The ~6/site median for test matches the hex lattice: each Li adsorption site has 6 neighbours; sites next to one of the two C-vacancies lose 1–3 hop directions (explaining the 3–5 bucket).

**Train/test overlap (counted at the R∪P site level):**
- All 16 train sites also appear in test (train ⊂ test at site level).
- 47 test sites are novel to train. These 47 are the actual cross-reactant H1 test.
- **Saddle-disjoint:** 0 saddle geometries appear in both sets (min cross-set Li(S)–Li(S) distance 0.94 Å). So even at the 16 shared sites, every training saddle is distinct from every test saddle.

Pairwise Li-Li distance between reactant sites is **perfectly bimodal**: either `<0.01 Å` (exact duplicate — the same site reused across many triplets to different products) or `>1.5 Å` (genuinely different adsorption sites). Any clustering threshold in that gap gives the same 63 sites.

**Evaluation protocol (leave-one-reactant-out style, per-site):** for each unique site in test, run the sampler once (`n_perturbations=32` candidates) from that site's representative endpoint, cluster candidates by PBC-RMSD, and Hungarian-match centroids to the site's known saddles (typically 3–7). Report ALL / NOVEL / SHARED aggregates separately — NOVEL (47 sites) is the real H1 number. Implemented in `examples/LiC/evaluate.py` via `saddlegen.utils.group_triplets_by_site(endpoints="RP")` and `match_sites(…)`.

Fully 2D-visualizable, interpretable by eye, small enough to iterate quickly.

## Data pipeline

- `torch.utils.data.Dataset` → either `TrajTripletDataset` (backend A) or `AseDbSaddleDataset` (backend B). Both yield the dict above as tensors; no ASE objects held past `__getitem__`.
- Graph built fresh per forward pass via `AtomicData.from_ase(Atoms(positions=x_t, ...), task_name=...)` with `charge`/`spin` set on the result. Correct handling of neighbor-list changes along the flow trajectory.
- Batching: fairchem's `data_list_collater` (or `atomicdata_list_to_batch` directly) concatenates atoms across samples with PyG-style `batch`. Heterogeneous N and M handled natively.
- **Batch homogeneity for MoE routing:** UMA's MOLE expert merge re-runs whenever `data.dataset` (task_name) changes. For training efficiency, use a per-task `torch.utils.data.Sampler` or sort batches by task so each batch is homogeneous. Mixed-task batches work but waste compute on the coefficient step.

## Training infrastructure

**Plain PyTorch + HuggingFace `accelerate`**, matching the GenFlows pattern. No Hydra, no Lightning, no TorchTNT.

- **Optimizer:** AdamW. `lr = 1e-3` (head-only, frozen backbone) or `lr = 1e-5` (end-to-end fine-tuning with UMA unfrozen). Cosine LR schedule, gradient clip `max_norm = 1.0`.
- **EMA:** decay `0.9999` on all trainable parameters.
- **Precision:** bf16 forward + fp32 optimizer state, to match UMA's training precision. Exact match to fairchem's config verified in first coding session.
- **Multi-node:** `accelerate launch --multi_gpu --num_machines=N ...` under SLURM. `accelerate` handles DDP, FSDP (if UMA + optimizer state doesn't fit per-GPU), gradient accumulation.
- **Checkpointing:** `accelerator.save_state()` / `load_state()`; FSDP-sharded under the hood for multi-node.
- **Gradient checkpointing:** `torch.utils.checkpoint` on the UMA backbone layers when fine-tuning; transparent to `accelerate`.

### Why not fairchem's full training stack (TorchTNT + Hydra + Ray)

Considered carefully. Fairchem's infra is designed for multi-task MLIP training (energy + force + stress, dataset-specific heads, Ray job orchestration). SaddleGen has one regression task (velocity), one head, one loss. Adopting `MLIPTrainEvalUnit` + `TrainEvalRunner` + Hydra would add ~1000 lines of framework machinery to hide the training loop. `accelerate` covers 100% of the actual technical requirements — including FSDP and multi-node at UMA-S-1.2 scale (6.6M active params, 290M total with all MoE experts merged) — while keeping the loop ~150 readable lines. The 290M-total figure drives FSDP planning when the full expert set must reside in memory; the 6.6M-active figure governs per-batch compute.

The migration path to fairchem's full stack remains open and is localized to `saddlegen/utils/training.py`. We are not painting ourselves into a corner.

## Evaluation

**Site-based (not R-only) grouping.** Test triplets are grouped by their **R ∪ P** Li adsorption sites — both endpoints of every triplet are legitimate "reactants" by microscopic reversibility, and counting saddles per site must include both for the numbers to match the physics (see §"First test case" for concrete counts). Implemented as `saddlegen.utils.group_triplets_by_site(endpoints="RP")`; the legacy R-only variant is available via `endpoints="R"` but should not be the default for reaction-discovery eval.

**Per-site Hungarian matching, with outlier-safe thresholding.** For each unique site, cluster candidates (medoid centroids) and run `scipy.optimize.linear_sum_assignment` against that site's known saddles. LSA minimises total cost and does not respect a per-pair threshold by itself — a single far-off centroid can cascade into a globally-optimal but individually-pathological assignment. Fix: mask above-threshold cost-matrix entries to `1e9` before LSA so it prefers sub-threshold pairings and only pairs across the threshold when unavoidable (then filtered post-LSA). Implemented in `saddlegen.utils.hungarian_match`.

**Primary metric:** RMSD under PBC between each predicted cluster centroid and its nearest known reference saddle (Hungarian assignment per site).

**Secondary metrics:**
- **Recall:** fraction of known saddles of each site recovered (RMSD < τ, e.g., τ = 0.1 Å for the Li/C test case).
- **Precision:** fraction of predicted cluster centroids that match a known saddle of that site.
- **Bonus discoveries:** for each centroid, also report the nearest saddle in the **global (train + test) pool** — a centroid matching a train saddle is a valid TS rediscovery, not a miss. Train/test are saddle-disjoint by construction in the Li/C case, so bonus hits from the train side are genuine.
- **ALL / NOVEL / SHARED split:** the test-site set partitions into "novel" (not in train at R∪P site level) and "shared" (in train). NOVEL is the cross-reactant H1 number; SHARED is held-out-saddle recovery at known reactants. `examples/LiC/evaluate.py` reports all three.
- **(Later)** Dimer-convergence rate: fraction of predicted centroids that Dimer refines to a valid first-order saddle point in ≤ 50 steps.

## Research hypotheses and risks

For the methods-section writeup. Each hypothesis has an explicit falsification test.

- **H1 — Local-environment transfer.** The GNN over UMA features generalizes TS knowledge across reactants with similar local chemistry. *Falsification:* leave-one-reactant-out on the Li/C test case should recover held-out TSs at RMSD below threshold. *Risk:* similar local environments in training data point to very different TSs.

- **H2 — Implicit latent via Gaussian perturbation.** Random Gaussian `ε_rs_pert` on mobile atoms selects different saddle modes without explicit conditioning. The hypothesis is that UMA's pretrained features encode Hessian-like soft-mode structure (they were trained on DFT forces), `GlobalAttn` allows distant atoms to coordinate which site reacts, and `VelocityHead` amplifies whichever mode has the largest projection in `ε_rs_pert` — effectively a learned Dimer. *Falsification:* saddle diversity at inference should scale with `σ_rs_pert` and `n_perturbations`. *Risk:* UMA features do not encode soft-mode structure, perturbations mode-average, the model returns one "mean" saddle per reactant.

- **H3 — Three-objective coexistence.** Objectives (1), (2), and (3) supervise disjoint regions of `(x_t, t)` space with probability 1 (their loci coincide only on measure-zero sets), so joint training does not produce gradient conflicts. *Falsification:* per-objective training losses should decrease independently; training is stable with weights `(0, 1, 2)`. *Risk:* in rare high-overlap cases (very small `σ_rs_pert` making obj (2) land near the clean ray), minor blur in velocities near the ray — fix by increasing `σ_rs_pert` or decreasing obj (2)'s weight.

- **H4 — Frozen vs. fine-tuned UMA.** Frozen UMA gives an acceptable baseline; fine-tuning improves further. *Risk:* fine-tuning destabilizes UMA's pretrained features; may need very low LR, warmup, or LoRA-style adapters.

- **H5 — Gaussian perturbation geometry.** Isotropic Gaussian on mobile atoms produces the sparse per-atom magnitude distribution that matches real TS displacements (few atoms move a lot, most barely). *Falsification:* on the Li/C test with two distant Li, Gaussian samples should regularly produce one-Li-dominant perturbations. *Risk:* for very large `M`, even Gaussian may under-represent the sparsest real TS directions; mitigation via sparse-Gaussian or Hessian-aware inference sampling reserved as future work.

- **H7 — Global attention resolves distant-site ambiguity.** `GlobalAttn` allows atoms separated by more than UMA's cutoff to exchange information, so the network does not predict simultaneous reactions at spatially-independent sites. *Falsification:* on the Li/C test with two distant Li adatoms, the model should predict single-Li-hop saddles (not simultaneous-hop artifacts). *Risk:* `O(N²)` attention cost grows for large unit cells; distance-sparse attention (attend only within 2× UMA cutoff) reserved as an optimization.

## Planned layout (subject to refinement during implementation)

```
saddlegen/
├── data/
│   ├── core.py                    # shared: MIC unwrap, triplet validation, R/P doubling, stats
│   ├── traj_dataset.py            # backend A: read .traj directly (TrajTripletDataset)
│   ├── convert_to_db.py           # backend B: .traj → ASE-DB CLI (supports .db / .aselmdb)
│   ├── db_dataset.py              # backend B: read ASE-DB (AseDbSaddleDataset)
│   └── transforms.py              # cell wrapping, Gaussian perturbation sampling (used at train/inference)
├── models/
│   ├── global_attn.py             # GlobalAttn (invariant-weighted equivariant self-attention)
│   └── velocity_head.py           # VelocityHead (time-embedded, equivariant)
├── flow/
│   ├── matching.py                # training objectives and loss computation
│   └── sampler.py                 # inference integrator (Euler default, RK45 optional)
├── utils/
│   ├── training.py                # accelerate-based train loop with EMA
│   ├── eval.py                    # RMSD-PBC, clustering, Hungarian, reactant-grouping
│   ├── backbone.py                # load_uma_backbone (fairchem loader + freeze)
│   ├── checkpointing.py           # load_ema_weights (EMA snapshot → inference modules)
│   └── refine.py                  # optional Dimer refine via fairchem Calculator
└── __init__.py

examples/
└── LiC/
    ├── train_set.traj             # [R1,S1,P1, R2,S2,P2, ...] no `side` keys — fall back to positional ordering
    ├── test_set.traj              # same format
    ├── train.py                   # argparse + wire saddlegen.* helpers; no generic code
    └── evaluate.py                # argparse + Li/C eval protocol (group by reactant, sample, match)
```

Package import name: `saddlegen` (lowercase, per PEP 8); the project is referred to as **SaddleGen** in prose.

## Reference codes (`other_codes/`, git-ignored)

Kept locally during design; **not tracked by git**. Will be removed before public release, along with this section.

| Directory | What it is | Role for SaddleGen |
|---|---|---|
| `fairchem/` (+ `fairchem-papers/`) | Meta FAIR's chemistry framework — UMA MLIP, eSCN-MD backbones, OMat25 data | **Architectural anchor.** Hard dependency. Provides UMA backbone, `AtomicData`, PBC-aware graph builder, ASE Calculator integration. |
| `mattergen/` (+ paper) | Microsoft's diffusion model for inorganic crystal generation with property conditioning | Methodology reference for conditional generation on periodic systems. |
| `flowmm/` (+ paper) | Riemannian flow matching on the crystal manifold (atom types + lattice) | Methodology reference; SaddleGen uses standard Euclidean flow matching because cell and atom identities are fixed (flat `ℝ^(3N)` displacement space). |
| `all-atom-diffusion-transformer/` (+ paper) | FAIR's ADiT — latent diffusion (Equiformer autoencoder + DiT) unified for molecules + materials | Methodology reference for decoupling representation learning from generation. **Not planned for deep read** — only relevant if we later explore latent-space flow matching as a follow-up to the Cartesian version. |

## Hardware target

NERSC Perlmutter, 4 × A100 per node. **Multi-node from day one** for the 30M-triplet run. The Li/C test case is a single-GPU (or single-node) job.

## License

TBD — likely MIT (matching GenFlows). fairchem is MIT.

## Next coding session — immediate tasks

The design is converged. The next working session should execute the following in order. CLAUDE.md (this file) is the authoritative spec; do not re-relitigate design decisions unless the user explicitly asks.

### Session handoff — status at last checkpoint

**Phase 1 and Phase 2 (steps 1–14) are complete.** All source code lives in `saddlegen/` and is syntax-clean; behavioural smoke tests pass on real Li/C data with a mock UMA backbone (gradients flow, sampler output respects frozen atoms, eval loop computes sensible recall/precision). CLAUDE.md below marks each step with ~~strikethrough~~ + "Done." notes as it was completed.

**What did not run at last checkpoint (environment-limited, not code-limited):**
- `accelerate` was not installed in the working Python env, so `train()`'s integration with `accelerator.prepare` / DDP / save-state was not exercised end-to-end. The non-`accelerate` pieces (`EMA`, LR schedule, `identity_collate`, config defaults) are verified; `train()` itself is standard `accelerate` API (~60 LOC of loop plumbing).
- UMA-S-1.2 was not loaded (needs `fairchem` + HuggingFace cache). `load_uma_backbone` symbol path (`predict_unit.model.module.backbone`) was verified against vendored `fairchem 2.19` source.
- bf16 autocast on GPU was not exercised (all tests ran fp32/CPU).

**To resume on a new machine:** (1) `pip install -e .` in the project root — brings in `fairchem-core`, `accelerate`, `ase`, `torch~=2.8`, etc. per `pyproject.toml`. (2) Ensure the UMA-S-1.2 checkpoint is reachable (HF token or pre-cached). (3) Run `python examples/LiC/train.py` — it's argparse-only, defaults are reasonable for the Li/C size. (4) Once trained, `python examples/LiC/evaluate.py` reports ALL / NOVEL / SHARED recall. Start with `--limit-reactants 3` and fewer epochs for a first smoke run.

**Latent-bug log — pitfalls already caught once, re-check if behaviour looks wrong:**
- **Silent CoM-projection bug** (fixed). Any system with 1 mobile atom and unconditional `v[mobile] -= v[mobile].mean()` gets its only mobile-atom velocity zeroed — no gradient, no learning, loss constant. Current code conditionals on `fixed.sum() == 0` per system; see §"Output projections".
- **AdaLN-zero step-0 gradient unlock** (not a bug, but surprising). `VelocityHead.time_mlp`'s last layer is zero-initialised; grad on the *first* MLP layer is exactly 0 at step 0 (no path through a zero-weighted matrix). Last layer updates on step 0, which unlocks earlier layers from step 1. If you see `|∇time_mlp[0]|` = 0 only on step 0 that is expected; if it stays zero past step 1 something else is broken.
- **LiC atom index 126 is the Li**, atoms 0–125 are the frozen C sheet. It's easy to assume atom 0 is the mobile one and get nonsense diagnostics.
- **Site grouping must be R∪P**, not R-only. R-only undercounts saddles per site by ~2× and looks like the data is worse than it is.
- **UMA layout `(N, (lmax+1)², C)` vs e3nn `(N, Irreps.dim)`** are not interchangeable; all SaddleGen modules use UMA's `SO3_Linear` to avoid layout-conversion plumbing. If you add e3nn ops, you need conversion helpers.
- **Negative z-component in the Li/C cell** (`-15 Å`) is intentional (vacuum direction, left-handed cell). MIC math and fairchem neighbor-list handle it; do not "fix" the sign.

### Phase 1 — fairchem deep read (no code yet, just pinning unknowns)

1. **Read `other_codes/fairchem/src/fairchem/core/models/` to locate `Linear_Force_Head`.** Record its exact architecture: MLP depth, width, activation, how it projects `l=1` irreps to `(N, 3)`. Our `VelocityHead` mirrors this, plus a sinusoidal time-embedding MLP whose output is added as a scalar bias on the invariant (`l=0`) channels only. Confirm equivariance preservation.
2. **Read `fairchem.core.pretrained_mlip.get_predict_unit`.** Understand exactly how UMA checkpoints are loaded, what object they return, and how to split backbone vs. head so we can swap ours in.
3. ~~**Read the `uma-s-1p2` config / model card.**~~ **Done in the prior session.** Pinned facts: bf16 training; cutoff 6.0 Å, max_neighbors=30 (non-strict); `task_name` set is `{omat, oc20, oc22, oc25, omol, odac, omc}` (oc22 confirmed present); no hard model-side atom-count cap (`max_atoms=700` is a sampler constraint in fairchem's training config only); 6.6M active / 290M total params; backbone is `K4L2` → output `(N, 9, 128)`; force normalization (`rmsd=1.423`) lives in UMA's heads, which we discard, so it does not apply to us — our own velocity scaling uses `delta_norm` from the LMDB stats.
4. **Read `AtomicData.from_ase` and `core/graph/` neighbor builder.** Confirm PBC handling and the exact field list required for a forward pass.
5. **Read `fairchem-core/pyproject.toml` to lock our dependency versions.** Produce `SaddleGen/pyproject.toml` with `fairchem-core` pinned to a specific version plus `ase`, `accelerate`, `lmdb`, `torch` matching fairchem's pins.

### Phase 2 — Li/C implementation (first real code)

6. ~~Implement `saddlegen/data/convert_traj_to_lmdb.py`~~ **Done.** Implemented as the dual-backend data layer described in §"Data format":
    - `saddlegen/data/core.py` — MIC unwrap, triplet validation, R→S / P→S doubling, sanitized metadata, `delta_norm` computation.
    - `saddlegen/data/traj_dataset.py` — `TrajTripletDataset` (backend A, reads `.traj` directly; used for Li/C).
    - `saddlegen/data/convert_to_db.py` — CLI that converts `.traj` triplets into ASE-DB (`.db` or `.aselmdb`); writes companion `*.stats.json` with `⟨‖Δ‖⟩`.
    - `saddlegen/data/db_dataset.py` — `AseDbSaddleDataset` (backend B, reads ASE-DB; used for the 30M run).
    - Both datasets yield the same sample dict. R→S and P→S are emitted as separate samples, so `len(dataset) == 2 × num_triplets`.
7. ~~Implement `saddlegen/data/transforms.py`~~ **Done.** Two pure-torch utilities used inside the flow loop:
    - `wrap_positions(positions, cell)` — fractional round-trip wrapping, autograd-safe and GPU-safe; called before every UMA forward and after every Euler step.
    - `gaussian_perturbation(mobile_mask, sigma, generator=None)` — isotropic `N(0, σ² I_{3M})` draw on mobile atoms, zero on frozen; used by objectives (1), (2), and inference.
    - MIC unwrap is not in this module — it lives in `core.py` and is applied once at conversion / `__getitem__` time, not inside the flow loop.
8. ~~Implement `saddlegen/data/lmdb_dataset.py`~~ **Superseded by step 6.**
9a. ~~Implement `saddlegen/models/global_attn.py`~~ **Done.** Invariant-weighted equivariant self-attention, ~95 lines. `Q, K` from `l=0` via `nn.Linear`; `V` and `W_out` via UMA's `SO3_Linear` (see note in §"Global attention" on the e3nn → `SO3_Linear` switch). Multi-head split across the channel dim. Batched-graph attention mask via `batch_idx`. `GlobalAttn(num_layers)` stacks residual `GlobalAttnLayer`s.
9b. ~~Implement `saddlegen/models/velocity_head.py`~~ **Done.** `VelocityHead(sphere_channels, input_lmax=2, depth=1)`, ~140 lines. `depth=1` at init is numerically equivalent to `Linear_Force_Head` thanks to zero-initialised time FiLM. See §"Head" for the time-FiLM equivariance proof and the `UMAGate` hand-rolled nonlinearity. Verified in-tree: equivariance error < 5e-7 at init and after 20 Adam steps; time signal reaches the output post-training.
10. ~~**Before writing `matching.py`:** read FlowMM ... Then implement `saddlegen/flow/matching.py`~~ **Done.** Read FlowMM's `model_pl.py` for training-loss tricks; adopted: uniform `t` sampling (no logit-normal), plain MSE (no time reweighting), per-atom averaging, value-based grad clip (to be applied in `training.py`). Skipped: their Riemannian manifold code, their dataset-level affine I/O normalisation (we have `delta_norm_mean` available if needed later).
    - `saddlegen/flow/matching.py` (~180 lines) implements the **three** straight-line OT objectives (CLAUDE.md diverged from the original "two-objective" plan). Per-sample Monte-Carlo objective draw via `k ~ Categorical(w_1, w_2, w_3)` — unbiased estimator of the weight-mixed loss with 1/3 the compute, and with `w_1=0` by default the sampler skips obj (1) entirely.
    - Exports: `FlowMatchingConfig`, `FlowMatchingLoss`, `sample_endpoints`, `build_atomic_data`, `apply_output_projections`. `FlowMatchingLoss` wraps `backbone + GlobalAttn + VelocityHead` and returns `{loss, per_obj_count, per_obj_loss, n_mobile}` for logging.
    - Output projections applied inside the loss module: (i) `v[fixed] = 0`; (ii) per-system CoM subtraction on mobile atoms via `index_add_` (vectorised, no Python loop over systems).
    - AtomicData construction per sample goes via ASE (`from_ase`), then fairchem's `data_list_collater` batches across samples. `otf_graph=True` rebuilds the neighbour list every forward, which is correct and necessary because `x_t` changes with flow time.
    - **AdaLN-zero gradient quirk:** `VelocityHead.time_mlp`'s final layer is zero-initialised, so at step 0 the gradient is zero on the first MLP layer (no path through a zero-weighted matrix). The last layer updates on step 0, which unlocks earlier layers from step 1 onward. Verified: loss drops 22.3 → 15.7 over 4 Adam steps on a mock-backbone smoke test.
11. ~~Implement `saddlegen/flow/sampler.py`~~ **Done.** `sample_saddles(reactant, backbone, global_attn, velocity_head, sigma_rs_pert, n_perturbations=32, K=50)`, ~130 lines, `@torch.no_grad()`. Batches all `n_perturbations` trajectories into one `AtomicData` per Euler step — cost is O(K × one-batched-UMA-forward), independent of perturbation count. Supports `return_trajectory=True` (yields `(K+1, n_perturbations, N, 3)`) and an explicit `torch.Generator` for reproducibility. Reuses `apply_output_projections` from `matching.py` so the inference-time output projections are byte-identical to training.
    - Verified in-tree: frozen atoms exactly immobile throughout integration (`v[fixed]=0`); trajectories diverge under different perturbations (no mode collapse); same generator seed → bit-identical outputs.
12. ~~Implement `saddlegen/utils/training.py`~~ **Done.** ~240 lines. Exports `TrainingConfig`, `EMA`, `identity_collate`, `train`.
    - `accelerate` is **lazily imported inside `train()`** so `EMA` / `TrainingConfig` / `identity_collate` are usable in environments where `accelerate` isn't installed (unit tests, minimal inference envs).
    - `TrainingConfig` defaults match CLAUDE.md: `ema_decay=0.9999`, `grad_clip_norm=1.0`, `mixed_precision="bf16"`, AdamW with `weight_decay=1e-5`. LR schedule is linear-warmup-then-cosine with a `min_lr_ratio` floor (default 0.01).
    - `EMA` is hand-rolled (not `torch.optim.swa_utils.AveragedModel`) for clean integration with accelerate's parameter preparation. Supports `update()`, `swap_in()` / `swap_out()` around eval, and `state_dict()` / `load_state_dict()` for checkpointing.
    - DataLoader uses `collate_fn=identity_collate` — a pass-through returning `list[dict]`, because samples have heterogeneous N and default collation would try to stack and fail.
    - Checkpointing: `accelerator.save_state(dir)` per `save_every_epochs` epochs, plus `ema.pt` and `meta.json` (epoch, global_step) written alongside. `resume_from=<ckpt_dir>` restores all three.
    - Validation: optional `val_dataset` + `val_every_epochs`; evaluation is done under EMA weights via `ema.swap_in()` / `swap_out()`.
    - Multi-node: works via `accelerate launch --multi_gpu --num_machines=N examples/li_on_carbon/train.py …`.
    - Verified in-tree (without accelerate): EMA update math, swap_in/out round-trip, state_dict round-trip, LR schedule (warmup → cosine → floor), identity_collate pass-through, config defaults.
13. ~~**Before writing `eval.py`:** read MatterGen ... Then implement `saddlegen/utils/eval.py`~~ **Done.** Read MatterGen's evaluation code (`evaluation/utils/utils.py`, `evaluation/metrics/structure.py`, `evaluation/evaluate.py`). **Divergences from MatterGen's protocol (deliberate)**:
    - Their RMSD uses pymatgen `StructureMatcher.get_rms_dist()` with composition-aware permutation search — needed because their generated crystals don't have aligned atom indices. **We don't need that** because our candidates and references share fixed atom ordering (same triplet, same indices). Fixed-correspondence RMSD under PBC is simpler, faster, and exact for our case.
    - They don't cluster — one-to-many greedy matching. **We do cluster** because the sampler yields many near-duplicate perturbation-driven candidates and we need distinct cluster centroids before scoring.
    - They use greedy one-to-many matching. **We use Hungarian** (scipy `linear_sum_assignment`) because CLAUDE.md asks for one-to-one optimal assignment of centroids ↔ references.
    - Their `structure_validity` (min interatomic distance < 0.5 Å) is the one piece genuinely useful for us; ported directly.
    - Skipped: diffusion schedulers, GemNet, property conditioning, space-group checks, charge balance — none apply to saddles.
    - `saddlegen/utils/eval.py` (~220 lines). Exports: `rmsd_pbc`, `pairwise_rmsd_pbc`, `cluster_by_rmsd`, `hungarian_match`, `validity_check`, `evaluate_predictions`, `aggregate_reactants`.
    - Cluster representatives are **medoids**, not means — averaging positions under PBC is ill-defined without a reference anchor, medoid side-steps the issue.
    - **Hungarian threshold gotcha:** LSA minimises total cost without respecting a per-pair threshold, so a single outlier can cascade into a globally-optimal but individually-pathological assignment. Fix: mask above-threshold entries to `1e9` in the cost matrix before LSA, so it prefers sub-threshold pairings first and only uses above-threshold pairings when unavoidable (then filtered post-LSA). Verified in-tree: outlier centroid is correctly left unmatched instead of forcing a miss on a well-matching pair.
    - Verified in-tree: RMSD-PBC (identical=0, 0.1Å uniform=0.1, PBC wraparound 9.98→0.02), pairwise symmetric and monotone, clustering (10 structs in 3 groups → 3 clusters with correct labels), Hungarian diagonal on clean, outlier-resistant, sparse-recall case (1/2 refs matched gives recall=0.5), micro-aggregate correct.
14. ~~Write `examples/LiC/train.py` and `evaluate.py`~~ **Done.** Example scripts are thin argparse + wiring around `saddlegen.*` primitives — no generic code leaks into the example. `convert.py` is omitted for Li/C since Backend A (`TrajTripletDataset`) reads `.traj` directly; conversion only pays off at 30M scale.
    - **Refactor during this step:** generic helpers extracted from the example scripts into the library so the same primitives are reusable across projects:
        - `saddlegen/utils/backbone.py` — `load_uma_backbone(name, device, freeze, eval_mode)` wraps fairchem's `get_predict_unit`, strips the `AveragedModel` wrapper + output heads, returns the `eSCNMDBackbone` directly.
        - `saddlegen/utils/checkpointing.py` — `load_ema_weights(ckpt_dir, modules, device)` copies the `ema.pt` shadow into the trainable params of `modules` (trainable-param order must match `EMA.__init__`'s iteration).
        - `saddlegen/data/core.py` — `atoms_to_sample_dict(atoms)` builds an inference-time sample dict (no saddle / metadata fields). `load_validated_triplets(paths)` returns the up-front list of validated `(R, S, P)` tuples; used by scripts that need to group triplets.
        - `saddlegen/utils/eval.py` — `group_triplets_by_reactant(triplets, cell, threshold=0.02)` clusters triplets by PBC-RMSD on reactant positions; used by evaluation to aggregate multiple triplets' saddles onto a single unique reactant.
    - **Evaluation protocol (implemented in `examples/LiC/evaluate.py`):** load train+test triplets, group test by reactant (PBC-RMSD < 0.02 Å), per unique reactant run the sampler → cluster candidates → Hungarian-match cluster centroids against that reactant's own known saddles (primary recall/precision/RMSD). Also report "bonus" matches against the full (train + test) saddle pool — any rediscovered TS from train is a valid TS, not a miss.
    - **Verified against real Li/C data:** 12 train triplets → 10 unique reactants; 171 test triplets → 59 unique reactants (≈3 saddles per reactant on average, as expected for Li hopping between C-sheet adsorption sites). `<‖Δ‖>` = 1.25 Å on train, giving σ_rs_pert ≈ 0.036 Å via the rule-of-thumb. 127 atoms per system, 126 frozen Cs + 1 mobile Li. No `side` keys in the source trajs — positional ordering used throughout.
    - **UMA charge/spin defaults corrected to (0, 0)** — the LiC traj `atoms.info` already carries `charge=0, spin=0`, and `spin=1` was never right for omat (UMA's `spin` is only physically used by the `omol` head).

### Phase 3 — 30M training (after Li/C works)

15. Scale the LMDB conversion to the full 30M dataset.
16. Multi-node `accelerate` launch config and SLURM wrapper.
17. Frozen-backbone baseline run.
18. Fine-tuning run (lr 1e-5, optional LoRA).
19. Ablations: `σ_rs_pert`, `σ_ts_pert`, `ε`, `w_1`, `w_3`, `K`, obj (1) on/off, `GlobalAttn` depth (and on/off as a sanity check of H7).

### Phase 4 — Paper

20. `refine.py` (optional Dimer refine using fairchem ASE Calculator) — only once we have good Phase 2 / 3 results.
21. Methods section drafted from this CLAUDE.md; results and figures from Phase 2 + 3.

## Accuracy improvement levers (to revisit if Li/C recall is poor or 30M results plateau)

When results disappoint, investigate roughly in this order — cheapest to most expensive in implementation effort, training cost, and risk of breaking what works.

**Tier 1 — hyperparameter sweeps (no code change, hours of GPU time):**
- `σ_rs_pert` — too small → no mode selection (collapses to mean saddle); too large → off-distribution starts. Sweep first.
- `ε` (obj-3 flow-time cutoff) — controls how aggressively obj (3) supervises the early ray.
- Loss weights `(w_1, w_2, w_3)` — try enabling obj (1), increasing `w_3`.
- Inference integration steps `K` — increase before suspecting model error.
- Time-sampling distribution in obj (2)/(3) — try logit-normal or beta-skewed in place of uniform.

**Tier 2 — small architectural changes (hours of dev, no UMA retraining):**
- `VelocityHead.depth` 1 → 2 → 3 (`SO3_Linear` + `Gate` stack, see §Head). Adds capacity above the frozen backbone.
- `GlobalAttn` depth 1 → 2 → 4, heads 8 → 16. Improves long-range coordination.
- Higher-order integrator (Euler → torchdiffeq RK45) at inference.
- Multiple supervision points per straight-line trajectory in obj (2)/(3) (currently one sample per draw — could draw 4 `t`s at once with shared `ε_rs_pert`).

**Tier 3 — partial unfreeze of UMA (significant dev + retraining):**
- `--finetune` end-to-end with `lr = 1e-5`, cosine warmup. Most likely lever if frozen baseline plateaus.
- LoRA-style adapters on UMA's MOLE layers — preserves pretrained weights, much lower memory.
- **Time-injection into UMA backbone (currently head-only).** Straight-line OT means `v_target` is constant in `t` along each trajectory, so head-only injection is theoretically defensible — but if accuracy is bad and integrator-error / capacity arguments are exhausted, this is the prime suspect. Two options, ascending intrusiveness:
  - *Input-level:* learned `t` → MLP → scalar bias added to atom embedding inside `csd_embedding` (alongside charge/spin). Equivariant by construction. Requires unfreezing `csd_embedding` and probably the first eSCN-MD block. ~30 lines.
  - *AdaLN at every block:* time-conditioned scalar `γ(t)` modulating each per-`l` RMSNorm (equivariant: scalars allowed on all `l`; shifts `β(t)` allowed only on `l=0`). What DiT/ADiT do for diffusion. Requires patching fairchem's eSCN-MD layer code. ~hundreds of lines, intrusive. Defer until the cheaper input-level variant is shown to be insufficient.

**Tier 4 — bigger backbone (substantial cost, last resort):**
- Swap UMA-S-1.2 → UMA-M-1.2 (K10L4: 10 layers, lmax=4 vs our K4L2's 4 layers, lmax=2). Strictly more capacity at every level. Multi-node from day one.
- Conservative formulation analog: parameterize a scalar potential `Φ(x)` and define `v = -∇_x Φ` via autograd. Mathematically clean but requires re-deriving the entire flow loss; speculative.

**Tier 5 — sampling / diversity (orthogonal to per-step accuracy):**
- Sparse-Gaussian or Hessian-aware perturbation at inference (current Gaussian may under-represent the sparsest TS modes for large `M`).
- Distance-sparse `GlobalAttn` (attend within 2× UMA cutoff) — for memory if dense attention OOMs on the 30M run.

## Open questions (all empirical; to be settled on the Li/C test case)

- **`σ_rs_pert`** — Gaussian std for obj (2) reactant-state perturbation and inference, in Å. Primary HP. Sweep roughly `σ_rs_pert ∈ {0.02, 0.05, 0.1, 0.2} · ⟨‖Δ‖⟩ / √(3M)` as anchor, then refine empirically.
- **`ε`** — flow-time cutoff for obj (3), dimensionless. Sweep `{0.01, 0.05, 0.1}`.
- **Loss weights** `(w_1, w_2, w_3)` — default `(0, 1, 2)`; ablate `w_1` on/off; sweep `w_3 ∈ {1, 2, 4}`.
- **`σ_ts_pert`** — Gaussian std for obj (1) TS-state perturbation, in Å. Default `σ_ts_pert = σ_rs_pert`.
- **Inference integration steps `K`** — default 50, sweep 20–200.
- **`GlobalAttn` depth** — default 1 layer, ablate 1–4. Number of heads default 8.
- **`VelocityHead.depth`** — default 1 (mirrors UMA's linear head exactly). For `depth ≥ 2`, stack `SO3_Linear → e3nn.nn.Gate → SO3_Linear`. Ablate `{1, 2, 3}` if capacity above the frozen backbone matters.
- **Time-injection point** — default head-only (math justified for straight-line OT). Tier-3 ablation: input-level injection into `csd_embedding` (requires fine-tuning). Tier-5: AdaLN at every backbone block (intrusive).
- **Distance-sparse `GlobalAttn`** — turn on only if dense attention becomes a bottleneck at the 30M scale.
- **Fine-tuning LR and whether to use LoRA adapters** — second-iteration concern.
