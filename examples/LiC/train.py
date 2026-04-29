"""
Train SaddleGen on the Li-on-C defective-graphene test case.

Data: `train_set.traj` with flat `[R1, S1, P1, R2, S2, P2, …]` ordering (no
`side` info required — `validate_triplet` falls back to positional ordering).
All reusable machinery lives in `saddlegen`; this script is only argparse +
wiring.

Sampling: ice-cream-cone of x_0 around the r_R → r_S axis. The per-sample
ball radius at r_S is `R_TS = min(alpha · |Δ|, R_max)`; default alpha=0.5
(30° cone half-angle) and R_max=1.0 Å.

Launch:
    python examples/LiC/train.py
multi-GPU / multi-node:
    accelerate launch --multi_gpu examples/LiC/train.py
"""

import argparse
from pathlib import Path

import torch

from saddlegen.data import TrajTripletDataset
from saddlegen.flow import FlowMatchingConfig, FlowMatchingLoss
from saddlegen.models import GlobalAttn, VelocityHead
from saddlegen.models.time_filmed_backbone import TimeFiLMBackbone
from saddlegen.utils import TrainingConfig, load_uma_backbone, train


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--output-dir", default=str(here / "runs" / "icecream_v0"))

    p.add_argument("--num-epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every-epochs", type=int, default=1000)

    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--attn-layers", type=int, default=0,
                   help="Number of GlobalAttn layers. Default 0 (no attention) — "
                        "Mode 1's partner direction already breaks the symmetry that "
                        "GlobalAttn was originally introduced to handle.")
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=1)

    # Training mode.
    p.add_argument("--mode", type=int, default=1,
                   help="0 = ice-cream-cone (legacy single-mobile-atom recipe); "
                        "1 = product-conditional (uses partner endpoint, no noise) — DEFAULT; "
                        "2 = Dimer-trajectory (NotImplementedError; dataset only).")
    p.add_argument("--delta-endpoint-channels", type=int, default=32,
                   help="Mode 1 only. Channel count for the partner-displacement "
                        "feature in VelocityHead. Default 32 — analogue of time_embed_dim.")

    # v1 architecture knobs.
    p.add_argument("--unfreeze-uma-last", action="store_true",
                   help="Unfreeze UMA's last message-passing block (blocks[-1]). "
                        "Trainable params on the backbone get a separate, much lower "
                        "LR (--uma-lr). Mode 1 v1+.")
    p.add_argument("--early-time-film", action="store_true",
                   help="Wrap the backbone with TimeFiLMBackbone, applying an "
                        "equivariant time-FiLM right before the last block. "
                        "Designed to be combined with --unfreeze-uma-last so the "
                        "unfrozen block sees a time-conditioned input. Mode 1 v1+.")
    p.add_argument("--uma-lr", type=float, default=1e-5,
                   help="Discriminative LR for unfrozen UMA params (when "
                        "--unfreeze-uma-last is set). Default 1e-5 — 100× lower "
                        "than the head's LR to avoid destroying pretrained features.")

    # Mode 0 — ice-cream-cone sampling of x_0.
    p.add_argument("--alpha", type=float, default=0.5,
                   help="cone half-angle = arcsin(alpha). Default 0.5 = 30°, "
                        "fits inside a C_6v wedge. Mode 0 only.")
    p.add_argument("--R-max", type=float, default=1.0,
                   help="Å. Absolute cap on the ball radius at r_S. Mode 0 only.")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] dataset: {args.train_traj}")
    dataset = TrajTripletDataset(
        args.train_traj,
        stats_cache=str(out_dir / "dataset_stats.json"),
    )
    print(f"[train] {len(dataset)} records ({dataset.num_triplets} triplets × 2 sides), "
          f"<||Δ||> = {dataset.delta_norm_mean:.4f} Å")

    M = int((~dataset[0]["fixed"]).sum().item())
    if args.mode == 0:
        print(f"[train] mode 0 — ice-cream-cone")
        print(f"[train] alpha={args.alpha}  R_max={args.R_max} Å  (M={M})")
    elif args.mode == 1:
        print(f"[train] mode 1 — product-conditional (no noise on x_0)")
        print(f"[train] delta_endpoint_channels={args.delta_endpoint_channels}  (M={M})")
    else:
        raise SystemExit(f"--mode {args.mode} is not yet wired into the trainer "
                         f"(mode 2 dataset is in place but the loss is not).")

    print(f"[train] loading backbone {args.backbone!r} onto {args.device}")
    raw_backbone = load_uma_backbone(
        args.backbone, device=args.device, freeze=True, eval_mode=True,
        unfreeze_last_block=args.unfreeze_uma_last,
    )
    sc, lmax = raw_backbone.sphere_channels, raw_backbone.lmax

    # Optionally wrap the backbone with the early time-FiLM (Mode 1 v1).
    if args.early_time_film:
        backbone = TimeFiLMBackbone(raw_backbone).to(args.device)
        print(f"[train] backbone wrapped with TimeFiLMBackbone (early time-FiLM "
              f"before blocks[-1])")
    else:
        backbone = raw_backbone

    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                       num_heads=args.attn_heads, num_layers=args.attn_layers).to(args.device)
    head_delta_C = args.delta_endpoint_channels if args.mode == 1 else 0
    head = VelocityHead(
        sphere_channels=sc, input_lmax=lmax, depth=args.head_depth,
        delta_endpoint_channels=head_delta_C,
    ).to(args.device)

    n_uma_unfrozen = sum(
        p.numel() for p in raw_backbone.parameters() if p.requires_grad
    )
    head_attn_params = list(attn.parameters()) + list(head.parameters())
    if args.early_time_film:
        head_attn_params += list(backbone.film.parameters())
    n_head_attn = sum(p.numel() for p in head_attn_params if p.requires_grad)
    print(f"[train] backbone K{raw_backbone.num_layers}L{lmax} "
          f"(sphere_channels={sc}), frozen={'partial' if n_uma_unfrozen else 'full'}")
    print(f"[train] attn_layers={args.attn_layers}  head_depth={args.head_depth}  "
          f"early_time_film={args.early_time_film}  unfreeze_uma_last={args.unfreeze_uma_last}")
    print(f"[train] trainable: head+attn(+early_film) = {n_head_attn:,}  "
          f"unfrozen UMA = {n_uma_unfrozen:,}")

    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(
            mode=args.mode,
            alpha=args.alpha, R_max_abs=args.R_max,
        ),
        backbone, attn, head,
    )

    # Discriminative LR via parameter groups when UMA is partially unfrozen.
    param_groups = None
    if args.unfreeze_uma_last:
        param_groups = [
            {
                "name": "head_attn_film",
                "params": [p for p in head_attn_params if p.requires_grad],
                "lr": args.learning_rate,
            },
            {
                "name": "uma_unfrozen",
                "params": [p for p in raw_backbone.parameters() if p.requires_grad],
                "lr": args.uma_lr,
            },
        ]

    train_cfg = TrainingConfig(
        output_dir=str(out_dir),
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm, ema_decay=args.ema_decay,
        mixed_precision=args.mixed_precision, seed=args.seed,
        log_every=args.log_every, save_every_epochs=args.save_every_epochs,
        extras={
            "mode": args.mode,
            "delta_endpoint_channels": head_delta_C,
            "alpha": args.alpha, "R_max": args.R_max,
            "backbone": args.backbone,
            "attn_layers": args.attn_layers, "attn_heads": args.attn_heads,
            "head_depth": args.head_depth,
            "early_time_film": bool(args.early_time_film),
            "unfreeze_uma_last": bool(args.unfreeze_uma_last),
            "uma_lr": args.uma_lr if args.unfreeze_uma_last else None,
        },
    )
    train(loss_module, dataset, train_cfg, param_groups=param_groups)


if __name__ == "__main__":
    main()
