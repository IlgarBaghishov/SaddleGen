"""
Train SaddleGen on the Li-on-C test case.

Data: `train_set.traj` with flat `[R1, S1, P1, R2, S2, P2, …]` ordering (no
`side` info required — `validate_triplet` falls back to positional ordering).
All reusable machinery lives in `saddlegen`; this script is only argparse +
wiring.

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
from saddlegen.utils import TrainingConfig, load_uma_backbone, train


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--output-dir", default=str(here / "runs" / "head_only_v0"))

    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every-epochs", type=int, default=50)

    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--attn-layers", type=int, default=1)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=1)

    p.add_argument("--sigma-rs-pert", type=float, default=None,
                   help="Å. If omitted, use rule-of-thumb --sigma-rs-factor · <||Δ||> / √(3M).")
    p.add_argument("--sigma-rs-factor", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--w1", type=float, default=0.0)
    p.add_argument("--w2", type=float, default=1.0)
    p.add_argument("--w3", type=float, default=2.0)

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

    # σ_rs_pert rule-of-thumb based on dataset stats (mobile-atom count from sample 0).
    M = int((~dataset[0]["fixed"]).sum().item())
    sigma_rs = (args.sigma_rs_pert if args.sigma_rs_pert is not None
                else args.sigma_rs_factor * dataset.delta_norm_mean / max(1, (3 * M) ** 0.5))
    print(f"[train] σ_rs_pert = {sigma_rs:.5f} Å   (M={M} mobile atoms)")

    print(f"[train] loading backbone {args.backbone!r} onto {args.device}")
    backbone = load_uma_backbone(args.backbone, device=args.device, freeze=True, eval_mode=True)
    sc, lmax = backbone.sphere_channels, backbone.lmax
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                       num_heads=args.attn_heads, num_layers=args.attn_layers).to(args.device)
    head = VelocityHead(sphere_channels=sc, input_lmax=lmax, depth=args.head_depth).to(args.device)
    print(f"[train] backbone K{backbone.num_layers}L{lmax} (sphere_channels={sc}), frozen")
    print(f"[train] trainable params: {sum(p.numel() for p in list(attn.parameters()) + list(head.parameters())):,}")

    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(sigma_rs_pert=sigma_rs, sigma_ts_pert=sigma_rs,
                           epsilon=args.epsilon,
                           loss_weights=(args.w1, args.w2, args.w3)),
        backbone, attn, head,
    )

    train_cfg = TrainingConfig(
        output_dir=str(out_dir),
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm, ema_decay=args.ema_decay,
        mixed_precision=args.mixed_precision, seed=args.seed,
        log_every=args.log_every, save_every_epochs=args.save_every_epochs,
        extras={
            "sigma_rs_pert": sigma_rs, "M": M, "backbone": args.backbone,
            "attn_layers": args.attn_layers, "attn_heads": args.attn_heads,
            "head_depth": args.head_depth, "epsilon": args.epsilon,
            "loss_weights": [args.w1, args.w2, args.w3],
        },
    )
    train(loss_module, dataset, train_cfg)


if __name__ == "__main__":
    main()
