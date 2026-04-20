"""
Load EMA weights from a training checkpoint into an inference-time module stack.

Training writes `ema.pt` alongside `accelerator.save_state(...)` output; this
file holds a flat list of shadow tensors parallel to the trainable parameters
(see `saddlegen.utils.training.EMA`). At inference time we want those shadow
weights — not the point-estimate weights in the accelerator checkpoint — because
they've been averaged over the training trajectory.
"""

from pathlib import Path

import torch
import torch.nn as nn


def load_ema_weights(
    ckpt_dir: str,
    modules: list[nn.Module],
    device: torch.device | str,
) -> None:
    """Copy EMA shadow weights from `<ckpt_dir>/ema.pt` into the trainable params of `modules`.

    Parameter order is inferred the same way as in `EMA.__init__`: iterate `modules`
    in the given order, take every parameter with `requires_grad=True`.
    """
    ckpt = Path(ckpt_dir)
    ema_path = ckpt / "ema.pt"
    if not ema_path.exists():
        raise FileNotFoundError(f"no ema.pt under {ckpt_dir}")
    sd = torch.load(ema_path, map_location=device)
    shadow = sd["shadow"]
    trainable = [p for m in modules for p in m.parameters() if p.requires_grad]
    if len(shadow) != len(trainable):
        raise RuntimeError(
            f"EMA state has {len(shadow)} tensors but {len(trainable)} trainable params "
            f"were found across the provided modules. Did the architecture change?"
        )
    with torch.no_grad():
        for p, s in zip(trainable, shadow):
            p.data.copy_(s.to(device))
