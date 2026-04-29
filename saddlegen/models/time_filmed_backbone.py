"""
A thin wrapper around the UMA backbone that injects an equivariant
time-FiLM right before the LAST message-passing block (`blocks[3]` for
UMA-S-1.2). Used by Mode 1 v1 onward.

Why here. UMA-S-1.2's only loss head is `MLP_EFS_Head`, which reads only
the l=0 channels of `blocks[3]`'s output for energy and derives forces by
autograd through positions. Layer-3's l≥1 outputs (the input to blocks[3])
are heavily implicated in the autograd path, so they're well-tuned. By
applying our learnable time-FiLM at this exact junction — and (optionally)
unfreezing `blocks[3]` so it can adapt — we let the backbone's final
message-passing pass become time-aware AND let it tune its l≥1 outputs
to be useful for downstream velocity prediction. See CLAUDE.md
§"Accuracy improvement levers > Tier 2 > UMA layer-4" for the analysis.

Construction:
    backbone = load_uma_backbone(..., unfreeze_last_block=True)
    wrapped  = TimeFiLMBackbone(backbone)
    feat     = wrapped(data, t_tensor, batch_idx)        # same dict as backbone(data)

The wrapper stashes `t` and `batch_idx` on the module and a
`forward_pre_hook` on `backbone.blocks[3]` reads them to construct the FiLM.
This avoids re-implementing UMA's full forward.

The pre-hook intercepts the input tensor `args[0]` (per-atom irreps,
shape (N, num_sph, sphere_channels)) and returns the FiLM'd version.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .time_film import TimeFiLM


class TimeFiLMBackbone(nn.Module):
    """Wrap a UMA `eSCNMDBackbone` so a learnable time-FiLM is applied to the
    input of `blocks[3]` (the last message-passing layer)."""

    def __init__(
        self,
        backbone: nn.Module,
        time_embed_dim: int = 64,
        time_mlp_hidden: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.film = TimeFiLM(
            channels=backbone.sphere_channels,
            time_embed_dim=time_embed_dim,
            time_mlp_hidden=time_mlp_hidden,
        )
        # Per-call state read by the pre-hook. Set in forward(...).
        self._t: torch.Tensor | None = None
        self._batch_idx: torch.Tensor | None = None
        # Register the pre-hook on the LAST block. UMA-S-1.2 has 4 blocks,
        # i.e. `blocks[3]` is the final one — see escn_md.py:803-816 where
        # the for-loop iterates `self.blocks[i]` and reassigns `x_message`.
        self._handle = backbone.blocks[-1].register_forward_pre_hook(
            self._film_pre_hook, with_kwargs=True,
        )

    @property
    def sphere_channels(self) -> int:
        return self.backbone.sphere_channels

    @property
    def lmax(self) -> int:
        return self.backbone.lmax

    @property
    def num_layers(self) -> int:
        return self.backbone.num_layers

    def _film_pre_hook(self, module, args, kwargs):
        """Intercept the first positional arg (x_message) and FiLM it.

        UMA's eSCNMD_Block.forward takes `x_message` as its first positional
        argument plus a number of auxiliary tensors (edge index, wigner,
        envelopes, etc.). We modify only `x_message`; other args pass through.
        """
        if self._t is None or self._batch_idx is None:
            raise RuntimeError(
                "TimeFiLMBackbone forward state not initialised — pre-hook "
                "fired without `forward()` having stashed t/batch_idx."
            )
        x_message = args[0]
        x_filmed = self.film(x_message, self._t, self._batch_idx)
        new_args = (x_filmed,) + args[1:]
        return new_args, kwargs

    def forward(
        self,
        data,
        t: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> dict:
        """Run the backbone with time-FiLM applied at the last block.

        Args:
            data: AtomicData / PyG Batch; what `backbone(data)` accepts.
            t: (B,) flow-time per system in [0, 1].
            batch_idx: (N,) atom-to-system map. Note `data.batch` is the same
                tensor in fairchem; we accept it explicitly to avoid coupling
                to the data layout.
        Returns the backbone's normal forward dict (with `node_embedding`).
        """
        self._t = t
        self._batch_idx = batch_idx
        try:
            return self.backbone(data)
        finally:
            self._t = None
            self._batch_idx = None
