"""
Load a pretrained fairchem/UMA backbone for use inside a SaddleGen model.

This is a thin wrapper around `fairchem.core.calculate.pretrained_mlip.get_predict_unit`
that strips the EMA wrapper and UMA's output heads, returning only the
`eSCNMDBackbone` we consume in `FlowMatchingLoss` and `sample_saddles`.
"""

import torch.nn as nn


def load_uma_backbone(
    name: str = "uma-s-1p2",
    device: str = "cuda",
    freeze: bool = True,
    eval_mode: bool = True,
) -> nn.Module:
    """Return the `eSCNMDBackbone` module from a pretrained UMA checkpoint.

    Args:
        name: HuggingFace model tag. Defaults to the small UMA-S-1.2 variant
            (6.6M active / 290M total params). For a larger backbone use
            `"uma-m-1p2"` etc., per fairchem's registry.
        device: "cuda", "cpu", or a specific CUDA index.
        freeze: if True, sets `requires_grad=False` on every backbone parameter.
            Turn this off to enable end-to-end fine-tuning.
        eval_mode: if True, calls `.eval()` on the returned module.
    """
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit

    predictor = get_predict_unit(name, device=device)
    # predictor.model is an AveragedModel(HydraModel); we want the backbone.
    backbone = predictor.model.module.backbone
    if freeze:
        for p in backbone.parameters():
            p.requires_grad_(False)
    if eval_mode:
        backbone.eval()
    return backbone
