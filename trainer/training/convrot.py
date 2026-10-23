"""Experimental INT8 ConvRot frozen-base LoCon training, without a ComfyUI install."""
from __future__ import annotations

import torch
from torch import nn


def apply_convrot(module, selection_config, device, dtype):
    """Use the same protected projections as SDNQ; adapters stay in the original basis.

    Kitchen rotates/quantizes activations for INT8 forward GEMMs and dequantizes
    the frozen weights for input gradients. Attention remains BF16. Only the
    independent LoCon bypass is updated; this is not quantized full finetuning.
    """
    try:
        from comfy_kitchen.tensor import QuantizedTensor
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout  # noqa: F401: registers layout
    except ImportError as exc:
        raise RuntimeError(
            "ConvRot requires comfy-kitchen==0.2.37 in the trainer environment. "
            "Install with: uv pip install --python <trainer-python> comfy-kitchen==0.2.37"
        ) from exc
    from sdnq.utils import check_param_name_in, check_quant_is_allowed

    if torch.device(device).type != "cuda" or dtype != torch.bfloat16:
        raise ValueError("Experimental ConvRot requires CUDA and BF16 computation")
    targets = [(name, layer) for name, layer in module.named_modules()
               if isinstance(layer, nn.Linear)
               and check_param_name_in(name + ".weight", selection_config.modules_to_not_convert) is None
               and check_quant_is_allowed(layer.__class__.__name__, layer.weight, selection_config)]
    if not targets:
        raise ValueError("ConvRot selection contains no eligible Linear projections")
    for name, layer in targets:
        if layer.in_features % 256:
            raise ValueError(f"ConvRot projection {name} requires input width divisible by 256")
    module.requires_grad_(False)
    for _, layer in targets:
        weight = layer.weight.to(device=device, dtype=dtype)
        layer.weight = nn.Parameter(QuantizedTensor.from_float(
            weight, "TensorWiseINT8Layout", per_channel=True,
            convrot=True, convrot_groupsize=256), requires_grad=False)
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.to(device=device, dtype=dtype)
        layer._trainer_convrot = True
    print(f"ConvRot: {len(targets)} frozen Linear projections, INT8 matmul, rotation group 256; "
          "BF16 attention/input-gradient computation. SDNQ storage/matmul options apply only to SDNQ.")
    return module
