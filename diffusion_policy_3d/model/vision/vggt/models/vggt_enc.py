# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from diffusion_policy_3d.model.vision.vggt.models.aggregator import Aggregator

VGGT_HF_REPO = "facebook/VGGT-1B"
VGGT_HF_FILENAME = "model.safetensors"


def resolve_vggt_weight(weight_path=None):
    """按 显式参数 -> 环境变量 VGGT_WEIGHT_PATH -> HuggingFace Hub 的顺序解析权重路径。"""
    weight_path = weight_path or os.environ.get("VGGT_WEIGHT_PATH")
    if weight_path is not None:
        weight_path = os.path.expanduser(weight_path)
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"VGGT weight not found at {weight_path}. "
                f"Unset VGGT_WEIGHT_PATH to download {VGGT_HF_REPO} from HuggingFace instead."
            )
        return weight_path

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=VGGT_HF_REPO, filename=VGGT_HF_FILENAME)


class VGGT_ENC(nn.Module, PyTorchModelHubMixin):
    """VGGT aggregator used as a frozen visual encoder.

    Only the aggregator trunk of VGGT-1B is kept: the camera / depth / track
    heads are dropped because the policy consumes patch tokens directly.
    """

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24):
        super().__init__()
        self.depth = depth
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth
        )

    def load_weight_freeze(self, weight_path=None):
        """Load pretrained VGGT-1B aggregator weights and freeze the encoder."""
        from safetensors.torch import load_file

        weight_path = resolve_vggt_weight(weight_path)
        weights = load_file(weight_path, device="cpu")
        agg_weights = {k: v for k, v in weights.items() if k.startswith("aggregator.")}
        if not agg_weights:
            raise RuntimeError(f"No 'aggregator.*' tensors found in {weight_path}")
        # strict=False: only the first `depth` blocks are instantiated when depth < 24.
        self.aggregator.load_state_dict(agg_weights, strict=False)
        for param in self.parameters():
            param.requires_grad = False

    def ptq_quantize(self, weight_dtype="int4"):
        """Post-training quantize the aggregator (int4 / int8 / fp8_e5m2).

        The released checkpoints were trained with this enabled, which is what
        keeps the frozen encoder inside ~20GB of VRAM. It is OFF by default;
        enable it with `policy.vggtq=true`. Requires the `coremltools` package.
        """
        from coremltools.optimize.torch.quantization import (
            PostTrainingQuantizer,
            PostTrainingQuantizerConfig,
        )

        config = PostTrainingQuantizerConfig.from_dict(
            {"global_config": {"weight_dtype": weight_dtype}}
        )
        self.aggregator = PostTrainingQuantizer(self.aggregator, config).compress()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor):
        """Extract aggregated patch tokens.

        Args:
            images: [S, 3, H, W] or [B, S, 3, H, W], values in [0, 1].

        Returns:
            Tensor of shape [B, S, N_tokens, 2 * embed_dim] -- the last layer of the
            aggregator token list. The leading `patch_start_idx` tokens are camera
            tokens and are kept so the projector can attend over them.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        return aggregated_tokens_list[-1]
