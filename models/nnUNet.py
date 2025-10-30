"""
IsoNet-compatible nnU-Net backbone.

This module mirrors the nnU-Net v2 PlainConvUNet architecture while keeping the
full decoder inside the main network (no detached classification head) and
replacing the final segmentation layer with a 3×3×3 convolution that produces a
single-channel logits volume. It is intended to be copied into IsoNet so that
weights pretrained there can later be transferred back into nnU-Net.
"""

from typing import Sequence, Optional, Mapping, Any

import torch
from torch import nn
from torch.nn import init as nn_init

from dynamic_network_architectures.architectures.unet import PlainConvUNet

DEFAULT_FEATURES_PER_STAGE: Sequence[int] = (32, 64, 128, 256, 320, 320)
DEFAULT_KERNEL_SIZES: Sequence[Sequence[int]] = (
    (3, 3, 3),
    (3, 3, 3),
    (3, 3, 3),
    (3, 3, 3),
    (3, 3, 3),
    (3, 3, 3),
)
DEFAULT_STRIDES: Sequence[Sequence[int]] = (
    (1, 1, 1),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
)
DEFAULT_N_CONV_PER_STAGE: Sequence[int] = (2, 2, 2, 2, 2, 2)
DEFAULT_N_CONV_PER_STAGE_DECODER: Sequence[int] = (2, 2, 2, 2, 2)
DEFAULT_CLASS_NAMES = [
    "ER",
    "mitochondria",
    "MT",
    "vesicle",
    "membrane",
    "ER_memb",
    "mito_memb",
    "MT_memb",
    "vesicle_memb",
    "actin",
]


__all__ = ["NNUNet"]


class NNUNet(nn.Module):
    """
    Wrapper around PlainConvUNet with the decoder kept intact and a custom
    3×3×3 segmentation layer for IsoNet pretraining.
    """

    def __init__(
        self,
        input_channels: int = 1,
        features_per_stage: Sequence[int] = DEFAULT_FEATURES_PER_STAGE,
        kernel_sizes: Sequence[Sequence[int]] = DEFAULT_KERNEL_SIZES,
        strides: Sequence[Sequence[int]] = DEFAULT_STRIDES,
        n_conv_per_stage: Sequence[int] = DEFAULT_N_CONV_PER_STAGE,
        n_conv_per_stage_decoder: Sequence[int] = DEFAULT_N_CONV_PER_STAGE_DECODER,
        conv_bias: bool = True,
        norm_op: Optional[type[nn.Module]] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[Mapping[str, Any]] = None,
        dropout_op: Optional[type[nn.Module]] = None,
        dropout_op_kwargs: Optional[Mapping[str, Any]] = None,
        nonlin: Optional[type[nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: Optional[Mapping[str, Any]] = None,
        deep_supervision: bool = False,
        initialize_weights: bool = True,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.out_channels = 1  # fixed single-channel head for IsoNet pretraining
        self.class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        self.learning_rate: Optional[float] = None
        self.metrics = {"train_loss": [], "val_loss": []}

        self.network = PlainConvUNet(
            input_channels=input_channels,
            n_stages=len(features_per_stage),
            features_per_stage=features_per_stage,
            conv_op=nn.Conv3d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=self.out_channels,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=dict(norm_op_kwargs),
            dropout_op=dropout_op,
            dropout_op_kwargs=None if dropout_op_kwargs is None else dict(dropout_op_kwargs),
            nonlin=nonlin,
            nonlin_kwargs=dict(nonlin_kwargs),
            deep_supervision=deep_supervision,
            nonlin_first=False,
        )

        if initialize_weights:
            PlainConvUNet.initialize(self.network)

        self.encoder = self.network.encoder
        self.decoder = self.network.decoder
        self._replace_final_seg_layer(
            in_channels=features_per_stage[0],
            initialize=initialize_weights,
        )

    def _replace_final_seg_layer(self, in_channels: int, initialize: bool) -> None:
        last_idx = len(self.decoder.seg_layers) - 1
        final_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.decoder.seg_layers[last_idx] = final_conv
        self.final = final_conv

        if initialize:
            nn_init.kaiming_normal_(final_conv.weight, nonlinearity="leaky_relu")
            if final_conv.bias is not None:
                nn_init.constant_(final_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
