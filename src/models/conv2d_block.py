
"""Convolution block."""

import mindspore.nn as nn
from mindspore.common.initializer import Normal
from .instance_norm_2d import InstanceNorm2d

class ConvBlock(nn.Cell):
    """
    Define convolution block.

    Args:
        channels (int): Number of input channels.
        out_channels (int): Number of input channels.
        kernel_size (int): Convolution kernel size. Default: 3.
        stride (int): Stride size of convolution kernel. Default: 1.
        padding (int): Enter the number of fills in the height and width directions. Default: 0.
        has_bias (bool): Whether to add bias. Default: False.

    Inputs:
        - **x** (tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        - **out** (tensor) - Extracted features.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> conv2d = ConvBlock(128,128)
    """

    def __init__(self, channels, out_channels, kernel_size=3, stride=1, padding=0, has_bias=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(channels, out_channels, kernel_size=kernel_size, stride=stride, pad_mode='same',
                              padding=padding,
                              weight_init=Normal(mean=0.0, sigma=0.02), has_bias=has_bias)
        self.ins_norm = InstanceNorm2d(out_channels, affine=False)
        self.activation = nn.LeakyReLU(alpha=0.2)

    def construct(self, x):
        """ build network """
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)
        return out
