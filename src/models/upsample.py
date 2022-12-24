
"""Define up-sampling operation."""

import mindspore.nn as nn


class UpSample(nn.Cell):
    """
    Define up-sampling and convolution module.

    Args:
        channels (int): Number of input channels.
        out_channels (int): Number of input channels.
        kernel_size (int): Convolution kernel size. Default: 3.
        has_bias (bool): Whether to add bias. Default: False.

    Inputs:
        - **x** (tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        - **out** (tensor) - Tensor output upsample.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> upsample = UpSample(128,128)
    """

    def __init__(self, channels, out_channels, kernel_size=3, has_bias=False):
        super(UpSample, self).__init__()

        self.conv = nn.Conv2d(channels, out_channels, stride=1, kernel_size=kernel_size, has_bias=has_bias)
        self.resize = nn.ResizeBilinear()

    def construct(self, x):
        """ build network """
        out = self.resize(x, scale_factor=2)
        out = self.conv(out)

        return out
