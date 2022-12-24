
"""Discriminator"""

import mindspore.nn as nn
from mindspore.common.initializer import Normal
from .instance_norm_2d import InstanceNorm2d


class Discriminator(nn.Cell):
    """
    Discriminator network.

    Args:
        channels (int): Base channel number per layer.
        n_dis (int): The number of discriminator layer.

    Inputs:
        - **x** (tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        - **out** (tensor) - Tensor output from the discriminator network.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> discriminator = Discriminator()
    """

    def __init__(self, channels, n_dis):
        super(Discriminator, self).__init__()
        self.has_bias = False

        layers = [
            nn.Conv2d(3, channels, kernel_size=3, stride=1, pad_mode='same', padding=0,
                      weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
            nn.LeakyReLU(alpha=0.2)
        ]

        for _ in range(1, n_dis):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, pad_mode='same', padding=0,
                          weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, pad_mode='same', padding=0,
                          weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
                InstanceNorm2d(channels * 4, affine=False),
                nn.LeakyReLU(alpha=0.2),
            ]
            channels *= 4

        layers += [
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, pad_mode='same', padding=0,
                      weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
            InstanceNorm2d(channels, affine=False),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, pad_mode='same', padding=0,
                      weight_init=Normal(mean=0, sigma=0.02), has_bias=self.has_bias),
        ]

        self.discriminator = nn.SequentialCell(layers)

    def construct(self, x):
        """ build network """
        out = self.discriminator(x)
        return out
