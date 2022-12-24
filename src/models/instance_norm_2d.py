
"""Define up-sampling operation."""

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import ops, Tensor
from mindspore.common import initializer as init


class InstanceNorm2d(nn.Cell):
    """myown InstanceNorm2d"""

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros'):
        super().__init__()
        self.num_features = num_features
        self.moving_mean = Parameter(init.initializer('zeros', num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(init.initializer('ones', num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(init.initializer(gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(init.initializer(beta_init, num_features), name="beta", requires_grad=affine)
        self.sqrt = ops.Sqrt()
        self.eps = Tensor(np.array([eps]), mindspore.float32)
        self.cast = ops.Cast()
    def construct(self, x):
        """calculate InstanceNorm output"""
        mean = ops.ReduceMean(keep_dims=True)(x, (2, 3))
        mean = self.cast(mean, mindspore.float32)
        tmp = x - mean
        tmp = tmp * tmp
        var = ops.ReduceMean(keep_dims=True)(tmp, (2, 3))
        std = self.sqrt(var+ self.eps)
        gamma_t = self.cast(self.gamma, mindspore.float32)
        beta_t = self.cast(self.beta, mindspore.float32)
        x = (x - mean) / std * gamma_t.reshape(1, -1, 1, 1) + beta_t.reshape(1, -1, 1, 1)
        return x
