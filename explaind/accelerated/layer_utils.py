from torch import Tensor, Size
from torch import nn
from typing import Union

import numbers

import torch
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter

from copy import deepcopy


__all__ = [
    "DropoutWHistory",
]

TESTING = False

_shape_t = Union[int, list[int], Size]

class _DropoutNdWHistory(nn.Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.inplace = inplace
        self.mask = None

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


class DropoutWHistory(_DropoutNdWHistory):
    r"""During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`.

    The zeroed elements are chosen independently for each forward call and are sampled from a Bernoulli distribution.

    Each channel will be zeroed out independently on every forward call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input: Tensor) -> Tensor:
        # todo: incomplete batch partitions need to be handled with correct masks 
        # through batch indexing
        # if self.training and self.mask is None:
        #     out = F.dropout(input, self.p, self.training, self.inplace)
        # elif self.training and self.mask is not None:
        #     out = input * self.mask
        # else:  # self.eval
        #     out = input * (1 - self.p)

        # # reverse engineer mask from entries that are zero in out but not in input
        # if self.mask is None and self.training:
        #     self.mask = ((input == 0) | (out != 0)).to(input.device)

        # ignore Dropout for now
        return input * (1 - self.p)
        # return out


class LayerNormWHistory(nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    .. image:: ../_static/img/nn/layer_norm.jpg
        :scale: 50 %

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_param=None,
        bias_param=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.mean_hist = None
        self.var_hist = None
        if self.elementwise_affine:
            if weight_param is not None:
                self.weight = deepcopy(weight_param)
            else:
                self.weight = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            if bias:
                if bias_param is not None:
                    self.bias = deepcopy(bias_param)
                else:
                    self.bias = Parameter(
                        torch.empty(self.normalized_shape, **factory_kwargs)
                    )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # if self.mean_hist is None:
        #     self.mean_hist = input.mean(dim=[i for i in range(-len(self.normalized_shape), 0)], keepdim=True)
        #     self.mean_hist = self.mean_hist.detach()
        #     self.mean_hist.requires_grad = False # solve this with.train vs .eval later
        # if self.var_hist is None:
        #     self.var_hist = input.var(dim=[i for i in range(-len(self.normalized_shape), 0)], keepdim=True, unbiased=False)
        #     self.var_hist = self.var_hist.detach()
        #     self.var_hist.requires_grad = False
        
        # input = (input - self.mean_hist) / (self.var_hist + self.eps).sqrt()

        m = input.mean(dim=[i for i in range(-len(self.normalized_shape), 0)], keepdim=True)
        v = input.var(dim=[i for i in range(-len(self.normalized_shape), 0)], keepdim=True, unbiased=False)
        input = (input - m) / (v + self.eps).sqrt()


        if self.elementwise_affine:
            if not self.weight.device == input.device:
                self.weight = self.weight.to(input.device)
            if not self.bias.device == input.device:
                self.bias = self.bias.to(input.device)
                
            input = input * self.weight + self.bias

        # if TESTING:
        #     l = F.layer_norm(
        #         input1, self.normalized_shape, self.weight, self.bias, self.eps
        #     )
        #     assert torch.allclose(l, input, atol=1e-6)

        return input

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )

