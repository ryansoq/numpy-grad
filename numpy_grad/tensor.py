"""Tensor — array-level autograd node wrapping np.ndarray."""

from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum grad along broadcasted axes so it matches `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "_parents", "_backward", "requires_grad")

    def __init__(self, data, requires_grad: bool = False, _parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self._parents = _parents
        self._backward = lambda: None
        self.requires_grad = requires_grad or any(
            p.requires_grad for p in _parents
        )

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def backward(self, grad: np.ndarray | None = None):
        if grad is None:
            assert self.data.size == 1, "scalar-only when grad is None"
            grad = np.ones_like(self.data)
        self.grad = grad

        topo, visited = [], set()

        def build(t: Tensor):
            if id(t) in visited:
                return
            visited.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        self.grad = None
        for p in self._parents:
            p.zero_grad()

    # --- arithmetic ---
    def __add__(self, other):
        from .ops import add
        return add(self, _wrap(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from .ops import sub
        return sub(self, _wrap(other))

    def __rsub__(self, other):
        from .ops import sub
        return sub(_wrap(other), self)

    def __mul__(self, other):
        from .ops import mul
        return mul(self, _wrap(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from .ops import div
        return div(self, _wrap(other))

    def __rtruediv__(self, other):
        from .ops import div
        return div(_wrap(other), self)

    def __neg__(self):
        from .ops import neg
        return neg(self)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, _wrap(other))

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only scalar exponent"
        from .ops import pow_scalar
        return pow_scalar(self, p)

    # --- shape ---
    def reshape(self, *shape):
        from .ops import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def transpose(self, *axes):
        from .ops import transpose
        if not axes:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return transpose(self, axes)

    @property
    def T(self):
        return self.transpose()

    def sum(self, axis=None, keepdims=False):
        from .ops import sum_op
        return sum_op(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        from .ops import mean_op
        return mean_op(self, axis, keepdims)

    def relu(self):
        from .ops import relu
        return relu(self)

    def exp(self):
        from .ops import exp
        return exp(self)

    def log(self):
        from .ops import log
        return log(self)


def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x)
