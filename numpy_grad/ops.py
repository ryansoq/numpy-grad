"""Primitive ops — array-level forward + backward.

Each op:
  1. Computes forward via numpy.
  2. Builds a child Tensor with parents.
  3. Closes a `_backward()` over parents that, when called, accumulates each
     parent's `.grad` from `out.grad`.

Broadcast handling: every gradient destined for a parent is reduced via
`_unbroadcast` to match the parent's original shape.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, _unbroadcast


def _accum(t: Tensor, g: np.ndarray):
    g = _unbroadcast(g, t.shape)
    t.grad = g if t.grad is None else t.grad + g


def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data, _parents=(a, b))

    def _bw():
        _accum(a, out.grad)
        _accum(b, out.grad)

    out._backward = _bw
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data - b.data, _parents=(a, b))

    def _bw():
        _accum(a, out.grad)
        _accum(b, -out.grad)

    out._backward = _bw
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data * b.data, _parents=(a, b))

    def _bw():
        _accum(a, out.grad * b.data)
        _accum(b, out.grad * a.data)

    out._backward = _bw
    return out


def div(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data / b.data, _parents=(a, b))

    def _bw():
        _accum(a, out.grad / b.data)
        _accum(b, -out.grad * a.data / (b.data ** 2))

    out._backward = _bw
    return out


def neg(a: Tensor) -> Tensor:
    out = Tensor(-a.data, _parents=(a,))

    def _bw():
        _accum(a, -out.grad)

    out._backward = _bw
    return out


def pow_scalar(a: Tensor, p: float) -> Tensor:
    out = Tensor(a.data ** p, _parents=(a,))

    def _bw():
        _accum(a, out.grad * p * a.data ** (p - 1))

    out._backward = _bw
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data @ b.data, _parents=(a, b))

    def _bw():
        # supports 2D @ 2D and batched (..., M, K) @ (..., K, N)
        if a.data.ndim >= 2 and b.data.ndim >= 2:
            _accum(a, out.grad @ np.swapaxes(b.data, -1, -2))
            _accum(b, np.swapaxes(a.data, -1, -2) @ out.grad)
        elif a.data.ndim == 1 and b.data.ndim == 1:
            _accum(a, out.grad * b.data)
            _accum(b, out.grad * a.data)
        elif a.data.ndim == 1:  # (K,) @ (K, N) -> (N,)
            _accum(a, out.grad @ b.data.T)
            _accum(b, np.outer(a.data, out.grad))
        else:  # (M, K) @ (K,) -> (M,)
            _accum(a, np.outer(out.grad, b.data))
            _accum(b, a.data.T @ out.grad)

    out._backward = _bw
    return out


def sum_op(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), _parents=(a,))

    def _bw():
        g = out.grad
        if axis is not None and not keepdims:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            for ax in sorted([ax % a.ndim for ax in axes]):
                g = np.expand_dims(g, axis=ax)
        _accum(a, np.broadcast_to(g, a.shape).copy())

    out._backward = _bw
    return out


def mean_op(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    out = Tensor(a.data.mean(axis=axis, keepdims=keepdims), _parents=(a,))
    if axis is None:
        n = a.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        n = int(np.prod([a.shape[ax] for ax in axes]))

    def _bw():
        g = out.grad
        if axis is not None and not keepdims:
            axes_local = (axis,) if isinstance(axis, int) else tuple(axis)
            for ax in sorted([ax % a.ndim for ax in axes_local]):
                g = np.expand_dims(g, axis=ax)
        _accum(a, np.broadcast_to(g, a.shape).copy() / n)

    out._backward = _bw
    return out


def exp(a: Tensor) -> Tensor:
    e = np.exp(a.data)
    out = Tensor(e, _parents=(a,))

    def _bw():
        _accum(a, out.grad * e)

    out._backward = _bw
    return out


def log(a: Tensor) -> Tensor:
    out = Tensor(np.log(a.data), _parents=(a,))

    def _bw():
        _accum(a, out.grad / a.data)

    out._backward = _bw
    return out


def relu(a: Tensor) -> Tensor:
    mask = a.data > 0
    out = Tensor(a.data * mask, _parents=(a,))

    def _bw():
        _accum(a, out.grad * mask)

    out._backward = _bw
    return out


def transpose(a: Tensor, axes=None) -> Tensor:
    out = Tensor(a.data.transpose(axes) if axes else a.data.T,
                 _parents=(a,))

    def _bw():
        if axes is None:
            _accum(a, out.grad.T)
        else:
            inv = np.argsort(axes)
            _accum(a, out.grad.transpose(inv))

    out._backward = _bw
    return out


def reshape(a: Tensor, shape) -> Tensor:
    out = Tensor(a.data.reshape(shape), _parents=(a,))

    def _bw():
        _accum(a, out.grad.reshape(a.shape))

    out._backward = _bw
    return out
