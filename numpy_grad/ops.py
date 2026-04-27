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


def sqrt(a: Tensor) -> Tensor:
    s = np.sqrt(a.data)
    out = Tensor(s, _parents=(a,))

    def _bw():
        _accum(a, out.grad * 0.5 / s)

    out._backward = _bw
    return out


def gelu(a: Tensor) -> Tensor:
    """GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    x = a.data
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    out = Tensor(0.5 * x * (1 + t), _parents=(a,))

    def _bw():
        # dGELU/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*c*(1 + 0.134145*x^2)
        d_inner = c * (1.0 + 0.134145 * x ** 2)
        d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * d_inner
        _accum(a, out.grad * d)

    out._backward = _bw
    return out


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax: subtract max before exp."""
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s, _parents=(a,))

    def _bw():
        # softmax Jacobian-vector product: dx = s * (g - sum(g * s, axis))
        g = out.grad
        sum_term = (g * s).sum(axis=axis, keepdims=True)
        _accum(a, s * (g - sum_term))

    out._backward = _bw
    return out


def embedding(table: Tensor, indices: np.ndarray) -> Tensor:
    """Gather rows of `table` by `indices`. Backward scatter-adds."""
    idx = np.asarray(indices, dtype=np.int64)
    out = Tensor(table.data[idx], _parents=(table,))

    def _bw():
        g = np.zeros_like(table.data)
        np.add.at(g, idx, out.grad)
        _accum(table, g)

    out._backward = _bw
    return out


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy across the leading dim. logits: (..., V), targets: (...,) ints.

    Computed as logsumexp - logits[target] in a numerically stable way, then averaged.
    """
    targets = np.asarray(targets, dtype=np.int64)
    flat_logits = logits.data.reshape(-1, logits.data.shape[-1])
    flat_targets = targets.reshape(-1)
    N, V = flat_logits.shape

    shifted = flat_logits - flat_logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss_val = -log_probs[np.arange(N), flat_targets].mean()

    out = Tensor(loss_val, _parents=(logits,))

    # cache softmax probs for backward
    probs = np.exp(log_probs)

    def _bw():
        g = probs.copy()
        g[np.arange(N), flat_targets] -= 1.0
        g = g / N * out.grad
        _accum(logits, g.reshape(logits.shape))

    out._backward = _bw
    return out


def masked_fill(a: Tensor, mask: np.ndarray, value: float) -> Tensor:
    """Replace positions where mask==True with `value`. Grad zeroes there."""
    mask = np.asarray(mask, dtype=bool)
    out_data = np.where(mask, value, a.data)
    out = Tensor(out_data, _parents=(a,))

    def _bw():
        _accum(a, np.where(mask, 0.0, out.grad))

    out._backward = _bw
    return out
