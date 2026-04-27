"""Composable layers — built from primitives in ops.py.

The point: ANY layer (Linear, MLP, LayerNorm, attention) is just a sequence
of primitive ops. The autograd engine doesn't know about layers — it only
sees the expression graph.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Module:
    """Base class — collects parameters via __dict__ walking."""

    def parameters(self):
        seen, out = set(), []

        def walk(obj):
            if isinstance(obj, Tensor) and obj.requires_grad and id(obj) not in seen:
                seen.add(id(obj))
                out.append(obj)
            elif isinstance(obj, Module):
                for v in obj.__dict__.values():
                    walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)

        walk(self)
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    """y = x @ W + b. He-style init."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        scale = np.sqrt(2.0 / in_dim)
        self.W = Tensor(
            np.random.randn(in_dim, out_dim) * scale, requires_grad=True
        )
        self.b = Tensor(np.zeros(out_dim), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        return out + self.b if self.b is not None else out


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x) if callable(layer) else layer.forward(x)
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class SGD:
    def __init__(self, params, lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    return (diff * diff).mean()
