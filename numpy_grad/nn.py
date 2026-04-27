"""Composable layers — built from primitives in ops.py.

The point: ANY layer (Linear, MLP, LayerNorm, attention) is just a sequence
of primitive ops. The autograd engine doesn't know about layers — it only
sees the expression graph.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor
from . import ops


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


def cross_entropy(logits: Tensor, targets) -> Tensor:
    return ops.cross_entropy_loss(logits, targets)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class SwiGLU(Module):
    """SwiGLU FFN (Shazeer 2020): SiLU(xW1) ⊙ (xW_gate) @ W2.

    Uses bias-free Linear layers. Per Shazeer's recommendation, the inner
    dim is 2/3 of d_ff to match parameter count of a 2-matrix GELU FFN.
    """

    def __init__(self, d_model: int, d_ff: int):
        d_inner = max(int(d_ff * 2 // 3), 1)
        self.w1 = Linear(d_model, d_inner, bias=False)
        self.gate = Linear(d_model, d_inner, bias=False)
        self.w2 = Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.gate(x))


def clip_grad_norm_(params, max_norm: float) -> float:
    """Global gradient clipping (in place). Returns the pre-clip total norm."""
    grads = [p.grad for p in params if p.grad is not None]
    total = float(np.sqrt(sum((g * g).sum() for g in grads)))
    if total > max_norm:
        scale = max_norm / (total + 1e-8)
        for g in grads:
            g *= scale
    return total


class Embedding(Module):
    """Token embedding table. Forward: gather rows by integer indices."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        scale = np.sqrt(2.0 / (num_embeddings + embedding_dim))
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * scale,
            requires_grad=True,
        )

    def forward(self, indices) -> Tensor:
        return ops.embedding(self.weight, indices)


class LayerNorm(Module):
    """LayerNorm over the last dim. y = gamma * (x - mean) / sqrt(var + eps) + beta"""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered * centered).mean(axis=-1, keepdims=True)
        normed = centered / (var + self.eps).sqrt()
        return normed * self.gamma + self.beta


class MultiHeadAttention(Module):
    """Causal multi-head self-attention via three separate projections.

    Three Linear layers (Wq/Wk/Wv) instead of one fused QKV — keeps the
    reshape graph simple, costs one extra matmul that BLAS handles in a blink.
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim

        # (B, T, D) -> (B, H, T, hd)
        def split_heads(t: Tensor) -> Tensor:
            return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

        q = split_heads(self.Wq(x))
        k = split_heads(self.Wk(x))
        v = split_heads(self.Wv(x))

        # scores: (B, H, T, T)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(hd))

        # causal mask: True where future
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = scores.masked_fill(mask, -1e9)

        attn = scores.softmax(axis=-1)
        out = attn @ v                                         # (B, H, T, hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.Wo(out)


class TransformerBlock(Module):
    """Pre-norm Transformer block: x + MHA(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None):
        d_ff = d_ff or 4 * d_model
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = Sequential(Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class AdamW:
    """AdamW with decoupled weight decay (Loshchilov 2019)."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.wd:
                p.data -= self.lr * self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
