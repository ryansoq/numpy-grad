"""micrograd convergence test — same scalar expression as cpp-grad/rust-grad.

Source: https://github.com/karpathy/micrograd README
Expected: g.data == 24.7041 (within fp tolerance)

Demonstrates that scalar (0-dim Tensor) is just a degenerate array — the same
engine handles it without any special code path.
"""

import pytest

from numpy_grad import Tensor


def test_micrograd_canonical():
    a = Tensor(-4.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    assert abs(float(g.data) - 24.7041) < 1e-4
    g.backward()
    # gradients from karpathy/micrograd README
    assert abs(float(a.grad) - 138.8338) < 1e-4
    assert abs(float(b.grad) - 645.5773) < 1e-4
