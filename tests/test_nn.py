"""Tiny MLP training test — fits y = sin(x) on 100 points."""

import numpy as np
import pytest

from numpy_grad import Tensor
from numpy_grad.nn import Linear, Sequential, ReLU, SGD, mse_loss


def test_mlp_fits_sine():
    rng = np.random.default_rng(42)
    x_np = rng.uniform(-3.0, 3.0, size=(100, 1))
    y_np = np.sin(x_np)

    model = Sequential(Linear(1, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 1))
    opt = SGD(model.parameters(), lr=0.05)

    x = Tensor(x_np)
    y = Tensor(y_np)

    losses = []
    for _ in range(500):
        opt.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    # initial loss ~0.5 (variance of sin), final should be < 0.05 if fit works
    assert losses[-1] < 0.05, f"final loss {losses[-1]:.4f} too high"
    assert losses[-1] < losses[0] * 0.2, "did not converge meaningfully"


def test_parameter_collection():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    params = model.parameters()
    # 2 Linear x (W + b) = 4 params
    assert len(params) == 4
    shapes = sorted([p.shape for p in params])
    assert shapes == [(2,), (4, 8), (8,), (8, 2)]
