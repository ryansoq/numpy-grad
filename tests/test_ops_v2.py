"""Phase 2 ops — sqrt/gelu/softmax/embedding/cross_entropy/masked_fill."""

import numpy as np

from numpy_grad import Tensor
from numpy_grad.ops import (
    cross_entropy_loss,
    embedding,
    masked_fill,
)
from tests.test_ops import gradcheck


def test_sqrt():
    gradcheck(lambda a: (a + 5.0).sqrt(), (3, 4))


def test_gelu():
    gradcheck(lambda a: a.gelu(), (3, 4), atol=1e-4)


def test_silu():
    gradcheck(lambda a: a.silu(), (3, 4), atol=1e-5)


def test_softmax_last():
    gradcheck(lambda a: a.softmax(axis=-1), (3, 4))


def test_softmax_axis0():
    gradcheck(lambda a: a.softmax(axis=0), (3, 4))


def test_embedding():
    rng = np.random.default_rng(0)
    table = Tensor(rng.standard_normal((10, 4)), requires_grad=True)
    idx = np.array([2, 5, 7, 5])
    out = embedding(table, idx)
    assert out.shape == (4, 4)
    out.sum().backward()
    # row 5 was indexed twice → grad row 5 should be 2 * ones
    assert np.allclose(table.grad[5], np.full(4, 2.0))
    assert np.allclose(table.grad[2], np.ones(4))
    assert np.allclose(table.grad[0], np.zeros(4))


def test_cross_entropy():
    rng = np.random.default_rng(0)
    logits = Tensor(rng.standard_normal((6, 5)), requires_grad=True)
    targets = np.array([0, 4, 2, 1, 3, 0])

    loss = cross_entropy_loss(logits, targets)
    loss.backward()

    # numerical check on a single logit
    eps = 1e-6
    L0 = float(loss.data)
    logits.data[0, 0] += eps
    L_plus = float(cross_entropy_loss(Tensor(logits.data.copy()), targets).data)
    logits.data[0, 0] -= 2 * eps
    L_minus = float(cross_entropy_loss(Tensor(logits.data.copy()), targets).data)
    logits.data[0, 0] += eps  # restore
    numeric = (L_plus - L_minus) / (2 * eps)
    assert abs(logits.grad[0, 0] - numeric) < 1e-5


def test_masked_fill():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    mask = np.array([[True, False], [False, True]])
    out = masked_fill(a, mask, -100.0)
    assert np.allclose(out.data, [[-100.0, 2.0], [3.0, -100.0]])
    out.sum().backward()
    # grad zero where masked, one where not
    assert np.allclose(a.grad, [[0.0, 1.0], [1.0, 0.0]])
