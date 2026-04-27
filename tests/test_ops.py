"""Per-op gradcheck — analytical grad must match central-difference numerical."""

import numpy as np
import pytest

from numpy_grad import Tensor


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference grad of scalar f(x) w.r.t. x."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f(x)
        x[idx] = orig - eps
        minus = f(x)
        x[idx] = orig
        g[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return g


def gradcheck(make_expr, *shapes, atol=1e-5, seed=0):
    """make_expr(*ndarrays) -> Tensor scalar. Verifies grad via central diff."""
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(s) for s in shapes]

    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = make_expr(*tensors)
    if out.data.ndim != 0:
        out = out.sum()
    out.backward()
    analytic = [t.grad.copy() for t in tensors]

    for i, a in enumerate(arrays):
        def f(xi, i=i):
            ts = [Tensor(arr.copy()) for arr in arrays]
            ts[i] = Tensor(xi.copy())
            r = make_expr(*ts)
            return r.data.sum()

        numeric = numerical_grad(f, a.copy())
        np.testing.assert_allclose(
            analytic[i], numeric, atol=atol,
            err_msg=f"grad mismatch on input {i}",
        )


def test_add():
    gradcheck(lambda a, b: a + b, (3, 4), (3, 4))


def test_add_broadcast():
    gradcheck(lambda a, b: a + b, (3, 4), (4,))


def test_sub():
    gradcheck(lambda a, b: a - b, (3, 4), (3, 4))


def test_mul():
    gradcheck(lambda a, b: a * b, (3, 4), (3, 4))


def test_mul_broadcast():
    gradcheck(lambda a, b: a * b, (3, 4), (1, 4))


def test_div():
    gradcheck(lambda a, b: a / b, (3, 4), (3, 4))


def test_neg():
    gradcheck(lambda a: -a, (3, 4))


def test_pow_scalar():
    gradcheck(lambda a: a ** 3, (2, 3))


def test_matmul_2d():
    gradcheck(lambda a, b: a @ b, (3, 4), (4, 5))


def test_matmul_batched():
    gradcheck(lambda a, b: a @ b, (2, 3, 4), (2, 4, 5))


def test_sum_all():
    gradcheck(lambda a: a.sum(), (3, 4))


def test_sum_axis():
    gradcheck(lambda a: a.sum(axis=0), (3, 4))


def test_sum_axis_keepdims():
    gradcheck(lambda a: a.sum(axis=1, keepdims=True), (3, 4))


def test_mean():
    gradcheck(lambda a: a.mean(), (3, 4))


def test_mean_axis():
    gradcheck(lambda a: a.mean(axis=0), (3, 4))


def test_exp():
    gradcheck(lambda a: a.exp(), (3, 4))


def test_log():
    # positive-only: shift to ensure data > 0
    gradcheck(lambda a: (a + 5.0).log(), (3, 4))


def test_relu():
    # gradcheck near 0 is noisy; test on a positive-shifted input
    gradcheck(lambda a: (a + 0.5).relu().sum() + (a - 0.5).relu().sum(),
              (3, 4), atol=1e-4)


def test_transpose():
    gradcheck(lambda a: a.T, (3, 4))


def test_reshape():
    gradcheck(lambda a: a.reshape(6, 2), (3, 4))


def test_chain():
    """Compound: y = ((x @ W).relu() + b).sum()"""
    def f(x, W, b):
        return ((x @ W).relu() + b).sum()
    gradcheck(f, (3, 4), (4, 5), (5,))
