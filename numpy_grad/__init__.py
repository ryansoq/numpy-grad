"""numpy-grad — array-level autograd in pure NumPy.

A reverse-mode autograd engine where each node wraps an `np.ndarray`. Same
expression API as cpp-grad / micrograd, but ops run on whole arrays so BLAS
handles matmul at full speed.

Quickstart:

    import numpy as np
    from numpy_grad import Tensor

    x = Tensor(np.random.randn(8, 4))
    W = Tensor(np.random.randn(4, 2), requires_grad=True)
    y = (x @ W).relu().sum()
    y.backward()
    print(W.grad.shape)  # (4, 2)
"""

from .tensor import Tensor
from . import nn

__all__ = ["Tensor", "nn"]
__version__ = "0.1.0"
