# numpy-grad

Array-level reverse-mode autograd in pure NumPy.

Same `Tensor` API as cpp-grad / micrograd, but each node wraps an `np.ndarray`
instead of a single float — so matmul runs on BLAS at full speed and the
Python overhead is paid per **op**, not per **element**.

## Why

Scalar-level autograd (cpp-grad, micrograd) is great for learning the algorithm
but dies past ~1k parameters because every `Value * Value` is a Python call.
Array-level autograd amortizes that overhead across the whole tensor — the
exact same trick PyTorch and JAX use in eager mode.

|                   | scalar (cpp-grad) | array (numpy-grad) |
|-------------------|-------------------|--------------------|
| 768×768 matmul    | ~590k graph nodes | 1 node             |
| Backend           | Python loop       | BLAS (OpenBLAS/MKL)|
| Realistic NN      | infeasible        | works              |

## Quickstart

```python
import numpy as np
from numpy_grad import Tensor
from numpy_grad.nn import Linear, Sequential, ReLU, SGD, mse_loss

# scalars work like micrograd — same expression syntax
a = Tensor(-4.0, requires_grad=True)
b = Tensor(2.0, requires_grad=True)
g = ((a + b) ** 2 + (a * b + b ** 3).relu()).sum()
g.backward()
print(a.grad, b.grad)

# arrays work via BLAS
model = Sequential(Linear(1, 16), ReLU(), Linear(16, 1))
opt = SGD(model.parameters(), lr=0.05)
x = Tensor(np.random.randn(100, 1))
y = Tensor(np.sin(x.data))
for _ in range(500):
    opt.zero_grad()
    loss = mse_loss(model(x), y)
    loss.backward()
    opt.step()
```

## Primitive ops

12 primitives, every layer composes from these:

`add` `sub` `mul` `div` `neg` `pow_scalar` `matmul` `sum` `mean` `exp` `log`
`relu` `transpose` `reshape`

Each registers a `_backward` closure during forward. `Tensor.backward()` walks
the topo-sorted graph in reverse and accumulates `grad`. Broadcast dimensions
are auto-reduced via `_unbroadcast`.

## Tests

```bash
python3 -m pytest tests/ -v
```

24 tests covering:
- per-op gradcheck vs central-difference numerical derivative
- micrograd canonical convergence (g ≈ 24.7041)
- end-to-end MLP fit on `y = sin(x)`

## Authors

Ryan & Nami ✨

## License

MIT
