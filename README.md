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

## How autograd works (5-minute tour)

Three concepts; the rest is bookkeeping.

### 1. A `Tensor` is a node in a graph

When you write `c = a * b`, you don't just compute `c` — you also record:
- *which* tensors fed `c` (its **parents**)
- *how* to push gradient back into them (its **`_backward` closure**)

So the user-facing expression `loss = ((x @ W).relu().sum() - y) ** 2`
quietly builds a DAG behind the scenes:

```
x        W
 \      /
  matmul                      ← Tensor: out=x@W,   _backward = "use W.T / x.T"
   |
  relu                        ← Tensor: out=max(0,·), _backward = "mask"
   |
  sum                         ← Tensor: out=Σ,    _backward = "broadcast 1s"
   |       y
    \     /
     sub                      ← Tensor: out=a-b,  _backward = "+grad / -grad"
      |
     **2                      ← Tensor: out=a²,   _backward = "2a · grad"
      |
     loss
```

### 2. Forward = build graph, Backward = walk it in reverse

`loss.backward()` does three things:
1. Seed `loss.grad = ones_like(loss.data)` — the output gradient is 1.0
2. Topologically sort all ancestors of `loss` (each Tensor visited once by id)
3. In reverse topo order, call each Tensor's `_backward()` closure — the
   closure pushes its contribution into its parents' `.grad`

By the time the walk hits `x` and `W`, their `.grad` slots have been
accumulated by every chain that flows through them.

### 3. Optimiser is a separate pass over `.grad`

`optimiser.step()` doesn't know anything about the graph. It just walks
the parameter list and mutates `p.data` using `p.grad` (with momentum,
weight decay, learning rate, etc.). Then `optimiser.zero_grad()` clears
`.grad` slots so next iteration starts fresh.

This three-way decoupling — **graph build → graph walk → parameter
update** — is exactly how PyTorch eager mode works internally. Reading
this codebase is reading a 200-line version of that.

## Program flow (text)

```
Tensor(data, requires_grad=True)         // wraps an np.ndarray + .grad slot
  |
  | __add__ / __mul__ / @ / .relu() ...   // user expression triggers ops
  v
op(parents...) in numpy_grad/ops.py
  |
  |- compute forward via numpy            // out_data = a.data @ b.data
  |- create child Tensor(out_data,        // result also a Tensor
  |       _parents=(a, b))                // remembers who fed it
  |- attach _backward closure to child:   // closure captures grads-to-accumulate
  |      def _bw():
  |          _accum(a, out.grad @ b.data.T)
  |          _accum(b, a.data.T @ out.grad)
  |      child._backward = _bw
  v
Repeat for every op → expression graph builds DAG of Tensors

# === when user calls loss.backward() ===

loss.backward(grad=None)
  |
  |- self.grad = ones_like(loss.data)     // seed the chain
  |- topo sort all ancestors via DFS      // each Tensor visited once by id
  |- for t in reversed(topo):             // unwind in reverse
  |      t._backward()                    // closure pushes grad into parents
  v
Every parameter Tensor now has p.grad populated.
Constants (no parents, requires_grad=False) get grad too but it's ignored.

# === optimiser is decoupled ===

optimiser.step()                          // AdamW / SGD in numpy_grad/nn.py
  |- for p in params:
  |      apply momentum/decay/lr to p.data using p.grad
  |- (does NOT touch the expression graph; pure parameter mutation)

optimiser.zero_grad()                     // clear .grad slots before next step
```

## Primitive ops

14 primitives, every layer composes from these:

`add` `sub` `mul` `div` `neg` `pow_scalar` `matmul` `sum` `mean` `exp` `log`
`sqrt` `gelu` `silu` `softmax` `relu` `embedding` `cross_entropy_loss`
`masked_fill` `transpose` `reshape`

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
