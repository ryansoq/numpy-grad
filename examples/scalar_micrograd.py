"""micrograd canonical example — scalar Tensors through the array engine.

Reproduces karpathy/micrograd README result: g ≈ 24.7041
"""

from numpy_grad import Tensor


def main():
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

    print(f"g = {float(g.data):.4f}  (expected 24.7041)")
    g.backward()
    print(f"a.grad = {float(a.grad):.4f}  (expected 138.8338)")
    print(f"b.grad = {float(b.grad):.4f}  (expected 645.5773)")


if __name__ == "__main__":
    main()
