"""Tiny MLP fits y = sin(x). Run: python3 examples/sine_fit.py"""

import numpy as np

from numpy_grad import Tensor
from numpy_grad.nn import Linear, Sequential, ReLU, SGD, mse_loss


def main():
    rng = np.random.default_rng(42)
    x_np = rng.uniform(-3.0, 3.0, size=(200, 1))
    y_np = np.sin(x_np)

    model = Sequential(Linear(1, 32), ReLU(), Linear(32, 32), ReLU(), Linear(32, 1))
    opt = SGD(model.parameters(), lr=0.05)

    x = Tensor(x_np)
    y = Tensor(y_np)

    for epoch in range(1000):
        opt.zero_grad()
        loss = mse_loss(model(x), y)
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d} | loss {float(loss.data):.6f}")

    print(f"final loss {float(loss.data):.6f}")


if __name__ == "__main__":
    main()
