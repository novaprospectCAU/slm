"""Tiny MLP demo using the minimal autograd Tensor engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow direct execution: python demos/demo_mlp.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.tensor import Tensor


def main() -> None:
    rng = np.random.default_rng(42)

    n = 128
    x_np = rng.normal(0, 1, size=(n, 2))
    y_np = (x_np[:, 0:1] * x_np[:, 1:2] > 0).astype(np.float64)

    x = Tensor(x_np, requires_grad=False)
    y = Tensor(y_np, requires_grad=False)

    w1 = Tensor(rng.normal(0, 0.2, size=(2, 16)), requires_grad=True)
    b1 = Tensor(np.zeros((1, 16)), requires_grad=True)
    w2 = Tensor(rng.normal(0, 0.2, size=(16, 1)), requires_grad=True)
    b2 = Tensor(np.zeros((1, 1)), requires_grad=True)
    params = [w1, b1, w2, b2]

    lr = 0.1
    steps = 400
    for step in range(steps):
        h = x.matmul(w1).add(b1).relu()
        pred = h.matmul(w2).add(b2)
        diff = pred.add(y.mul(-1.0))
        loss = diff.mul(diff).mean()

        for p in params:
            p.grad = None
        loss.backward()
        for p in params:
            if p.grad is None:
                continue
            p.data = p.data - lr * p.grad

        if step % 100 == 0 or step == steps - 1:
            print(f"step={step:03d} loss={float(loss.data):.6f}")


if __name__ == "__main__":
    main()
