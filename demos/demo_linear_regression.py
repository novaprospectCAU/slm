"""Linear regression demo with custom Tensor + Linear layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.nn import Linear
from src.tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred.add(target.mul(-1.0))
    return diff.mul(diff).mean()


def main() -> None:
    rng = np.random.default_rng(123)

    n = 128
    x_np = rng.normal(size=(n, 3))
    true_w = np.array([[1.5], [-2.0], [0.7]])
    true_b = np.array([[0.25]])
    y_np = x_np @ true_w + true_b

    x = Tensor(x_np, requires_grad=False)
    y = Tensor(y_np, requires_grad=False)
    model = Linear(3, 1, seed=0)

    lr = 0.1
    steps = 200
    for step in range(steps):
        pred = model(x)
        loss = mse_loss(pred, y)

        for p in model.parameters():
            p.grad = None
        loss.backward()

        for p in model.parameters():
            p.data = p.data - lr * p.grad

        if step % 50 == 0 or step == steps - 1:
            print(f"step={step:03d} loss={float(loss.data):.6f}")


if __name__ == "__main__":
    main()

