import numpy as np

from src.nn import Linear
from src.tensor import Tensor


def finite_difference_grad(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = fn(x.copy())
        x[idx] = orig - eps
        minus = fn(x.copy())
        x[idx] = orig
        grad[idx] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad


def test_linear_forward_shape() -> None:
    layer = Linear(3, 2, seed=0)
    x = Tensor(np.zeros((5, 3)))
    y = layer(x)
    assert y.shape == (5, 2)


def test_linear_gradcheck_weight_and_bias() -> None:
    rng = np.random.default_rng(10)
    x_np = rng.normal(size=(4, 3))
    w_np = rng.normal(size=(3, 2))
    b_np = rng.normal(size=(1, 2))

    x = Tensor(x_np, requires_grad=False)
    layer = Linear(3, 2, seed=0)
    layer.weight.data = w_np.copy()
    layer.bias.data = b_np.copy()

    out = layer(x).sum()
    out.backward()

    num_dw = finite_difference_grad(lambda w: float((x_np @ w + b_np).sum()), w_np.copy())
    num_db = finite_difference_grad(lambda b: float((x_np @ w_np + b).sum()), b_np.copy())
    np.testing.assert_allclose(layer.weight.grad, num_dw, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(layer.bias.grad, num_db, rtol=1e-5, atol=1e-5)


def test_linear_training_step_reduces_loss() -> None:
    rng = np.random.default_rng(99)
    n = 64
    in_features = 3
    out_features = 1

    x_np = rng.normal(size=(n, in_features))
    true_w = np.array([[2.0], [-1.0], [0.5]])
    true_b = np.array([[0.3]])
    y_np = x_np @ true_w + true_b

    x = Tensor(x_np, requires_grad=False)
    y = Tensor(y_np, requires_grad=False)
    layer = Linear(in_features, out_features, seed=1)

    def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
        diff = pred.add(target.mul(-1.0))
        return diff.mul(diff).mean()

    loss0 = float(mse_loss(layer(x), y).data)
    lr = 0.1
    for _ in range(120):
        pred = layer(x)
        loss = mse_loss(pred, y)

        for p in layer.parameters():
            p.grad = None
        loss.backward()
        for p in layer.parameters():
            p.data = p.data - lr * p.grad

    loss1 = float(mse_loss(layer(x), y).data)
    assert loss1 < loss0
