import numpy as np

from src.tensor.tensor import Tensor


def finite_difference_grad(
    fn,
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
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


def test_gradcheck_add() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4))
    y = rng.normal(size=(3, 4))

    tx = Tensor(x.copy(), requires_grad=True)
    ty = Tensor(y.copy(), requires_grad=True)
    out = tx.add(ty).sum()
    out.backward()

    num_dx = finite_difference_grad(lambda z: float((z + y).sum()), x.copy())
    num_dy = finite_difference_grad(lambda z: float((x + z).sum()), y.copy())
    np.testing.assert_allclose(tx.grad, num_dx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ty.grad, num_dy, rtol=1e-5, atol=1e-5)


def test_gradcheck_mul() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 5))
    y = rng.normal(size=(2, 5))

    tx = Tensor(x.copy(), requires_grad=True)
    ty = Tensor(y.copy(), requires_grad=True)
    out = tx.mul(ty).sum()
    out.backward()

    num_dx = finite_difference_grad(lambda z: float((z * y).sum()), x.copy())
    num_dy = finite_difference_grad(lambda z: float((x * z).sum()), y.copy())
    np.testing.assert_allclose(tx.grad, num_dx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ty.grad, num_dy, rtol=1e-5, atol=1e-5)


def test_gradcheck_matmul() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3))
    y = rng.normal(size=(3, 4))

    tx = Tensor(x.copy(), requires_grad=True)
    ty = Tensor(y.copy(), requires_grad=True)
    out = tx.matmul(ty).sum()
    out.backward()

    num_dx = finite_difference_grad(lambda z: float((z @ y).sum()), x.copy())
    num_dy = finite_difference_grad(lambda z: float((x @ z).sum()), y.copy())
    np.testing.assert_allclose(tx.grad, num_dx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ty.grad, num_dy, rtol=1e-5, atol=1e-5)


def test_gradcheck_sum() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 3))

    tx = Tensor(x.copy(), requires_grad=True)
    out = tx.sum()
    out.backward()

    num_dx = finite_difference_grad(lambda z: float(z.sum()), x.copy())
    np.testing.assert_allclose(tx.grad, num_dx, rtol=1e-5, atol=1e-5)


def test_gradcheck_mean() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 3))

    tx = Tensor(x.copy(), requires_grad=True)
    out = tx.mean()
    out.backward()

    num_dx = finite_difference_grad(lambda z: float(z.mean()), x.copy())
    np.testing.assert_allclose(tx.grad, num_dx, rtol=1e-5, atol=1e-5)


def test_gradcheck_relu() -> None:
    rng = np.random.default_rng(5)
    # Shift away from zero to avoid non-differentiable points.
    x = rng.normal(size=(3, 4))
    x[np.abs(x) < 0.2] += np.sign(x[np.abs(x) < 0.2]) * 0.2 + 0.2

    tx = Tensor(x.copy(), requires_grad=True)
    out = tx.relu().sum()
    out.backward()

    num_dx = finite_difference_grad(lambda z: float(np.maximum(z, 0.0).sum()), x.copy())
    np.testing.assert_allclose(tx.grad, num_dx, rtol=1e-5, atol=1e-5)

