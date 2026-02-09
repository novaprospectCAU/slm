import numpy as np

from src.tensor.tensor import Tensor


def test_grad_accumulates_when_tensor_reused() -> None:
    x_val = np.array([[2.0, -3.0], [0.5, 4.0]], dtype=np.float64)
    x = Tensor(x_val.copy(), requires_grad=True)

    # y = x*x + x  -> dy/dx = 2x + 1 (elementwise), then summed.
    y = x.mul(x).add(x).sum()
    y.backward()

    expected = 2.0 * x_val + 1.0
    np.testing.assert_allclose(x.grad, expected, rtol=1e-8, atol=1e-8)

