"""Linear layer implemented with the custom Tensor engine."""

from __future__ import annotations

import numpy as np

from src.tensor import Tensor


class Linear:
    """Fully-connected layer: y = xW + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")

        rng = np.random.default_rng(seed)
        limit = np.sqrt(1.0 / in_features)
        w = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out.add(self.bias)
        return out

    def parameters(self) -> list[Tensor]:
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

