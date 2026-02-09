"""Minimal Tensor and autograd engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


BackwardFn = Callable[[], None]


def _as_array(data: float | int | np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum broadcasted gradient back to the original operand shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


@dataclass(eq=False)
class Tensor:
    """Tensor with reverse-mode autodiff support for core operations."""

    data: np.ndarray | float | int
    requires_grad: bool = False
    _prev: set["Tensor"] = field(default_factory=set, repr=False)
    _op: str = field(default="", repr=False)
    _backward: BackwardFn = field(default=lambda: None, repr=False)
    grad: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.data = _as_array(self.data)

    def __hash__(self) -> int:
        return id(self)

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        self.grad = grad if self.grad is None else self.grad + grad

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def add(self, other: "Tensor | float | int") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev={self, other},
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def mul(self, other: "Tensor | float | int") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev={self, other},
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev={self, other},
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _prev={self},
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                expanded = np.broadcast_to(grad, self.shape)
            else:
                axes: tuple[int, ...]
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                expanded = np.broadcast_to(grad, self.shape)
            self._accumulate_grad(expanded)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            count = self.data.size
        elif isinstance(axis, int):
            count = self.data.shape[axis]
        else:
            count = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims).mul(1.0 / count)

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            _prev={self},
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def backward(self, grad: np.ndarray | float | int | None = None) -> None:
        """Backpropagate gradients from this tensor."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar outputs")
            grad = np.ones_like(self.data)
        else:
            grad = _as_array(grad)

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = grad
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: "Tensor | float | int") -> "Tensor":
        return self.add(other)

    def __radd__(self, other: "Tensor | float | int") -> "Tensor":
        return self.add(other)

    def __mul__(self, other: "Tensor | float | int") -> "Tensor":
        return self.mul(other)

    def __rmul__(self, other: "Tensor | float | int") -> "Tensor":
        return self.mul(other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)
