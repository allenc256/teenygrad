import numpy as np
from numpy.typing import ArrayLike
from typing import Self, Sequence, Callable

class Tensor:
    name: str
    _value: np.ndarray
    _requires_grad: bool
    _grad: np.ndarray | None
    _backward_fn: Callable[[Self], None] | None
    _ancestors: list[Self]

    def __init__(
        self,
        initial_value: ArrayLike,
        requires_grad: bool = False,
        name: str = '',
        _backward_fn: Callable[[Self], None] | None = None,
        _ancestors: Sequence[Self] = (),
    ) -> None:
        if any(a._requires_grad for a in _ancestors):
            if not requires_grad:
                raise ValueError('requires_grad must be set if at least one ancestor requires gradients')
            if _backward_fn is None:
                raise ValueError('_backward_fn must be set if at least one ancestor requires gradients')

        self.name = name
        self._value = np.copy(initial_value)
        self._ancestors = list(_ancestors)
        self._requires_grad = requires_grad
        self._grad = None
        self._backward_fn = _backward_fn

    @property
    def shape(self) -> tuple[int, ...]:
        return self._value.shape
    
    @property
    def grad(self) -> np.ndarray | None:
        return self._grad
    
    def __add__(self, other: Self | ArrayLike) -> 'Tensor':
        _other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward_fn(out: Self) -> None:
            assert out._grad is not None
            if self._grad is not None:
                self._grad += _squash_gradient(out._grad, self._grad.shape)
            if _other._grad is not None:
                _other._grad += _squash_gradient(out._grad, _other._grad.shape)

        return Tensor(
            self._value + _other._value,
            requires_grad=(self._requires_grad or _other._requires_grad),
            name='+',
            _ancestors=[self, _other],
            _backward_fn=_backward_fn,
        )
    
    def __mul__(self, other: Self | ArrayLike) -> 'Tensor':
        _other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward_fn(out: Self) -> None:
            if self._grad is not None:
                self._grad += _squash_gradient(out._grad * _other._value, self._grad.shape)
            if _other._grad is not None:
                _other._grad += _squash_gradient(self._value * out._grad, _other._grad.shape)

        return Tensor(
            self._value * _other._value,
            requires_grad=(self._requires_grad or _other._requires_grad),
            name='*',
            _ancestors=[self, _other],
            _backward_fn=_backward_fn,
        )
    
    def sum(self) -> 'Tensor':
        def _backward_fn(out: Self) -> None:
            if self._grad is not None:
                self._grad += out._grad

        return Tensor(
            np.copy(np.sum(self._value)),
            requires_grad=self._requires_grad,
            name='sum',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )
    
    def log(self) -> 'Tensor':
        def _backward_fn(out: Self) -> None:
            if self._grad is not None:
                self._grad += out._grad / self._value

        return Tensor(
            np.log(self._value),
            requires_grad=self._requires_grad,
            name='log',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )
    
    def exp(self) -> 'Tensor':
        def _backward_fn(out: Self) -> None:
            if self._grad is not None:
                self._grad += np.exp(self._value) * out._grad

        return Tensor(
            np.exp(self._value),
            requires_grad=self._requires_grad,
            name='exp',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )
    
    def pow(self, other: float | int) -> 'Tensor':
        def _backward_fn(out: Self) -> None:
            assert out._grad is not None
            if self._grad is not None:
                self._grad += _squash_gradient((other * self._value ** (other - 1)) * out._grad, self._grad.shape)

        return Tensor(
            self._value ** other,
            requires_grad=self._requires_grad,
            name='pow',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )

    def relu(self):
        def _backward_fn(out: Self) -> None:
            assert out._grad is not None
            if self._grad is not None:
                self._grad += (out._value > 0) * out._grad

        return Tensor(
            (self._value > 0) * self._value,
            requires_grad=self._requires_grad,
            name='relu',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )
    
    def __pow__(self, other: float | int) -> 'Tensor':
        return self.pow(other)
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
   
    def backward(self) -> None:
        nodes = self._topological_sort()

        for n in nodes:
            if n._requires_grad:
                n._grad = np.zeros(n._value.shape, dtype=n._value.dtype)

        self._grad = np.ones(self.shape, dtype=self._value.dtype)

        for n in nodes:
            if n._grad is not None and n._backward_fn is not None:
                n._backward_fn(n)

    def _topological_sort(self) -> list[Self]:
        visited_list = []
        visited_set = set()
        def _visit(node: Self) -> None:
            if node in visited_set:
                return
            visited_set.add(node)
            for ancestor in node._ancestors:
                _visit(ancestor)
            visited_list.append(node)
        _visit(self)
        visited_list.reverse()
        return visited_list
    
    def __str__(self) -> str:
        return f'Tensor({self._value}, name=\'{self.name}\')'

def _squash_gradient(grad: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    if len(shape) == 0:
        return np.sum(grad)
    axis = [i for i in range(len(shape)) if shape[i] == 1]
    if len(axis) == 0:
        return grad
    return np.sum(grad, axis=tuple(axis), keepdims=True)
