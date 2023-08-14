import numpy as np
from numpy.typing import ArrayLike
from typing import Self, Sequence, Callable

class Tensor:
    _value: np.ndarray
    _requires_grad: bool
    _grad: np.ndarray | None
    _backward_fn: Callable[[np.ndarray, np.ndarray], None] | None
    _ancestors: list[Self]
    _op: str

    def __init__(
        self,
        initial_value: ArrayLike,
        requires_grad: bool = False,
        _backward_fn: Callable[[np.ndarray, np.ndarray], None] | None = None,
        _ancestors: Sequence[Self] = (),
        _op: str = '',
    ) -> None:
        if any(a._requires_grad for a in _ancestors):
            if not requires_grad:
                raise ValueError('requires_grad must be set if at least one ancestor requires gradients')
            if _backward_fn is None:
                raise ValueError('_backward_fn must be set if at least one ancestor requires gradients')
        
        self._value = np.copy(initial_value)
        self._ancestors = list(_ancestors)
        self._requires_grad = requires_grad
        self._grad = None
        self._backward_fn = _backward_fn
        self._op = _op

    @property
    def shape(self) -> tuple[int, ...]:
        return self._value.shape
    
    @property
    def grad(self) -> np.ndarray | None:
        return self._grad
    
    def __add__(self, other: Self | ArrayLike) -> 'Tensor':
        _other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward_fn(out_grad: np.ndarray, out_value: np.ndarray) -> None:
            if self._grad is not None:
                self._grad += _squash_gradient(out_grad, self._grad.shape)
            if _other._grad is not None:
                _other._grad += _squash_gradient(out_grad, _other._grad.shape)

        return Tensor(
            self._value + _other._value,
            requires_grad=(self._requires_grad or _other._requires_grad),
            _ancestors=[self, _other],
            _backward_fn=_backward_fn,
            _op='+',
        )
    
    def __mul__(self, other: Self | ArrayLike) -> 'Tensor':
        _other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward_fn(out_grad: np.ndarray, out_value: np.ndarray) -> None:
            if self._grad is not None:
                self._grad += _squash_gradient(out_grad * _other._value, self._grad.shape)
            if _other._grad is not None:
                _other._grad += _squash_gradient(self._value * out_grad, _other._grad.shape)

        return Tensor(
            self._value * _other._value,
            requires_grad=(self._requires_grad or _other._requires_grad),
            _ancestors=[self, _other],
            _backward_fn=_backward_fn,
            _op='*',
        )
    
    def sum(self) -> 'Tensor':
        def _backward_fn(out_grad: np.ndarray, out_value: np.ndarray) -> None:
            if self._grad is not None:
                self._grad += out_grad

        return Tensor(
            np.copy(np.sum(self._value)),
            requires_grad=self._requires_grad,
            _ancestors=[self],
            _backward_fn=_backward_fn,
            _op='sum',
        )
    
    def log(self) -> 'Tensor':
        def _backward_fn(out_grad: np.ndarray, out_value: np.ndarray) -> None:
            if self._grad is not None:
                self._grad += out_grad / self._value

        return Tensor(
            np.log(self._value),
            requires_grad=self._requires_grad,
            _ancestors=[self],
            _backward_fn=_backward_fn,
            _op='log',
        )
    
    def pow(self, other: ArrayLike) -> 'Tensor':
        _other = other if isinstance(other, np.ndarray) else np.copy(other)

        def _backward_fn(out_grad: np.ndarray, out_value: np.ndarray) -> None:
            if self._grad is not None:
                self._grad += _squash_gradient((_other * self._value ** (_other - 1)) * out_grad, self._grad.shape)

        return Tensor(
            self._value ** _other,
            requires_grad=self._requires_grad,
            _ancestors=[self],
            _backward_fn=_backward_fn,
            _op='pow',
        )

    def relu(self):
        def _backward_fn(out_grad: np.ndarray, out_value: np.ndarray) -> None:
            if self._grad is not None:
                self._grad += (out_value > 0) * out_grad

        return Tensor(
            (self._value > 0) * self._value,
            requires_grad=self._requires_grad,
            _ancestors=[self],
            _backward_fn=_backward_fn,
            _op='relu',
        )
    
    def __pow__(self, other: ArrayLike) -> 'Tensor':
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
   
    def backward(self) -> None:
        nodes = self._topological_sort()

        for n in nodes:
            if n._requires_grad:
                n._grad = np.zeros(n._value.shape, dtype=n._value.dtype)

        self._grad = np.ones(self.shape, dtype=self._value.dtype)

        for n in nodes:
            if n._grad is not None and n._backward_fn is not None:
                n._backward_fn(n._grad, n._value)

    # FIXME: detect cycles
    def _topological_sort(self) -> list[Self]:
        visited_list = []
        visited_set = set()
        def _visit(node: Self) -> None:
            if node in visited_set:
                return
            visited_list.append(node)
            visited_set.add(node)
            for ancestor in node._ancestors:
                _visit(ancestor)
        _visit(self)
        return visited_list
    
    def __str__(self) -> str:
        return f'Tensor({self._value}, op=\'{self._op}\')'

def _squash_gradient(grad: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    if len(shape) == 0:
        return np.sum(grad)
    axis = [i for i in range(len(shape)) if shape[i] == 1]
    if len(axis) == 0:
        return grad
    return np.sum(grad, axis=tuple(axis), keepdims=True)
