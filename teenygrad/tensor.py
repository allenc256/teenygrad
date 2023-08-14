import numpy as np
from numpy.typing import ArrayLike
from typing import Sequence, Callable

class Tensor:
    """Analog of `torch.Tensor`.

    This class stores all internal values and gradients as numpy arrays. This class includes a simple autograd
    implementation for supporting back-propagation over various operations over tensors.

    Attributes:
        name: The name of the tensor (for debugging purposes).
        value: The value of the tensor.
        grad: The current gradient of the tensor relative to the scalar loss function, if it has been calculated.
            This gradient should have the same shape as the value array. If it has not been calculated, this will
            be set to None.
    """
    name: str
    value: np.ndarray
    grad: np.ndarray | None
    _requires_grad: bool
    _backward_fn: Callable[['Tensor'], None] | None
    _ancestors: list['Tensor']

    def __init__(
        self,
        value: ArrayLike,
        requires_grad: bool | None = None,
        name: str = '',
        _backward_fn: Callable[['Tensor'], None] | None = None,
        _ancestors: Sequence['Tensor'] = (),
    ) -> None:
        """Initializes the tensor.

        Args:
            value: The initial value for the tensor.
            requires_grad: Whether gradients should be computed for the tensor (e.g., whether the tensor corresponds
                to weights or constant data).
            name: The name of the tensor.
        """
        # Validate that requires_grad and _backward_fn are set correctly.
        ancestors_requires_grad = any(a._requires_grad for a in _ancestors)
        if ancestors_requires_grad:
            if requires_grad == False:
                raise ValueError('requires_grad cannot be False if at least one ancestor requires gradients')
            if _backward_fn is None:
                raise ValueError('_backward_fn must be set if at least one ancestor requires gradients')

        self.name = name
        self.value = np.copy(value)
        self.grad = None
        self._ancestors = list(_ancestors)
        self._requires_grad = requires_grad or ancestors_requires_grad
        self._backward_fn = _backward_fn

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self) -> int:
        return self.value.ndim

    def sum(self) -> 'Tensor':
        """Analog of `torch.Tensor.sum()`."""
        def _backward_fn(out: 'Tensor') -> None:
            if self.grad is not None:
                self.grad += out.grad

        return Tensor(
            np.copy(np.sum(self.value)),
            name='sum',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )

    def log(self) -> 'Tensor':
        """Analog of `torch.Tensor.log()`."""
        def _backward_fn(out: 'Tensor') -> None:
            if self.grad is not None:
                self.grad += out.grad / self.value

        return Tensor(
            np.log(self.value),
            name='log',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )

    def exp(self) -> 'Tensor':
        """Analog of `torch.Tensor.exp()`."""
        def _backward_fn(out: 'Tensor') -> None:
            if self.grad is not None:
                self.grad += np.exp(self.value) * out.grad

        return Tensor(
            np.exp(self.value),
            name='exp',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )

    def pow(self, other: float | int) -> 'Tensor':
        """Analog of `torch.Tensor.pow()`."""
        def _backward_fn(out: 'Tensor') -> None:
            assert out.grad is not None
            if self.grad is not None:
                self.grad += _unbroadcast_gradient((other * self.value ** (other - 1)) * out.grad, self.grad.shape)

        return Tensor(
            self.value ** other,
            name='pow',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )

    def relu(self):
        """Analog of `torch.Tensor.relu()`."""
        def _backward_fn(out: 'Tensor') -> None:
            assert out.grad is not None
            if self.grad is not None:
                self.grad += (out.value > 0) * out.grad

        return Tensor(
            (self.value > 0) * self.value,
            name='relu',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )

    def sigmoid(self):
        """Analog of `torch.Tensor.sigmoid()`."""
        def _backward_fn(out: 'Tensor') -> None:
            if self.grad is not None:
                self.grad += (out.value * (1 - out.value)) * out.grad

        return Tensor(
            1 / (1 + np.exp(-self.value)),
            name='sigmoid',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Analog of `torch.Tensor.matmul()`."""
        return self.tensordot(other, 1)

    def tensordot(self, other: 'Tensor', dims: int = 2) -> 'Tensor':
        """Analog of `torch.Tensor.tensordot()`."""
        def _backward_fn(out: 'Tensor') -> None:
            assert out.grad is not None
            if self.grad is not None:
                # Figure out how many dimensions to contract going backwards.
                to_contract_ndim = (out.grad.ndim + other.value.ndim - self.grad.ndim) // 2

                # Gradient update is tensor contraction of output gradient and other's tranpose.
                self.grad += np.tensordot(out.grad, _transpose(other.value, to_contract_ndim), axes=to_contract_ndim)

            if other.grad is not None:
                # Figure out how many dimensions to contract going backwards.
                to_contract_ndim = (out.grad.ndim + self.value.ndim - other.grad.ndim) // 2

                # Gradient update is tensor contraction of self's transpose and output gradient.
                other.grad += np.tensordot(_transpose(self.value, self.value.ndim - to_contract_ndim), out.grad, axes=to_contract_ndim)

        return Tensor(
            np.tensordot(self.value, other.value, axes=dims),
            name='tensordot',
            _ancestors=[self, other],
            _backward_fn=_backward_fn,
        )

    def permute(self, dims: Sequence[int]) -> 'Tensor':
        """Analog of `torch.Tensor.permute()`."""
        def _backward_fn(out: 'Tensor') -> None:
            assert out.grad is not None
            if self.grad is not None:
                self.grad += _inverse_permute(out.grad, dims)

        return Tensor(
            np.transpose(self.value, axes=dims),
            name='permute',
            _ancestors=[self],
            _backward_fn=_backward_fn,
        )

    def __add__(self, other: 'Tensor' | ArrayLike) -> 'Tensor':
        _other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward_fn(out: 'Tensor') -> None:
            assert out.grad is not None
            if self.grad is not None:
                self.grad += _unbroadcast_gradient(out.grad, self.grad.shape)
            if _other.grad is not None:
                _other.grad += _unbroadcast_gradient(out.grad, _other.grad.shape)

        return Tensor(
            self.value + _other.value,
            name='+',
            _ancestors=[self, _other],
            _backward_fn=_backward_fn,
        )

    def __mul__(self, other: 'Tensor' | ArrayLike) -> 'Tensor':
        _other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward_fn(out: 'Tensor') -> None:
            if self.grad is not None:
                self.grad += _unbroadcast_gradient(out.grad * _other.value, self.grad.shape)
            if _other.grad is not None:
                _other.grad += _unbroadcast_gradient(out.grad * self.value, _other.grad.shape)

        return Tensor(
            self.value * _other.value,
            name='*',
            _ancestors=[self, _other],
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

    def __matmul__(self, other): # self @ other
        return self.matmul(other)

    def __rmatmul__(self, other): # other @ self
        return other.matmul(self)

    def backward(self) -> None:
        """Performs back-propagation starting from the given tensor.

        This method will accumulate gradients for the transitive closure of all ancestors in the tensor's computation
        graph. Back-propagation must start from a scalar tensor (typically representing a loss), as all gradients
        are assumed to be computed relative to a scalar loss function.

        Only nodes where ``requires_grad`` has been set to True will have gradients accumulated into.

        Like the corresponding pytorch method, this method does not automatically zero-out gradient buffers.
        It is the responsibility of callers to zero-out gradients (e.g., by setting ``my_tensor.grad = None``).
        """
        # Make sure we're starting from a scalar.
        if self.ndim != 0:
            raise ValueError('gradients can only be computed from scalars')

        # Compute the sorted transitive closure of ancestors.
        nodes = self._topological_sort()

        # Initialize any gradient buffers as necessary.
        for n in nodes:
            if n._requires_grad and n.grad is None:
                n.grad = np.zeros(n.value.shape, dtype=n.value.dtype)

        # Initialize the root gradient to one.
        self.grad = np.ones((), dtype=self.value.dtype)

        # Accumulate gradients by running the backwards function.
        for n in nodes:
            if n.grad is not None and n._backward_fn is not None:
                n._backward_fn(n)

    def _topological_sort(self) -> list['Tensor']:
        """Topologically sorts the ancestor graph.

        The sort order is such that the root node (i.e., self) appears first, and all leaf nodes appear later.
        """
        visited_list = []
        visited_set = set()
        def _visit(node: 'Tensor') -> None:
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
        return f'Tensor({self.value}, name=\'{self.name}\')'

def _unbroadcast_gradient(grad: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Squashes gradients in order to account for broadcasting.

    This method sums gradients along dimensions that would be broadcasted during some operation. For example,
    if the target shape is a scalar, then this method should sum along all dimensions of the gradient and return
    a scalar.

    Args:
        grad: The gradient to be squashed.
        shape: The target shape that the gradient should be squashed to.

    Returns:
        A numpy array with the target shape.
    """
    # Prepend extra dimensions of size 1 to shape as necessary to match broadcast rules.
    missing_dims = grad.ndim - len(shape)
    bcast_shape = ([1] * missing_dims) + list(shape)

    # Find all axes with size 1, and sum along those.
    axis = [i for i in range(len(bcast_shape)) if bcast_shape[i] == 1]
    if len(axis) == 0:
        return grad
    squashed = np.sum(grad, axis=tuple(axis), keepdims=True)

    # Reshape to target shape.
    return squashed.reshape(shape)

def _transpose(a: np.ndarray, dims: int) -> np.ndarray:
    """Transposes a set number of dimensions.

    The last `dims` dimensions of the array are moved to the front of the array. For example, if the input
    array has shape (5, 4, 3) and `dims` is 2, then the output array will have shape (4, 3, 5).

    Args:
        a: The numpy array to transpose.
        dims: The number of dimensions to transpose.
    """
    axes = list(range(a.ndim))
    axes = axes[-dims:] + axes[:-dims]
    return np.transpose(a, axes)

def _inverse_permute(a: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Performs an inverse permutation of axes.

    This method is effectively the inverse of `np.transpose()`. That is, if you pass an array with `dims` set
    as the axes to `np.transpose()`, you can recover the original array by calling this method on the output.

    Args:
        a: The numpy array to inverse-permute.
        dims: The permutation of the axes for the forward direction.
    """
    idxs = [0] * len(dims)
    for i, from_i in enumerate(dims):
        idxs[from_i] = i
    return np.transpose(a, idxs)
