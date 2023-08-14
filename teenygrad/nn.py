import numpy as np
from typing import Sequence
from teenygrad.tensor import Tensor

class Module:
    """Analog of `torch.nn.Module`."""
    def forward(self, input: Tensor) -> Tensor:
        """Analog of `torch.nn.Module.forward()`."""
        raise NotImplementedError()

    def parameters(self) -> Sequence[Tensor]:
        """Analog of `torch.nn.Module.parameters()`."""
        raise NotImplementedError()

    def zero_grad(self):
        """Analog of `torch.nn.Module.zero_grad()`."""
        for p in self.parameters():
            p.grad = None

class Sequential(Module):
    """Analog of `torch.nn.Sequential`."""
    _layers: list[Module]

    def __init__(self, *args: Module) -> None:
        self._layers = list(args)

    def forward(self, x: Tensor) -> Tensor:
        activations = x
        for l in self._layers:
            activations = l.forward(activations)
        return activations

    def parameters(self) -> Sequence[Tensor]:
        return [p for l in self._layers for p in l.parameters()]

class Linear(Module):
    """Analog of `torch.nn.Linear`."""
    _weights: Tensor
    _biases: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        # N.B., initialize all weights/biases using a uniform distribution over [-1, 1].
        self._weights = Tensor(np.random.rand(in_features, out_features) * 2 - 1, requires_grad=True)
        self._biases = Tensor(np.random.rand(out_features) * 2 - 1, requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return input @ self._weights + self._biases

    def parameters(self) -> Sequence[Tensor]:
        return [self._weights, self._biases]

class ReLU(Module):
    """Analog of `torch.nn.ReLU`."""
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()

    def parameters(self) -> Sequence[Tensor]:
        return []

class Sigmoid(Module):
    """Analog of `torch.nn.Sigmoid`."""
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()

    def parameters(self) -> Sequence[Tensor]:
        return []
