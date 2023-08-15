# teenygrad

Small library which implements a subset of the PyTorch tensor and
autograd libraries using numpy. This project is meant to serve as a
demonstration only.

This project was heavily inspired by
[micrograd](https://github.com/karpathy/micrograd) and
[tinygrad](https://github.com/tinygrad/tinygrad).

## Examples

### Logistic Regression

We'll demonstrate computing gradients for logistic regression. Suppose
we have an initial setup of inputs, outputs, and model parameters for
logistic regression like so:

```
import numpy as np

X_np = np.random.rand(10, 3)            # input data
y_np = np.random.rand(10)               # output data
w_np = np.random.rand(3) * 2 - 1        # weight parameters
b_np = np.random.rand()                 # bias parameter

def logistic_regression(X, w, b):
    return (X @ w + b).sigmoid()

def binary_cross_entropy_loss(y, y_est):
    return (-(y * y_est.log() + (1 - y) * (1 - y_est).log())).sum() / y.shape[0]
```

We can compute estimates, loss, and gradients using PyTorch like so:

```
import torch

# Convert to tensors.
X = torch.tensor(X_np)
y = torch.tensor(y_np)
w = torch.tensor(w_np, requires_grad=True)
b = torch.tensor(b_np, requires_grad=True)

# Compute estimated values and loss.
y_est = logistic_regression(X, w, b)
loss = binary_cross_entropy_loss(y, y_est)

# Compute gradients.
loss.backward()
print(loss, w.grad, b.grad)
```

And the corresponding code using teenygrad (just replace `torch.tensor` with
`teenygrad.Tensor`):

```
import teenygrad

# Convert to tensors.
X = teenygrad.Tensor(X_np)
y = teenygrad.Tensor(y_np)
w = teenygrad.Tensor(w_np, requires_grad=True)
b = teenygrad.Tensor(b_np, requires_grad=True)

# Compute estimated values and loss.
y_est = logistic_regression(X, w, b)
loss = binary_cross_entropy_loss(y, y_est)

# Compute gradients.
loss.backward()
print(loss, w.grad, b.grad)
```

### Neural Network

See [`demo.ipynb`](demo.ipynb) for an example of training a 2 layer
neural network for binary classification using teenygrad.

