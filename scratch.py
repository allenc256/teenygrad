import numpy as np
import torch
import teenygrad as tg

X_np = np.random.rand(3, 3)
Y_np = np.random.rand(3)

X_torch = torch.tensor(X_np, requires_grad=True)
Y_torch = torch.tensor(Y_np, requires_grad=True)
A_torch = X_torch + Y_torch
A_torch.retain_grad()
loss_torch = A_torch.sum()

X_tg = tg.Tensor(X_np, requires_grad=True)
Y_tg = tg.Tensor(Y_np, requires_grad=True)
A_tg = X_tg + Y_tg
loss_tg = A_tg.sum()

loss_torch.backward()
loss_tg.backward()

assert A_torch.grad is not None
assert X_torch.grad is not None
assert Y_torch.grad is not None
assert A_torch.shape == A_tg.shape
print('A_grad_err:', (A_torch.grad.numpy() - A_tg.grad).sum())
print('X_grad_err:', (X_torch.grad.numpy() - X_tg.grad).sum())
print('Y_grad_err:', (Y_torch.grad.numpy() - Y_tg.grad).sum())

# W_np = np.random.rand(3, 3)
# W_tg = tg.Tensor(W_np, requires_grad=True)
# W_torch = torch.tensor(W_np, requires_grad=True)

# x_np = np.random.rand(3, 3)
# x_tg = tg.Tensor(x_np, requires_grad=True)
# x_torch = torch.tensor(x_np, requires_grad=True)

# y_tg = W_tg.matmul(x_tg)
# y_torch = torch.matmul(W_torch, x_torch)

# loss_tg = y_tg.sum()
# loss_torch = y_torch.sum()

# loss_tg.backward()
# loss_torch.backward()

# print('x:', x_np)
# print('loss_tg:', loss_tg)
# print('loss_torch:', loss_torch)
# print('W_tg.grad:', W_tg.grad)
# print('W_torch.grad:', W_torch.grad)
# print('x_tg.grad:', x_tg.grad)
# print('x_torch.grad:', x_torch.grad)

# def sigmoid(logits):
#     return 1 / (1 + (-logits).exp())

# def binary_cross_entropy(actual, estimated):
#     return -(actual * estimated.log() + (1 - actual) * (1 - estimated).log())

# actual_np = np.random.rand(4)
# actual_tg = tg.Tensor(actual_np)
# actual_torch = torch.tensor(actual_np)

# logits_np = np.random.rand(4)
# logits_tg = tg.Tensor(logits_np, requires_grad=True)
# logits_torch = torch.tensor(logits_np, requires_grad=True)
# logits_torch.retain_grad()

# estimated_tg = sigmoid(logits_tg)
# estimated_torch = sigmoid(logits_torch)
# estimated_torch.retain_grad()

# loss_tg = binary_cross_entropy(actual_tg, estimated_tg).sum()
# loss_torch = binary_cross_entropy(actual_torch, estimated_torch).sum()

# loss_tg.backward()
# loss_torch.backward()

# print('values')
# print('------')
# print('actual:', actual_np)
# print('loss_torch:', loss_torch)
# print('loss_tg:', loss_tg)
# print('estimated_torch.grad:', estimated_torch.grad)
# print('estimated_tg.grad:', estimated_tg.grad)
# print('logits_torch.grad:', logits_torch.grad)
# print('logits_tg.grad:', logits_tg.grad)
