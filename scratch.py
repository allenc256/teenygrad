import numpy as np
import torch
import teenygrad as tg

def sigmoid(logits):
    return 1 / (1 + (-logits).exp())

def binary_cross_entropy(actual, estimated):
    return -(actual * estimated.log() + (1 - actual) * (1 - estimated).log())

actual_np = np.random.rand(4)
actual_tg = tg.Tensor(actual_np)
actual_torch = torch.tensor(actual_np)

logits_np = np.random.rand(4)
logits_tg = tg.Tensor(logits_np, requires_grad=True)
logits_torch = torch.tensor(logits_np, requires_grad=True)
logits_torch.retain_grad()

estimated_tg = sigmoid(logits_tg)
estimated_torch = sigmoid(logits_torch)
estimated_torch.retain_grad()

loss_tg = binary_cross_entropy(actual_tg, estimated_tg).sum()
loss_torch = binary_cross_entropy(actual_torch, estimated_torch).sum()

loss_tg.backward()
loss_torch.backward()

print('values')
print('------')
print('actual:', actual_np)
print('loss_torch:', loss_torch)
print('loss_tg:', loss_tg)
print('estimated_torch.grad:', estimated_torch.grad)
print('estimated_tg.grad:', estimated_tg.grad)
print('logits_torch.grad:', logits_torch.grad)
print('logits_tg.grad:', logits_tg.grad)
