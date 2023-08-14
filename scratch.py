import numpy as np
import torch
import teenygrad as tg

actual_np = np.random.rand(4)
actual_tg = tg.Tensor(actual_np)
actual_torch = torch.tensor(actual_np)

estimated_np = np.random.rand(4)
estimated_tg = tg.Tensor(estimated_np, requires_grad=True)
estimated_torch = torch.tensor(estimated_np, requires_grad=True)
estimated_torch.retain_grad()

def cross_entropy(actual, estimated):
    return (-(actual * estimated.log() + (1 - actual) * (1 - estimated).log())).sum()

loss_tg = cross_entropy(actual_tg, estimated_tg)
loss_torch = cross_entropy(actual_torch, estimated_torch)

loss_tg.backward()
loss_torch.backward()

print('values')
print('------')
print('actual:', actual_np)
print('estimated:', estimated_np)
print('loss_torch:', loss_torch)
print('loss_tg:', loss_tg)
print('estimated_torch.grad:', estimated_torch.grad)
print('estimated_tg.grad:', estimated_tg.grad)
