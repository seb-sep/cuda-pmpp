
import torch
from torch.utils.cpp_extension import load


module = load(
    name='m',
    sources=['main.cpp', 'sum_reduce.cu'],
    verbose=True
)

a = torch.tensor([0.5, 0.5, 0.5, 0.5], device='cuda')

print(module.sum(a), a.sum())