# Look at this test for inspiration
# https://github.com/pytorch/pytorch/blob/main/test/test_cpp_extensions_jit.py

import torch
from torch.utils.cpp_extension import load_inline, load

module = load(
    name = 'm',
    sources = ['ch3/main.cpp', 'ch3/ch3.cu', 'ch3/square.cu'],
    # extra_cflags=['-02'],
    verbose=True
)

# A = torch.tensor([[1., 2.],
#                   [3., 4.]], device='cuda')

# B = torch.tensor([[2., 0.],
#                   [2., 0.]], device='cuda')

A = torch.rand((2, 2), device='cuda')
B = torch.rand((2, 2), device='cuda')

print(module.gemm(A, B))