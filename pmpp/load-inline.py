# Look at this test for inspiration
# https://github.com/pytorch/pytorch/blob/main/test/test_cpp_extensions_jit.py

import torch
from torch.utils.cpp_extension import load_inline, load

module = load(
    name = 'm',
    sources = ['main.cpp', 'ch3.cu', 'square.cu'],
    # extra_cflags=['-02'],
    verbose=True
)

A = torch.tensor([[1., 2.],
                  [3., 4.]], device='cuda')

B = torch.tensor([[2., 0.],
                  [2., 0.]], device='cuda')

print(module.gemm(A, B))