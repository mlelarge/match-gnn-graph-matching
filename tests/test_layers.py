import pytest
import torch

from models.layers import *

bs = 12
f = 16
n = 20
ATOL = 1e-2

@pytest.fixture
def make_b():
    return [torch.randn(bs, f, n, n) + torch.randint(5,(bs, f, n,n)), torch.randn(bs, f, n) + torch.randint(5,(bs, f, n))]

def test_normalize(make_b):
    for b in make_b:
        bn = normalize(b)
        assert bn.shape == b.shape
        assert torch.allclose(bn.mean(dim = (-1,-2), keepdim=True), torch.zeros_like(b), atol=ATOL)
        assert torch.allclose((4*n)*bn.var(dim = (-1,-2), keepdim=True), torch.ones_like(b), atol=ATOL)

b = torch.randn(bs,f,n)
def test_MlpBlock_Node(b=b):
    node_emb = MlpBlock_Node(f, 2*f, 3)
    b_out = node_emb(b)
    assert b_out.shape == (bs, 2*f, n)
    m12 = b_out.mean(dim=(-2,-1), keepdim=True)
    v12 = b_out.var(dim=(-2,-1), keepdim=True,unbiased=False)
    assert torch.allclose(m12, torch.zeros(bs,2*f,n), atol=ATOL)
    assert torch.allclose((4*n)*v12, torch.ones(bs,2*f,n), atol=ATOL)

