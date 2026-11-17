import math
import torch
from src.vertex_appearance import VertexAppearance, fourier_direction_3bands


def test_fourier_order():
    d = torch.tensor([[0.,0.,0.],[1.,0.,0.]])
    expected = torch.cat([torch.cat(((d*w).sin(), (d*w).cos()), -1) for w in (math.pi,2*math.pi,4*math.pi)], -1)
    torch.testing.assert_close(fourier_direction_3bands(d), expected)


def test_initialization_and_two_steps():
    torch.manual_seed(2)
    m = VertexAppearance(8)
    assert sum(p.numel() for p in m.decoder.parameters()) == 1923
    xyz = torch.randn(8,3)
    assert m(xyz, torch.zeros(3)).count_nonzero() == 0
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for _ in range(2):
        opt.zero_grad()
        (m(xyz, torch.zeros(3))-1).square().mean().backward()
        opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())
