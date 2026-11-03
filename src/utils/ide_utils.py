"""Differentiable, normalized directional harmonics for reflection shading.

The encoding follows MultiNeRF's IDE, using a normalized recurrence rather
than expanded Legendre polynomials to avoid cancellation at high degrees.
"""

import math

import torch
from torch import nn


def _validate_degree(degree):
    if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= 5:
        raise ValueError("IDE degree must be an integer between 1 and 5.")


def get_ml_array(deg_view):
    import numpy as np

    _validate_degree(deg_view)
    return np.array([(m, 2 ** i) for i in range(deg_view)
                     for m in range(2 ** i + 1)]).T


class IntegratedDirectionalEncoding(nn.Module):
    def __init__(self, degree=5):
        super().__init__()
        _validate_degree(degree)
        self.degree = degree
        self.levels = tuple(2 ** i for i in range(degree))
        self.output_dim = 2 * sum(l + 1 for l in self.levels)
        self.l_max = self.levels[-1]

        # Y_m^m = coefficient[m] * (x + iy)^m, including Condon-Shortley phase.
        diagonal = [1.0 / math.sqrt(4.0 * math.pi)]
        for m in range(1, self.l_max + 1):
            diagonal.append(-math.sqrt((2 * m + 1) / (2 * m)) * diagonal[-1])
        a = torch.zeros(self.l_max + 1, self.l_max + 1, dtype=torch.float64)
        b = torch.zeros_like(a)
        for l in range(1, self.l_max + 1):
            for m in range(l):
                a[l, m] = math.sqrt((4 * l * l - 1) / (l * l - m * m))
                if m < l - 1:
                    b[l, m] = math.sqrt(
                        ((2 * l + 1) * ((l - 1) ** 2 - m * m)) /
                        ((2 * l - 3) * (l * l - m * m)))
        sigma = [0.5 * l * (l + 1) for l in self.levels for _ in range(l + 1)]

        # Derived constants move with the module, but need not be checkpointed.
        for name, value in (
                ('diagonal', torch.tensor(diagonal, dtype=torch.float64)),
                ('recurrence_a', a), ('recurrence_b', b),
                ('diagonal_mask', torch.eye(self.l_max + 1, dtype=torch.float64)),
                ('sigma', torch.tensor(sigma, dtype=torch.float64))):
            self.register_buffer(name, value, persistent=False)

    def forward(self, xyz, kappa_inv):
        """Encode unit directions [..., 3] with nonnegative smoothing [..., 1]."""
        if xyz.dtype not in (torch.float32, torch.float64):
            raise TypeError("IDE directions must be float32 or float64.")
        if xyz.ndim == 0 or xyz.shape[-1] != 3:
            raise ValueError("IDE directions must have shape [..., 3].")
        if kappa_inv.ndim == 0 or kappa_inv.shape[-1] != 1:
            raise ValueError("IDE smoothing must have shape [..., 1].")
        torch.broadcast_shapes(xyz.shape[:-1], kappa_inv.shape[:-1])
        kappa_inv = kappa_inv.to(xyz)
        a = self.recurrence_a.to(xyz)
        b = self.recurrence_b.to(xyz)
        mask = self.diagonal_mask.to(xyz)

        xy = torch.complex(xyz[..., 0:1], xyz[..., 1:2])
        z = xyz[..., 2:3]
        # Each power is computed once, even though several levels reuse it.
        diagonal = torch.cat([xy ** m for m in range(self.l_max + 1)], dim=-1)
        diagonal = diagonal * self.diagonal.to(xyz)
        previous = torch.zeros_like(diagonal)
        previous2 = torch.zeros_like(diagonal)
        selected = []
        for l in range(self.l_max + 1):
            current = a[l] * z * previous - b[l] * previous2 + mask[l] * diagonal
            if l in self.levels:
                selected.append(current[..., :l + 1])
            previous2, previous = previous, current

        harmonics = torch.cat(selected, dim=-1)
        ide = harmonics * torch.exp(-self.sigma.to(xyz) * kappa_inv)
        return torch.cat([ide.real, ide.imag], dim=-1)


def generate_ide_fn_pytorch(deg_view):
    """Return a callable module with an ``output_dim`` and movable buffers."""
    return IntegratedDirectionalEncoding(deg_view)
