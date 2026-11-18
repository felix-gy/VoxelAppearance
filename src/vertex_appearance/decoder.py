"""Shared vertex appearance with deterministic Fourier direction encoding."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def fourier_direction_3bands(d):
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError("Expected directions [P,3]")
    phase = d[:, None, :] * d.new_tensor([math.pi, 2 * math.pi, 4 * math.pi])[None, :, None]
    return torch.cat((phase.sin(), phase.cos()), dim=-1).reshape(d.shape[0], 18)


class VertexAppearance(nn.Module):
    def __init__(self, num_vertices):
        super().__init__()
        self.vertex_features = nn.Parameter(torch.randn(num_vertices, 8) * 0.01)
        self.decoder = nn.Sequential(nn.Linear(26, 64), nn.ReLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, vertex_xyz, camera_origin, vertex_indices=None):
        features = self.vertex_features if vertex_indices is None else self.vertex_features[vertex_indices]
        d = F.normalize(vertex_xyz - camera_origin.reshape(1, 3), dim=-1, eps=1e-8)
        return self.decoder(torch.cat((features, fourier_direction_3bands(d)), -1))


    def forward_subset(self, vertex_xyz, camera_origin, indices):
        return self(vertex_xyz[indices], camera_origin, vertex_indices=indices)


class VertexTwoBases(VertexAppearance):
    def __init__(self, num_vertices, num_voxels):
        super().__init__(num_vertices)
        self.decoder[-1] = nn.Linear(64, 6)
        with torch.no_grad():
            self.decoder[-1].weight[:3].zero_()
            self.decoder[-1].weight[3:].normal_(std=0.01)
            self.decoder[-1].bias.zero_()
        mix = torch.zeros(num_voxels, 2)
        mix[:, 0] = 1
        self.mix = nn.Parameter(mix)

    @torch.no_grad()
    def initialize_first_basis(self, weights):
        self.vertex_features.copy_(weights["vertex_features"])
        self.decoder[0].weight.copy_(weights["decoder.0.weight"])
        self.decoder[0].bias.copy_(weights["decoder.0.bias"])
        self.decoder[-1].weight[:3].copy_(weights["decoder.2.weight"])
        self.decoder[-1].bias[:3].copy_(weights["decoder.2.bias"])
        self.mix[:, 0].fill_(1)
        self.mix[:, 1].zero_()

    def forward(self, vertex_xyz, camera_origin, vertex_indices=None):
        return super().forward(vertex_xyz, camera_origin, vertex_indices).unflatten(-1, (2, 3))

    def voxel_residual(self, bases, keys, voxel_indices=None):
        mix = self.mix if voxel_indices is None else self.mix[voxel_indices]
        first = bases[:, 0][keys] * mix[:, 0, None, None]
        return torch.addcmul(first, bases[:, 1][keys], mix[:, 1, None, None])


class VertexRGB(nn.Module):
    def __init__(self, num_vertices):
        super().__init__()
        self.vertex_rgb = nn.Parameter(torch.zeros(num_vertices, 3))

    def forward(self, vertex_xyz, camera_origin):
        return self.vertex_rgb

    def forward_subset(self, vertex_xyz, camera_origin, indices):
        return self.vertex_rgb[indices]
