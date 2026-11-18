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
    def __init__(self, num_vertices, feature_dim=8):
        super().__init__()
        self.vertex_features = nn.Parameter(torch.randn(num_vertices, feature_dim) * 0.01)
        self.decoder = nn.Sequential(nn.Linear(feature_dim + 18, 64), nn.ReLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, vertex_xyz, camera_origin, vertex_indices=None):
        features = self.vertex_features if vertex_indices is None else self.vertex_features[vertex_indices]
        d = F.normalize(vertex_xyz - camera_origin.reshape(1, 3), dim=-1, eps=1e-8)
        return self.decoder(torch.cat((features, fourier_direction_3bands(d)), -1))


    def forward_subset(self, vertex_xyz, camera_origin, indices):
        return self(vertex_xyz[indices], camera_origin, vertex_indices=indices)


class VertexTwoBases(VertexAppearance):
    appearance_mode = "vertex_two_bases"
    def __init__(self, num_vertices, num_voxels):
        super().__init__(num_vertices)
        self.decoder[-1] = nn.Linear(64, 6)
        with torch.no_grad():
            self.decoder[-1].weight[:3].zero_()
            self.decoder[-1].weight[3:].normal_(std=0.01)
            self.decoder[-1].bias.zero_()
        self._initialize_voxel_parameters(num_voxels)

    def _initialize_voxel_parameters(self, num_voxels):
        self.mix = nn.Parameter(torch.zeros(num_voxels, 2))
        with torch.no_grad():
            self._reset_voxel_parameters()

    def _reset_voxel_parameters(self):
        self.mix[:, 0].fill_(1)
        self.mix[:, 1].zero_()

    @property
    def voxel_parameter(self):
        return self.mix

    @torch.no_grad()
    def initialize_first_basis(self, weights):
        self.vertex_features.copy_(weights["vertex_features"])
        self.decoder[0].weight.copy_(weights["decoder.0.weight"])
        self.decoder[0].bias.copy_(weights["decoder.0.bias"])
        self.decoder[-1].weight[:3].copy_(weights["decoder.2.weight"])
        self.decoder[-1].bias[:3].copy_(weights["decoder.2.bias"])
        self._reset_voxel_parameters()

    def forward(self, vertex_xyz, camera_origin, vertex_indices=None):
        return super().forward(vertex_xyz, camera_origin, vertex_indices).unflatten(-1, (2, 3))

    def voxel_residual(self, bases, keys, voxel_indices=None):
        mix = self.mix if voxel_indices is None else self.mix[voxel_indices]
        first = bases[:, 0][keys] * mix[:, 0, None, None]
        return torch.addcmul(first, bases[:, 1][keys], mix[:, 1, None, None])


class VertexBoundedResidual(VertexTwoBases):
    appearance_mode = "vertex_two_bases_bounded"

    def __init__(self, num_vertices, num_voxels, eta=0.25):
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        super().__init__(num_vertices, num_voxels)
        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float32))

    def _initialize_voxel_parameters(self, num_voxels):
        self.voxel_gate = nn.Parameter(torch.zeros(num_voxels))

    def _reset_voxel_parameters(self):
        self.voxel_gate.zero_()

    @property
    def voxel_parameter(self):
        return self.voxel_gate

    def correction_input(self, bases, keys, voxel_indices=None):
        gate = self.voxel_gate if voxel_indices is None else self.voxel_gate[voxel_indices]
        return gate[:, None, None] * bases[:, 1][keys]

    def voxel_residual(self, bases, keys, voxel_indices=None):
        return bases[:, 0][keys] + self.eta * self.correction_input(bases, keys, voxel_indices).tanh()


class VertexRGB(nn.Module):
    def __init__(self, num_vertices):
        super().__init__()
        self.vertex_rgb = nn.Parameter(torch.zeros(num_vertices, 3))

    def forward(self, vertex_xyz, camera_origin):
        return self.vertex_rgb

    def forward_subset(self, vertex_xyz, camera_origin, indices):
        return self.vertex_rgb[indices]
