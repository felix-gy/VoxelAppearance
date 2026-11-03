# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sparse_voxel_gears.constructor import SVConstructor
from src.sparse_voxel_gears.properties import SVProperties
from src.sparse_voxel_gears.renderer import SVRenderer
from src.sparse_voxel_gears.adaptive import SVAdaptive
from src.sparse_voxel_gears.io import SVInOut
from src.sparse_voxel_gears.pooling import SVPooling
from src.utils.ide_utils import generate_ide_fn_pytorch


class SpecularMLP(nn.Module):
    def __init__(self, ide_dim, bottleneck_dim=8, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ide_dim + bottleneck_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, ide_encoding, bottleneck_vec, normal_dot_view):
        return self.net(torch.cat(
            [ide_encoding, bottleneck_vec, normal_dot_view], dim=-1))


class SparseVoxelModel(SVConstructor, SVProperties, SVRenderer, SVAdaptive, SVInOut, SVPooling):
    MIN_ROUGHNESS = 1e-4

    def __init__(self, n_samp_per_vox=1, ss=1.5, white_background=False,
                 black_background=False, ide_degree=5, hidden_dim=64,
                 bottleneck_dim=8, normal_eps=1e-4, ide_dim=None):
        """Create shared appearance modules; voxel tensors are allocated later.

        model_init(), octpath_init() and load() all initialize the per-voxel
        appearance and move the shared modules to the voxel device.
        """
        super().__init__()
        self.n_samp_per_vox = n_samp_per_vox
        self.ss = ss
        self.white_background = white_background
        self.black_background = black_background
        self.per_voxel_attr_lst = ['octpath', 'octlevel', '_subdiv_p']
        self.per_voxel_param_lst = [
            '_albedo', '_roughness', '_specular_tint', '_bottleneck',
        ]
        self.grid_pts_param_lst = ['_geo_grid_pts']
        self.scene_center = None
        self.scene_extent = None
        self.inside_extent = None
        self.octpath = None
        self.octlevel = None
        self._geo_grid_pts = None
        self._subdiv_p = None
        for name in self.per_voxel_param_lst:
            setattr(self, name, None)
        self.loaded_legacy_sh = False
        self._configure_appearance(ide_degree, hidden_dim, bottleneck_dim, normal_eps)
        if ide_dim is not None and ide_dim != self._ide_fn.output_dim:
            raise ValueError(
                f"ide_degree={ide_degree} produces {self._ide_fn.output_dim} "
                f"features, not ide_dim={ide_dim}.")

    def _configure_appearance(self, ide_degree, hidden_dim, bottleneck_dim,
                              normal_eps, device='cpu'):
        for name, value in [('hidden_dim', hidden_dim), ('bottleneck_dim', bottleneck_dim)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(normal_eps, bool) or not math.isfinite(normal_eps) or normal_eps <= 0:
            raise ValueError("normal_eps must be positive and finite.")
        self.ide_degree = ide_degree
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.normal_eps = normal_eps
        self._ide_fn = generate_ide_fn_pytorch(ide_degree).to(
            device=device, dtype=torch.float32)
        self._specular_mlp = SpecularMLP(
            self._ide_fn.output_dim, bottleneck_dim, hidden_dim).to(
                device=device, dtype=torch.float32)

    @property
    def appearance_config(self):
        return dict(ide_degree=self.ide_degree, hidden_dim=self.hidden_dim,
                    bottleneck_dim=self.bottleneck_dim, normal_eps=self.normal_eps)

    def _init_appearance(self, rgb=0.5, specular_tint=0.04, roughness=0.3):
        """Allocate appearance in every layout initialization path.

        Tensor RGB inputs retain their connection to autograd, as do the
        geometry inputs accepted by octpath_init().
        """
        device = self.octpath.device
        self._ide_fn.to(device=device, dtype=torch.float32)
        self._specular_mlp.to(device=device, dtype=torch.float32)

        def value_tensor(value, channels):
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
            return torch.broadcast_to(value, (self.num_voxels, channels))

        for name, value in [('_albedo', rgb), ('_specular_tint', specular_tint)]:
            value = value_tensor(value, 3)
            if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError(f"{name} initialization must be finite and in [0, 1].")
            raw = torch.logit(value.clamp(1e-6, 1 - 1e-6)).contiguous()
            setattr(self, name, raw.requires_grad_())
        roughness = value_tensor(roughness, 1) - self.MIN_ROUGHNESS
        if not torch.isfinite(roughness).all() or (roughness <= 0).any():
            raise ValueError(f"Initial roughness must exceed {self.MIN_ROUGHNESS}.")
        self._roughness = (
            roughness + torch.log(-torch.expm1(-roughness))).contiguous().requires_grad_()
        self._bottleneck = torch.randn(
            self.num_voxels, self.bottleneck_dim,
            dtype=torch.float32, device=device).mul_(0.01).requires_grad_()

    @property
    def albedo(self):
        return torch.sigmoid(self._albedo)

    @property
    def specular_tint(self):
        return torch.sigmoid(self._specular_tint)

    @property
    def roughness(self):
        return F.softplus(self._roughness) + self.MIN_ROUGHNESS
