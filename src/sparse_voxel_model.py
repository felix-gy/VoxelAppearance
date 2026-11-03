# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn.functional as F

from src.sparse_voxel_gears.constructor import SVConstructor
from src.sparse_voxel_gears.properties import SVProperties
from src.sparse_voxel_gears.renderer import SVRenderer
from src.sparse_voxel_gears.adaptive import SVAdaptive
from src.sparse_voxel_gears.io import SVInOut
from src.sparse_voxel_gears.pooling import SVPooling
from src.utils.ide_utils import generate_ide_fn_pytorch
import torch.nn as nn


class SpecularMLP(nn.Module):
    def __init__(self, ide_dim=72, bottleneck_dim=8, scalar_dim=1, hidden_dim=64):
        super().__init__()
        in_features = ide_dim + bottleneck_dim + scalar_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, ide_encoding, bottleneck_vec, scalar):
        # Concatenamos la dirección física con el contexto espacial local
        x = torch.cat([ide_encoding, bottleneck_vec, scalar], dim=-1)
        return self.net(x)

class SparseVoxelModel(SVConstructor, SVProperties, SVRenderer, SVAdaptive, SVInOut, SVPooling):

    def __init__(self,
                 n_samp_per_vox=1,       # Number of sampled points per visited voxel
                 sh_degree=3,            # Use 3 * (k+1)^2 params per voxels for view-dependent colors
                 ss=1.5,                 # Super-sampling rates for anti-aliasing
                 white_background=False, # Assum white background
                 black_background=False, # Assum black background
                 ide_dim =72,
                 hidden_dim=64,
                 bottleneck_dim=8

                 ):
        '''
        Setup of the model meta. At this point, no voxel is allocated.
        Use the following methods to allocate voxels and parameters.

        1. `model_load` defined in `src/sparse_voxel_gears/io.py`.
           Load the saved models from a given path.

        2. `model_init` defined in `src/sparse_voxel_gears/constructor.py`.
           Heuristically initial the sparse grid layout and parameters from the training datas.
        '''
        super().__init__()

        self.n_samp_per_vox = n_samp_per_vox
        self.max_sh_degree = sh_degree
        self.ss = ss
        self.white_background = white_background
        self.black_background = black_background

        # List the variable names
        self.per_voxel_attr_lst = [
            'octpath', 'octlevel',
            '_subdiv_p',
        ]
        self.per_voxel_param_lst = [
            '_sh0', '_shs','_albedo', '_roughness', '_specular_tint', '_bottleneck',
        ]
        self.grid_pts_param_lst = [
            '_geo_grid_pts',
        ]

        # To be init from model_init
        self.scene_center = None
        self.scene_extent = None
        self.inside_extent = None
        self.octpath = None
        self.octlevel = None
        self.active_sh_degree = sh_degree

        self._geo_grid_pts = None
        self._sh0 = None
        self._shs = None
        self._subdiv_p = None

        # REF
        self._albedo = None
        self._roughness = None
        self._specular_tint = None
        self._bottleneck= None

        self._ide_fn = generate_ide_fn_pytorch(deg_view=5)

        self._specular_mlp = SpecularMLP(ide_dim=ide_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, scalar_dim=1)

    @property
    def albedo(self):
        return torch.sigmoid(self._albedo)

    @property
    def specular_tint(self):
        return torch.sigmoid(self._specular_tint)

    @property
    def roughness(self):
        return F.softplus(self._roughness) + 1e-4

