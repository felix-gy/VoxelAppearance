"""Parameters and layout operations for the RBF4 appearance field."""

import torch
from torch import nn

from src.sparse_voxel_gears.adaptive import (
    agg_voxel_into_grid_pts,
    mask_cat_perm,
    subdivide_by_interp,
)

MODE = 'rbf4_iq_sine'
ACTIVATION_ITERATION = 1000
ARCHITECTURE = (22, 64, 3)
FREQUENCIES = (1., 2., 4., 8.)
SCALE_LIMITS = (-2., 2.)


class RBF4IQSine(nn.Module):
    def __init__(self, n_vertices, device):
        super().__init__()
        device = torch.device(device)
        # Allocate the field without advancing the baseline CPU or CUDA RNG.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            options = dict(device=device, dtype=torch.float32)
            self.rbf_features = nn.Parameter(0.01 * torch.randn(n_vertices, 4, **options))
            self.rbf_log_scales = nn.Parameter(torch.zeros(n_vertices, 3, **options))
            self.rbf_phase = nn.Parameter(torch.zeros(4, **options))
            self.fc1 = nn.Linear(22, 64, **options)
            self.fc2 = nn.Linear(64, 3, **options)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        self.register_buffer('frequencies', torch.tensor(FREQUENCIES, **options), persistent=False)

    def raster_tensors(self, vox_key):
        mlp = torch.cat([p.reshape(-1) for p in (*self.fc1.parameters(), *self.fc2.parameters())])
        return (vox_key, self.rbf_features, self.rbf_log_scales, self.rbf_phase, mlp, self.frequencies)

    @torch.no_grad()
    def project_scales(self):
        self.rbf_log_scales.clamp_(*SCALE_LIMITS)

    @torch.no_grad()
    def remap(self, old_keys, new_keys, old_vox_key, new_vox_key, kept_idx, subdivide_idx=None):
        # Match exact integer coordinates across the two layouts.
        _, idx = torch.cat((old_keys, new_keys)).unique(dim=0, return_inverse=True)
        old_idx, new_idx = idx.split((len(old_keys), len(new_keys)))
        lookup = old_idx.new_full((len(idx),), -1)
        lookup[old_idx] = torch.arange(len(old_idx), device=old_idx.device)
        source = lookup[new_idx]
        surviving = source >= 0

        if subdivide_idx is not None:
            corners = self.rbf_features[old_vox_key]
            children = subdivide_by_interp(corners[subdivide_idx])
            contributions = mask_cat_perm(corners, kept_idx=kept_idx, cat_tensor=children)
            features = agg_voxel_into_grid_pts(len(new_keys), new_vox_key, contributions)
        else:
            features = self.rbf_features.new_empty(len(new_keys), 4)

        scales = self.rbf_log_scales.new_zeros(len(new_keys), 3)
        features[surviving] = self.rbf_features[source[surviving]]
        scales[surviving] = self.rbf_log_scales[source[surviving]]
        self.rbf_features = nn.Parameter(features)
        self.rbf_log_scales = nn.Parameter(scales)


def activate_rbf4(model, iteration, optimizer):
    """Replace the SH angular group using the same bias-correction clock."""
    if model.appearance_mode != MODE or model.rbf_active or iteration < ACTIVATION_ITERATION:
        return
    angular = optimizer.param_groups[2]
    shs, = angular['params']
    step = optimizer.state.pop(shs, {}).get('step', 0)
    angular['params'] = list(model.rbf_field.parameters())
    for p in angular['params']:
        optimizer.state[p] = {
            'step': step,
            'exp_avg': torch.zeros_like(p),
            'exp_avg_sq': torch.zeros_like(p),
        }
    model.rbf_active = True
