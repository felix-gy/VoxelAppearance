# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import warnings

import torch


APPEARANCE_PARAMS = ('_albedo', '_roughness', '_specular_tint', '_bottleneck')


class SVInOut:
    def save(self, path, quantize=False):
        """Save all appearance, shared weights and geometry without per-voxel SH."""
        os.makedirs(os.path.dirname(os.fspath(path)) or '.', exist_ok=True)
        state_dict = dict(
            format_version=2,
            appearance_config=self.appearance_config,
            render_config={name: getattr(self, name) for name in (
                'n_samp_per_vox', 'ss', 'white_background', 'black_background')},
            specular_mlp={k: v.detach().cpu().contiguous()
                          for k, v in self._specular_mlp.state_dict().items()},
            quantized=quantize,
        )
        for name in ('scene_center', 'inside_extent', 'scene_extent', 'octpath',
                     'octlevel', '_geo_grid_pts', *APPEARANCE_PARAMS):
            value = getattr(self, name)
            state_dict[name] = None if value is None else value.detach().cpu().contiguous()
        if self._geo_grid_pts is None:
            state_dict['frozen_vox_geo'] = self.frozen_vox_geo.detach().cpu().contiguous()
        if quantize:
            quantize_state_dict(state_dict)
        torch.save(state_dict, path)
        self.latest_save_path = path

    def load(self, path, device='cuda'):
        """Restore REF checkpoints, or warm-start from legacy SH constant color.

        Legacy view-dependent SH coefficients cannot be converted exactly to
        this architecture; the explicit warning describes that loss.
        """
        state = torch.load(path, map_location='cpu', weights_only=False)
        version = state.get('format_version', 1)
        if version not in (1, 2):
            raise ValueError(f"Unsupported model checkpoint version: {version}.")
        if state.get('quantized', False):
            dequantize_state_dict(state)
        for name in ('n_samp_per_vox', 'ss', 'white_background', 'black_background'):
            if name in state.get('render_config', {}):
                setattr(self, name, state['render_config'][name])
        self._invalidate_voxel_cache()
        self.loaded_path = path
        for name in ('scene_center', 'inside_extent', 'scene_extent'):
            setattr(self, name, state[name].to(device=device, dtype=torch.float32))
        self.octpath = state['octpath'].to(device=device, dtype=torch.int64)
        self.octlevel = state['octlevel'].to(device=device, dtype=torch.int8)
        self._geo_grid_pts = state.get('_geo_grid_pts')
        if self._geo_grid_pts is not None:
            self._geo_grid_pts = self._geo_grid_pts.to(
                device=device, dtype=torch.float32).contiguous().requires_grad_()
            if self._geo_grid_pts.shape != (self.num_grid_pts, 1):
                raise ValueError("Checkpoint geometry does not match its voxel layout.")
        elif 'frozen_vox_geo' in state:
            self.frozen_vox_geo = state['frozen_vox_geo'].to(
                device=device, dtype=torch.float32).contiguous()
            if self.frozen_vox_geo.shape != (self.num_voxels, 8):
                raise ValueError("Checkpoint corner geometry must have shape [N, 8].")
        else:
            raise ValueError("Checkpoint has no voxel geometry.")

        self.loaded_legacy_sh = version == 1 and '_sh0' in state
        if self.loaded_legacy_sh:
            warnings.warn(
                "Loading a legacy SH checkpoint as a REF warm start: constant "
                "RGB initializes albedo; directional SH color is discarded and "
                "specular appearance must be trained again.", UserWarning)
            self._configure_appearance(**self.appearance_config, device=device)
            self._init_appearance((state['_sh0'].to(device) * 0.28209479177387814 + 0.5).clamp(0, 1))
        else:
            if version != 2:
                raise ValueError("Checkpoint is neither a REF model nor a supported legacy SH model.")
            self._configure_appearance(**state['appearance_config'], device=device)
            channels = (3, 1, 3, self.bottleneck_dim)
            for name, count in zip(APPEARANCE_PARAMS, channels):
                value = state[name].to(device=device, dtype=torch.float32).contiguous()
                if value.shape != (self.num_voxels, count):
                    raise ValueError(f"Invalid checkpoint shape for {name}: {tuple(value.shape)}.")
                setattr(self, name, value.requires_grad_())
            self._specular_mlp.load_state_dict(state['specular_mlp'], strict=True)
        self._subdiv_p = torch.ones(
            self.num_voxels, 1, dtype=torch.float32, device=device).requires_grad_()

    def save_iteration(self, model_path, iteration, quantize=False):
        path = os.path.join(model_path, 'checkpoints', f'iter{iteration:06d}_model.pt')
        self.save(path, quantize=quantize)
        self.latest_save_iter = iteration

    def load_iteration(self, model_path, iteration=-1):
        if iteration == -1:
            directory = os.path.join(model_path, 'checkpoints')
            matches = [re.fullmatch(r'iter(\d+)_model\.pt', f) for f in os.listdir(directory)]
            iterations = [int(m.group(1)) for m in matches if m is not None]
            if not iterations:
                raise FileNotFoundError(f'No model checkpoints found in {directory}.')
            iteration = max(iterations)
        path = os.path.join(model_path, 'checkpoints', f'iter{iteration:06d}_model.pt')
        self.load(path)
        self.loaded_iter = iteration
        return iteration


def quantize_state_dict(state):
    # Shared network weights are small and remain full precision.
    for name in ('_geo_grid_pts', 'frozen_vox_geo', *APPEARANCE_PARAMS):
        value = state.get(name)
        if torch.is_tensor(value):
            state[name] = quantization(value)


def dequantize_state_dict(state):
    for name in ('_geo_grid_pts', 'frozen_vox_geo', *APPEARANCE_PARAMS):
        value = state.get(name)
        if isinstance(value, dict):
            state[name] = dequantization(value)
    # Read the old per-channel SH quantization only for checkpoint migration.
    for name in ('_sh0', '_shs'):
        if isinstance(state.get(name), list):
            parts = [dequantization(value) for value in state[name]]
            state[name] = (torch.cat(parts, dim=1) if parts else
                           torch.empty(len(state['octpath']), 0, 3))


def quantization(src_tensor, max_iter=10):
    src_shape = src_tensor.shape
    if src_tensor.numel() == 0:
        return dict(index=src_tensor.to(torch.uint8), codebook=src_tensor.new_zeros(256))
    src_vals = src_tensor.flatten().contiguous()
    order = src_vals.argsort()
    quantile_ind = (torch.linspace(0,1,257, device=src_tensor.device) * (len(order) - 1)).long().clamp_(0, len(order)-1)
    codebook = src_vals[order[quantile_ind]].contiguous()
    codebook[0] = -torch.inf
    ind = torch.searchsorted(codebook, src_vals)

    codebook = codebook[1:]
    ind = (ind - 1).clamp_(0, 255)

    diff_l = (src_vals - codebook[ind-1]).abs()
    diff_m = (src_vals - codebook[ind]).abs()
    ind = ind - 1 + (diff_m < diff_l)
    ind.clamp_(0, 255)

    for _ in range(max_iter):
        codebook = torch.zeros_like(codebook).index_reduce_(
            dim=0,
            index=ind,
            source=src_vals,
            reduce='mean',
            include_self=False)
        diff_l = (src_vals - codebook[ind-1]).abs()
        diff_r = (src_vals - codebook[(ind+1).clamp_max_(255)]).abs()
        diff_m = (src_vals - codebook[ind]).abs()
        upd_mask = torch.minimum(diff_l, diff_r) < diff_m
        if upd_mask.sum() == 0:
            break
        shift = (diff_r < diff_l) * 2 - 1
        ind[upd_mask] += shift[upd_mask]
        ind.clamp_(0, 255)

    codebook = torch.zeros_like(codebook).index_reduce_(
        dim=0,
        index=ind,
        source=src_vals,
        reduce='mean',
        include_self=False)

    return dict(
        index=ind.reshape(src_shape).to(torch.uint8),
        codebook=codebook,
    )

def dequantization(quant_dict):
    return quant_dict['codebook'][quant_dict['index'].long()]
