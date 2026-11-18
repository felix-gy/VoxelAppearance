# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import torch

from src.sparse_voxel_gears.deferred import DEFERRED_MODES, restore_deferred

class SVInOut:

    def save(self, path, quantize=False):
        '''
        Save the necessary attributes and parameters for reproducing rendering.
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.appearance_mode in DEFERRED_MODES:
            raise ValueError("Use save_deferred for deferred appearance models")
        state_dict = self.model_state()

        if quantize:
            quantize_state_dict(state_dict)
            state_dict['quantized'] = True
        else:
            state_dict['quantized'] = False

        for k, v in state_dict.items():
            if torch.is_tensor(v):
                state_dict[k] = v.cpu()
        torch.save(state_dict, path)
        self.latest_save_path = path

    def model_state(self):
        return {
            'active_sh_degree': self.active_sh_degree,
            'scene_center': self.scene_center.detach().contiguous(),
            'inside_extent': self.inside_extent.detach().contiguous(),
            'scene_extent': self.scene_extent.detach().contiguous(),
            'octpath': self.octpath.detach().contiguous(),
            'octlevel': self.octlevel.detach().contiguous(),
            '_geo_grid_pts': self._geo_grid_pts.detach().contiguous(),
            '_sh0': self._sh0.detach().contiguous(),
            '_shs': self._shs.detach().contiguous(),
        }

    def load(self, path):
        '''
        Load the saved models.
        '''
        self.loaded_path = path
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        if self.appearance_mode in DEFERRED_MODES:
            restore_deferred(self, state_dict)
            return state_dict
        if "appearance_mode" in state_dict:
            raise ValueError("Expected SH checkpoint. Select the matching deferred mode to load a deferred model.")
        self.load_model_state(state_dict)
        return None

    def load_model_state(self, state_dict):
        if state_dict.get('quantized', False):
            dequantize_state_dict(state_dict)

        self.active_sh_degree = state_dict['active_sh_degree']

        self.scene_center = state_dict['scene_center'].cuda()
        self.inside_extent = state_dict['inside_extent'].cuda()
        self.scene_extent = state_dict['scene_extent'].cuda()

        self.octpath = state_dict['octpath'].cuda()
        self.octlevel = state_dict['octlevel'].cuda().to(torch.int8)

        self._geo_grid_pts = state_dict['_geo_grid_pts'].cuda().requires_grad_()
        self._sh0 = state_dict['_sh0'].cuda().requires_grad_()
        self._shs = state_dict['_shs'].cuda().requires_grad_()

        # Subdivision priority trackor
        self._subdiv_p = torch.ones(
            [self.num_voxels, 1],
            dtype=torch.float32, device="cuda").requires_grad_()

    def save_iteration(self, model_path, iteration, quantize=False):
        path = os.path.join(model_path, "checkpoints", f"iter{iteration:06d}_model.pt")
        self.save(path, quantize=quantize)
        self.latest_save_iter = iteration

    def load_iteration(self, model_path, iteration=-1, *, load_training=False):
        if iteration == -1:
            # Find the maximum iteration if it is -1.
            fnames = os.listdir(os.path.join(model_path, "checkpoints"))
            loaded_iter = max(int(m[1]) for fname in fnames
                              if (m := re.fullmatch(r"iter(\d+)_model\.pt", fname)))
        else:
            loaded_iter = iteration

        path = os.path.join(model_path, "checkpoints", f"iter{loaded_iter:06d}_model.pt")
        checkpoint = self.load(path)
        if checkpoint is not None and checkpoint["step"] != loaded_iter:
            raise ValueError("Deferred checkpoint step mismatch")
        self.loaded_training = checkpoint["training"] if load_training and checkpoint else None

        self.loaded_iter = loaded_iter

        return loaded_iter


# Quantization utilities to reduce size when saving model.
# It can reduce ~70% model size with minor PSNR drop.
def quantize_state_dict(state_dict):
    state_dict['_geo_grid_pts'] = quantization(state_dict['_geo_grid_pts'])
    state_dict['_sh0'] = [quantization(v) for v in state_dict['_sh0'].split(1, dim=1)]
    state_dict['_shs'] = [quantization(v) for v in state_dict['_shs'].split(1, dim=1)]

def dequantize_state_dict(state_dict):
    state_dict['_geo_grid_pts'] = dequantization(state_dict['_geo_grid_pts'])
    state_dict['_sh0'] = torch.cat(
        [dequantization(v) for v in state_dict['_sh0']], dim=1)
    state_dict['_shs'] = torch.cat(
        [dequantization(v) for v in state_dict['_shs']], dim=1)

def quantization(src_tensor, max_iter=10):
    src_shape = src_tensor.shape
    src_vals = src_tensor.flatten().contiguous()
    order = src_vals.argsort()
    quantile_ind = (torch.linspace(0,1,257) * (len(order) - 1)).long().clamp_(0, len(order)-1)
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
