# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import hashlib
import os
import re
import tempfile

import torch

from src.sparse_voxel_gears.rbf4_iq_sine import (
    ACTIVATION_ITERATION, ARCHITECTURE, FREQUENCIES, MODE, SCALE_LIMITS,
)

RBF_FORMAT = {
    'format': MODE,
    'version': 1,
    'architecture': ARCHITECTURE,
    'frequencies': FREQUENCIES,
    'log_scale_limits': SCALE_LIMITS,
    'view_direction': 'normalized_world_camera_to_sample',
    'direction_fourier': (1., 2., 4.),
    'direction_layout': 'sin_xyz_then_cos_xyz_per_frequency',
}


def layout_fingerprint(model):
    digest = hashlib.sha256()
    for tensor in (model.octpath, model.octlevel, model.grid_pts_key, model.vox_key):
        array = tensor.detach().cpu().numpy()
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.astype(array.dtype.newbyteorder('<'), copy=False).tobytes())
    return digest.hexdigest()


def atomic_save(state, path):
    directory = os.path.dirname(path) or '.'
    with tempfile.NamedTemporaryFile(dir=directory, prefix='rbf4-', suffix='.tmp', delete=False) as temp:
        temp_path = temp.name
    try:
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


class SVInOut:

    def save(self, path, quantize=False, training_state=None):
        '''
        Save the necessary attributes and parameters for reproducing rendering.
        '''
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        neural = self.appearance_mode == MODE
        if neural and quantize:
            raise ValueError("RBF4 checkpoints require full precision")
        if neural and self.rbf_field is None:
            raise ValueError("RBF4 field has not been initialized")
        state_dict = {
            'active_sh_degree': self.active_sh_degree,
            'scene_center': self.scene_center.data.contiguous(),
            'inside_extent': self.inside_extent.data.contiguous(),
            'scene_extent': self.scene_extent.data.contiguous(),
            'octpath': self.octpath.data.contiguous(),
            'octlevel': self.octlevel.data.contiguous(),
            '_geo_grid_pts': self._geo_grid_pts.data.contiguous(),
            '_sh0': self._sh0.data.contiguous(),
            '_shs': self._shs.data.contiguous(),
        }

        if quantize:
            quantize_state_dict(state_dict)
            state_dict['quantized'] = True
        else:
            state_dict['quantized'] = False

        for k, v in state_dict.items():
            if torch.is_tensor(v):
                state_dict[k] = v.cpu()
        if neural:
            priority = self.subdivision_priority
            atomic_save({
                **RBF_FORMAT,
                'active': self.rbf_active,
                'layout_fingerprint': layout_fingerprint(self),
                'model': state_dict,
                'render_settings': {
                    'n_samp_per_vox': self.n_samp_per_vox,
                    'ss': self.ss,
                    'white_background': self.white_background,
                    'black_background': self.black_background,
                },
                'field': {name: value.cpu() for name, value in self.rbf_field.state_dict().items()},
                'subdivision_priority': None if priority is None else priority.detach().cpu(),
                'training': training_state,
            }, path)
        else:
            torch.save(state_dict, path)
        self.latest_save_path = path

    def load(self, path):
        '''
        Load the saved models.
        '''
        self.loaded_path = path
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        checkpoint = state_dict
        neural = self.appearance_mode == MODE
        if neural:
            if any(checkpoint.get(key) != value for key, value in RBF_FORMAT.items()):
                raise ValueError("Expected a compatible rbf4_iq_sine checkpoint")
            if type(checkpoint.get('active')) is not bool:
                raise ValueError("Invalid RBF4 activation state")
            state_dict = checkpoint['model']
            settings = checkpoint['render_settings']
            self.n_samp_per_vox = settings['n_samp_per_vox']
            self.ss = settings['ss']
            self.white_background = settings['white_background']
            self.black_background = settings['black_background']
        elif 'format' in checkpoint or 'active_sh_degree' not in checkpoint:
            raise ValueError("Expected an SVRaster SH checkpoint. Use the RBF4 reader for RBF4 models.")

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

        if neural:
            if layout_fingerprint(self) != checkpoint.get('layout_fingerprint'):
                raise ValueError("RBF4 checkpoint layout fingerprint mismatch")
            self.init_rbf_field()
            target = self.rbf_field.state_dict()
            field = checkpoint['field']
            if field.keys() != target.keys() or any(
                field[name].shape != value.shape or field[name].dtype != value.dtype
                for name, value in target.items()
            ):
                raise ValueError("RBF4 checkpoint parameter layout mismatch")
            self.rbf_field.load_state_dict(field)
            self.rbf_active = checkpoint['active']
            priority = checkpoint.get('subdivision_priority')
            if priority is not None:
                self._subdiv_p.grad = priority.cuda()
        return checkpoint.get('training') if neural else None

    def save_iteration(self, model_path, iteration, quantize=False, training_state=None):
        path = os.path.join(model_path, "checkpoints", f"iter{iteration:06d}_model.pt")
        self.save(path, quantize=quantize, training_state=training_state)
        self.latest_save_iter = iteration

    def load_iteration(self, model_path, iteration=-1, *, load_training=False):
        if iteration == -1:
            # Find the maximum iteration if it is -1.
            fnames = os.listdir(os.path.join(model_path, "checkpoints"))
            iterations = [int(match[1]) for fname in fnames
                          if (match := re.fullmatch(r"iter(\d+)_model\.pt", fname))]
            if not iterations:
                raise FileNotFoundError(f"No model checkpoints in {model_path}")
            loaded_iter = max(iterations)
        else:
            loaded_iter = iteration

        path = os.path.join(model_path, "checkpoints", f"iter{loaded_iter:06d}_model.pt")
        training_state = self.load(path)
        if training_state is not None and training_state.get('iteration') != loaded_iter:
            raise ValueError("RBF4 checkpoint iteration mismatch")
        if self.appearance_mode == MODE and self.rbf_active != (loaded_iter >= ACTIVATION_ITERATION):
            raise ValueError("RBF4 checkpoint activation does not match its iteration")

        self.loaded_training_state = training_state if load_training else None
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
