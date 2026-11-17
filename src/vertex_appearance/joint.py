"""Opt-in adaptive Vertex-MLP model; the original trainer owns its schedule."""
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.sparse_voxel_model import SparseVoxelModel
from src.sparse_voxel_gears.adaptive import agg_voxel_into_grid_pts, subdivide_by_interp
from .decoder import VertexAppearance
from .checkpoint import layout_fingerprint


class JointVoxelModel(SparseVoxelModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vertex_module = None
        self.joint_training_state = None
        self.joint_config = None

    def activate_vertex(self):
        if self.vertex_module is not None:
            return False
        # Initializing the replacement must not consume the baseline's RNG stream.
        with torch.random.fork_rng():
            self.vertex_module = VertexAppearance(self.num_grid_pts).cuda()
        self.active_sh_degree = 0
        self._shs.requires_grad_(False)
        return True

    def angular_parameters(self):
        return (list(self.vertex_module.parameters()) if self.vertex_module is not None
                else [self._shs])

    def render(self, camera, **options):
        if self.vertex_module is not None and options.get('color_mode') is None and 'vertex_vox_fn' not in options:
            return super().render_vertex(camera, self.vertex_module, joint_training=True, **options)
        return super().render(camera, **options)

    def reset_sh_from_cameras(self, cameras):
        super().reset_sh_from_cameras(cameras)
        if self.vertex_module is not None:
            with torch.no_grad():
                self.vertex_module.decoder[-1].weight.zero_()
                self.vertex_module.decoder[-1].bias.zero_()

    @torch.no_grad()
    def _adapt_vertex(self, mask, subdivide):
        if self.vertex_module is None:
            return (super().subdividing(mask) if subdivide else super().pruning(mask))
        mask = mask.reshape(-1)
        if (subdivide and not mask.any()) or (not subdivide and mask.all()):
            return
        keys = self.grid_pts_key.clone()
        old = self.vertex_module.vertex_features.detach()
        values = old[self.vox_key]
        if subdivide:
            values = torch.cat((values[~mask], subdivide_by_interp(values[mask])), 0)
            super().subdividing(mask)
        else:
            values = values[~mask]
            super().pruning(mask)
        self._shs.requires_grad_(False)
        features = agg_voxel_into_grid_pts(self.num_grid_pts, self.vox_key, values)
        # Preserve old vertices exactly, even across mixed-resolution interfaces.
        both = torch.cat((keys, self.grid_pts_key), 0)
        _, inverse = both.unique(dim=0, return_inverse=True)
        mapping = torch.full((int(inverse.max())+1,), -1, device=keys.device, dtype=torch.long)
        mapping[inverse[:len(keys)]] = torch.arange(len(keys), device=keys.device)
        source = mapping[inverse[len(keys):]]
        valid = source >= 0
        features[valid] = old[source[valid]]
        self.vertex_module.vertex_features = nn.Parameter(features)

    def pruning(self, mask):
        return self._adapt_vertex(mask, False)

    def subdividing(self, mask):
        return self._adapt_vertex(mask, True)

    def save(self, path, quantize=False):
        if quantize:
            raise ValueError('Joint checkpoints require float32; omit --save_quantized')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=Path(path).parent, suffix='.tmp')
        os.close(fd)
        try:
            super().save(temporary, quantize=False)
            payload = torch.load(temporary, map_location='cpu', weights_only=False)
            payload['vertex_joint'] = dict(
                schema_version=1, active=self.vertex_module is not None,
                module=None if self.vertex_module is None else {k:v.detach().cpu() for k,v in self.vertex_module.state_dict().items()},
                layout=layout_fingerprint(self), config=self.joint_config,
                training=self.joint_training_state,
                priority=self._subdiv_p.grad.detach().cpu() if self._subdiv_p.grad is not None else None)
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.latest_save_path = path

    def load(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        joint = payload.get('vertex_joint')
        if joint is None or joint.get('schema_version') != 1:
            raise ValueError('Expected a joint checkpoint; use the v2 loader for frozen appearance')
        super().load(path)
        self.vertex_module = None
        if layout_fingerprint(self) != joint['layout']:
            raise ValueError('Joint checkpoint layout mismatch')
        if joint['active']:
            self.activate_vertex()
            self.vertex_module.load_state_dict(joint['module'])
        self.joint_config = joint['config']
        self.joint_training_state = joint['training']
        if joint.get('priority') is not None:
            self._subdiv_p.grad = joint['priority'].cuda()


def training_state(optimizer, scheduler, iteration, camera_order, render_options):
    return dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                iteration=iteration, camera_order=camera_order,
                render_options={k:v for k,v in render_options.items() if k != 'gt_color'},
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def restore_training(state, optimizer, scheduler):
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
