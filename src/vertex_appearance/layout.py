"""Validate the fixed global vertex layout before allocating appearance state."""
import torch


def validate_layout(model):
    p, v = model.num_grid_pts, model.num_voxels
    key, xyz = model.vox_key, model.grid_pts_xyz
    if p < 1 or v < 1 or key.shape != (v, 8) or key.dtype != torch.int64:
        raise ValueError('Expected nonempty global int64 voxel keys [V,8]')
    if xyz.shape != (p, 3) or xyz.dtype != torch.float32 or not xyz.is_cuda:
        raise ValueError('Expected CUDA float32 vertex coordinates [P,3]')
    if key.device != xyz.device or key.min().item() < 0 or key.max().item() >= p:
        raise ValueError('Vertex keys are out of range or on another device')
    if not torch.isfinite(xyz).all():
        raise ValueError('Nonfinite vertex coordinates')
    if model._geo_grid_pts.shape != (p, 1) or model._sh0.shape != (v, 3):
        raise ValueError('Geometry or SH0 shape does not match the layout')
    # A bounded geometric probe avoids allocating another [V,8,3] buffer.
    idx = torch.linspace(0, v-1, min(v, 4096), device=xyz.device).long()
    bits = xyz.new_tensor([[k >> 2, (k >> 1) & 1, k & 1] for k in range(8)])
    expected = model.vox_center[idx, None, :] + (bits[None] - 0.5) * model.vox_size[idx, None, :]
    tolerance = max(1e-6, abs(model.scene_extent.item()) * 1e-6)
    if not torch.allclose(xyz[key[idx]], expected, atol=tolerance, rtol=1e-6):
        raise ValueError('Vertex corner order does not match the CUDA trilinear order')
