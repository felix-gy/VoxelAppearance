"""CPU integration checks with reference octree/gather operations.

Run: python -m unittest discover -s tests -v
CUDA rasterization is tested separately when a GPU is available.
"""

import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import sph_harm_y
import torch

import svraster_cuda
from src.config import cfg
from src.sparse_voxel_model import SparseVoxelModel
from src.sparse_voxel_gears import constructor
from src.sparse_voxel_gears.io import quantization
from src.utils.ide_utils import generate_ide_fn_pytorch, get_ml_array
from src.utils.optimizer_utils import (
    create_voxel_optimizer, create_specular_optimizer, create_scheduler,
)


MAX_LEVEL = svraster_cuda.meta.MAX_NUM_LEVELS


def encode_octpath(ijk, levels):
    levels = levels.reshape(-1, 1).long()
    path = torch.zeros_like(levels)
    for depth in range(1, MAX_LEVEL + 1):
        bits = (ijk >> (levels - depth).clamp_min(0)) & 1
        child = (bits[:, 0:1] << 2) | (bits[:, 1:2] << 1) | bits[:, 2:3]
        path |= torch.where(levels >= depth, child << (3 * (MAX_LEVEL - depth)), 0)
    return path


def decode_octpath(path, levels):
    ijk = torch.zeros(len(path), 3, dtype=torch.int64, device=path.device)
    for depth in range(1, MAX_LEVEL + 1):
        child = (path >> (3 * (MAX_LEVEL - depth))) & 7
        bits = torch.cat([(child >> 2) & 1, (child >> 1) & 1, child & 1], dim=-1)
        ijk = torch.where(levels >= depth, (ijk << 1) | bits, ijk)
    return ijk


def gather_geometry(keys, idx, grid):
    # Poison unselected rows: shader code must never read them.
    out = grid.new_full((len(keys), 8), float('nan'))
    out[idx] = grid[keys[idx]].squeeze(-1)
    return out


class IDETests(unittest.TestCase):
    def test_values_against_independent_spherical_harmonics(self):
        generator = torch.Generator().manual_seed(83)
        xyz = torch.nn.functional.normalize(torch.randn(100, 3, generator=generator), dim=-1)
        xyz = torch.cat([xyz, torch.tensor([[0., 0., 1.], [0., 0., -1.]])])
        theta = np.arccos(xyz[:, 2].double().clamp(-1, 1).numpy())
        phi = np.arctan2(xyz[:, 1].numpy(), xyz[:, 0].numpy())
        for degree, dimension in enumerate((4, 10, 20, 38, 72), start=1):
            with self.subTest(degree=degree):
                harmonics = np.stack([
                    sph_harm_y(int(l), int(m), theta, phi)
                    for m, l in get_ml_array(degree).T], axis=-1)
                reference = np.concatenate([harmonics.real, harmonics.imag], axis=-1)
                encoding = generate_ide_fn_pytorch(degree).float()
                actual = encoding(xyz, torch.zeros(len(xyz), 1))
                self.assertEqual(actual.shape, (len(xyz), dimension))
                np.testing.assert_allclose(actual.numpy(), reference, atol=2e-5, rtol=2e-5)

    def test_gradients_and_degenerate_directions_are_finite(self):
        encoding = generate_ide_fn_pytorch(5).double()
        xyz = torch.tensor([[0.3, 0.4, math.sqrt(0.75)]], dtype=torch.float64, requires_grad=True)
        rough = torch.tensor([[0.07]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(encoding, (xyz, rough), fast_mode=True))
        poles = torch.tensor([[0., 0., 1.], [0., 0., -1.], [0., 0., 0.]], requires_grad=True)
        rough = torch.full((3, 1), 1e-4, requires_grad=True)
        encoding.float()(poles, rough).square().sum().backward()
        self.assertTrue(poles.grad.isfinite().all())
        self.assertTrue(rough.grad.isfinite().all())


class AppearanceIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.patches = [
            patch.object(svraster_cuda.utils, 'ijk_2_octpath', encode_octpath),
            patch.object(svraster_cuda.utils, 'octpath_2_ijk', decode_octpath),
            patch.object(svraster_cuda.renderer.GatherGeoParams, 'apply', gather_geometry),
        ]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        self.indices = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
        self.camera = torch.tensor([2., 1., -2.])

    def make_model(self, **kwargs):
        model = SparseVoxelModel(ide_degree=3, hidden_dim=16, bottleneck_dim=5, **kwargs)
        model.ijkl_init(torch.zeros(3), 1., self.indices, 1)
        with torch.no_grad():
            xyz = model.grid_pts_xyz
            model._geo_grid_pts.copy_((xyz * torch.tensor([0.4, 0.2, 0.3])).sum(-1, keepdim=True))
        return model

    def test_all_initialization_routes_and_configurable_dimensions(self):
        model = self.make_model()
        self.assertEqual(model._bottleneck.shape, (3, 5))
        self.assertEqual(model._specular_mlp.net[0].in_features, 20 + 5 + 1)
        self.assertFalse(hasattr(model, '_sh0'))
        self.assertFalse(hasattr(model, '_shs'))
        torch.testing.assert_close(model.roughness, torch.full((3, 1), 0.3))
        # model_init's layout allocation is CUDA-only; supply a tiny CPU layout.
        path, level = model.octpath.clone(), model.octlevel.clone()
        with patch.object(constructor, 'octlayout_inside_uniform', return_value=(path, level)):
            model.model_init(torch.tensor([[-0.5] * 3, [0.5] * 3]), 0,
                             albedo_init=0.2, roughness_init=0.1)
        torch.testing.assert_close(model.albedo, torch.full((3, 3), 0.2))
        self.assertEqual(model._bottleneck.shape, (3, 5))
        model.points_init(torch.zeros(3), 1., torch.tensor([
            [-0.3, -0.3, -0.3], [-0.2, -0.2, -0.2], [0.3, -0.3, -0.3]]),
            octlevel=1, rgb=torch.tensor([[0.2] * 3, [0.4] * 3, [0.8] * 3]))
        self.assertEqual(model.num_voxels, 2)
        torch.testing.assert_close(model.albedo[:, 0].sort().values, torch.tensor([0.3, 0.8]))
        with self.assertRaises(ValueError):
            SparseVoxelModel(ide_degree=3, ide_dim=72)

    def test_visible_subset_empty_subset_and_viewdir(self):
        model = self.make_model()
        idx = torch.tensor([2, 0])
        result = model.vox_fn(idx, self.camera)
        self.assertEqual(result['rgbs'].shape, (3, 3))
        self.assertTrue(result['rgbs'].isfinite().all())
        torch.testing.assert_close(result['rgbs'][1], torch.zeros(3))
        dirs = model.vox_center - self.camera
        explicit = model.vox_fn(idx, None, viewdir=dirs)
        torch.testing.assert_close(explicit['rgbs'], result['rgbs'])
        subset = model.vox_fn(idx, None, viewdir=dirs[idx])
        torch.testing.assert_close(subset['rgbs'], result['rgbs'])
        empty = model.vox_fn(torch.empty(0, dtype=torch.long), self.camera)
        torch.testing.assert_close(empty['rgbs'], torch.zeros(3, 3))
        albedo = model.vox_fn(idx, self.camera, color_mode='albedo')['rgbs']
        torch.testing.assert_close(albedo[idx], model.albedo[idx])
        with self.assertRaisesRegex(ValueError, 'SH colors have been replaced'):
            model.vox_fn(idx, self.camera, color_mode='sh0')

    def test_flat_geometry_has_finite_zero_normal_gradients(self):
        model = self.make_model()
        idx = torch.tensor([0, 2])
        corners = torch.full((2, 8), -10., requires_grad=True)
        model.evaluate_color(idx, corners, self.camera).sum().backward()
        torch.testing.assert_close(corners.grad, torch.zeros_like(corners))
        self.assertTrue(all(p.grad is None or p.grad.isfinite().all()
                            for p in model._specular_mlp.parameters()))

    def test_checkpointed_chunks_match_colors_and_all_gradients(self):
        model = self.make_model()
        idx = torch.tensor([2, 0])
        directions = model.vox_center - self.camera
        params = [model._geo_grid_pts, *(getattr(model, n) for n in model.per_voxel_param_lst),
                  *model._specular_mlp.parameters()]
        model.color_checkpoint = False
        expected = model.vox_fn(idx, self.camera)['rgbs']
        expected_grads = torch.autograd.grad(expected.square().sum(), params)
        model.color_chunk_size = 1
        for use_checkpoint in (False, True):
            model.color_checkpoint = use_checkpoint
            for viewdir in (None, directions, directions[idx]):
                with self.subTest(checkpoint=use_checkpoint, viewdir=viewdir is not None):
                    actual = model.vox_fn(idx, self.camera, viewdir=viewdir)['rgbs']
                    actual_grads = torch.autograd.grad(actual.square().sum(), params)
                    torch.testing.assert_close(actual, expected)
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-4)
        with torch.no_grad():
            torch.testing.assert_close(model.vox_fn(idx, self.camera)['rgbs'], expected)

    def test_checkpoint_retains_less_activation_data_and_handles_frozen_geometry(self):
        model = self.make_model(color_chunk_size=1)
        model.freeze_vox_geo()
        idx = torch.arange(model.num_voxels)

        def retained_elements(checkpoint_enabled):
            model.color_checkpoint = checkpoint_enabled
            saved_bytes = []

            def pack(tensor):
                saved_bytes.append(tensor.numel() * tensor.element_size())
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                rgb = model.vox_fn(idx, self.camera)['rgbs']
            rgb.sum().backward()
            self.assertIsNotNone(model._albedo.grad)
            self.assertIsNotNone(next(model._specular_mlp.parameters()).grad)
            return sum(saved_bytes)

        full = retained_elements(False)
        checkpointed = retained_elements(True)
        self.assertLess(checkpointed, full / 4)
        with self.assertRaises(ValueError):
            SparseVoxelModel(color_chunk_size=0)

    def test_initialization_uses_float32_even_if_default_dtype_differs(self):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = self.make_model()
        finally:
            torch.set_default_dtype(original_dtype)
        for value in [model._geo_grid_pts, *model._specular_mlp.parameters(),
                      *(getattr(model, n) for n in model.per_voxel_param_lst)]:
            self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(model.vox_fn(torch.arange(3), self.camera)['rgbs'].isfinite().all())

    def test_optimizer_updates_every_appearance_field_and_geometry(self):
        model = self.make_model()
        optimizer = create_voxel_optimizer(model, cfg.optimizer, torch.optim.Adam)
        specular_optimizer = create_specular_optimizer(model, cfg.optimizer)
        params = [model._geo_grid_pts] + [getattr(model, n) for n in model.per_voxel_param_lst]
        before = [p.detach().clone() for p in params]
        net_before = [p.detach().clone() for p in model._specular_mlp.parameters()]
        idx = torch.arange(model.num_voxels)
        optimizer.zero_grad(set_to_none=True)
        specular_optimizer.zero_grad(set_to_none=True)
        loss = (model.vox_fn(idx, self.camera)['rgbs'] - 0.1).square().mean()
        loss.backward()
        for value in params:
            self.assertIsNotNone(value.grad)
            self.assertTrue(value.grad.isfinite().all())
        optimizer.step()
        specular_optimizer.step()
        for old, value in zip(before, params):
            self.assertFalse(torch.equal(old, value))
        self.assertTrue(any(not torch.equal(old, value) for old, value in
                            zip(net_before, model._specular_mlp.parameters())))
        network_param = next(model._specular_mlp.parameters())
        moments = copy.deepcopy(specular_optimizer.state_dict())
        model.pruning(torch.tensor([False, True, False]))
        optimizer = create_voxel_optimizer(model, cfg.optimizer, torch.optim.Adam)
        self.assertIs(next(model._specular_mlp.parameters()), network_param)
        torch.testing.assert_close(specular_optimizer.state[network_param]['exp_avg'],
                                   moments['state'][0]['exp_avg'])
        self.assertIs(optimizer.param_groups[1]['params'][0], model._albedo)
        specular_optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(p.grad is None for p in model._specular_mlp.parameters()))

    def test_pruning_subdivision_and_frozen_geometry_preserve_appearance(self):
        model = self.make_model()
        with torch.no_grad():
            model._albedo.copy_(torch.arange(9).reshape(3, 3))
        model.freeze_vox_geo()
        parent = {n: getattr(model, n)[0].clone() for n in model.per_voxel_param_lst}
        network = model._specular_mlp
        model.subdividing(torch.tensor([True, False, False]))
        self.assertEqual(model.num_voxels, 10)
        self.assertEqual(model.frozen_vox_geo.shape, (10, 8))
        for name, value in parent.items():
            torch.testing.assert_close(getattr(model, name)[2:], value.expand(8, -1))
        model.pruning(torch.tensor([True] + [False] * 9))
        self.assertEqual(model._bottleneck.shape, (9, 5))
        self.assertIs(model._specular_mlp, network)
        model.unfreeze_vox_geo()
        self.assertTrue(model.vox_fn(torch.arange(9), self.camera)['rgbs'].isfinite().all())

    def test_checkpoint_roundtrip_with_and_without_quantization(self):
        model = self.make_model(ss=1.1, white_background=True, normal_eps=0.005)
        idx = torch.arange(model.num_voxels)
        expected = model.vox_fn(idx, self.camera)['rgbs'].detach()
        with tempfile.TemporaryDirectory() as tmp:
            for quantize in (False, True):
                with self.subTest(quantize=quantize):
                    path = Path(tmp) / 'model.pt'
                    model.save(path, quantize=quantize)
                    saved = torch.load(path, weights_only=False)
                    self.assertNotIn('_sh0', saved)
                    self.assertNotIn('_shs', saved)
                    restored = SparseVoxelModel()  # Different architecture on purpose.
                    restored.load(path, device='cpu')
                    self.assertEqual(restored.appearance_config, model.appearance_config)
                    self.assertEqual(restored.ss, model.ss)
                    self.assertTrue(restored.white_background)
                    actual = restored.vox_fn(idx, self.camera)['rgbs']
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0.01 if quantize else 0)
                    for value in restored._specular_mlp.parameters():
                        self.assertEqual(value.device.type, 'cpu')

    def test_resuming_both_optimizers_reproduces_the_next_step(self):
        model = self.make_model()
        voxel_optimizer = create_voxel_optimizer(model, cfg.optimizer, torch.optim.Adam)
        network_optimizer = create_specular_optimizer(model, cfg.optimizer)

        def step(current_model, voxel_opt, network_opt):
            voxel_opt.zero_grad(set_to_none=True)
            network_opt.zero_grad(set_to_none=True)
            rgb = current_model.vox_fn(torch.arange(3), self.camera)['rgbs']
            (rgb - 0.2).square().mean().backward()
            voxel_opt.step()
            network_opt.step()

        step(model, voxel_optimizer, network_optimizer)
        moments = copy.deepcopy((voxel_optimizer.state_dict(), network_optimizer.state_dict()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resume.pt'
            model.save(path)
            resumed = SparseVoxelModel()
            resumed.load(path, device='cpu')
        resumed_voxel = create_voxel_optimizer(resumed, cfg.optimizer, torch.optim.Adam)
        resumed_network = create_specular_optimizer(resumed, cfg.optimizer)
        resumed_voxel.load_state_dict(moments[0])
        resumed_network.load_state_dict(moments[1])
        step(model, voxel_optimizer, network_optimizer)
        step(resumed, resumed_voxel, resumed_network)
        for name in ['_geo_grid_pts', *model.per_voxel_param_lst]:
            torch.testing.assert_close(getattr(resumed, name), getattr(model, name))
        for restored, expected in zip(resumed._specular_mlp.parameters(), model._specular_mlp.parameters()):
            torch.testing.assert_close(restored, expected)

    def test_latest_checkpoint_ignores_other_files_and_records_actual_iteration(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_iteration(tmp, 2)
            model.save_iteration(tmp, 7)
            (Path(tmp) / 'checkpoints' / 'notes999.txt').write_text('not a checkpoint')
            restored = SparseVoxelModel()
            original_load = restored.load
            with patch.object(restored, 'load', side_effect=lambda p: original_load(p, device='cpu')):
                self.assertEqual(restored.load_iteration(tmp), 7)
            self.assertEqual(restored.loaded_iter, 7)

    def test_unshared_corners_reduce_save_and_adapt(self):
        model = self.make_model()
        corner_input = torch.randn(3, 8, requires_grad=True)
        model.octpath_init(torch.zeros(3), 1., model.octpath, model.octlevel,
                           density=corner_input, reduce_density=True)
        self.assertEqual(model._geo_grid_pts.shape, (model.num_grid_pts, 1))
        model._geo_grid_pts.sum().backward()
        self.assertTrue(corner_input.grad.isfinite().all())
        # A differentiable reduced input is non-leaf. Freezing should cache
        # detached corners without illegally changing that tensor's grad flag.
        original_grid = model._geo_grid_pts
        model.freeze_vox_geo()
        model.unfreeze_vox_geo()
        self.assertIs(model._geo_grid_pts, original_grid)
        model.octpath_init(torch.zeros(3), 1., model.octpath, model.octlevel,
                           density=corner_input.detach(), reduce_density=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'corners.pt'
            model.save(path)
            restored = SparseVoxelModel()
            restored.load(path, device='cpu')
        torch.testing.assert_close(restored.frozen_vox_geo, corner_input.detach())
        restored.subdividing(torch.tensor([True, False, False]))
        restored.pruning(torch.tensor([True] + [False] * 9))
        restored.unfreeze_vox_geo()
        self.assertEqual(restored._geo_grid_pts.shape, (restored.num_grid_pts, 1))

    def test_legacy_sh_checkpoint_is_explicit_warm_start(self):
        model = self.make_model()
        rgb = torch.tensor([[0.2, 0.4, 0.6]]).expand(3, 3)
        legacy = {n: getattr(model, n).detach() for n in (
            'scene_center', 'inside_extent', 'scene_extent', 'octpath', 'octlevel', '_geo_grid_pts')}
        legacy.update(_sh0=(rgb - 0.5) / 0.28209479177387814,
                      _shs=torch.zeros(3, 15, 3), active_sh_degree=3, quantized=True)
        legacy['_geo_grid_pts'] = quantization(legacy['_geo_grid_pts'])
        for name in ('_sh0', '_shs'):
            legacy[name] = [quantization(v) for v in legacy[name].split(1, dim=1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.pt'
            torch.save(legacy, path)
            with self.assertWarnsRegex(UserWarning, 'directional SH color is discarded'):
                model.load(path, device='cpu')
        self.assertTrue(model.loaded_legacy_sh)
        self.assertFalse(hasattr(model, '_sh0'))
        torch.testing.assert_close(model.albedo, rgb, atol=1e-5, rtol=1e-5)
        self.assertTrue(model.vox_fn(torch.arange(3), self.camera)['rgbs'].isfinite().all())

    def test_unclipped_training_image_matches_target_resolution_and_has_gradient(self):
        model = self.make_model(ss=2)
        # Use actual render() post-processing, replacing only the CUDA rasterizer.
        color = torch.full((3, 8, 8), 1.5, requires_grad=True)
        camera = type('Camera', (), dict(image_width=4, image_height=4, tanfovx=1.,
            tanfovy=1., cx=2., cy=2., w2c=torch.eye(4), c2w=torch.eye(4), near=.02))()
        outputs = (color, torch.zeros(3, 8, 8), torch.zeros(3, 8, 8),
                   torch.zeros(1, 8, 8), torch.zeros(3, 1))
        with patch.object(svraster_cuda.renderer, 'rasterize_voxels', return_value=outputs):
            result = model.render(camera)
        self.assertEqual(result['raw_color'].shape, (3, 8, 8))
        self.assertEqual(result['color_unclamped'].shape, (3, 4, 4))
        torch.testing.assert_close(result['color'], torch.ones(3, 4, 4))
        result['color_unclamped'].square().mean().backward()
        self.assertGreater(color.grad.abs().sum().item(), 0)


class SchedulerAndConfigTests(unittest.TestCase):
    def test_lr_schedule_survives_voxel_optimizer_recreation_and_resume(self):
        config = cfg.optimizer.clone()
        config.lr_decay_ckpt = [2, 4]
        config.lr_decay_mult = 0.1
        param = torch.ones(1, requires_grad=True)
        reference = torch.optim.Adam([param], lr=0.01)
        scheduler = create_scheduler(reference, config)
        for completed in range(6):
            fresh = torch.optim.Adam([param], lr=0.01)
            resumed = create_scheduler(fresh, config, completed)
            self.assertEqual(resumed.last_epoch, completed)
            self.assertAlmostEqual(fresh.param_groups[0]['lr'], reference.param_groups[0]['lr'])
            reference.step()
            scheduler.step()

    def test_legacy_config_migration(self):
        import src.config as config_module
        with tempfile.TemporaryDirectory() as tmp, patch.object(config_module, 'cfg', cfg.clone()):
            path = Path(tmp) / 'config.yaml'
            path.write_text('model:\n  sh_degree: 3\noptimizer:\n  sh0_lr: 0.02\n'
                            'init:\n  sh0_init: 0.4\n  sh_degree_init: 0\n'
                            'procedure:\n  reset_sh_ckpt: [1000]\n')
            with self.assertWarnsRegex(UserWarning, 'Migrated legacy SH'):
                config_module.update_config(str(path))
            config = config_module.cfg
            self.assertNotIn('sh_degree', config.model)
            self.assertEqual(config.optimizer.albedo_lr, 0.02)
            self.assertEqual(config.init.albedo_init, 0.4)
            self.assertEqual(config.procedure.reset_albedo_ckpt, [1000])
            config.procedure.reset_albedo_ckpt = []
            config_module.update_config([], ['--bottleneck_dim', '6'])
            self.assertEqual(config.model.bottleneck_dim, 6)

    def test_marching_cubes_extracts_plane_without_eager_cuda_allocation(self):
        from src.utils.marching_cubes_utils import torch_marching_cubes_voxels
        corners = torch.tensor([[x, y, z] for x in (0., 1.) for y in (0., 1.)
                                for z in (0., 1.)])
        vertices, faces = torch_marching_cubes_voxels(
            corners[:, 0].unsqueeze(0), corners.unsqueeze(0), iso=0.5)
        self.assertEqual(len(faces), 2)
        torch.testing.assert_close(vertices[:, 0], torch.full((len(vertices),), 0.5))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA rasterization needs a GPU')
class CUDASmokeTests(unittest.TestCase):
    def test_real_rasterizer_backward_optimizer_and_checkpoint(self):
        from src.cameras import MiniCam
        model = SparseVoxelModel(ss=1., black_background=True, ide_degree=3)
        ijk = torch.tensor([[x, y, z] for x in range(2) for y in range(2)
                            for z in range(2)], device='cuda')
        model.ijkl_init(torch.zeros(3, device='cuda'), 1., ijk, 1, density=-2.)
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = -2.
        camera = MiniCam(pose, fovx=math.pi / 3, fovy=math.pi / 3, width=16, height=16)
        vox_opt = create_voxel_optimizer(model, cfg.optimizer)
        net_opt = create_specular_optimizer(model, cfg.optimizer)
        result = model.render(camera)
        (result['color_unclamped'] - 0.2).square().mean().backward()
        for p in [model._geo_grid_pts, model._albedo, model._roughness,
                  model._specular_tint, model._bottleneck, *model._specular_mlp.parameters()]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(p.grad.isfinite().all())
        vox_opt.step()
        net_opt.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cuda.pt'
            model.save(path)
            restored = SparseVoxelModel()
            restored.load(path)
            torch.testing.assert_close(restored.render(camera)['color'], model.render(camera)['color'])


if __name__ == '__main__':
    unittest.main()
