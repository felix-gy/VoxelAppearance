"""Small CUDA checks for the RBF4 path. Run with python -m pytest -s."""

import math

import pytest
import torch

import svraster_cuda
from src.cameras import MiniCam
from src.sparse_voxel_gears.rbf4_iq_sine import MODE, activate_rbf4
from src.sparse_voxel_model import SparseVoxelModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def fixture_model(voxels=1, samples=1):
    model = SparseVoxelModel(
        n_samp_per_vox=samples, ss=1, black_background=True, appearance_mode=MODE,
    )
    # At level one, corners 0 and 4 are adjacent along x.
    paths = torch.tensor([0, 4][:voxels], device="cuda")
    paths <<= 3 * (svraster_cuda.meta.MAX_NUM_LEVELS - 1)
    model.octpath_init([0., 0., 3.5], 4., paths, 1, density=-0.4, rgb=0.45)
    model.init_rbf_field()
    pose = torch.eye(4)
    pose[:2, 3] = -1.
    camera = MiniCam(pose.numpy(), 0.85, 0.85, 16, 16)
    return model, camera


def optimizer_for(model):
    return svraster_cuda.sparse_adam.SparseAdam([
        {'params': [model._geo_grid_pts]},
        {'params': [model._sh0]},
        {'params': model.angular_parameters()},
    ], lr=0.01)


def reference(model, t, ray):
    field = model.rbf_field
    corners = t.new_tensor([[i >> 2, (i >> 1) & 1, i & 1] for i in range(8)])
    keys = model.vox_key[0]
    scales = field.rbf_log_scales[keys].clamp(-2., 2.)
    phi = 1 / (1 + ((t - corners).square() * (-2 * scales).exp()).sum(-1))
    alpha = phi / phi.sum()
    spatial = (field.rbf_features[keys] *
               (alpha[:, None] * field.frequencies + field.rbf_phase).sin()).sum(0)
    direction = torch.cat([fn((2**k) * ray) for k in range(3) for fn in (torch.sin, torch.cos)])
    encoded = torch.cat((spatial, direction))
    return phi, alpha, encoded, field.fc2(field.fc1(encoded).relu())


def test_reference_and_initial_render():
    torch.manual_seed(17)
    cuda_state, cpu_state = torch.cuda.get_rng_state(), torch.get_rng_state()
    model, camera = fixture_model(samples=3)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_state)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert sum(p.numel() for p in model.rbf_field.parameters()) == 7*model.num_grid_pts + 1671

    direction = torch.tensor([0., 0., 1.], device="cuda")
    for point in ([0., 0., 0.], [1., 1., 1.], [0.3, 0., 0.6], [0.5, 0.5, 0.5]):
        phi, alpha, encoded, residual = reference(model, direction.new_tensor(point), direction)
        assert torch.all(phi > 0)
        torch.testing.assert_close(alpha.sum(), alpha.new_tensor(1.))
        assert encoded.shape == (22,) and encoded.isfinite().all()
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=0, rtol=0)

    with torch.no_grad():
        original = model.render(camera, output_depth=True, output_normal=True, output_T=True, track_max_w=True)
        model.rbf_active = True
        neural = model.render(camera, output_depth=True, output_normal=True, output_T=True, track_max_w=True)
        for key in ('color', 'depth', 'normal', 'T', 'max_w'):
            assert original[key].isfinite().all()
            torch.testing.assert_close(neural[key], original[key], atol=1e-6, rtol=0)
        assert original['max_w'].max() > 0


def test_backward_against_reference_and_finite_differences():
    torch.manual_seed(23)
    model, camera = fixture_model(samples=3)
    model.rbf_active = True
    field = model.rbf_field
    with torch.no_grad():
        field.rbf_features.mul_(50)
        field.rbf_log_scales.uniform_(-0.2, 0.2)
        field.rbf_phase.fill_(0.18)
        field.fc1.weight[0].zero_()
        field.fc1.weight[0, :4].fill_(0.4)
        field.fc1.bias[0] = 3.0
        field.fc2.weight[:, 0] = 0.10

    pixel = (9, 6)
    row, col = pixel
    ray = camera.position.new_tensor([
        (col + 0.5 - camera.cx) * 2 * camera.tanfovx / camera.image_width,
        (row + 0.5 - camera.cy) * 2 * camera.tanfovy / camera.image_height,
        1.,
    ])
    ray = ray / ray.norm()
    size = model.vox_size[0]
    corner = model.vox_center[0] - size / 2
    enter = (corner - camera.position) / ray
    exit = (corner + size - camera.position) / ray
    midpoint = (torch.minimum(enter, exit).max() + torch.maximum(enter, exit).min()) / 2
    t = (camera.position + midpoint * ray - corner) / size

    weights = ray.new_tensor([1., 0.7, -0.4])

    def render_objective():
        image = model.render(camera, output_T=True, debug=True)
        return (weights * image['raw_color'][:, row, col]).sum(), image['raw_T'][0, row, col]

    actual, transmittance = render_objective()
    *_, residual = reference(model, t, ray)
    expected = (1 - transmittance.detach()) * (weights * (0.45 + residual)).sum()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    parameters = list(field.parameters())
    reference_grads = torch.autograd.grad(expected, parameters)
    actual.backward()
    for p, expected_grad in zip(parameters, reference_grads):
        assert p.grad.isfinite().all()
        torch.testing.assert_close(p.grad, expected_grad, atol=1e-4, rtol=1e-2)
    for p in (model._geo_grid_pts, model._sh0):
        assert p.grad.isfinite().all() and p.grad.abs().max() > 0

    for p, index in ((field.rbf_features, (0, 2)), (field.rbf_log_scales, (0, 0)),
                     (field.rbf_phase, (2,))):
        with torch.no_grad():
            original = p[index].clone()
            p[index] = original + 0.002
            plus, _ = render_objective()
            p[index] = original - 0.002
            minus, _ = render_objective()
            p[index] = original
        assert p.grad[index].abs() > 1e-5
        torch.testing.assert_close(p.grad[index], (plus - minus) / 0.004, atol=1e-4, rtol=1e-2)


def test_overfit_and_checkpoint(tmp_path):
    torch.manual_seed(33)
    torch.cuda.reset_peak_memory_stats()
    model, camera = fixture_model()
    model.rbf_active = True
    model.freeze_vox_geo()
    model._sh0.requires_grad_(False)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60], gamma=0.3)

    with torch.no_grad():
        initial = model.render(camera, output_T=True)
        alpha = 1 - initial['T']
        target_rgb = alpha.new_tensor([0.62, 0.31, 0.53]).view(3, 1, 1)
        target = alpha * target_rgb
        start_loss = (initial['color'] - target).square().mean().item()

    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.render(camera)['color'] - target).square().mean()
        loss.backward()
        optimizer.step()
        model.rbf_field.project_scales()
        scheduler.step()
    assert loss.item() < 0.01 * start_loss

    model.save_iteration(tmp_path, 1079, training_state={
        'iteration': 1079,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    })
    loaded = SparseVoxelModel(ss=1, black_background=True, appearance_mode=MODE)
    assert loaded.load_iteration(tmp_path, load_training=True) == 1079
    restored_optimizer = optimizer_for(loaded)
    restored_optimizer.load_state_dict(loaded.loaded_training_state['optimizer'])
    restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(restored_optimizer, [60])
    restored_scheduler.load_state_dict(loaded.loaded_training_state['scheduler'])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for original, restored in zip(model.angular_parameters(), loaded.angular_parameters()):
        assert restored_optimizer.state[restored]['step'] == optimizer.state[original]['step']
        torch.testing.assert_close(restored_optimizer.state[restored]['exp_avg'],
                                   optimizer.state[original]['exp_avg'])
    with pytest.raises(ValueError, match="SH checkpoint"):
        SparseVoxelModel().load_iteration(tmp_path)

    with torch.no_grad():
        loaded.freeze_vox_geo()
        image = loaded.render(camera)['color']
        torch.testing.assert_close(image, model.render(camera)['color'], atol=1e-5, rtol=0)
        psnr = -10 * math.log10((image - target).square().mean().item())
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            loaded.render(camera)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / 100
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"16x16, 1 voxel, 1 view: loss {start_loss:.5g} -> {loss.item():.5g}, "
          f"PSNR {psnr:.2f} dB, {ms:.4f} ms/render, {1000/ms:.1f} FPS, "
          f"{peak_mib:.3f} MiB peak PyTorch CUDA allocation")


def test_activation_and_adaptation():
    torch.manual_seed(42)
    model, camera = fixture_model(voxels=2)
    field = model.rbf_field
    optimizer = optimizer_for(model)
    activate_rbf4(model, 999, optimizer)
    assert not model.rbf_active
    model.render(camera)['color'].mean().backward()
    optimizer.step()
    warmup_step = optimizer.state[model._shs]['step']
    assert warmup_step == 1

    activate_rbf4(model, 1000, optimizer)
    assert model.rbf_active and model.active_sh_degree == 0
    assert all(optimizer.state[p]['step'] == warmup_step for p in field.parameters())
    optimizer.zero_grad(set_to_none=True)
    model.render(camera)['color'].mean().backward()
    upstream = (field.rbf_features, field.rbf_log_scales, field.rbf_phase, field.fc1.weight)
    assert all(p.grad.abs().max() == 0 for p in upstream)
    assert field.fc2.weight.grad.abs().max() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.render(camera)['color'].mean().backward()
    assert all(p.grad.abs().max() > 0 for p in upstream)

    with torch.no_grad():
        field.rbf_log_scales.fill_(2.5)
        field.project_scales()
        assert (field.rbf_log_scales == 2.).all()
        old_keys = list(map(tuple, model.grid_pts_key.tolist()))
        old_features = dict(zip(old_keys, field.rbf_features.clone()))
        phase = field.rbf_phase
        weights = field.fc1.weight
        first_corners = field.rbf_features[model.vox_key[0]].clone()
        center_key = tuple(model.grid_pts_key[model.vox_key[0]].float().mean(0).long().tolist())

    model.subdividing(torch.tensor([True, False], device="cuda"))
    assert model.num_voxels == 9
    assert field.rbf_phase is phase and field.fc1.weight is weights
    for i, key in enumerate(map(tuple, model.grid_pts_key.tolist())):
        if key in old_features:
            assert torch.equal(field.rbf_features[i], old_features[key])
            assert (field.rbf_log_scales[i] == 2.).all()
        else:
            assert (field.rbf_log_scales[i] == 0.).all()
        if key == center_key:
            torch.testing.assert_close(field.rbf_features[i], first_corners.mean(0))

    mask = torch.zeros(9, dtype=torch.bool, device="cuda")
    mask[0] = True
    before_pruning = dict(zip(map(tuple, model.grid_pts_key.tolist()), field.rbf_features.detach()))
    model.pruning(mask)
    assert model.num_voxels == 8
    for key, feature in zip(map(tuple, model.grid_pts_key.tolist()), field.rbf_features):
        assert torch.equal(feature, before_pruning[key])
    optimizer = optimizer_for(model)
    optimizer.zero_grad(set_to_none=True)
    model.render(camera, lambda_dist=0.01, lambda_ascending=0.01,
                 lambda_R_concen=0.01, gt_color=torch.full((3, 16, 16), 0.4, device="cuda"),
                 output_depth=True, output_normal=True)['color'].mean().backward()
    for p in (field.rbf_features, field.rbf_log_scales, field.rbf_phase, field.fc1.weight,
              model._geo_grid_pts, model._sh0, model._subdiv_p):
        assert p.grad is not None and p.grad.isfinite().all()
