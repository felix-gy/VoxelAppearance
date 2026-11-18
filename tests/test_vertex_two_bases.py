import pytest
import torch
from test_vertex_cuda import scene

from src.vertex_appearance.checkpoint import load_checkpoint, save_checkpoint
from src.vertex_appearance.decoder import VertexAppearance, VertexTwoBases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_two_bases_render_gradients_and_checkpoint(tmp_path):
    torch.manual_seed(37)
    model, camera = scene()
    one = VertexAppearance(model.num_grid_pts).cuda()
    two = VertexTwoBases(model.num_grid_pts, model.num_voxels).cuda()
    with torch.no_grad():
        one.decoder[-1].weight.normal_(std=0.02)
        two.initialize_first_basis(one.state_dict())
    bases = two(model.grid_pts_xyz, camera.position)
    assert bases.shape == (model.num_grid_pts, 2, 3)
    assert two.mix.shape == (model.num_voxels, 2)
    residual = two.voxel_residual(bases, model.vox_key)
    assert residual.shape == (model.num_voxels, 8, 3)
    assert residual.isfinite().all()
    torch.testing.assert_close(residual, one(model.grid_pts_xyz, camera.position)[model.vox_key],
                               atol=1e-6, rtol=0)

    def render(module):
        return model.render_vertex(camera, module, decode_all_vertices=True)["color"]

    torch.testing.assert_close(render(one), render(two), atol=1e-5, rtol=0)
    optimizer = torch.optim.Adam(two.parameters(), lr=0.01)
    losses = []
    for step in range(8):
        optimizer.zero_grad()
        loss = (render(two) - 0.72).square().mean()
        loss.backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in two.parameters())
        assert two.mix.grad[:, 1].abs().max() > 0
        second = two.decoder[-1].weight.grad[3:].abs().max()
        assert (second == 0) if step == 0 else (second > 0)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    identity = {"layout": "fixture"}
    path = tmp_path / "appearance.pt"
    save_checkpoint(path, two, optimizer, 8, {"mode": "vertex_two_bases"},
                    identity, torch.Generator(), frozen_sh0=model._sh0)
    restored = VertexTwoBases(model.num_grid_pts, model.num_voxels).cuda()
    loaded = load_checkpoint(path, restored, identity)
    assert torch.equal(model._sh0, loaded["frozen_sh0"].cuda())
    torch.testing.assert_close(render(two), render(restored), atol=1e-5, rtol=0)
    with pytest.raises(ValueError, match="mode mismatch"):
        load_checkpoint(path, one, identity)
    with pytest.raises(ValueError, match="geometry/layout"):
        load_checkpoint(path, restored, {"layout": "other"})


def test_neighbor_penalty_is_zero_for_equal_coefficients_and_smooths_difference():
    from src.vertex_appearance.neighbors import coefficient_penalty

    model, _ = scene()
    module = VertexTwoBases(model.num_grid_pts, model.num_voxels).cuda()
    penalty, edges = coefficient_penalty(model, strength=0.001, batch_size=4096)
    assert edges == 12
    assert penalty(module, 1).item() == 0
    with torch.no_grad():
        module.mix[0, 1] = 0.5
    loss = penalty(module, 1)
    assert loss.item() > 0
    torch.testing.assert_close(loss, penalty(module, 1), rtol=0, atol=0)
    loss.backward()
    assert module.mix.grad[0, 1] > 0 and module.mix.grad[:, 1].sum().abs() < 1e-8


def test_bounded_residual_identity_gate_gradient_and_limit(tmp_path):
    from src.vertex_appearance.decoder import VertexBoundedResidual

    torch.manual_seed(7)
    model, camera = scene()
    one = VertexAppearance(model.num_grid_pts).cuda()
    with torch.no_grad():
        one.decoder[-1].weight.normal_(std=0.02)
    bounded = VertexBoundedResidual(model.num_grid_pts, model.num_voxels).cuda()
    bounded.initialize_first_basis(one.state_dict())

    def render(module):
        return model.render_vertex(camera, module, decode_all_vertices=True)["color"]

    torch.testing.assert_close(render(one), render(bounded), atol=1e-5, rtol=0)
    optimizer = torch.optim.Adam(bounded.parameters(), lr=0.01)
    (render(bounded) - 0.72).square().mean().backward()
    assert bounded.voxel_gate.grad.abs().max() > 0
    assert torch.count_nonzero(bounded.decoder[-1].weight.grad[3:]) == 0
    optimizer.step()
    optimizer.zero_grad()
    (render(bounded) - 0.72).square().mean().backward()
    assert bounded.decoder[-1].weight.grad[3:].abs().max() > 0

    with torch.no_grad():
        bounded.voxel_gate.fill_(10000.)
        bases = bounded(model.grid_pts_xyz, camera.position)
        correction = bounded.voxel_residual(bases, model.vox_key) - bases[:, 0][model.vox_key]
        assert correction.abs().max() <= 0.25 + 1e-7
        torch.testing.assert_close(bounded.eta, torch.tensor(0.25, device="cuda"))
    path = tmp_path / "bounded.pt"
    mode = {"mode": bounded.appearance_mode}
    save_checkpoint(path, bounded, optimizer, 1, mode, {}, torch.Generator())
    recovered = VertexBoundedResidual(model.num_grid_pts, model.num_voxels, eta=0.1).cuda()
    load_checkpoint(path, recovered, {})
    torch.testing.assert_close(recovered.voxel_gate, bounded.voxel_gate)
    torch.testing.assert_close(recovered.eta, bounded.eta)
    torch.testing.assert_close(render(recovered), render(bounded), atol=1e-5, rtol=0)
