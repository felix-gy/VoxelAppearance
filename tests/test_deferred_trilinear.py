import pytest
import torch
from test_deferred_mlp import fixture

from src.sparse_voxel_gears.deferred import LINEAR_MODE, NHT_MODE, save_deferred
from src.sparse_voxel_model import SparseVoxelModel
from train_deferred import make_optimizer, restore_training, training_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("mode", [NHT_MODE, LINEAR_MODE])
def test_trilinear_forward_backward_and_resume(mode, tmp_path):
    torch.manual_seed(41)
    model, camera = fixture((0,))
    with torch.no_grad():
        reference = model.render(camera, color_mode="sh0")["color"]
    model.init_deferred(mode)
    pack = model.render(camera, return_deferred_buffers=True)
    torch.testing.assert_close(pack["color"], reference, rtol=1e-5, atol=1e-5)
    y, x = 7, 8
    ray = camera.compute_rd()[:, y, x]
    ray = ray / ray.norm()
    size = model.vox_size[0]
    corner = model.vox_center[0] - 0.5 * size
    enter = (corner - camera.position) / ray
    leave = (corner + size - camera.position) / ray
    t = (torch.minimum(enter, leave).max() + torch.maximum(enter, leave).min()) / 2
    q = (camera.position + t * ray - corner) / size
    corners = q.new_tensor([[j >> 2, (j >> 1) & 1, j & 1] for j in range(8)])
    weights = torch.where(corners.bool(), q, 1 - q).prod(1)
    torch.testing.assert_close(weights.sum(), q.new_tensor(1.))
    z = weights @ model.deferred.features[0]
    descriptor = torch.cat((z.sin(), z.cos())) if mode == NHT_MODE else z
    opacity = pack["A"][0, y, x]
    assert opacity > 0 and (pack["A"] == 0).any()
    assert descriptor.shape == (8,)
    torch.testing.assert_close(pack["F"][:, y, x], opacity * descriptor, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(pack["B"][:, y, x], (1 - 0.55 * opacity).expand(3))
    assert (pack["F"][:, pack["A"][0] == 0] == 0).all()

    optimizer = make_optimizer(model)

    def loss_for(current):
        return (current.render(camera)["color"] - (1 - 0.3 * pack["A"].detach())).square().mean()

    loss_for(model).backward()
    assert model.deferred.features.grad.abs().max() == 0
    optimizer.step()
    optimizer.zero_grad()
    loss_for(model).backward()
    for param in (model.deferred.features, model._sh0, model.deferred.decoder[0].weight,
                  model.deferred.decoder[-1].bias):
        flat = param.view(-1)
        grad = param.grad.flatten()
        index = grad.abs().argmax().item()
        assert grad.isfinite().all() and grad[index].abs() > 0
        with torch.no_grad():
            base = flat[index].clone()
            for h in (1e-3, 1e-2):
                flat[index] = base + h
                plus = loss_for(model)
                flat[index] = base - h
                minus = loss_for(model)
                torch.testing.assert_close(grad[index], (plus - minus) / (2*h), rtol=1e-2, atol=1e-4)
            flat[index] = base

    def update(current, optim):
        optim.zero_grad()
        loss = loss_for(current)
        assert loss.isfinite()
        loss.backward()
        optim.step()
        return loss.item()

    losses = [update(model, optimizer) for _ in range(10)]
    assert losses[-1] < losses[0]
    assert model._geo_grid_pts.grad is None and model._shs.grad is None
    save_deferred(model, tmp_path, 11, "", training_state(optimizer, ["fixture"]))
    loaded = SparseVoxelModel(white_background=True, ss=1, appearance_mode=mode)
    loaded.load_iteration(tmp_path, load_training=True)
    other = make_optimizer(loaded)
    restore_training(loaded.loaded_training, other, ["fixture"])
    torch.testing.assert_close(loaded.render(camera)["color"], model.render(camera)["color"],
                               atol=1e-5, rtol=1e-5)
    assert update(loaded, other) == pytest.approx(update(model, optimizer), abs=1e-6)
    with pytest.raises(ValueError, match="compatible"):
        SparseVoxelModel(appearance_mode="deferred_mlp").load_iteration(tmp_path)
