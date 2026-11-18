import pytest
import torch
from test_deferred_mlp import fixture

from src.sparse_voxel_gears.adaptive import subdivide_by_interp
from src.sparse_voxel_gears.deferred import JOINT_MODE, save_deferred
from src.sparse_voxel_model import SparseVoxelModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("path", ["features", "opacity", "combined"])
def test_joint_density_and_priority(path):
    torch.manual_seed(4)
    model, camera = fixture((0, 1))
    model.init_deferred(JOINT_MODE)
    with torch.no_grad():
        model.deferred.features.normal_(0, 0.4)
        model.deferred.decoder[-1].weight.normal_(0, 0.15)
        model.deferred.decoder[-1].bias.fill_(0.13)

    def objective():
        pkg = model.render(camera, return_deferred_buffers=True)
        if path != "combined":
            f = pkg["F"] if path == "features" else pkg["F"].detach()
            a = pkg["A"].detach() if path == "features" else pkg["A"]
            b = pkg["B"].detach()
            # Hold the independent branches fixed for numeric perturbations too.
            return model.deferred(f if path == "features" else fixed_f, fixed_b,
                                  fixed_a if path == "features" else a, direction)[:, 7, 8].sum()
        return pkg["raw_color"][:, 7, 8].sum()

    initial = model.render(camera, return_deferred_buffers=True)
    fixed_a, fixed_b, fixed_f = (initial[k].detach() for k in ("A", "B", "F"))
    direction = camera.compute_rd()
    objective().backward()
    density = model._geo_grid_pts
    idx = density.grad.abs().argmax().item()
    assert density.grad.abs().max() > 1e-5
    assert model.subdivision_priority.abs().max() > 0
    params = [density]
    if path == "combined":
        params.extend((model.deferred.features, model._sh0, model.deferred.decoder[-1].weight))
    for param in params:
        flat, grad = param.view(-1), param.grad.flatten()
        assert grad.isfinite().all() and grad.abs().max() > 0
        index = idx if param is density else grad.abs().argmax().item()
        with torch.no_grad():
            value = flat[index].clone()
            h = 0.002
            flat[index] = value + h
            plus = objective()
            flat[index] = value - h
            minus = objective()
            flat[index] = value
        torch.testing.assert_close(grad[index], (plus-minus)/(2*h), atol=1e-4, rtol=1e-2)


def test_adaptation_activation_and_roundtrip(tmp_path):
    model, camera = fixture((0, 1))
    model.appearance_mode = JOINT_MODE
    model.active_sh_degree = 0
    model.render(camera)["color"].mean().backward()
    save_deferred(model, tmp_path, 999, "", {"warmup": True})
    warmup = SparseVoxelModel(white_background=True, ss=1, appearance_mode=JOINT_MODE)
    assert warmup.load_iteration(tmp_path, load_training=True) == 999
    assert warmup.deferred is None and warmup.loaded_training == {"warmup": True}
    model.subdividing(torch.tensor([True, False], device="cuda"))
    base = model.render(camera)["color"].detach()
    model.init_deferred(JOINT_MODE)
    torch.testing.assert_close(model.render(camera)["color"], base, atol=1e-5, rtol=1e-5)
    model.render(camera)["color"].mean().backward()

    decoder = model.deferred.decoder
    old = model.deferred.features.detach().clone()
    prune = torch.arange(model.num_voxels, device="cuda") == 1
    model.pruning(prune)
    torch.testing.assert_close(model.deferred.features, old[~prune], atol=0, rtol=0)
    parent = model.deferred.features.detach().clone()
    subdivide = torch.arange(model.num_voxels, device="cuda") == 0
    model.subdividing(subdivide)
    expected = torch.cat((parent[1:], subdivide_by_interp(parent[:1])))
    torch.testing.assert_close(model.deferred.features, expected, atol=0, rtol=0)
    assert model.deferred.decoder is decoder
    optimizer = torch.optim.Adam([model._geo_grid_pts, model._sh0, *model.deferred.parameters()])
    optimizer.zero_grad()
    model.render(camera, lambda_R_concen=0.01, lambda_dist=0.01, lambda_ascending=0.01,
                 gt_color=camera.c2w.new_full((3, 16, 16), 0.5),
                 output_depth=True, output_normal=True)["color"].mean().backward()
    optimizer.step()
    assert model._geo_grid_pts.grad.isfinite().all()
    save_deferred(model, tmp_path, 1001, "", {"optimizer": optimizer.state_dict()})
    active = SparseVoxelModel(white_background=True, ss=1, appearance_mode=JOINT_MODE)
    assert active.load_iteration(tmp_path, load_training=True) == 1001
    torch.testing.assert_close(active.render(camera)["color"], model.render(camera)["color"],
                               atol=1e-5, rtol=1e-5)
    resumed = torch.optim.Adam([active._geo_grid_pts, active._sh0, *active.deferred.parameters()])
    resumed.load_state_dict(active.loaded_training["optimizer"])
    resumed.zero_grad()
    active.render(camera)["color"].mean().backward()
    resumed.step()
