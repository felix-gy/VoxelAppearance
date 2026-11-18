import pytest
import torch

import svraster_cuda
from src.cameras import MiniCam
from src.sparse_voxel_gears.deferred import MODE, save_deferred
from src.sparse_voxel_model import SparseVoxelModel
from train_deferred import make_optimizer, restore_training, training_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def fixture(ids=(0, 1), white=True):
    model = SparseVoxelModel(white_background=white, black_background=not white, ss=1.)
    paths = torch.tensor(ids, device="cuda") << (3 * (svraster_cuda.meta.MAX_NUM_LEVELS - 1))
    model.octpath_init([0., 0., 3.5], 4., paths, 1, density=-3., rgb=0.45)
    pose = torch.eye(4)
    pose[:2, 3] = -1.
    return model, MiniCam(pose.numpy(), 1.55, 1.55, 16, 16)


def test_composition_and_initial_identity():
    model, camera = fixture()
    kwargs = dict(output_depth=True, output_normal=True, output_T=True)
    with torch.no_grad():
        model._shs.fill_(0.05)
        sh3 = model.render(camera, color_mode="sh3")["color"]
        reference = model.render(camera, color_mode="sh0", ss=1.25, **kwargs)
        model.init_deferred()
        torch.testing.assert_close(model.render(camera, color_mode="sh3")["color"], sh3,
                                   atol=0, rtol=0)
        model.deferred.features[0].fill_(0.2)
        model.deferred.features[1].fill_(0.7)
        image = model.render(camera, ss=1.25, **kwargs)
        for key in ("color", "depth", "normal", "T"):
            torch.testing.assert_close(image[key], reference[key], atol=1e-5, rtol=1e-5)
        buffers = model.render(camera, return_deferred_buffers=True, output_T=True)
        transmittance = []
        for i in range(2):
            single, _ = fixture((i,))
            transmittance.append(single.render(camera, output_T=True)["T"])
        t0, t1 = transmittance
        w0, w1 = 1 - t0, t0 * (1 - t1)
        torch.testing.assert_close(buffers["F"], (0.2*w0 + 0.7*w1).expand_as(buffers["F"]))
        torch.testing.assert_close(buffers["A"], w0 + w1)
        torch.testing.assert_close(buffers["T"], t0 * t1)
        torch.testing.assert_close(buffers["B"], (0.45*(w0+w1) + t0*t1).expand_as(buffers["B"]))
        assert (buffers["A"] == 0).any() and (w1 > 0).any()
        model.deferred.decoder[-1].bias.fill_(-0.3)
        corrected = model.render(camera)["color"]
        assert (corrected[:, buffers["A"][0] == 0] == 1).all()


def test_gradients():
    torch.manual_seed(13)
    model, camera = fixture(white=False)
    baseline = model.render(camera, color_mode="sh0")["color"].detach()
    model.init_deferred()
    optimizer = make_optimizer(model)

    def objective():
        return model.render(camera)["color"].mean()

    torch.testing.assert_close(model.render(camera)["color"], baseline, atol=1e-5, rtol=1e-5)
    objective().backward()
    assert model.deferred.features.grad.abs().max() == 0
    assert model.deferred.decoder[0].weight.grad.abs().max() == 0
    optimizer.step()
    optimizer.zero_grad()
    objective().backward()
    parameters = (model.deferred.features, model._sh0, model.deferred.decoder[0].weight,
                  model.deferred.decoder[-1].weight)
    for param in parameters:
        flat = param.view(-1)
        grad = param.grad.flatten()
        index = grad.abs().argmax().item()
        assert grad[index].abs() > 0
        with torch.no_grad():
            value = flat[index].clone()
            for delta in (1e-3, 1e-2):
                flat[index] = value + delta
                plus = objective()
                flat[index] = value - delta
                minus = objective()
                torch.testing.assert_close(grad[index], (plus-minus)/(2*delta), atol=1e-4, rtol=1e-2)
            flat[index] = value
    assert model._geo_grid_pts.grad is None and model._shs.grad is None


def test_fit_checkpoint_and_resume(tmp_path):
    model, camera = fixture(white=False)
    model.init_deferred()
    optimizer = make_optimizer(model)
    fixed = {key: value.clone() for key, value in model.model_state().items()
             if torch.is_tensor(value) and key != "_sh0"}
    with torch.no_grad():
        target = 0.65 * model.render(camera, return_deferred_buffers=True)["A"]

    def step(current, optim):
        optim.zero_grad()
        image = current.render(camera)["color"]
        loss = (image - target).square().mean()
        loss.backward()
        optim.step()
        return loss.item()

    losses = [step(model, optimizer) for _ in range(20)]
    assert 0 < losses[-1] < losses[0]
    for key, value in fixed.items():
        assert torch.equal(model.model_state()[key], value)
    save_deferred(model, tmp_path, 20, "", training_state(optimizer, ["fixture"]))
    loaded = SparseVoxelModel(black_background=True, ss=1., appearance_mode=MODE)
    assert loaded.load_iteration(tmp_path, load_training=True) == 20
    other = make_optimizer(loaded)
    restore_training(loaded.loaded_training, other, ["fixture"])
    for left, right in zip(model.deferred.parameters(), loaded.deferred.parameters()):
        assert torch.equal(left, right)
    torch.testing.assert_close(loaded.render(camera)["color"], model.render(camera)["color"],
                               atol=1e-5, rtol=1e-5)
    assert step(loaded, other) == pytest.approx(step(model, optimizer), abs=1e-6)
    with pytest.raises(ValueError, match="Expected SH"):
        SparseVoxelModel().load_iteration(tmp_path)
