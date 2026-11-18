"""Run the controlled frozen-geometry one/two-basis experiment."""
import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image

from src.utils import loss_utils
from src.vertex_appearance.checkpoint import load_checkpoint, save_checkpoint
from src.vertex_appearance.decoder import VertexAppearance, VertexTwoBases
from src.vertex_appearance.runtime import fit, setup
from src.vertex_appearance.neighbors import face_pairs

SOURCES = {"materials_t": "out/materials_t", "lego": "out/lego-original"}


@torch.no_grad()
def score(render, cameras):
    result, images = [], {}
    for camera in cameras:
        image = render(camera)
        gt = camera.image.cuda()
        result.append(dict(view=camera.image_name, psnr=loss_utils.psnr_score(image, gt).item(),
                           ssim=loss_utils.ssim_score(image, gt).item(),
                           lpips=loss_utils.correct_lpips_loss(image, gt).item()))
        if camera.image_name in ("r_0.png", "r_100.png"):
            images[camera.image_name] = pixels(image)
    return dict(per_view=result, **{k: sum(r[k] for r in result)/len(result)
                                   for k in ("psnr", "ssim", "lpips")}), images


def pixels(image):
    return (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)


@torch.no_grad()
def timings(model, module, cameras):
    events = {name: torch.cuda.Event(enable_timing=True) for name in (
        "preprocess_start", "preprocess_end", "appearance_start", "appearance_end", "gather_end",
        "raster_start", "raster_end", "post_end")}
    for cam in cameras:
        model.render_vertex(cam, module, decode_all_vertices=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    stats = []
    for _ in range(8):
        for camera in cameras:
            start = time.perf_counter()
            model.render_vertex(camera, module, decode_all_vertices=True, profile_events=events)
            torch.cuda.synchronize()
            stats.append(dict(
                total_ms=1000*(time.perf_counter()-start),
                decoder_ms=events["appearance_start"].elapsed_time(events["appearance_end"]),
                gather_mix_sh0_ms=events["appearance_end"].elapsed_time(events["gather_end"]),
                raster_ms=events["raster_start"].elapsed_time(events["raster_end"]),
                preprocess_ms=events["preprocess_start"].elapsed_time(events["preprocess_end"])))
    mean = {k: sum(row[k] for row in stats)/len(stats) for k in stats[0]}
    return dict(**mean, fps=1000/mean["total_ms"],
                peak_mib=torch.cuda.max_memory_allocated()/2**20)


@torch.no_grad()
def face_residuals(model, module, camera):
    bases = module(model.grid_pts_xyz, camera.position)
    jumps = []
    for axis, left, right in face_pairs(model):
        high = [j for j in range(8) if j & (1 << (2-axis))]
        low = [j for j in range(8) if not j & (1 << (2-axis))]
        key_l, key_r = model.vox_key[left][:, high], model.vox_key[right][:, low]
        assert torch.equal(key_l, key_r)
        if isinstance(module, VertexTwoBases):
            a = module.voxel_residual(bases, key_l, left)
            b = module.voxel_residual(bases, key_r, right)
        else:
            a, b = bases[key_l], bases[key_r]
        jumps.append((a-b).mean(1))
    jump = torch.cat(jumps)
    return dict(faces=len(jump), rms=jump.square().mean().sqrt().item(),
                max_abs=jump.abs().max().item())


def run(scene, root):
    root.mkdir(parents=True, exist_ok=True)
    source = Path(SOURCES[scene])
    geometry = source / "checkpoints/iter020000_model.pt"
    base = yaml.safe_load((source / "config.yaml").read_text())
    base["data"].update(res_width=256, res_downscale=0., eval=True)
    base["model"].update(ss=1., white_background=True)
    config = dict(seed=3721, steps=300, validation_interval=300, checkpoint_interval=300,
                  mode="vertex_trilinear_fourier3", geometry_checkpoint=str(geometry),
                  resolved_base=base, lr_vertex_features=0.00025, lr_mlp=0.00025,
                  lr_mix=0.00025)
    config_path = root / "protocol.yaml"
    config_path.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(config=config_path, appearance_checkpoint=None, resume=None)
    config, _, model, one, pack, identity = setup(args)
    train = sorted(pack.get_train_cameras(), key=lambda cam: cam.image_name)[:32]
    test = sorted(pack.get_test_cameras(), key=lambda cam: cam.image_name)[:4]
    assert (len(train), len(test)) == (32, 4)
    assert all((cam.image_width, cam.image_height) == (256, 256) for cam in train + test)
    result, images = {}, {}
    for mode in ("sh0", "sh3"):
        result[mode], images[mode] = score(lambda cam: model.render(cam, color_mode=mode)["color"], test)
    if scene == "materials_t":
        warm_path = Path("output/vertex-mlp-materials-t/vertex-seed3721-step300.pt")
        warm = torch.load(warm_path, map_location="cpu", weights_only=True)
        assert Path(warm["source"]) == geometry and warm["seed"] == 3721 and warm["step"] == 300
        assert warm["mode"] == config["mode"] and warm["train_views"] == [cam.image_name for cam in train]
        assert warm["sh0"].shape == model._sh0.shape
        one.load_state_dict(warm["appearance"])
        model._sh0.copy_(warm["sh0"])
        result["sh0_frozen"], images["sh0_frozen"] = score(lambda cam: model.render(cam, color_mode="sh0")["color"], test)
        initial, _ = score(lambda cam: model.render_vertex(cam, one, decode_all_vertices=True)["color"], test)
        historical = json.loads((warm_path.parent / "seed3721.json").read_text())["checkpoints"]["300"]["test"]
        for actual, previous in zip(initial["per_view"], historical["per_view"], strict=True):
            assert actual["view"] == previous["view"]
            assert abs(actual["psnr"]-previous["psnr"]) <= 1e-4
        del warm
    two = VertexTwoBases(model.num_grid_pts, model.num_voxels).cuda()
    two.initialize_first_basis(one.state_dict())
    with torch.no_grad():
        errors = [(model.render_vertex(c, one, decode_all_vertices=True)["color"] -
                   model.render_vertex(c, two, decode_all_vertices=True)["color"]).abs().max().item()
                  for c in test]
    assert max(errors) <= 1e-5
    result.update(initial_max_rgb_error=max(errors), identity=identity, gpu=torch.cuda.get_device_name(),
                  train_views=[c.image_name for c in train], test_views=[c.image_name for c in test])
    two.cpu()
    for name, module in (("one", one), ("two", two)):
        module.cuda()
        optimizer = torch.optim.Adam(module.parameters(), lr=0.00025, betas=(0.1, 0.99), eps=1e-15)
        order = [i % len(train) for i in range(300)]
        loss_utils.metric_networks.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        smoke = None
        for step, loss in fit(model, module, train, order, optimizer, 0, 300, dense=True):
            if step in (1, 15, 100, 300):
                print(scene, name, step, loss, torch.cuda.max_memory_allocated()/2**20, flush=True)
            if step == 15:
                smoke = dict(loss=loss, peak_mib=torch.cuda.max_memory_allocated()/2**20)
        assert all(int(state["step"]) == 300 for state in optimizer.state.values())
        peak = torch.cuda.max_memory_allocated()/2**20
        mode = "vertex_two_bases" if name == "two" else config["mode"]
        arm_config = dict(config, mode=mode, comparison=dict(dense=True, seed=3721, betas=[0.1, 0.99],
                          eps=1e-15, train_views=result["train_views"], test_views=result["test_views"]))
        arm_config["appearance"] = dict(config["appearance"], layers=[26, 64, 6 if name == "two" else 3])
        checkpoint = root / f"{name}-step300.pt"
        save_checkpoint(checkpoint, module, optimizer, 300, arm_config, identity,
                        torch.Generator().manual_seed(3721), frozen_sh0=model._sh0)
        del optimizer
        module.zero_grad(set_to_none=True)
        before = timings(model, module, test)
        seams = face_residuals(model, module, test[0])
        render = lambda cam: model.render_vertex(cam, module, decode_all_vertices=True)["color"]
        scores, images[name] = score(render, test)
        with torch.no_grad():
            image = render(test[0])
            load_checkpoint(checkpoint, module, identity)
            torch.testing.assert_close(image, render(test[0]), atol=1e-5, rtol=0)
        result[name] = dict(**scores, timings=before, seams=seams, smoke=smoke,
                            training_peak_mib=peak, parameters=sum(v.numel() for v in module.parameters()))
        (root / "metrics.json").write_text(json.dumps(result, indent=2)+"\n")
        module.cpu()
        print(scene, name, scores, before, seams, flush=True)

    from PIL import ImageDraw
    for camera in test:
        if camera.image_name not in images["one"]:
            continue
        panels = [("GT", pixels(camera.image))]+[(k, images[k][camera.image_name])
                                                for k in ("sh0", "sh0_frozen", "sh3", "one", "two") if k in images]
        canvas = Image.new("RGB", (len(panels)*256, 278))
        for index, (label, image) in enumerate(panels):
            canvas.paste(Image.fromarray(image), (index*256, 22))
            ImageDraw.Draw(canvas).text((index*256+5, 5), label, fill="white")
        canvas.save(root / f"comparison-{Path(camera.image_name).stem}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=SOURCES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.scene, args.output)
