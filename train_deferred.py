"""Fit and benchmark deferred appearance on a frozen SVRaster checkpoint."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from yacs.config import CfgNode

from src.config import cfg
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_gears.deferred import MODE, save_deferred
from src.sparse_voxel_model import SparseVoxelModel
from src.utils import loss_utils
from src.utils.system_utils import seed_everything


def make_optimizer(model):
    return torch.optim.Adam([
        {"params": [model._sh0], "lr": cfg.optimizer.sh0_lr},
        {"params": model.deferred.parameters(), "lr": cfg.optimizer.shs_lr},
    ], betas=(cfg.optimizer.optim_beta1, cfg.optimizer.optim_beta2), eps=cfg.optimizer.optim_eps)


def photo_loss(image, target):
    regularizer = cfg.regularizer
    if regularizer.use_l1:
        loss = loss_utils.l1_loss(image, target)
    elif regularizer.use_huber:
        loss = loss_utils.huber_loss(image, target, regularizer.huber_thres)
    else:
        loss = (image - target).square().mean()
    return regularizer.lambda_photo * loss


def training_state(optimizer, camera_names):
    return dict(optimizer=optimizer.state_dict(), camera_names=camera_names,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def restore_training(state, optimizer, camera_names):
    if state["camera_names"] != camera_names:
        raise ValueError("Training cameras differ from the saved sequence")
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])


@torch.no_grad()
def benchmark(model, cameras, color_mode):
    name = color_mode or MODE
    events = []
    handles = []
    if color_mode is None:
        decoder = model.deferred.decoder
        def start(_module, _inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append([event])

        def end(_module, _inputs, _output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events[-1].append(event)

        handles = [decoder.register_forward_pre_hook(start), decoder.register_forward_hook(end)]

    try:
        model.render(cameras[0], color_mode=color_mode)
        torch.cuda.synchronize()
        events.clear()
        torch.cuda.reset_peak_memory_stats()
        timings, psnr, ssim = [], [], []
        for camera in cameras:
            target = camera.image.cuda()
            start_time = time.perf_counter()
            image = model.render(camera, color_mode=color_mode)["color"]
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start_time)
            psnr.append(loss_utils.psnr_score(image, target).item())
            ssim.append(loss_utils.ssim_score(image[None], target[None]).item())
        mean_time = sum(timings) / len(timings)
        return name, dict(
            psnr=sum(psnr) / len(psnr), ssim=sum(ssim) / len(ssim),
            ms_per_frame=1000 * mean_time, fps=1 / mean_time,
            mlp_ms=sum(a.elapsed_time(b) for a, b in events) / len(events) if events else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        )
    finally:
        for handle in handles:
            handle.remove()


def run(args):
    root = Path(args.model_path)
    if args.resume:
        # Read model settings from the checkpoint itself.
        candidates = sorted((root / "checkpoints").glob("iter*_model.pt"))
        if not candidates:
            raise FileNotFoundError(f"No deferred checkpoint under {root}")
        latest = max(candidates, key=lambda p: int(p.stem.split("_")[0][4:]))
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint.get("appearance_mode") != MODE:
            raise ValueError("Expected a deferred_mlp checkpoint")
        cfg.merge_from_other_cfg(CfgNode.load_cfg(checkpoint["config"]))
        camera_limit = checkpoint["training"]["camera_limit"]
    else:
        source_config = CfgNode.load_cfg((Path(args.source_model) / "config.yaml").read_text())
        unused = sorted(set(source_config) - set(cfg))
        if unused:
            print("Ignoring unsupported source config sections:", ", ".join(unused))
        cfg.merge_from_other_cfg(CfgNode({key: value for key, value in source_config.items()
                                         if key in cfg}))
        if args.res_width is not None:
            cfg.data.res_width = args.res_width
            cfg.data.res_downscale = 0.
        if args.ss is not None:
            cfg.model.ss = args.ss
        camera_limit = args.views

    seed_everything(cfg.procedure.seed)
    mode = MODE if args.resume else "sh"
    model = SparseVoxelModel(n_samp_per_vox=cfg.model.n_samp_per_vox,
                             sh_degree=cfg.model.sh_degree, ss=cfg.model.ss,
                             white_background=cfg.model.white_background,
                             black_background=cfg.model.black_background,
                             appearance_mode=mode)
    if args.resume:
        from src.sparse_voxel_gears.deferred import restore_deferred
        restore_deferred(model, checkpoint)
        previous_step = checkpoint["step"]
    else:
        model.load_iteration(args.source_model, args.source_iteration)
        previous_step = 0

    pack = DataPack(source_path=cfg.data.source_path, image_dir_name=cfg.data.image_dir_name,
                    mask_dir_name=cfg.data.mask_dir_name, res_downscale=cfg.data.res_downscale,
                    res_width=cfg.data.res_width, skip_blend_alpha=cfg.data.skip_blend_alpha,
                    alpha_is_white=cfg.model.white_background, data_device=cfg.data.data_device,
                    use_test=cfg.data.eval, test_every=cfg.data.test_every)
    train = sorted(pack.get_train_cameras(), key=lambda cam: cam.image_name)[:camera_limit]
    test = sorted(pack.get_test_cameras(), key=lambda cam: cam.image_name)[:camera_limit]
    if not train or (args.benchmark and not test):
        raise ValueError("Training cameras and requested held-out cameras must be available")
    if args.benchmark and args.resume:
        raise ValueError("Benchmark against the original SH checkpoint on a fresh run")

    names = [cam.image_name for cam in train]
    model.freeze_vox_geo()
    results = {}
    if not args.resume:
        if args.benchmark:
            results.update(benchmark(model, test, mode) for mode in ("sh0", "sh3"))
        model.init_deferred()
    optimizer = make_optimizer(model)
    if args.resume:
        restore_training(checkpoint["training"], optimizer, names)
        del checkpoint

    ignored = [key for key, value in cfg.regularizer.items()
               if key.startswith("lambda_") and key != "lambda_photo" and value]
    if cfg.regularizer.rand_bg:
        ignored.append("rand_bg")
    if cfg.auto_exposure.enable:
        ignored.append("auto_exposure")
    print("Appearance-only training. Inactive baseline options:", ", ".join(ignored) or "none")

    history = []
    for step in range(previous_step + 1, previous_step + args.steps + 1):
        camera = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = photo_loss(model.render(camera)["color"], camera.image.cuda())
        if not loss.isfinite():
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    state = training_state(optimizer, names)
    state["camera_limit"] = camera_limit
    path = save_deferred(model, root, step, cfg.dump(), state)
    (root / "config.yaml").write_text(cfg.dump())
    print(f"Saved {path}. Loss {history[0]:.6g} → {history[-1]:.6g}")
    del state, optimizer, loss
    model.deferred.zero_grad(set_to_none=True)
    model._sh0.grad = None
    if args.benchmark:
        results.update([benchmark(model, test, None)])
        report = dict(gpu=torch.cuda.get_device_name(), voxels=model.num_voxels,
                      resolution=[test[0].image_width, test[0].image_height], ss=model.ss,
                      train=names, test=[cam.image_name for cam in test], loss=history,
                      metrics=results)
        (root / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model")
    parser.add_argument("--source_iteration", type=int, default=-1)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--views", type=int)
    parser.add_argument("--res_width", type=int)
    parser.add_argument("--ss", type=float)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or (args.views is not None and args.views < 1):
        parser.error("steps and views must be positive")
    if args.resume == bool(args.source_model):
        parser.error("Specify either --source_model or --resume")
    run(args)
