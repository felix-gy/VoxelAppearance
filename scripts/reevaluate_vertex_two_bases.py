"""Compare one basis, two bases and regularized two bases at fixed checkpoints."""
import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from compare_vertex_two_bases import SOURCES, face_residuals, score
from src.utils import loss_utils
from src.vertex_appearance.checkpoint import load_checkpoint, save_checkpoint
from src.vertex_appearance.decoder import VertexTwoBases
from src.vertex_appearance.neighbors import coefficient_penalty
from src.vertex_appearance.runtime import fit, setup

MILESTONES = (300, 1000)
REGULARIZER = dict(strength=0.001, seed=7919, batch_size=4096)
ARMS = ("one", "two", "regularized")


def publish_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def run(scene, seed, root, stop=1000, arms=ARMS):
    root.mkdir(parents=True, exist_ok=True)
    source = Path(SOURCES[scene])
    geometry = source / "checkpoints/iter020000_model.pt"
    base = yaml.safe_load((source / "config.yaml").read_text())
    base["data"].update(res_width=256, res_downscale=0., eval=True)
    base["model"].update(ss=1., white_background=True)
    config = dict(seed=seed, steps=MILESTONES[-1], validation_interval=300, checkpoint_interval=300,
                  mode="vertex_trilinear_fourier3", geometry_checkpoint=str(geometry),
                  resolved_base=base, lr_vertex_features=0.00025, lr_mlp=0.00025, lr_mix=0.00025)
    protocol = root / "protocol.yaml"
    protocol.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(config=protocol, appearance_checkpoint=None, resume=None)
    config, _, model, one, pack, identity = setup(args)
    train = sorted(pack.get_train_cameras(), key=lambda c: c.image_name)[:86]
    test = sorted(pack.get_test_cameras(), key=lambda c: c.image_name)[:4]
    assert (len(train), len(test)) == (86, 4)
    assert all((c.image_width, c.image_height) == (256, 256) for c in train + test)
    train_names, test_names = ([c.image_name for c in views] for views in (train, test))

    if scene == "materials_t":
        warm_root = Path("output/vertex-mlp-materials-t")
        path = warm_root / f"vertex-seed{seed}-step300.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert Path(state["source"]) == geometry and state["seed"] == seed and state["step"] == 300
        assert state["mode"] == config["mode"] and state["train_views"] == train_names[:32]
        one.load_state_dict(state["appearance"])
        model._sh0.copy_(state["sh0"])
        historical = json.loads((warm_root / f"seed{seed}.json").read_text())["checkpoints"]["300"]["test"]
        actual, _ = score(lambda c: model.render_vertex(c, one, decode_all_vertices=True)["color"], test)
        for a, b in zip(actual["per_view"], historical["per_view"], strict=True):
            assert a["view"] == b["view"] and abs(a["psnr"]-b["psnr"]) <= 1e-4
        del state

    two = VertexTwoBases(model.num_grid_pts, model.num_voxels).cuda()
    two.initialize_first_basis(one.state_dict())
    with torch.no_grad():
        for cam in test:
            a = model.render_vertex(cam, one, decode_all_vertices=True)["color"]
            b = model.render_vertex(cam, two, decode_all_vertices=True)["color"]
            torch.testing.assert_close(a, b, atol=1e-5, rtol=0)

    one.cpu()
    two.cpu()
    # Independent modules must start from identical bases and coefficients.
    initial_two = {k: v.clone() for k, v in two.state_dict().items()}
    comparison = dict(dense=True, seed=seed, betas=[0.1, 0.99], eps=1e-15,
                      train_views=train_names, test_views=test_names)
    publish_json(root / "protocol.json", dict(**comparison, geometry=identity, milestones=MILESTONES,
                 coefficient_penalty=REGULARIZER))
    order = [i % len(train) for i in range(MILESTONES[-1])]
    for arm in arms:
        module = one if arm == "one" else two
        if arm != "one":
            module.load_state_dict(initial_two)
        module.cuda()
        arm_comparison = dict(comparison)
        penalty = None
        if arm == "regularized":
            penalty, edges = coefficient_penalty(model, **REGULARIZER)
            arm_comparison.update(coefficient_penalty=REGULARIZER, training_edges=edges)
        arm_config = dict(config, comparison=arm_comparison)
        if arm != "one":
            arm_config.update(mode="vertex_two_bases",
                              appearance=dict(config["appearance"], layers=[26, 64, 6]))
        optimizer = torch.optim.Adam(module.parameters(), lr=0.00025, betas=(0.1, 0.99), eps=1e-15)
        rng = torch.Generator().manual_seed(seed)
        arm_dir = root / arm
        arm_dir.mkdir(exist_ok=True)
        completed = [step for step in MILESTONES if (arm_dir / f"step{step:04}.json").is_file()]
        start = max(completed, default=0)
        if start >= stop:
            module.cpu()
            continue
        if start:
            loaded = load_checkpoint(arm_dir / f"step{start:04}.pt", module, identity, optimizer, rng)
            # A shortened horizon keeps the same camera cycle and optimizer state.
            saved_config = dict(loaded["config"], steps=arm_config["steps"])
            if saved_config != arm_config:
                raise ValueError("Saved checkpoint does not match requested protocol")
            del loaded
        loss_utils.metric_networks.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for step, loss in fit(model, module, train, order, optimizer, start, stop,
                              dense=True, penalty=penalty):
            if step % 100 == 0:
                print(scene, seed, arm, step, f"loss={loss:.7f}", flush=True)
            if step not in MILESTONES:
                continue
            assert all(int(state["step"]) == step for state in optimizer.state.values())
            peak = torch.cuda.max_memory_allocated() / 2**20
            save_checkpoint(arm_dir / f"step{step:04}.pt", module, optimizer, step,
                            arm_config, identity, rng, frozen_sh0=model._sh0)
            module.zero_grad(set_to_none=True)
            scores, _ = score(lambda c: model.render_vertex(c, module, decode_all_vertices=True)["color"], test)
            seams = face_residuals(model, module, test[0])
            publish_json(arm_dir / f"step{step:04}.json", dict(scene=scene, seed=seed, arm=arm,
                         step=step, peak_train_mib=peak, training_edges=arm_comparison.get("training_edges"),
                         **scores, seams=seams))
            print("MILESTONE", scene, seed, arm, step, scores["psnr"], seams["rms"], flush=True)
            loss_utils.metric_networks.clear()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        del optimizer
        module.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=SOURCES)
    parser.add_argument("--seed", type=int, choices=(3721, 3722), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop", type=int, choices=MILESTONES, default=MILESTONES[-1])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    run(args.scene, args.seed, args.output, args.stop, args.arms)
