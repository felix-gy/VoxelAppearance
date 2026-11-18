"""Finish the two-seed Lego comparison and diagnose vertex-basis contributions."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch

from compare_vertex_two_bases import face_residuals, score
from reevaluate_vertex_two_bases import ARMS, MILESTONES, publish_json, run
from src.utils.loss_utils import metric_networks, psnr_score
from src.vertex_appearance.checkpoint import load_checkpoint, save_checkpoint
from src.vertex_appearance.decoder import VertexAppearance, VertexTwoBases
from src.vertex_appearance.runtime import fit, setup

SEEDS = (3721, 3722)
WIDE = "wide_one"


@torch.no_grad()
def widen(source, width):
    wide = VertexAppearance(len(source.vertex_features), feature_dim=width).to(source.vertex_features.device)
    wide.vertex_features[:, :8].copy_(source.vertex_features)
    wide.decoder[0].weight[:, :8].copy_(source.decoder[0].weight[:, :8])
    wide.decoder[0].weight[:, 8:width].zero_()
    wide.decoder[0].weight[:, width:].copy_(source.decoder[0].weight[:, 8:])
    wide.decoder[0].bias.copy_(source.decoder[0].bias)
    wide.decoder[-1].load_state_dict(source.decoder[-1].state_dict())
    return wide


@torch.no_grad()
def measure_psnr(render, cameras):
    rows = []
    for cam in cameras:
        image = render(cam)
        assert image.isfinite().all()
        rows.append(dict(view=cam.image_name, psnr=psnr_score(image, cam.image.cuda()).item()))
    return rows


def render_dense(model, module, cam):
    return model.render_vertex(cam, module, decode_all_vertices=True)["color"]


@torch.no_grad()
def ablate_second_basis(model, module, cameras):
    coefficient = module.mix[:, 1].clone()
    quantiles = torch.quantile(coefficient, coefficient.new_tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    summary = dict(mean=coefficient.mean().item(), std=coefficient.std(unbiased=False).item(),
                   rms=coefficient.square().mean().sqrt().item(), mean_abs=coefficient.abs().mean().item(),
                   min=coefficient.min().item(), max=coefficient.max().item(),
                   fraction_abs_gt_0_01=(coefficient.abs() > .01).float().mean().item(),
                   quantiles=dict(zip(("p01", "p05", "p50", "p95", "p99"), quantiles.tolist())))
    rows = []
    for cam in cameras:
        bases = module(model.grid_pts_xyz, cam.position)
        amplitudes = [bases[:, i].square().mean().sqrt().item() for i in range(2)]
        del bases
        full = render_dense(model, module, cam)
        try:
            module.mix[:, 1].zero_()
            first_only = render_dense(model, module, cam)
        finally:
            module.mix[:, 1].copy_(coefficient)
        diff = full - first_only
        gt = cam.image.cuda()
        rows.append(dict(view=cam.image_name, basis1_rms=amplitudes[0], basis2_rms=amplitudes[1],
                         image_rms=diff.square().mean().sqrt().item(),
                         image_max_abs=diff.abs().max().item(),
                         psnr_full=psnr_score(full, gt).item(),
                         psnr_without_basis2=psnr_score(first_only, gt).item()))
    return dict(lambda2=summary, per_view=rows)


def read_protocol(root):
    cfg, _, model, one, pack, identity = setup(SimpleNamespace(
        config=root/"protocol.yaml", resume=None, appearance_checkpoint=None))
    protocol = json.loads((root/"protocol.json").read_text())

    def select(split, views):
        by_name = {cam.image_name:cam for cam in views}
        return [by_name[name] for name in protocol[split + "_views"]]

    return cfg, model, one, identity, select("train", pack.get_train_cameras()), select("test", pack.get_test_cameras())


def run_wide(root, cfg, model, one, identity, train, test, stop):
    # Minimize the difference in parameter count while keeping the hidden width 64.
    p, v = model.num_grid_pts, model.num_voxels
    target = 8*p + 26*64 + 64 + 64*6 + 6 + 2*v
    width = max(8, 8 + round((target - (8*p + 26*64 + 64 + 64*3 + 3))/(p + 64)))
    module = widen(one, width)
    with torch.no_grad():
        torch.testing.assert_close(render_dense(model, one, test[0]), render_dense(model, module, test[0]),
                                   atol=1e-5, rtol=0)
    one.cpu()
    directory = root / WIDE
    directory.mkdir(exist_ok=True)
    comparison = dict(dense=True, seed=cfg["seed"], betas=[0.1, 0.99], eps=1e-15,
                      train_views=[cam.image_name for cam in train], test_views=[cam.image_name for cam in test])
    architecture = dict(cfg["appearance"], feature_dim=width, layers=[width+18, 64, 3])
    config = dict(cfg, comparison=comparison, appearance=architecture)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.00025, betas=(0.1, 0.99), eps=1e-15)
    generator = torch.Generator().manual_seed(cfg["seed"])
    start = max((n for n in MILESTONES if (directory/f"step{n:04}.json").is_file()), default=0)
    if start and start < stop:
        state = load_checkpoint(directory/f"step{start:04}.pt", module, identity, optimizer, generator)
        if state["config"] != config:
            raise ValueError("Wide-control checkpoint does not match the protocol")
        del state
    order = [step % len(train) for step in range(stop)]
    for step, loss in fit(model, module, train, order, optimizer, start, stop, dense=True):
        if step % 100 == 0:
            print(cfg["seed"], WIDE, step, f"loss={loss:.7f}", flush=True)
        if step not in MILESTONES:
            continue
        save_checkpoint(directory/f"step{step:04}.pt", module, optimizer, step, config,
                        identity, generator, frozen_sh0=model._sh0)
        module.zero_grad(set_to_none=True)
        values, _ = score(lambda cam: render_dense(model, module, cam), test)
        publish_json(directory/f"step{step:04}.json",
                     dict(**values, parameters=sum(p.numel() for p in module.parameters()),
                          target_parameters=target, feature_dim=width,
                          seams=face_residuals(model, module, test[0])))
        metric_networks.clear()
    module.cpu()


def diagnose(root, cfg, model, one, identity, train, test, stop):
    one.cpu()
    for arm in (*ARMS, WIDE):
        for step in (x for x in MILESTONES if x <= stop):
            stem = root / arm / f"step{step:04}"
            output = stem.with_suffix(".diagnostics.json")
            if output.exists():
                continue
            state = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
            if state["identity"] != identity or state["step"] != step:
                raise ValueError("Checkpoint geometry or step mismatch")
            comparison = state["config"]["comparison"]
            if comparison["train_views"] != [cam.image_name for cam in train] or                     comparison["test_views"] != [cam.image_name for cam in test]:
                raise ValueError("Checkpoint camera order mismatch")
            if any(int(item["step"]) != step for item in state["optimizer"]["state"].values()):
                raise ValueError("Checkpoint Adam counters do not match its step")
            if not torch.equal(model._sh0, state["frozen_sh0"].to(model._sh0.device)):
                raise ValueError("Checkpoint frozen SH0 mismatch")
            if arm in ("two", "regularized"):
                module = VertexTwoBases(model.num_grid_pts, model.num_voxels)
            else:
                width = state["vertex_module"]["vertex_features"].shape[1]
                module = VertexAppearance(model.num_grid_pts, feature_dim=width)
            module.load_state_dict(state["vertex_module"])
            del state
            module.cuda().eval()
            train_scores = measure_psnr(lambda cam: render_dense(model, module, cam), train)
            test_scores = measure_psnr(lambda cam: render_dense(model, module, cam), test)
            previous = json.loads(stem.with_suffix(".json").read_text())
            if abs(mean(x["psnr"] for x in test_scores) - previous["psnr"]) > 1e-4:
                raise ValueError("Reloaded rendering does not reproduce reserved-view PSNR")
            data = dict(seed=cfg["seed"], arm=arm, step=step, train_psnr=mean(r["psnr"] for r in train_scores),
                        test_psnr=mean(r["psnr"] for r in test_scores),
                        ssim=previous["ssim"], lpips=previous["lpips"], seam_rms=previous["seams"]["rms"],
                        parameters=sum(p.numel() for p in module.parameters()),
                        train_per_view=train_scores, test_per_view=test_scores)
            if isinstance(module, VertexTwoBases):
                data["basis2"] = ablate_second_basis(model, module, test)
            publish_json(output, data)
            print("DIAGNOSTICS", cfg["seed"], arm, step, data["train_psnr"], data["test_psnr"], flush=True)
            del module


def report(root, output):
    rows = [json.loads(path.read_text()) for path in sorted(root.glob("seed*/*/*.diagnostics.json"))]
    if not rows:
        return
    fields = ("seed", "step", "arm", "parameters", "train_psnr", "test_psnr", "ssim", "lpips", "seam_rms")
    with (output/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Lego: diagnóstico de una y dos bases", "",
             "Mismo checkpoint SH y SH0 congelado, 86 cámaras de entrenamiento en ciclo, cuatro reservadas, "
             "256×256, ss=1, y semillas 3721 y 3722. Control de capacidad con el ancho entero de latentes "
             "más cercano al total de dos bases, 64 neuronas ocultas y una salida RGB. El número de "
             "parámetros del control puede diferir del de dos bases.", "",
             "| Semilla | Paso | Brazo | Parámetros | PSNR train | PSNR reservado | Brecha | SSIM | LPIPS | RMS salto |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: (r["seed"], r["step"], r["arm"])):
        lines.append(f"| {r['seed']} | {r['step']} | {r['arm']} | {r['parameters']} | "
                     f"{r['train_psnr']:.4f} | {r['test_psnr']:.4f} | {r['train_psnr']-r['test_psnr']:.4f} | "
                     f"{r['ssim']:.6f} | {r['lpips']:.6f} | {r['seam_rms']:.6f} |")
    lookup = {(r["seed"], r["step"], r["arm"]):r for r in rows}
    lines += ["", "## Comparación emparejada en vistas reservadas", "",
              "| Semilla | Paso | Dos − una (dB) | Dos − una ancha (dB) | Regularizada − dos (dB) |",
              "| --- | --- | ---: | ---: | ---: |"]
    for seed in SEEDS:
        for step in MILESTONES:
            d = {arm:lookup.get((seed, step, arm)) for arm in (*ARMS, WIDE)}
            if not all(d.values()):
                continue
            lines.append(f"| {seed} | {step} | {d['two']['test_psnr']-d['one']['test_psnr']:+.4f} | "
                         f"{d['two']['test_psnr']-d[WIDE]['test_psnr']:+.4f} | "
                         f"{d['regularized']['test_psnr']-d['two']['test_psnr']:+.4f} |")
    lines += ["", "## Segunda base y ablación", "",
              "RMS de las bases sobre vértices globales. RMS de imagen es la diferencia entre el render intacto "
              "y el mismo render con el segundo coeficiente a cero. Incluye composición y clamp. "
              "Los JSON guardan resultados por cámara y la distribución de coeficientes, incluidos percentiles.", "",
              "| Semilla | Paso | Brazo | RMS base 1 | RMS base 2 | Media abs. λ₂ | RMS λ₂ | P05 λ₂ | P95 λ₂ | RMS imagen | Δ PSNR con base 2 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if "basis2" not in r:
            continue
        b = r["basis2"]
        l, views = b["lambda2"], b["per_view"]
        vals = [mean(v[k] for v in views) for k in ("basis1_rms", "basis2_rms")]
        vals += [l["mean_abs"], l["rms"], l["quantiles"]["p05"], l["quantiles"]["p95"],
                 mean(v["image_rms"] for v in views),
                 mean(v["psnr_full"]-v["psnr_without_basis2"] for v in views)]
        lines.append(f"| {r['seed']} | {r['step']} | {r['arm']} | " + " | ".join(f"{x:.6f}" for x in vals) + " |")
    (output/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/vertex-two-bases-86/lego"))
    parser.add_argument("--output", type=Path, default=Path("output/vertex-bases-lego-diagnostics"))
    parser.add_argument("--stop", type=int, choices=MILESTONES, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/".diagnostics.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in SEEDS:
            directory = args.root / f"seed{seed}"
            run("lego", seed, directory, stop=args.stop)
            cfg, model, one, identity, train, test = read_protocol(directory)
            run_wide(directory, cfg, model, one, identity, train, test, args.stop)
            diagnose(directory, cfg, model, one, identity, train, test, args.stop)
            report(args.root, args.output)
            del model, one


if __name__ == "__main__":
    main()
