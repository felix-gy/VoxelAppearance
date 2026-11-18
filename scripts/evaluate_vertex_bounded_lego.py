"""Compare a bounded residual against the same one-base warm start on Lego."""
import argparse
import gc
import json
from pathlib import Path
from statistics import mean

import torch

from compare_vertex_two_bases import face_residuals, score
from diagnose_vertex_bases_lego import measure_psnr, read_protocol, render_dense
from reevaluate_vertex_two_bases import publish_json
from src.utils.loss_utils import metric_networks
from src.vertex_appearance.checkpoint import load_checkpoint, save_checkpoint
from src.vertex_appearance.decoder import VertexBoundedResidual
from src.vertex_appearance.runtime import fit

MILESTONES = (300, 500, 1000)
ETA = 0.25


@torch.no_grad()
def correction_statistics(model, module, cameras):
    results = []
    for cam in cameras:
        bases = module(model.grid_pts_xyz, cam.position)
        squares = bases.new_zeros(())
        saturated = torch.zeros((), device=bases.device, dtype=torch.int64)
        max_abs = bases.new_zeros(())
        for start in range(0, model.num_voxels, 131072):
            stop = min(start+131072, model.num_voxels)
            q = module.correction_input(bases, model.vox_key[start:stop],
                                        slice(start, stop))
            correction = module.eta * q.tanh()
            squares += correction.square().sum()
            saturated += (q.abs() > 2).sum()
            max_abs = torch.maximum(max_abs, correction.abs().max())
        count = model.num_voxels * 8 * 3
        results.append(dict(view=cam.image_name, correction_rms=(squares/count).sqrt().item(),
                            correction_max=max_abs.item(), saturated_fraction=saturated.item()/count))
    gate = module.voxel_gate.detach()
    quantiles = torch.quantile(gate, gate.new_tensor([.05, .50, .95])).tolist()
    return dict(per_view=results, gate_rms=gate.square().mean().sqrt().item(),
                gate_p05=quantiles[0], gate_median=quantiles[1], gate_p95=quantiles[2],
                **{key:mean(r[key] for r in results) for key in ("correction_rms", "saturated_fraction")})


def run_seed(source, output):
    cfg, model, control, identity, train, test = read_protocol(source)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = source / "one/step0300.pt"
    warm = load_checkpoint(checkpoint, control, identity)
    if warm["step"] != 300 or warm["config"]["seed"] != cfg["seed"]:
        raise ValueError("Unexpected warm-start seed or iteration")
    if not torch.equal(model._sh0, warm["frozen_sh0"].cuda()):
        raise ValueError("Warm-start SH0 differs from source geometry")
    protocol = warm["config"]["comparison"]
    if protocol["train_views"] != [c.image_name for c in train] or protocol["test_views"] != [c.image_name for c in test]:
        raise ValueError("Warm-start cameras differ from the experiment")
    del warm

    bounded = VertexBoundedResidual(model.num_grid_pts, model.num_voxels, ETA).cuda()
    bounded.initialize_first_basis(control.state_dict())
    with torch.no_grad():
        initial_errors = [(render_dense(model, control, cam) - render_dense(model, bounded, cam))
                          .abs().max().item() for cam in test]
    assert max(initial_errors) <= 1e-5
    publish_json(output/"protocol.json", dict(seed=cfg["seed"], source=str(checkpoint),
                 identity=identity, eta=ETA, initial_max_rgb_error=max(initial_errors),
                 train_views=protocol["train_views"], test_views=protocol["test_views"],
                 steps=list(MILESTONES), adam=dict(lr=0.00025, betas=[0.1,0.99], eps=1e-15)))
    bounded.cpu()
    order = [i % len(train) for i in range(MILESTONES[-1])]
    for name, module in (("one", control), ("bounded", bounded)):
        directory = output / name
        directory.mkdir(exist_ok=True)
        start = max((n for n in MILESTONES if (directory/f"step{n:04}.json").is_file()), default=0)
        if start == MILESTONES[-1]:
            module.cpu()
            continue
        module.cuda()
        optimizer = torch.optim.Adam(module.parameters(), lr=0.00025, betas=(0.1,0.99), eps=1e-15)
        generator = torch.Generator().manual_seed(cfg["seed"])
        config = dict(cfg, comparison=protocol, steps=MILESTONES[-1])
        if name == "bounded":
            config.update(mode=module.appearance_mode, eta=ETA,
                          appearance=dict(cfg["appearance"], layers=[26, 64, 6]))
        if start:
            saved = load_checkpoint(directory/f"step{start:04}.pt", module, identity, optimizer, generator)
            if saved["config"] != config:
                raise ValueError("Checkpoint does not match the frozen protocol")
            del saved
        torch.cuda.reset_peak_memory_stats()
        for step, loss in fit(model, module, train, order, optimizer, start, MILESTONES[-1], dense=True):
            if step % 100 == 0:
                print(cfg["seed"], name, step, f"loss={loss:.7f}", flush=True)
            if step not in MILESTONES:
                continue
            assert all(int(item["step"]) == step for item in optimizer.state.values())
            peak = torch.cuda.max_memory_allocated()/2**20
            save_checkpoint(directory/f"step{step:04}.pt", module, optimizer, step, config,
                            identity, generator, frozen_sh0=model._sh0)
            module.zero_grad(set_to_none=True)
            render = lambda cam: render_dense(model, module, cam)
            metrics, _ = score(render, test)
            dataset = measure_psnr(render, train)
            result = dict(**metrics, training_psnr=mean(row["psnr"] for row in dataset),
                          training_per_view=dataset, peak_training_mib=peak,
                          parameters=sum(p.numel() for p in module.parameters()),
                          seams=face_residuals(model, module, test[0]))
            if name == "bounded":
                result["correction"] = correction_statistics(model, module, test)
                assert all(x["correction_max"] <= ETA for x in result["correction"]["per_view"])
            publish_json(directory/f"step{step:04}.json", result)
            print("MILESTONE", cfg["seed"], name, step, metrics["psnr"], result["seams"]["rms"], flush=True)
            metric_networks.clear()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        del optimizer
        module.cpu()


def report(output):
    rows = {}
    for path in output.glob("seed*/*/step*.json"):
        rows[int(path.parents[1].name[4:]), path.parent.name, int(path.stem[4:])] = json.loads(path.read_text())
    lines = ["# Lego: Vertex-MLP con residual acotado", "",
             f"**Hitos publicados: {len(rows)}/12.**", "",
             "Ambos brazos arrancaron desde el checkpoint de una base del paso 300 de cada semilla. "
             "Los hitos representan pasos adicionales. La geometría, densidad y SH0 siguen congelados. "
             "Se usaron 86 cámaras cíclicas y cuatro reservadas a 256×256, ss=1, fondo blanco, una muestra "
             "de densidad, MSE y Adam nuevo a 0,00025, betas (0,1, 0,99), epsilon 1e−15. "
             "La segunda salida empieza con pesos de desviación estándar 0,01, sesgo cero y voxel_gate cero.", "",
             "El decoder mezcla en PyTorch mediante B₁ + 0,25 tanh(a B₂), antes de la interpolación y composición "
             "CUDA existentes. El modo vertex_two_bases_bounded guarda voxel_gate y eta en el estado. "
             "La amplitud está acotada por canal en las esquinas y tras la interpolación. La saturación se mide "
             "como la fracción de esquinas RGB con |a B₂| > 2, sobre todos los vóxeles y las cuatro vistas "
             "reservadas. El RMS de saltos se mide en centros de caras compartidas del mismo nivel.", "",
             "| Semilla | Paso | Brazo | PSNR train | PSNR reservado | SSIM | LPIPS | RMS salto | RMS corrección | Saturación |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (seed, name, step), row in sorted(rows.items(), key=lambda kv:(kv[0][0],kv[0][2],kv[0][1])):
        correction = row.get("correction", {})
        rms = f"{correction['correction_rms']:.6f}" if correction else "—"
        sat = f"{100*correction['saturated_fraction']:.6f}%" if correction else "—"
        lines.append(f"| {seed} | {step} | {name} | {row['training_psnr']:.4f} | {row['psnr']:.4f} | "
                     f"{row['ssim']:.6f} | {row['lpips']:.6f} | {row['seams']['rms']:.6f} | {rms} | {sat} |")
    lines.extend(["", "## Comparación emparejada", "",
                  "| Semilla | Paso | Δ PSNR reservado | Δ SSIM | Δ LPIPS | Δ RMS salto |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for seed in (3721,3722):
        for step in MILESTONES:
            a, b = (rows.get((seed, kind, step)) for kind in ("bounded", "one"))
            if a and b:
                lines.append(f"| {seed} | {step} | {a['psnr']-b['psnr']:+.4f} | {a['ssim']-b['ssim']:+.6f} | "
                             f"{a['lpips']-b['lpips']:+.6f} | {a['seams']['rms']-b['seams']['rms']:+.6f} |")
    lines.extend(["", "## Resumen de las dos semillas", ""])
    for step in MILESTONES:
        pairs = [(rows.get((seed, "bounded", step)), rows.get((seed, "one", step)))
                 for seed in (3721, 3722)]
        if not all(a and b for a, b in pairs):
            continue
        delta = {key: mean(a[key] - b[key] for a, b in pairs)
                 for key in ("psnr", "ssim", "lpips")}
        jump = mean(a["seams"]["rms"] for a, _ in pairs)
        correction = mean(a["correction"]["correction_rms"] for a, _ in pairs)
        saturation = 100 * mean(a["correction"]["saturated_fraction"] for a, _ in pairs)
        lines.append(f"- **Paso {step}.** Residual acotado menos una base: "
                     f"{delta['psnr']:+.4f} dB, {delta['ssim']:+.6f} SSIM y "
                     f"{delta['lpips']:+.6f} LPIPS. RMS del salto {jump:.6f}, "
                     f"RMS de la corrección {correction:.6f} y saturación {saturation:.6f}%.")
    lines.extend(["", "La comparación pareada usa una base como control. Las métricas históricas de dos "
                  "bases libres usan otro punto de partida y solo sirven de contexto.", "",
                  "Resultados por vista, metadatos y checkpoints en el directorio del experimento.", ""])
    (output/"report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/vertex-two-bases-86/lego"))
    parser.add_argument("--output", type=Path, default=Path("output/vertex-bounded-residual-lego"))
    parser.add_argument("--seed", type=int, choices=(3721, 3722), nargs="+", default=[3721, 3722])
    args = parser.parse_args()
    for seed in args.seed:
        run_seed(args.source/f"seed{seed}", args.output/f"seed{seed}")
        report(args.output)
