"""Build the paired 86-camera comparison report from completed milestones."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

SCENES = ("materials_t", "lego")
SEEDS = (3721, 3722)
MODES = ("one", "two", "regularized")
STEPS = (300, 1000)
METRICS = ("psnr", "ssim", "lpips", "rms")


def magnitude(row, name):
    return row["seams"]["rms"] if name == "rms" else row[name]


def delta_row(a, b):
    return {name: magnitude(a, name) - magnitude(b, name) for name in METRICS}


def report(root, out):
    rows = {}
    for path in sorted(root.glob("*/seed*/*/step*.json")):
        r = json.loads(path.read_text())
        if r["step"] in STEPS:
            rows[r["scene"], r["seed"], r["step"], r["arm"]] = r

    lines = [
        "# Vertex-MLP con 86 cámaras y suavizado de coeficientes", "",
        f"**Hitos completos: {len(rows)}/24.**", "",
        "## Protocolo", "",
        "Rama `Vertex-MLP-Dos-Bases`, RTX 3060. Se comparan una base, dos bases y dos bases con penalización. Cada brazo usa las primeras 86 cámaras de entrenamiento por nombre y las mismas cuatro reservadas de la prueba anterior, en orden cíclico, a 256×256, fondo blanco, ss=1 y una muestra por vóxel.",
        "",
        "Las semillas son 3721 y 3722. En Materials T, ambos brazos de dos bases y el control de una base arrancan del Vertex-MLP histórico de 300 pasos de la semilla correspondiente y congelan su mismo SH0. Los pasos de las tablas son actualizaciones adicionales para Materials T. En Lego se congela el checkpoint SH 20.000 de `out/lego-original` y cada semilla parte de la inicialización compartida de una base.",
        "",
        "Densidad, geometría, SH0 y topología se congelan. Se mantiene el rasterizador Vertex-MLP, la decodificación densa para comparar y MSE con Adam a 0,00025, betas (0,1, 0,99), epsilon 1e−15. Los estados de Adam empiezan en cero y continúan entre hitos. Las dos variantes de dos bases comparten la inicialización completa, incluidos los pesos pequeños de la segunda base y mezcla (1,0).",
        "",
        "El brazo regularizado añade `0.001 × mean((mix[v] − mix[u])²)` sobre 4096 aristas muestreadas por paso de un conjunto fijo de caras vecinas del mismo nivel. Se usan ambos coeficientes de cada vóxel. El conjunto se construye con semilla 7919 desde 65.536 vóxeles muestreados y vecinos en los tres ejes positivos. El muestreo de cada paso es determinista por índice y se puede reconstruir al reanudar.",
        "",
        "La evaluación usa PSNR, SSIM y LPIPS-VGG normalizado, promediados sobre cuatro vistas. El RMS mide el salto RGB del residual en el centro de caras vecinas del mismo nivel, antes de SH0 y clamp. Sus caras se muestrean con semilla independiente de la regularización. La prueba no mide saltos entre niveles distintos.",
        "",
        "## Resultados por escena, semilla y paso", "",
        "| Escena | Semilla | Paso | Brazo | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | RMS de salto ↓ | Pico entrenamiento (MiB) |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    pairs = []
    camera_pairs = []
    for scene in SCENES:
        for seed in SEEDS:
            for step in STEPS:
                for mode in MODES:
                    row = rows.get((scene, seed, step, mode))
                    if row:
                        lines.append(f"| {scene} | {seed} | {step} | {mode} | {row['psnr']:.4f} | "
                                     f"{row['ssim']:.6f} | {row['lpips']:.6f} | {row['seams']['rms']:.6f} | "
                                     f"{row['peak_train_mib']:.0f} |")
                for mode, base in (("two", "one"), ("regularized", "two"), ("regularized", "one")):
                    a, b = (rows.get((scene, seed, step, kind)) for kind in (mode, base))
                    if a is None or b is None:
                        continue
                    assert [x["view"] for x in a["per_view"]] == [x["view"] for x in b["per_view"]]
                    assert a["seams"]["faces"] == b["seams"]["faces"]
                    info = dict(scene=scene, seed=seed, step=step, comparison=f"{mode} − {base}")
                    pairs.append(dict(**info, **delta_row(a, b)))
                    for x, y in zip(a["per_view"], b["per_view"], strict=True):
                        camera_pairs.append(dict(**info, view=x["view"],
                                                 **{k:x[k]-y[k] for k in METRICS[:-1]}))
    lines += ["", "## Diferencias pareadas", "",
              "Signos: PSNR y SSIM positivos favorecen el primer brazo. LPIPS y RMS negativos lo favorecen. Una media de dos semillas no estima por sí sola significación estadística.",
              "", "| Escena | Paso | Comparación | Semillas | Δ PSNR (dB) | Δ SSIM | Δ LPIPS | Δ RMS |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    by_group = defaultdict(list)
    for row in pairs:
        by_group[row["scene"], row["step"], row["comparison"]].append(row)
    for (scene, step, name), rs in sorted(by_group.items()):
        vals = " | ".join(f"{mean(r[k] for r in rs):+.6f}" for k in METRICS)
        lines.append(f"| {scene} | {step} | {name} | {len(rs)} | {vals} |")

    lines += ["", "## Trayectoria de PSNR al pasar de una a dos bases", ""]
    for scene in SCENES:
        groups = [by_group.get((scene, step, "two − one"), []) for step in STEPS]
        if any(len(g) != 2 for g in groups):
            lines.append(f"- **{scene}.** Comparación longitudinal pendiente hasta completar las dos semillas de cada hito.")
            continue
        trajectory = [mean(r["psnr"] for r in g) for g in groups]
        delta = trajectory[-1] - trajectory[-2]
        if abs(trajectory[-1]) < .01:
            label = "la diferencia final quedó por debajo de 0,01 dB"
        elif abs(delta) < .01:
            label = "la diferencia cambió menos de 0,01 dB entre 300 y 1000"
        elif delta > 0:
            label = "la ventaja de dos bases aumentó entre 300 y 1000"
        else:
            label = "la diferencia se movió a favor de una base entre 300 y 1000"
        lines.append(f"- **{scene}.** Δ medio a 300 y 1000: "
                     f"{trajectory[0]:+.4f} y {trajectory[1]:+.4f} dB. {label}.")
    lines += ["", "## Efecto del suavizado de coeficientes", ""]
    for scene in SCENES:
        group = by_group.get((scene, 1000, "regularized − two"), [])
        if len(group) != 2:
            lines.append(f"- **{scene}.** Comparación al paso 1000 pendiente.")
            continue
        values = {k: mean(r[k] for r in group) for k in METRICS}
        raw = mean(rows[scene, seed, 1000, "two"]["seams"]["rms"] for seed in SEEDS)
        smooth = mean(rows[scene, seed, 1000, "regularized"]["seams"]["rms"] for seed in SEEDS)
        direction = ("bajó el RMS en las dos semillas" if all(r["rms"] < 0 for r in group)
                     else "el cambio de RMS varió entre semillas" if min(r["rms"] for r in group) < 0
                     else "el RMS se mantuvo o subió en las dos semillas")
        lines.append(f"- **{scene}.** El suavizado {direction}. El RMS medio pasó de "
                     f"{raw:.6f} a {smooth:.6f}. Las diferencias medias fueron "
                     f"{values['psnr']:+.4f} dB, {values['ssim']:+.6f} SSIM y "
                     f"{values['lpips']:+.6f} LPIPS.")

    lines += ["", "## Implementación y validación", "",
              "`neighbors.py` localiza vecinos del mismo nivel y crea la penalización diferenciable. El entrenador congelado admite un término opcional que se suma al MSE. El brazo regularizado conserva la arquitectura de dos bases y el formato de apariencia. Los checkpoints conservan la configuración de muestreo y permiten reproducir la secuencia al reanudar.",
              "",
              "Pasaron 25 pruebas CUDA, incluidas las regresiones de Vertex-MLP y la nueva prueba de gradiente de la penalización. En cada experimento se comprobó identidad del render inicial con tolerancia 1e−5 y se verificó el contador de Adam de cada tensor entrenable en cada hito.",
              "", "## Artefactos y reproducción", "",
              "Cada hito publica un JSON por brazo con las cuatro métricas, resultados por cámara y el pico de entrenamiento del tramo. Los checkpoints incluyen apariencia, identidad, SH0 congelado y estado Adam. `paired.csv` y `paired-per-view.csv` contienen los deltas por semilla y por cámara.",
              "", "```bash",
              'export PYTHONPATH="$PWD/cuda:$PWD${PYTHONPATH:+:$PYTHONPATH}"',
              "python scripts/reevaluate_vertex_two_bases.py lego --seed 3721 --stop 1000 \\",
              "  --output output/vertex-two-bases-86/lego/seed3721",
              "python scripts/report_vertex_two_bases_86.py",
              "```", ""]
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(out)
    for filename, data in [("paired.csv", pairs), ("paired-per-view.csv", camera_pairs)]:
        if data:
            with (root/filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    print(f"Report: {len(rows)}/24 milestones", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/vertex-two-bases-86"))
    parser.add_argument("--report", type=Path, default=Path("docs/Vertex-MLP-86-Camaras.md"))
    args = parser.parse_args()
    report(args.input, args.report)
