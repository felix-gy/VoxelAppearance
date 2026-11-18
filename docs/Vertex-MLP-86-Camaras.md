# Vertex-MLP con 86 cámaras y suavizado de coeficientes

**Hitos completos: 18/24.**

## Protocolo

Rama `Vertex-MLP-Dos-Bases`, RTX 3060. Se comparan una base, dos bases y dos bases con penalización. Cada brazo usa las primeras 86 cámaras de entrenamiento por nombre y las mismas cuatro reservadas de la prueba anterior, en orden cíclico, a 256×256, fondo blanco, ss=1 y una muestra por vóxel.

Las semillas son 3721 y 3722. En Materials T, ambos brazos de dos bases y el control de una base arrancan del Vertex-MLP histórico de 300 pasos de la semilla correspondiente y congelan su mismo SH0. Los pasos de las tablas son actualizaciones adicionales para Materials T. En Lego se congela el checkpoint SH 20.000 de `out/lego-original` y cada semilla parte de la inicialización compartida de una base.

Densidad, geometría, SH0 y topología se congelan. Se mantiene el rasterizador Vertex-MLP, la decodificación densa para comparar y MSE con Adam a 0,00025, betas (0,1, 0,99), epsilon 1e−15. Los estados de Adam empiezan en cero y continúan entre hitos. Las dos variantes de dos bases comparten la inicialización completa, incluidos los pesos pequeños de la segunda base y mezcla (1,0).

El brazo regularizado añade `0.001 × mean((mix[v] − mix[u])²)` sobre 4096 aristas muestreadas por paso de un conjunto fijo de caras vecinas del mismo nivel. Se usan ambos coeficientes de cada vóxel. El conjunto se construye con semilla 7919 desde 65.536 vóxeles muestreados y vecinos en los tres ejes positivos. El muestreo de cada paso es determinista por índice y se puede reconstruir al reanudar.

La evaluación usa PSNR, SSIM y LPIPS-VGG normalizado, promediados sobre cuatro vistas. El RMS mide el salto RGB del residual en el centro de caras vecinas del mismo nivel, antes de SH0 y clamp. Sus caras se muestrean con semilla independiente de la regularización. La prueba no mide saltos entre niveles distintos.

## Resultados por escena, semilla y paso

| Escena | Semilla | Paso | Brazo | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | RMS de salto ↓ | Pico entrenamiento (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| materials_t | 3721 | 300 | one | 23.0188 | 0.869270 | 0.152912 | 0.000000 | 5206 |
| materials_t | 3721 | 300 | two | 23.0342 | 0.869683 | 0.152834 | 0.004941 | 5295 |
| materials_t | 3721 | 300 | regularized | 23.0342 | 0.869683 | 0.152829 | 0.004910 | 5297 |
| materials_t | 3721 | 1000 | one | 23.0933 | 0.870510 | 0.152721 | 0.000000 | 5206 |
| materials_t | 3721 | 1000 | two | 23.0437 | 0.866861 | 0.158742 | 0.059072 | 5295 |
| materials_t | 3721 | 1000 | regularized | 23.0444 | 0.866897 | 0.158684 | 0.058763 | 5297 |
| materials_t | 3722 | 300 | one | 22.9946 | 0.869808 | 0.152893 | 0.000000 | 5206 |
| materials_t | 3722 | 300 | two | 23.0079 | 0.870119 | 0.152916 | 0.004721 | 5295 |
| materials_t | 3722 | 300 | regularized | 23.0080 | 0.870122 | 0.152909 | 0.004698 | 5297 |
| materials_t | 3722 | 1000 | one | 23.0046 | 0.870251 | 0.152367 | 0.000000 | 5206 |
| materials_t | 3722 | 1000 | two | 22.9556 | 0.866343 | 0.158854 | 0.053123 | 5295 |
| materials_t | 3722 | 1000 | regularized | 22.9561 | 0.866379 | 0.158786 | 0.052875 | 5297 |
| lego | 3721 | 300 | one | 26.8189 | 0.950931 | 0.050562 | 0.000000 | 7854 |
| lego | 3721 | 300 | two | 27.0276 | 0.954816 | 0.047440 | 0.010215 | 7985 |
| lego | 3721 | 300 | regularized | 27.0278 | 0.954821 | 0.047433 | 0.010193 | 7987 |
| lego | 3722 | 300 | one | 26.8577 | 0.951068 | 0.050389 | 0.000000 | 7854 |
| lego | 3722 | 300 | two | 27.0505 | 0.954716 | 0.047499 | 0.009564 | 7985 |
| lego | 3722 | 300 | regularized | 27.0507 | 0.954722 | 0.047495 | 0.009548 | 7987 |

## Diferencias pareadas

Signos: PSNR y SSIM positivos favorecen el primer brazo. LPIPS y RMS negativos lo favorecen. Una media de dos semillas no estima por sí sola significación estadística.

| Escena | Paso | Comparación | Semillas | Δ PSNR (dB) | Δ SSIM | Δ LPIPS | Δ RMS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| lego | 300 | regularized − one | 2 | +0.200983 | +0.003772 | -0.003011 | +0.009871 |
| lego | 300 | regularized − two | 2 | +0.000234 | +0.000005 | -0.000006 | -0.000019 |
| lego | 300 | two − one | 2 | +0.200749 | +0.003767 | -0.003006 | +0.009889 |
| materials_t | 300 | regularized − one | 2 | +0.014407 | +0.000364 | -0.000033 | +0.004804 |
| materials_t | 300 | regularized − two | 2 | +0.000045 | +0.000002 | -0.000006 | -0.000027 |
| materials_t | 300 | two − one | 2 | +0.014363 | +0.000362 | -0.000027 | +0.004831 |
| materials_t | 1000 | regularized − one | 2 | -0.048685 | -0.003742 | +0.006191 | +0.055819 |
| materials_t | 1000 | regularized − two | 2 | +0.000649 | +0.000036 | -0.000063 | -0.000278 |
| materials_t | 1000 | two − one | 2 | -0.049335 | -0.003778 | +0.006254 | +0.056098 |

## Trayectoria de PSNR al pasar de una a dos bases

- **materials_t.** Δ medio a 300 y 1000: +0.0144 y -0.0493 dB. la diferencia se movió a favor de una base entre 300 y 1000.
- **lego.** Comparación longitudinal pendiente hasta completar las dos semillas de cada hito.

## Efecto del suavizado de coeficientes

- **materials_t.** El suavizado bajó el RMS en las dos semillas. El RMS medio pasó de 0.056098 a 0.055819. Las diferencias medias fueron +0.0006 dB, +0.000036 SSIM y -0.000063 LPIPS.
- **lego.** Comparación al paso 1000 pendiente.

## Implementación y validación

`neighbors.py` localiza vecinos del mismo nivel y crea la penalización diferenciable. El entrenador congelado admite un término opcional que se suma al MSE. El brazo regularizado conserva la arquitectura de dos bases y el formato de apariencia. Los checkpoints conservan la configuración de muestreo y permiten reproducir la secuencia al reanudar.

Pasaron 25 pruebas CUDA, incluidas las regresiones de Vertex-MLP y la nueva prueba de gradiente de la penalización. En cada experimento se comprobó identidad del render inicial con tolerancia 1e−5 y se verificó el contador de Adam de cada tensor entrenable en cada hito.

## Artefactos y reproducción

Cada hito publica un JSON por brazo con las cuatro métricas, resultados por cámara y el pico de entrenamiento del tramo. Los checkpoints incluyen apariencia, identidad, SH0 congelado y estado Adam. `paired.csv` y `paired-per-view.csv` contienen los deltas por semilla y por cámara.

```bash
export PYTHONPATH="$PWD/cuda:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/reevaluate_vertex_two_bases.py lego --seed 3721 --stop 1000 \
  --output output/vertex-two-bases-86/lego/seed3721
python scripts/report_vertex_two_bases_86.py
```
