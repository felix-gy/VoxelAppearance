# SVRaster diferido con NHT trilineal

Rama `Deferred-NHT-Trilineal` desde `MLP-Diferida` en `7f44699`. El experimento parte de la iteración SH 20.000 de Materials T y optimiza únicamente SH0, características de apariencia y la MLP. Conserva el diferido original y el modo SH predeterminado.

## Cálculo y gradientes

Para una intersección aceptada con intervalo `[a,b]`, se calcula el punto medio `o + (a + (b−a)/2)d` y sus coordenadas locales `q` en el vóxel. Los ocho pesos `N_j(q)` usan el orden de esquinas del baseline, con Z en el bit menos significativo. Con una muestra de densidad, el punto coincide con la muestra original. Con más muestras, el punto medio sigue determinando la apariencia y la cuadratura de densidad se conserva.

```text
z = Σ_j N_j(q) h_j
e_NHT = [sin(z), cos(z)]       # z de 4 canales
e_control = z                 # z de 8 canales
F = Σ_v (T_v α_v) e_v
B = Σ_v (T_v α_v) c_SH0,v + T_final c_fondo
A = 1 − T_final
C = clamp(B + A · MLP([F, B, Fourier3(d)]), 0, 1)
```

SH0 se evalúa con la convención del diferido original. El fondo se aporta a RGB y los latentes se acumulan sin normalización por opacidad. La MLP compartida `29→32→32→3` tiene 2.115 parámetros, usa ReLU y empieza con la capa de salida a cero. Se aplica por píxel antes del posprocesamiento y clamp originales. Los latentes se inicializan con desviación estándar 0,01. Las esquinas pertenecen a cada vóxel y sus valores pueden ser discontinuos en caras compartidas.

El backward mantiene fijos los pesos volumétricos. Para NHT, `g_z = g_sin cos(z) − g_cos sin(z)`; para el control, `g_z = g_F`. Cada gradiente de esquina recibe `T_v α_v N_j(q) g_z` mediante acumulación atómica. Los gradientes RGB siguen el camino de SH0 existente en modo diferido. El código recalcula la consulta local en backward y no almacena las intersecciones. La MLP se deriva con PyTorch.

## Inventario de código

| Archivo y símbolos | Tipo | Motivo e implicación |
| --- | --- | --- |
| `cuda/src/deferred_trilinear.cuh` — constantes del layout, `midpoint`, `interpolate`, `accumulate`, `backward` | Añadido | Comparte la definición local entre forward y backward. Las especializaciones de cuatro y ocho canales separan la variante armónica del control lineal. |
| `cuda/src/forward.cu` — `renderCUDA`, `rasterize_voxels` | Modificado | Admite entradas por vóxel de 35 y 67 float32 (RGB seguido de 8×4 u 8×8 valores en orden esquina/canal). Conserva la salida diferida planar de 11 canales: tres de RGB y ocho latentes. Interpola y acumula en la misma pasada. |
| `cuda/src/backward.cu` — `renderCUDA`, `rasterize_voxels_backward` | Modificado | Lee gradientes de los 11 canales y los acumula en un tensor por vóxel del ancho de entrada. Reutiliza el peso original, recalcula el punto medio y deriva seno/coseno cuando corresponde. No habilita gradientes de geometría. |
| `cuda/svraster_cuda/renderer.py` — `rasterize_voxels` | Modificado | Valida anchos 3, 11, 35 y 67. Aplica a toda variante diferida los requisitos de geometría congelada y pérdidas compatibles. Mantiene las firmas públicas y privadas de la extensión. |
| `src/sparse_voxel_gears/deferred.py` — modos, `FEATURE_SHAPES`, `DeferredAppearance`, `save_deferred`, `restore_deferred` | Añadido y modificado | Selecciona la forma de parámetros y registra la variante en el checkpoint. Verifica modo, versión, layout, forma y dtype. El diferido original conserva su esquema v1 y puede cargar sus checkpoints anteriores. |
| `src/sparse_voxel_model.py` — constructor e `init_deferred` | Modificado | Selecciona el campo opcional y conserva las reglas de congelación. |
| `src/sparse_voxel_gears/renderer.py` — `vox_fn`, `render` | Modificado | Empaqueta las esquinas sin alterar el orden, selecciona el modo del modelo y permite medir el intervalo CUDA del rasterizador, incluyendo preparación y codificación local. |
| `src/sparse_voxel_gears/io.py` — `save`, `load` | Modificado | Despacha los tres modos diferidos por formato y mantiene la lectura SH. La persistencia diferida sigue usando reemplazo atómico. |
| `train_deferred.py` — `benchmark`, `run`, CLI | Modificado | Añade `--appearance-mode` y `--eval-views`, restaura la selección al reanudar y permite mediciones en hitos reanudados. Registra el primer paso y el pico de CUDA del tramo, etapas de render, calidad y conteos. SH0 y SH3 se miden únicamente en el flujo original de benchmark de `deferred_mlp`. |
| `render.py` — CLI | Modificado | Permite evaluar checkpoints NHT y trilineales con su selector explícito. |
| `tests/test_deferred_trilinear.py` | Añadido | Usa la fixture CUDA original para contrastar ambos modos con PyTorch, diferencias finitas y reanudación. |

No se eliminó ninguna función ni formato previo. Los costos añadidos dependen de las intersecciones aceptadas y del ancho de los latentes. La medición de CUDA asignada por PyTorch no incluye la memoria de otros procesos.

## Validación y reproducción

La extensión compiló en CUDA Toolkit 12.5 con PyTorch 2.5.0+cu124. Pasaron cinco pruebas CUDA, incluidas las tres regresiones originales de SH y MLP diferida. Las dos pruebas nuevas comprobaron referencia de composición, fondo vacío, gradientes con perturbaciones 1e−3 y 1e−2, activación de gradientes después de actualizar la salida, diez pasos adicionales de ajuste y reanudación. Usaron `atol=1e-4`, `rtol=1e-2` para diferencias finitas y `atol=rtol=1e-5` para el render recargado.

```bash
(cd cuda && python setup.py build_ext --inplace)
export PYTHONPATH="$PWD/cuda${PYTHONPATH:+:$PYTHONPATH}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_deferred_mlp.py tests/test_deferred_trilinear.py

# Repetir con deferred_trilinear para el control.
mode=deferred_nht_trilinear
run_dir="output/materials-t-32-$mode"
python train_deferred.py --appearance-mode "$mode" --source_model out/materials_t \
  --source_iteration 20000 --model_path "$run_dir" --steps 1 --views 32 \
  --eval-views 4 --res_width 256 --ss 1
python train_deferred.py --model_path "$run_dir" --resume --steps 29 --benchmark
python train_deferred.py --model_path "$run_dir" --resume --steps 270 --benchmark
python render.py "$run_dir" --appearance-mode "$mode" --iteration 300
```

## Materials T

RTX 3060, 2.937.583 vóxeles, 256×256, `ss=1`, una muestra de densidad, fondo blanco y semilla 3721. Las 32 vistas de entrenamiento se visitaron en el mismo orden cíclico que el diferido base. La evaluación reservada usó las primeras cuatro vistas por nombre del split test, `r_0.png`, `r_1.png`, `r_10.png` y `r_100.png`. Se usó MSE de peso 1 y Adam de tasas constantes 0,01 para SH0 y 0,00025 para latentes y MLP, con betas (0,1, 0,99) y epsilon 1e−15.

Los conteos de características son 23.500.664 en la base (8V), 94.002.656 en NHT (32V) y 188.005.312 en el control (64V). Cada variante añade también 2.115 parámetros de la MLP y entrena los 8.812.749 escalares SH0 existentes. El control tiene el doble de latentes que NHT, por lo que la ablación cambia también la capacidad y el costo de entrenamiento.

### Calidad por hito

| Modo | Paso | PSNR entrenamiento (dB) | SSIM entrenamiento | PSNR reservado (dB) | SSIM reservado |
| --- | ---: | ---: | ---: | ---: | ---: |
| SH0, referencia registrada | — | — | — | 22,495 | 0,88132 |
| SH3, referencia registrada | — | — | — | 26,161 | 0,94832 |
| Diferido base, registro previo | 30 | 25,023 | 0,92876 | 22,648 | 0,88227 |
| Diferido base, registro previo | 300 | 29,341 | 0,96827 | 23,133 | 0,87000 |
| NHT | 30 | 25,024 | 0,92875 | 22,651 | 0,88229 |
| NHT | 300 | 29,282 | 0,96798 | 23,158 | 0,87022 |
| Control trilineal | 30 | 25,023 | 0,92876 | 22,648 | 0,88227 |
| Control trilineal | 300 | 29,358 | 0,96848 | 23,135 | 0,87008 |

Se reutilizaron los registros de SH0, SH3 y diferido base. No se midió LPIPS en este protocolo. En ambos modos nuevos subió el PSNR reservado entre 30 y 300, pero cayó el SSIM. Por el criterio acordado, ambos se detuvieron a 300. A ese paso, NHT ganó 0,025 dB y 0,00022 SSIM frente al diferido base. Son diferencias pequeñas observadas en cuatro vistas, con una sola semilla. NHT y control también dieron resultados de calidad cercanos entre sí. El experimento no aísla el efecto armónico a igual número de parámetros.

### Costo al paso 300

| Modo | ms completos | FPS | ms raster + codificación | ms MLP | Pico inferencia (MiB) | Pico entrenamiento (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Diferido base, registro previo | 17,74 | 56,36 | — | 0,54 | 1.562,08 | — |
| NHT | 21,20 | 47,18 | 20,53 | 0,46 | 2.099,79 | 3.250,87 |
| Control trilineal | 25,74 | 38,86 | 24,99 | 0,45 | 2.817,79 | 5.046,28 |

El paso 1 sirvió de prueba de memoria en escena real e incluyó la primera actualización de Adam: 3.039,68 MiB para NHT y 4.835,09 MiB para el control. La memoria de entrenamiento de la tabla corresponde al tramo 31–300. El tiempo completo se midió en las cuatro vistas reservadas con sincronización CUDA. La etapa raster incluye la preparación de datos y la interpolación/codificación local. La MLP se cronometró por separado. Fourier direccional queda incluido en el tiempo completo. El control costó más en esta prueba aunque su descriptor omite las funciones armónicas. Su ancho de entrada y los parámetros por vóxel son mayores.

Para los checkpoints 30 y 300 verifiqué los tensores congelados contra el checkpoint SH, y las listas de cámaras contra el registro previo. También cargué el checkpoint NHT 300 con `render.py` y rendericé las 200 vistas del split test en modo FPS como comprobación del lector. Los checkpoints y métricas de ambos hitos están en `output/materials-t-32-deferred_nht_trilinear/` y `output/materials-t-32-deferred_trilinear/`.
