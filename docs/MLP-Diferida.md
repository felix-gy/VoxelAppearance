# SVRaster con MLP diferida

Rama `MLP-Diferida` desde `main` en `74e8907`. El modo `deferred_mlp` parte de un modelo SH y optimiza SH0, ocho latentes por vóxel y una MLP compartida. La densidad, el octree y los coeficientes SH superiores quedan congelados. El modo SH sigue siendo el predeterminado.

## Cambios de código

| Archivo | Tipo y símbolos | Motivo e implicación |
| --- | --- | --- |
| `cuda/src/forward.cu` | Modificado. `renderCUDA`, `render`, `rasterize_voxels_procedure`, `rasterize_voxels` | Acepta tres canales SH u once canales empaquetados. Reutiliza el recorrido original y acumula ocho latentes por píxel en registros por hilo. La salida interna contiene RGB seguido de latentes y el fondo se añade a RGB. El caso diferido sin vóxeles devuelve transmitancia uno. |
| `cuda/src/backward.cu` | Modificado. `renderCUDA`, `render`, `rasterize_voxels_backward` | Reconstruye los pesos con la recurrencia original de transmitancia. En modo diferido escribe once gradientes por vóxel con atómicos y omite las derivadas de densidad y prioridad. Los tres canales SH conservan el backward original. |
| `cuda/svraster_cuda/renderer.py` | Modificado. `rasterize_voxels` | Valida el ancho del tensor empaquetado y exige densidad y prioridad congeladas para los once canales. Rechaza las pérdidas del kernel incompatibles con este backward. La firma de la extensión y la tupla de salidas se conservan. |
| `src/sparse_voxel_gears/deferred.py` | Añadido. `DeferredAppearance`, `layout_digest`, `save_deferred`, `restore_deferred` | Implementa Fourier y el decoder PyTorch, y registra las características. Guarda un checkpoint versionado y atómico. Verifica el hash del layout y las formas y tipos de la apariencia antes de cargarla. |
| `src/sparse_voxel_model.py` | Modificado y añadido. Constructor e `init_deferred` | Registra el modo y crea el decoder tras cargar SH. Congela densidad, prioridad y SH superiores. |
| `src/sparse_voxel_gears/renderer.py` | Modificado. `vox_fn` y `render` | Empaqueta SH0 y latentes, separa las imágenes compuestas y aplica el decoder antes del posprocesamiento original. `return_deferred_buffers=True` expone `B`, `F` y `A` a resolución de rasterización. Conserva los modos SH explícitos para comparación. |
| `src/sparse_voxel_gears/adaptive.py` | Modificado. `pruning` y `subdividing` | Añade un rechazo explícito de la adaptación mientras está inicializada la apariencia diferida. Los algoritmos de poda y subdivisión de SH se conservan. |
| `src/sparse_voxel_gears/io.py` | Modificado y añadido. `save`, `load`, `load_iteration`, `model_state` y `load_model_state` | Extrae el esquema base para reutilizarlo. El lector elige el formato según el modo y la búsqueda de iteraciones omite temporales. Los lectores SH reciben un error explícito con checkpoints diferidos. |
| `train_deferred.py` | Añadido. `make_optimizer`, `photo_loss`, `training_state`, `restore_training`, `benchmark`, `run` y CLI | Inicializa desde SH, entrena apariencia, reanuda y mide las tres rutas. Reutiliza el cargador de datos y la pérdida fotométrica configurada. Tiene grupos Adam independientes para SH0 y la apariencia nueva. |
| `render.py` | Modificado. CLI y creación del modelo | `--appearance-mode deferred_mlp` selecciona explícitamente el checkpoint diferido para evaluación. |
| `tests/test_deferred_mlp.py` | Añadido | Valida identidades numéricas, gradientes, ajuste corto, persistencia y compatibilidad SH. |
| Código eliminado | Ningún archivo ni función pública | Se extrajo la serialización común y se generalizó el layout interno para reutilizar el rasterizador. |

## Composición y gradientes

Para cada intersección aceptada el kernel calcula el peso original `w = T·α`. En el layout interno `[11,H,W]`, los canales 0–2 contienen `B = Σ w·c_SH0 + T_final·fondo` y los canales 3–10 contienen `F = Σ w·f_v`. El renderer calcula `A = 1 − T_final`. Las características son float32 contiguas `[V,8]` y se inicializan con desviación estándar 0,01. No se divide `F` por `A`. Los modos de fondo determinista conservan la composición del baseline. El fondo blanco se incluye en CUDA y el fondo de media de imagen, cuando corresponde, se incorpora antes del decoder.

La MLP `29→32→32→3` tiene 2.115 parámetros, dos ReLU y una salida lineal inicializada a cero. La entrada concatena `F`, `B` y la dirección mundial normalizada, con seno XYZ y coseno XYZ para cada frecuencia `1,2,4`. El decoder se evalúa en batch una vez por píxel rasterizado. Forma `B + A·ΔRGB`, seguido del redimensionado y clamp del renderer. Un píxel vacío conserva el fondo. La profundidad y las normales se producen antes del decoder.

Con los pesos de composición fijos, el rasterizador acumula `w·∂L/∂F` y `w·∂L/∂B` para cada vóxel visible. La parte RGB usa la convención `max(0, 0.28209479177387814·SH0 + 0.5)` y deriva su clamp en esta ruta. Autograd deriva la MLP, la composición por imagen y el posprocesamiento. El backward de densidad diferida queda fuera de esta fase.

El entrenamiento usa MSE, L1 o Huber según configuración y emplea `sh0_lr` y `shs_lr` con los betas y epsilon configurados. El checkpoint contiene geometría, SH0, SH superiores de referencia, parámetros diferidos, configuración, índice del paso, secuencia de cámaras, optimizador y RNG de Python, NumPy, Torch CPU y CUDA. Se publica mediante un temporal y renombrado. La configuración de Ship incluía `graph_refine` y `pose_opt`, secciones ajenas al baseline que el entrenador de apariencia informó y omitió. El entrenador también informó los regularizadores de esa ejecución que quedaron inactivos: SSIM en la pérdida, concentración de transmitancia y color, distorsión y TV de densidad.

## Verificación

Compilado con PyTorch 2.5.0+cu124 y Toolkit CUDA 12.5. Las pruebas CUDA de uno y dos vóxeles verificaron composición y fondo vacío, identidad inicial con supersampling, profundidad y normales, compatibilidad SH3, diferencias finitas para latentes, SH0 y MLP con `atol=1e-4` y `rtol=1e-2`, y el flujo de gradientes tras el primer paso de la salida. El ajuste sintético de 20 pasos redujo la pérdida manteniendo exactamente los tensores congelados. La recarga restauró los parámetros del decoder y reprodujo la imagen con `atol=rtol=1e-5`. Se comprobó el siguiente paso de optimización. En una copia del checkpoint de Ship, la CLI reanudó desde el paso 30 y publicó el paso 31. El lector de evaluación cargó el checkpoint 30 y renderizó las 200 vistas de prueba en su modo de FPS.

### Experimento Ship 5000

RTX 3060, 570.968 vóxeles, imágenes de 256×256, `ss=1`, cuatro vistas de entrenamiento y cuatro reservadas, seleccionadas por nombre. Tras 30 pasos, la pérdida del primer y último paso fue 0,002528 y 0,000934, respectivamente. Los tiempos se midieron por fotograma con sincronización CUDA y un render de calentamiento por modo. La columna MLP mide el decoder con eventos CUDA. El tiempo completo incluye rasterización, Fourier y MLP cuando corresponde. La VRAM de la tabla es el pico asignado por PyTorch durante la evaluación.

| Modo | PSNR (dB) | SSIM | ms por fotograma | FPS | MLP (ms) | Pico MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SH0 | 25,257 | 0,81688 | 3,459 | 289,1 | — | 385,93 |
| SH3 | 27,471 | 0,87849 | 3,729 | 268,2 | — | 385,93 |
| MLP diferida | 24,952 | 0,80872 | 4,077 | 245,3 | 0,442 | 423,83 |

La MLP diferida quedó por debajo de ambas referencias en calidad medida y FPS en esta ejecución corta. Estos resultados corresponden a Ship y al presupuesto indicado. La acumulación adicional crece con las intersecciones y los buffers del decoder con la resolución rasterizada. El número de vóxeles permanece fijo durante el ajuste.

## Reproducción

```bash
(cd cuda && python setup.py build_ext --inplace)
export PYTHONPATH="$PWD/cuda${PYTHONPATH:+:$PYTHONPATH}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_deferred_mlp.py
python train_deferred.py --source_model out/ship_0 --source_iteration 5000 \
  --model_path output/mlp-diferida-ship --steps 30 --views 4 --res_width 256 --ss 1 --benchmark
python render.py output/mlp-diferida-ship --appearance-mode deferred_mlp --iteration 30
python train_deferred.py --model_path output/mlp-diferida-ship --resume --steps 1
```
