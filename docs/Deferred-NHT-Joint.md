# NHT trilineal conjunto para Lego

Rama `Deferred-NHT-Joint`, desde `38b0471` de `Deferred-NHT-Trilineal`. El modo optativo `deferred_nht_joint` usa el entrenador original. Las iteraciones 1–1000 usan SH0. Al final de la 1000 se ejecuta primero la adaptación programada y luego se inicializan los latentes y el decoder con salida cero sobre la topología resultante. El primer render NHT es el de la iteración 1001. Las promociones SH quedan omitidas en este modo.

## Gradientes y adaptación

Se conserva el descriptor NHT local `e_v = [sin(z_v), cos(z_v)]`, donde `z_v` interpola los cuatro latentes de las ocho esquinas en el punto medio del segmento rayo–vóxel. La MLP `29→32→32→3` usa el mapa compuesto `F = Σ w_v e_v`, RGB base `B` y Fourier-3 de la dirección. Produce el residual por píxel antes del posprocesamiento habitual. Su factor `A = 1 − T_final` se diferencia por PyTorch.

En CUDA, el gradiente de cada peso incluye `dot(g_B, c_SH0) + dot(g_F, e_v)`. El gradiente final de transmitancia incluye el camino por `A` y el fondo y alimenta la misma recurrencia inversa que el baseline. Se conservan las ramas de profundidad, normales, distorsión y orden. La concentración de color local sigue usando SH0 y la TV se aplica en el entrenador a los parámetros de densidad. La prioridad acumula `abs(alpha * dL/dalpha)` con el término NHT incluido. Se mantiene la linealización de alpha del baseline en saturación. Las pruebas de diferencias finitas se ejecutan fuera de discontinuidades y saturación.

Poda conserva exactamente los latentes de las filas supervivientes. Subdivisión calcula cada esquina hija evaluando el campo latente trilineal del padre, con el mismo orden de hijos y esquinas de la geometría. El decoder conserva sus pesos. Las salidas por píxel pueden variar al cambiar la topología, los puntos medios y los pesos volumétricos. SparseAdam usa el grupo angular original y `shs_lr` para latentes y decoder. Tras adaptación se recrean optimizador y scheduler en el punto original y se restaura el estado del scheduler. Esto conserva el comportamiento previo de reiniciar los momentos Adam durante la adaptación. La activación tampoco adelanta el RNG del warmup, porque la asignación del campo ocurre después de la primera adaptación.

## Inventario de cambios

| Archivo | Cambio | Razón e implicación |
| --- | --- | --- |
| `cuda/src/backward.cu`, `backward.h` | Modificado. Kernel, lanzador y punto de entrada reciben un selector explícito de NHT conjunto. | El backward utiliza una fila de 44 gradientes por vóxel (8 densidades, 3 RGB, 1 prioridad, 32 latentes). Recalcula el descriptor y acumula el gradiente que faltaba hacia densidad y prioridad. El forward conserva el layout NHT de 35 entradas y 11 salidas. |
| `cuda/svraster_cuda/renderer.py` | Modificado. Validación y `_RasterizeVoxels.backward`. | Habilita densidad y regularizadores solo en el modo conjunto entre los diferidos y propaga el selector a CUDA. La extensión requiere recompilación. |
| `src/sparse_voxel_gears/deferred.py` | Añadido y modificado. `JOINT_MODE`, calendario, `save_deferred` y `restore_deferred`. | Extiende el formato atómico con fase y prioridad. El warmup guarda SH0 y geometría sin latentes. Desde el checkpoint 1000 se guardan también el decoder y los latentes. Se validan identidad de modo, layout y fase. |
| `src/sparse_voxel_model.py`, `src/sparse_voxel_gears/renderer.py` | Modificado. Inicialización y despacho. | La fase inactiva usa SH0. El modo activo conserva densidad y prioridad entrenables, admite las opciones de fondo originales y comparte el forward con NHT congelado. `render.py` reconoce el modo mediante el catálogo existente. |
| `src/sparse_voxel_gears/adaptive.py` | Modificado. `pruning` y `subdividing`. | Remapea las filas locales con el mismo orden de poda y subdivisión del octree. Registra un parámetro nuevo para los latentes remapeados y conserva el módulo compartido. |
| `train.py` | Modificado. CLI, selección angular, transición y checkpoint. | Usa la lógica original de cámaras, pérdidas y adaptación. Guarda optimizador, scheduler, prioridad, secuencia de cámaras, exposición y RNG. Reanuda en la siguiente iteración del checkpoint. |
| `tests/test_deferred_nht_joint.py` | Añadido. | Valida ambos caminos de densidad y el gradiente combinado, características, SH0, MLP, prioridad, remapeo y checkpoints. |

No se eliminó un modo previo ni se añadió un entrenador alternativo. Los checkpoints conjuntos requieren precisión completa. El modo SH y los diferidos congelados conservan su esquema y sus rutas.

## Validación ejecutada

- Se compiló la extensión con CUDA Toolkit 12.5 y PyTorch 2.5.0+cu124. Pasaron **9 pruebas CUDA**, incluidas las regresiones SH y diferidas previas y las pruebas conjuntas. Se usaron `atol=1e-4` y `rtol=1e-2` para diferencias finitas.
- En una fixture de dos imágenes de 16×16, el `train.py` real podó de 8 a 4 vóxeles y subdividió a 32 al final de la iteración 1000. El checkpoint 999 conservó SH0 y el 1000 incluyó NHT con salida a cero. Reanudar desde 999 reprodujo la geometría y apariencia de la iteración 1001 con `atol=1e-6`; coincidieron scheduler y RNG CUDA. `render.py` escribió la imagen de prueba del checkpoint conjunto.
- El entrenamiento completo de Lego y su consumo pico de memoria quedan sin medir.

## Comandos

Desde la raíz del repositorio y con el entorno SVRaster activo:

```bash
(cd cuda && python setup.py build_ext --inplace)
export PYTHONPATH="$PWD/cuda${PYTHONPATH:+:$PYTHONPATH}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_deferred_mlp.py tests/test_deferred_trilinear.py tests/test_deferred_nht_joint.py

python train.py --appearance-mode deferred_nht_joint \
  --model_path output/lego-nht-joint \
  --cfg_files cfg/synthetic_nerf.yaml \
  --source_path data/nerf_synthetic/lego \
  --checkpoint_iterations 1000 5000 10000 \
  --test_iterations 1000 5000 10000 20000

python train.py --appearance-mode deferred_nht_joint \
  --model_path output/lego-nht-joint --load_iteration -1

python render.py output/lego-nht-joint \
  --appearance-mode deferred_nht_joint --iteration -1 --skip_train
python eval.py output/lego-nht-joint
```

Para abrir una cámara interactiva en Viser:

```bash
python viz.py output/lego-nht-joint \
  --appearance-mode deferred_nht_joint --iteration 20000 --port 7007
```

Abrir `http://localhost:7007` en el navegador. El visor renderiza el checkpoint
seleccionado con la misma ruta NHT que usa `render.py`. Se puede cambiar la
resolución y el supersampling en la interfaz.
