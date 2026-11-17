# Integración Vertex-MLP Fourier-3 trilineal

## Protocolo

Base: `out/materials_t/checkpoints/iter020000_model.pt`; configuración de
`out/materials_t/config.yaml`. MSE + 0.02 SSIM, ss=1.5, fondo blanco,
semilla 3721. Geometría fija. El formato base permanece independiente.

## Estado y trazabilidad

Implementación en curso. No se han validado todavía kernels, entrenamiento ni
rendimiento de la variante. No interpretar los hiperparámetros como resultados.

## Cambios

- `src/vertex_appearance/decoder.py`: Fourier de tres bandas y MLP compartida
  26→64→3; última capa cero. Incluye control RGB por vértice.
- Los buffers del rasterizador permanecen globales; el orden de esquinas es
  000,001,010,011,100,101,110,111.

## Implicaciones

4.338.721 vértices, 2.937.583 vóxeles. Gather RGB: aproximadamente 269 MiB.
Las activaciones y Adam requieren memoria adicional. CUDA debe compilarse desde
este checkout: el entorno Conda svraster apuntaba inicialmente a otro proyecto.

## Validación inicial

Compilación con Conda `svraster`, PyTorch 2.5.0/cu124 y toolkit 12.5.
Paquete y binario cargados desde este checkout. `python -m pytest tests -q`:
5 pruebas aprobadas (Fourier, inicialización, regresión SH0, invariancia de
geometría, diferencias finitas y overfit sintético).

## Cambios CUDA

Se añaden `vertex_forward` y `vertex_backward` como especializaciones separadas
basadas en los kernels originales; no se modifican sus implementaciones SH.
El transporte interno empaqueta base RGB y 8 residuales en `[V,27]`. El buffer
backward usa 36 valores (8 geometría, 3 base, 1 prioridad, 24 residuales).
Esta desviación del buffer independiente propuesto preserva la interfaz de
retornos existente, pero duplica código CUDA que debe mantenerse sincronizado.
La MLP no se ejecuta en CUDA personalizado y se usa el mismo `interp_w` de densidad.

## Uso

```bash
conda activate svraster
CUDA_HOME=/usr/local/cuda-12.5 MAX_JOBS=2 python -m pip install -e cuda/ --no-build-isolation
python -m pytest tests -q
python train_vertex_fourier3.py --output output/vertex_materials --steps 5000
python render_vertex_fourier3.py --output output/vertex_materials/eval --appearance-checkpoint output/vertex_materials/checkpoints/iter005000_appearance.pt
```

El control usa `--mode vertex_rgb_trilinear`. La ruta original sigue disponible
mediante `render.py`; los checkpoints geométricos no se reescriben. No hay push.

## Entrenamiento y persistencia implementados

- `runtime.py`: configuración YAML específica, carga del protocolo base,
  congelación, Adam, selección reproducible de cámaras, evaluación y medición.
  Se usa un diccionario YAML en lugar de ampliar el estado global YACS, para
  aislar completamente el experimento del entrenamiento adaptativo original.
- `checkpoint.py`: SHA-256 del archivo base, hash de layout/orden/coordenadas,
  estado neuronal, Adam y estados aleatorios CPU/CUDA/cámaras. La carga rechaza
  identidades incompatibles y permite reubicar el checkpoint base.
- Scripts raíz separados para entrenar y renderizar; control con RGB libre.
- El instrumento de tiempos distingue raster (incluye preprocesamiento y
  empaquetado SH0), posprocesamiento, apariencia y gather. Fourier y MLP se
  miden además aisladamente. El total siempre incluye todas esas operaciones.

Prueba real de dos pasos: 200 vistas test evaluadas, PSNR 23.48459,
SSIM 0.900107, LPIPS 0.077448 y LPIPS corregido 0.091256.
Pico asignado durante esta prueba: 5.470.391.296 bytes. Estos valores son una
prueba de funcionamiento, no el resultado final del experimento.

Se produjo OOM al solapar accidentalmente evaluación y debug en dos procesos;
se repitió el debug con GPU libre. No se cambió precisión ni algoritmo.

## Validaciones ampliadas y commits

Tras recompilar las comprobaciones de entrada CUDA: **11 pruebas aprobadas**.
Se cubren además RGB constante frente a SH, gradiente cero bajo clamp y
persistencia de la imagen. Se conserva el protocolo de épocas completas
barajadas, con secuencia reproducible desde semilla y paso al reanudar.

Commits locales iniciales:
- `50610e7`: decoder Fourier y guía.
- `8a43a40`: forward/backward CUDA y regresiones.
- `11abc56`: entrenamiento, persistencia, evaluación e instrumentación.

## Límites

La continuidad garantizada es la del residual entre vóxeles conformes con
vértices compartidos. No se garantiza continuidad en interfaces adaptativas ni
en el color total si cambia SH0 entre vóxeles. Los resultados dependen del
checkpoint geométrico congelado. No se soportan AMP, más de una muestra,
regularizadores CUDA ni actualización de topología en el modo neuronal.

## Protocolo definitivo

El debug inicial usó selección con reemplazo. Se corrigió a épocas barajadas
como el baseline y se inició una corrida definitiva independiente desde cero,
`output/vertex_materials_final`, para comparar de forma consistente con el
control `output/vertex_materials_control`. Los runs smoke/debug y de reanudación
no se usarán como resultados finales. El debug de 1000 pasos llegó a un pico
asignado de 5.530.730.496 bytes.

## Inventario de interfaces y archivos

| Grupo | Archivos | Implicación |
|---|---|---|
| Decoder | `src/vertex_appearance/{__init__,decoder}.py` | Fourier y módulos entrenables independientes |
| Persistencia | `src/vertex_appearance/checkpoint.py` | Nuevo schema v2; no modifica SVInOut |
| Experimento | `src/vertex_appearance/runtime.py` | Config propia, training, evaluación y benchmark |
| Entradas | `train_vertex_fourier3.py`, `render_vertex_fourier3.py` | Nuevas CLI, no cambia train.py |
| Configuración | `cfg/vertex_fourier3_trilinear.yaml` | Parámetros del protocolo y arquitectura |
| Renderer | `src/sparse_voxel_gears/renderer.py` | Nuevo render_vertex, profiling opcional; color_mode preservado |
| Autograd | `cuda/svraster_cuda/renderer.py` | Dispatch de modo y gradiente RGB empaquetado |
| Kernels | `cuda/src/vertex_{forward,backward}.{cu,h}` | Interpolación y backward nuevos, recompilación requerida |
| Extensión | `cuda/binding.cpp`, `cuda/setup.py` | Exporta y compila kernels adicionales |
| Tests | `tests/test_{fourier3,trilinear_reference,vertex_checkpoint,vertex_cuda}.py` | Referencias, regresiones y diferenciación |

Para medir: añadir `--benchmark` al comando de render con checkpoint.
Los checkpoints guardan las frecuencias, convención angular y protocolo base
resuelto; las modificaciones posteriores del YAML base no alteran su inferencia.

Commit `31d0ac9`: validación estricta de buffers CUDA, configuración efectiva,
pruebas de render persistente y corrección del protocolo de cámaras.

El benchmark mueve el módulo neuronal a CPU durante las mediciones SH0/SH3,
para no atribuir su memoria de parámetros a esos baselines. Se mantienen fuera
de los commits los checkpoints, imágenes y artefactos binarios.
