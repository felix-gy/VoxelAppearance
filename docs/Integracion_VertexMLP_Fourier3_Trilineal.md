# Integración Vertex-MLP Fourier-3 trilineal en SVRaster

## Estado y alcance

Implementación completada y validada en Conda `svraster`: decoder, forward y
backward CUDA, entrenamiento de apariencia, inferencia, persistencia,
reanudación, pruebas automáticas y comparación con SH0, SH3 y RGB por vértice.
Ambas variantes completaron 5000 pasos. El resultado funcional no implica
superar al baseline SH3; los resultados medidos se muestran abajo.

## Protocolo y entorno

- Base: `out/materials_t/checkpoints/iter020000_model.pt`, sin cuantización.
- Configuración base: `out/materials_t/config.yaml`.
- Datos: `data/nerf_synthetic/materials`; 200 vistas test.
- MSE + 0.02·(1−SSIM), usando la implementación SSIM existente.
- Fondo blanco, autoexposición desactivada, ss=1.5 fijo, una muestra por segmento.
- Semilla 3721; cámaras en épocas completas barajadas, idénticas para las variantes.
- Adam: LR características/RGB 0.01, LR MLP 0.001, betas (0.9,0.99), eps 1e-8.
- Sin poda, subdivisión, regularizadores CUDA/geométricos ni scheduler.
- Validación cada 500 pasos, checkpoint cada 1000 y al finalizar.
- RTX 3060 12 GiB; PyTorch 2.5.0+cu124; toolkit de compilación CUDA 12.5.
- Se corrigió la instalación editable: paquete y binario ahora provienen de este
  checkout, no de `/home/felix/Projects/svraster`.

El checkpoint tiene P=4.338.721 vértices y V=2.937.583 vóxeles. H y MLP suman
34.711.691 parámetros; el control aprende únicamente 3P valores RGB.

## Arquitectura y contratos

Cámara → direcciones mundo cámara→vértice → Fourier → MLP compartida → gather
por vox_key → interpolación CUDA en el punto medio original → composición.

- `H[P,8]`, Fourier `[P,18]`, entrada `[P,26]`, MLP `26→64→3` (1923 parámetros).
- Frecuencias π,2π,4π; cada banda contiene sin(XYZ),cos(XYZ), sin dirección cruda.
- Última capa cero; sin sigmoid/tanh. Primera capa y H pueden tener gradiente
  cero en el primer backward y recibirlo después del primer paso.
- Residual `[P,3]`, gather `[V,8,3]`, float32 contiguo en CUDA.
- Transporte interno `[V,27]`: RGB base seguido de los ocho RGB residuales.
- SH0 se evalúa mediante el mecanismo original (factor SH, +0.5 y clamp mínimo).
- Color de muestra: max(SH0 + suma de pesos trilineales por residual, 0).
- Orden de esquinas: 000,001,010,011,100,101,110,111; k=4bx+2by+bz.
- Se reutilizan los pesos del punto medio de densidad; alfa, sus umbrales,
  ordenamiento, T, depth, normales, fondos y clamp final permanecen originales.

Los buffers usan índices **globales**, aunque vox_fn recibe índices visibles.
freeze_vox_geo produce [V,8]; no se compacta. La validación inicial comprueba
formas, tipos, límites de índices, coordenadas finitas y hasta 4096 vóxeles para
verificar geométricamente el orden de esquinas, sin duplicar el gather global.

El backward empaqueta 36 valores por vóxel: 8 de geometría, 3 base, 1 prioridad
y 24 residuales. Acumula por operaciones atómicas, aplica la máscara preclamp>0
y los pesos originales, sin duplicar alfa/transmitancia. Autograd del gather
suma vértices compartidos y deriva la MLP. El wrapper desacopla geometría, SH0
y prioridad: solo la apariencia recibe gradientes, incluso desde otros scripts.

## Inventario de cambios e implicaciones

| Archivos | Modificación e implicación |
|---|---|
| `src/vertex_appearance/decoder.py`, `__init__.py` | Fourier y módulos VertexAppearance/VertexRGB independientes |
| `src/vertex_appearance/layout.py` | Validación del layout antes de reservar apariencia |
| `src/vertex_appearance/checkpoint.py` | Schema v2 de apariencia, identidad geométrica y reanudación |
| `src/vertex_appearance/runtime.py` | Configuración, entrenamiento, evaluación e instrumentación |
| `train_vertex_fourier3.py`, `render_vertex_fourier3.py` | Nuevas CLI; entrenamiento original independiente |
| `cfg/vertex_fourier3_trilinear.yaml` | Protocolo y arquitectura validados |
| `src/sparse_voxel_gears/renderer.py` | Nuevo render_vertex y eventos opcionales; render original conservado |
| `cuda/svraster_cuda/renderer.py` | Dispatch del modo y transporte autograd del RGB empaquetado |
| `cuda/src/vertex_forward.cu`, `.h` | Nueva ruta de color espacial; recompilación requerida |
| `cuda/src/vertex_backward.cu`, `.h` | Gradientes por esquinas y validación de entradas |
| `cuda/binding.cpp`, `cuda/setup.py` | Nuevas exportaciones y unidades CUDA |
| `tests/test_fourier3.py`, `test_trilinear_reference.py` | Referencias de codificación, pesos y acumulación |
| `tests/test_vertex_checkpoint.py`, `test_vertex_cuda.py` | Persistencia, regresión, diferenciación y overfit |
| `requirements-dev.txt` | Dependencia pytest de desarrollo |
| Este documento | Cambios, decisiones, resultados y reproducción |

SVInOut, el formato geométrico, las propiedades del modelo, train.py y los
kernels SH originales no cambian. La interfaz conserva los cinco resultados
color,depth,normal,T,max_w y el diccionario del renderer con sus campos raw_*.

## Decisiones respecto de la especificación

- Método render_vertex y appearance_mode independiente de color_mode.
- Configuración YAML propia en lugar de ampliar el estado global YACS:
  defaults del archivo propio → protocolo base → overrides explícitos.
  En reanudación/inferencia prevalece la configuración resuelta del checkpoint.
- Kernels separados basados en los originales, para preservar su ruta SH.
  Implica duplicación que debe mantenerse sincronizada si cambia el rasterizador.
- Buffer RGB/gradientes empaquetado en lugar de un retorno adicional; conserva
  la aridad de autograd y las salidas públicas.
- Las pruebas directas del kernel y autograd sustituyen la ruta provisional de
  promedio de esquinas, que no forma parte de la implementación final.
- Se usan los valores fotométricos de materials_t, no los L1/SSIM ilustrativos.
- Los runs smoke/debug previos no son el experimento final. Un debug inicial
  muestreaba con reemplazo: se corrigió y se reinició desde cero el run definitivo.
- Un solapamiento accidental de dos ejecuciones reales agotó VRAM; se repitió
  con GPU libre, sin cambiar precisión ni algoritmo.

## Persistencia y compatibilidad

Checkpoints independientes con versión, modo, configuración resuelta, estado
neuronal, Adam, paso y estados aleatorios CPU/CUDA/generador. Las cámaras se
reconstruyen reproduciblemente desde semilla y paso. La carga valida P,V,
hash de topología/orden de grid_pts_key/coordenadas mundo y SHA-256 del archivo
geométrico. Se puede reubicar el mismo archivo mediante --geometry-checkpoint.
No se permite reutilizar H con otro orden de vértices sin remapeo explícito.

No hay caché de RGB entre renders. Inferencia usa eval y no_grad. El control
VertexRGB aprende [P,3] sin Fourier/MLP y comparte el mismo kernel y protocolo.
El benchmark mueve apariencia a CPU durante SH0/SH3 para excluir sus parámetros
inutilizados de la memoria asignada a esos baselines.

## Validación

**17 pruebas aprobadas**, incluidas las CUDA. Se verificaron Fourier,
normalización angular, inicialización y gradientes tras dos pasos, pesos
trilineales, acumulación compartida, regresión SH0, invariancia geométrica,
diferencias finitas, clamp, guardar/cargar/render, frustum vacío/parcial,
esquinas geométricas, índices inválidos, aislamiento SH y overfit de una/dos vistas.
El layout real de materials_t también pasó su validación. Se comparó además
una vista SH3 real entre la extensión del checkout anterior y la nueva:
error absoluto máximo RGB=0 y T=0. La reanudación real
funcionó; las corridas comparativas definitivas comenzaron desde cero.

Tolerancias: regresión y persistencia 1e-5; gradientes sintéticos float32
atol=1e-4, rtol=1e-2 lejos de umbrales. Advertencias de pytest/MiniCam existentes
no indican fallos numéricos. git diff --check sin errores.

## Resultados comparativos finales

Métricas sobre imágenes float32, mismo protocolo en los cuatro métodos; PNG
exportados para inspección. PSNR/SSIM/LPIPS promediados sobre 200 vistas test.

| Método | PSNR | SSIM | LPIPS | LPIPS corregido | FPS total |
|---|---:|---:|---:|---:|---:|
| SH0 | 23.65449 | 0.901724 | 0.077471 | 0.091297 | 99.76 |
| SH3 | 29.96786 | 0.954652 | 0.035469 | 0.043374 | 81.26 |
| Vertex-MLP Fourier-3 | 27.89134 | 0.935391 | 0.059812 | 0.072690 | 16.80 |
| RGB por vértice | 24.94174 | 0.921024 | 0.065948 | 0.077455 | 58.89 |

La variante neuronal mejora SH0 en 4.24 dB,
pero queda 2.08 dB por debajo de SH3 y es
más lenta. Frente al control RGB mejora 2.95 dB.
Estos resultados no demuestran una ventaja sobre SH3.

## Rendimiento y memoria

Eventos CUDA, 10 vistas de calentamiento y 50 mediciones de vistas test.
Total incluye apariencia, gather, raster y posprocesamiento, excluyendo I/O.
Raster incluye preprocesamiento geométrico, SH y empaquetado de entradas.
Fourier/MLP se miden además aisladamente: 7.30 ms
para direcciones/Fourier y 36.19 ms para
concatenación/MLP. Apariencia completa: 43.50 ms;
gather: 2.26 ms.

| Método | Total ms | Raster ms | Post ms | Pico inferencia, bytes |
|---|---:|---:|---:|---:|
| SH0 | 10.02 | 9.74 | 0.26 | 1,508,395,008 |
| SH3 | 12.31 | 12.04 | 0.25 | 1,508,395,008 |
| Vertex-MLP Fourier-3 | 59.52 | 13.49 | 0.25 | 4,117,459,968 |
| RGB por vértice | 16.98 | 14.32 | 0.26 | 2,175,752,704 |

Pico asignado entrenamiento: 5,531,939,328 bytes neuronal y 3,270,554,112 bytes
control. Los parámetros neuronales ocupan 138.846.764 bytes; los RGB de vértices
52.064.652 y el gather global 282.007.968. parameters_bytes del JSON se refiere
solo al módulo de apariencia; el pico GPU incluye el modelo base y temporales.

## Reproducción y artefactos

```bash
conda activate svraster
python -m pip install -r requirements-dev.txt
CUDA_HOME=/usr/local/cuda-12.5 MAX_JOBS=2 python -m pip install -e cuda/ --no-build-isolation
python -m pytest tests -q
python train_vertex_fourier3.py --output output/vertex_materials_final --steps 5000
python train_vertex_fourier3.py --output output/vertex_materials_control --mode vertex_rgb_trilinear --steps 5000
python render_vertex_fourier3.py --output output/vertex_materials_final/comparison --appearance-checkpoint output/vertex_materials_final/checkpoints/iter005000_appearance.pt
python render_vertex_fourier3.py --output output/vertex_materials_final/benchmark --appearance-checkpoint output/vertex_materials_final/checkpoints/iter005000_appearance.pt --benchmark
```

Reanudar: añadir --resume RUTA_APPEARANCE.pt --steps TOTAL al entrenamiento.
Para el control usar su checkpoint y directorio al renderizar/medir. Volver a
la ruta original: python render.py out/materials_t --skip_train.

Artefactos en output/vertex_materials_final y output/vertex_materials_control:
config.yaml, checkpoints/iter005000_appearance.pt, test/iter005000/metrics.json,
PNG renders/gt, memory.json y benchmark/benchmark.json. La comparación SH0/SH3
está en output/vertex_materials_final/comparison. Datos, binarios, imágenes y
checkpoints quedan fuera de Git.

## Limitaciones

CUDA float32, una muestra, geometría fija; sin AMP, RBF, poda/subdivisión ni
regularizadores CUDA. Se decodifican todos los vértices por vista. Las
activaciones y el gather global dominan memoria; Fourier/MLP añaden latencia.
La continuidad del residual está garantizada en interfaces conformes con
vértices compartidos, no en interfaces adaptativas ni en el color total si
SH0 cambia entre vóxeles. La calidad depende de la geometría congelada.

## Commits locales de integración

- `50610e7`: Add Fourier vertex appearance decoder and integration guide
- `8a43a40`: Add trilinear vertex CUDA forward and backward with regression tests
- `11abc56`: Add fixed geometry appearance training checkpoints and evaluation tools
- `31d0ac9`: Validate vertex buffers and checkpoint rendering and preserve camera epochs
- `e372eee`: Cover camera direction and empty frustum and document integration implications
- `0673d8f`: Exclude unused appearance parameters from baseline GPU memory measurements
- `f86857e`: Add development dependencies and two view appearance validation
- `3295343`: Record complete vertex appearance validation results
- `31d942f`: Enforce frozen base gradients in vertex rendering wrapper
- `364d3a9`: Document full materials experiment and baseline performance comparisons
- `916c795`: Validate global vertex layout before appearance allocation


El commit que consolida este informe se identifica en git log. No se realizó push.
