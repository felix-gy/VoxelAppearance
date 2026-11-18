# SVRaster con apariencia RBF4

La rama `RBF-App` parte de `main` en `74e8907`. Añade el modo optativo `rbf4_iq_sine` al rasterizador y entrenador originales. El modo predeterminado sigue siendo SH. El archivo `Plan_SVRaster_RBF_4_Canales.md` se conservó intacto.

## Apariencia por intersección

Una intersección rayo–vóxel aporta un color calculado en el punto medio del intervalo `[a,b]`. Con origen mundial `o`, rayo normalizado `d`, esquina mínima `v_min` y longitud de lado `l`, la coordenada local es `t = (o + ((a+b)/2)d − v_min)/l`. Los ocho centros `q_i` son las esquinas binarias del cubo, en orden XYZ con Z como bit menos significativo.

```text
eta_ia = clamp(log_scale_ia, −2, 2)
phi_i = 1 / (1 + sum_a ((t_a − q_ia)^2 exp(−2 eta_ia)))
S = sum_i phi_i
alpha_i = phi_i / S
f_c = sum_i feature_ic sin(alpha_i m_c + phase_c)
m = [1, 2, 4, 8] radianes
```

La dirección sale de la cámara hacia la muestra en coordenadas mundo. Para cada frecuencia `k` de `[1, 2, 4]`, concatenamos `sin(k*d_x), sin(k*d_y), sin(k*d_z), cos(k*d_x), cos(k*d_y), cos(k*d_z)`. Los cuatro canales espaciales y los dieciocho direccionales alimentan la MLP `22 → 64 → 3`. La activación intermedia es ReLU. La salida por intersección es `max(0, c_SH0 + MLP(f, enc(d)))`, con `c_SH0 = max(0, 0.5 + 0.28209479177387814 * sh0)` calculado por la evaluación SH original. La última capa a cero reproduce el color SH0 al activarse.

La densidad usa la cuadratura y los pesos trilineales originales. Para dos o tres puntos de cuadratura se integra la densidad en todos los puntos previstos y se evalúa la apariencia una vez en el punto medio de la intersección. El mismo color resultante se usa en composición volumétrica y en el cálculo de gradientes de alfa, densidad y prioridad.

### Parámetros y layout

| Dato | Forma y almacenamiento | Inicialización |
| --- | --- | --- |
| `rbf_features` | `[P,4]` float32 contiguo | Normal con desviación estándar 0,01 |
| `rbf_log_scales` | `[P,3]` float32 contiguo | Cero |
| `rbf_phase` | `[4]` float32 contiguo, compartido | Cero |
| MLP | `W1[64,22], b1[64], W2[3,64], b2[3]` float32 | Primera capa con inicialización de PyTorch, segunda capa con pesos y sesgos a cero |
| Frecuencias espaciales | Buffer fijo `[4]` float32 | `[1,2,4,8]` |
| Vértices | `vox_key[N,8]` int64 contiguo | Índices globales en el orden original, asociados a `grid_pts_key[P,3]` |

La MLP aporta 1.667 parámetros y el modo añade `7P + 1671` entrenables en total. Autograd empaqueta la MLP por filas en el orden de la tabla, con offsets 0, 1408, 1472 y 1664. Cada hilo usa datos temporales propios para un rayo. Las especializaciones CUDA comparten el recorrido y la composición del rasterizador. En RBF4 se precalcula la dirección por rayo y se evalúa el codificador para las intersecciones que superan el umbral de alfa. No se reserva un tensor global de intersecciones ni se añaden buffers de características a la memoria compartida del bloque.

### Backward

Se recalculan bases, codificación y activaciones en el rasterizado inverso. La derivada del clamp final vale uno si el RGB previo al clamp es positivo y cero en caso contrario. El gradiente hacia SH0 continúa por el evaluador SH original, que aplica el factor SH0 y su regla existente para el clamp interno. ReLU usa derivada cero en el origen.

```text
u_ic = alpha_i m_c + phase_c
g_feature_ic = g_f_c sin(u_ic)
g_phase_c = sum_i g_f_c feature_ic cos(u_ic)
g_alpha_i = sum_c g_f_c feature_ic cos(u_ic) m_c
g_phi_i = (g_alpha_i − sum_j alpha_j g_alpha_j) / S
g_eta_ia = 2 g_phi_i phi_i^2 (t_a − q_ia)^2 exp(−2 eta_ia)
```

La última derivada se aplica si la escala almacenada está dentro de `[-2,2]`, extremos incluidos. El entrenador proyecta las escalas después de cada paso. Las características y fases se optimizan sin cotas adicionales. Se acumulan gradientes por vértice con atómicos. Los hilos activos de cada warp reducen las contribuciones a un coeficiente común de la MLP o de la fase antes de la escritura global. Las sumas atómicas pueden introducir variaciones en el orden de redondeo. El autograd mantiene el contrato original para rayos, octree y cámaras, y devuelve gradientes de densidad, SH0, prioridad y del campo neuronal.

## Entrenamiento y adaptación

La opción `--rbf4-iq-sine` fuerza SH0 durante las iteraciones 1–999. La inicialización del campo restaura el RNG de PyTorch de CPU y del dispositivo del campo, por lo que su creación no adelanta las secuencias del warmup. Antes del render de la iteración 1000, la MLP y los parámetros RBF ocupan el tercer grupo de SparseAdam. El contador de pasos de SH angular pasa a los nuevos parámetros, que empiezan con momentos cero. La inicialización de la capa de salida bloquea temporalmente el gradiente hacia la primera capa y el codificador. Después del primer paso de la capa de salida, ambos pueden recibir gradientes. Las promociones posteriores de grado SH se omiten en este modo.

Se conservan `geo_lr`, `sh0_lr`, `shs_lr`, `optim_beta1`, `optim_beta2`, `optim_eps`, `lr_decay_ckpt` y `lr_decay_mult` de la configuración elegida. Los valores predeterminados son 0,025, 0,010, 0,00025, 0,1, 0,99, 1e−15, `[19000]` y 0,1, respectivamente. También se conservan la selección de cámaras, el supersampling, los fondos, exposición, geometría, calendario geométrico, *reset* de SH0 y las pérdidas configuradas. Entre ellas están las pérdidas fotométricas y de SSIM, profundidad, máscaras, normales, concentración del color y de la transmitancia, orden, distorsión y variación total de la densidad. Los coeficientes SH superiores del baseline permanecen en el modelo para el warmup y se guardan para conservar su layout. Después de la activación, su grupo se dedica al campo neuronal.

Poda y subdivisión ejecutan las decisiones originales antes del remapeo de apariencia. Las coincidencias de claves enteras copian exactamente características y log-escalas. Un vértice nuevo recibe la media de las contribuciones interpoladas de sus padres, y una log-escala de cero. Fases y MLP mantienen sus tensores. El optimizador se vuelve a crear y el scheduler se restaura en el punto previsto por el entrenador original. Este remapeo conserva parámetros supervivientes. La apariencia puede presentar discontinuidades en caras compartidas, donde todas las bases son positivas. Un cambio de layout puede alterar el RGB aun con características interpoladas.

## Checkpoints y evaluación

El formato `rbf4_iq_sine`, versión 1, registra arquitectura, frecuencias, límites, convención y orden direccional, activación y SHA-256 del layout. El fingerprint cubre caminos, niveles, coordenadas enteras y enlaces de vértices. La geometría, parámetros, ajustes de render y prioridad se guardan juntos. Durante el entrenamiento también se guardan iteración, optimizador, scheduler, secuencia de cámaras, exposición y RNG de Python, NumPy, Torch CPU y CUDA. El archivo temporal se crea en el directorio de destino y se publica con `os.replace`. La búsqueda de checkpoints ignora los temporales.

La reanudación restaura el estado y empieza en la iteración siguiente. La evaluación y el visor usan el rasterizador de entrenamiento y descartan el estado de optimización tras la carga. Los lectores que esperan el esquema SH fallan explícitamente en esta rama o por ausencia de sus claves de modelo en versiones antiguas. El lector RBF4 exige su esquema, versión y layout. Los checkpoints RBF4 usan precisión completa. El formato no incorpora conversiones desde latentes de ocho canales ni desde RGB por vértice.

## Inventario

| Archivo y símbolos afectados | Clasificación | Comportamiento, razón e implicación |
| --- | --- | --- |
| `cuda/src/rbf4_iq_sine.cuh` — `Field`, `Gradients`, `Sample`, `make_field`, `offset_squared`, `inverse_scale_squared`, `direction_encoding`, `evaluate`, `evaluate_color`, `color`, `warp_atomic`, `backward` y constantes de layout | Añadido | Define y valida el contrato de tensores y evalúa y deriva el nuevo campo. |
| `cuda/src/forward.cu`, `cuda/src/forward.h` — `renderCUDA`, `FwRendFunc`, `render`, `rasterize_voxels_procedure`, `rasterize_voxels` | Reemplazado | El lanzador especializa el mismo recorrido para color SH constante o color RBF4 en el punto medio. La composición, ordenamiento, alfa, profundidad, normal, fondo y seguimiento de pesos se conservan. |
| `cuda/src/backward.cu`, `cuda/src/backward.h` — `renderCUDA`, `BwRendFunc`, `render`, `rasterize_voxels_backward` | Reemplazado | El recorrido inverso recompone el color por intersección, ejecuta la cadena neuronal y suma sus gradientes. Se conservan ramas de densidad, regularización, SH0 y prioridad. |
| `cuda/svraster_cuda/renderer.py` — `rasterize_voxels`, `_RasterizeVoxels.forward`, `_RasterizeVoxels.backward` | Reemplazado | El puente añade tensores RBF opcionales y devuelve sus gradientes en el mismo orden. La API Python de render y la lista de salidas por imagen se conservan. Las firmas privadas de la extensión cambian y requieren recompilación. |
| `src/sparse_voxel_gears/rbf4_iq_sine.py` — `RBF4IQSine`, `raster_tensors`, `project_scales`, `remap`, `activate_rbf4` y constantes `MODE`, `ACTIVATION_ITERATION`, `ARCHITECTURE`, `FREQUENCIES`, `SCALE_LIMITS` | Añadido | Registra parámetros, empaqueta tensores, conserva claves y aplica transición y proyección. Reutiliza la interpolación y reducción de la adaptación original. |
| `src/sparse_voxel_model.py` — `SparseVoxelModel.__init__`, `init_rbf_field`, `angular_parameters` | Reemplazado y añadido | El constructor incorpora modo y estado de activación. Los métodos inicializan el campo y seleccionan parámetros del grupo angular. |
| `src/sparse_voxel_gears/renderer.py` — `SVRenderer.vox_fn` | Reemplazado | Selecciona base SH0 y presenta el campo al rasterizador cuando está activo. La ruta de estadísticas y los modos de diagnóstico conservan el despacho original. |
| `src/sparse_voxel_gears/adaptive.py` — `SVAdaptive.pruning`, `SVAdaptive.subdividing` | Reemplazado | Incorpora el remapeo al final de la operación geométrica. Los criterios, orden y agregación original de densidad se conservan. |
| `src/sparse_voxel_gears/io.py` — `RBF_FORMAT`, `layout_fingerprint`, `atomic_save`, `SVInOut.save`, `load`, `save_iteration`, `load_iteration` | Añadido y reemplazado | Extiende persistencia con un esquema versionado, guardado atómico, validación y restauración optativa del estado de entrenamiento. El esquema SH original y sus utilidades de cuantización se conservan. |
| `train.py` — `training`, `create_trainer` interno y CLI | Reemplazado | Selecciona RBF4, controla activación, proyección de escalas y checkpoint conjunto. Reanuda con la configuración de la ejecución cuando no se indican otros archivos. Conserva la ejecución SH predeterminada. |
| `render.py` — CLI y creación de `SparseVoxelModel` | Reemplazado | El flag RBF4 habilita la carga explícita de esta variante para render y evaluación. |
| `viz.py`, `viz_rbf4.py` — `SVRasterViewer.__init__`, `SVRasterViewer.render_viser_camera`, `viz.main` y entrada RBF4 | Reemplazado y añadido | Reutiliza la interfaz de cámara con un lector RBF4 explícito. El control de grado SH aparece en el visor SH. |
| `tests/test_rbf4_iq_sine.py`, `docs/RBF-App.md` | Añadido | Proporcionan cuatro comprobaciones pequeñas y el presente informe. |
| `src/config.py`, `cfg/`, `cuda/src/sh_compute.cu`, `cuda/src/preprocess.cu`, `cuda/src/geo_params_gather.cu`, `cuda/svraster_cuda/sparse_adam.py`, `eval.py` | Conservado | Siguen definiendo el calendario, SH0, visibilidad, densidad, SparseAdam y las métricas sobre imágenes escritas por el renderer. |
| Archivos y funciones públicos del baseline | Eliminado | Ninguno. |

## Verificación y alcance

Compilación local con PyTorch 2.5.0+cu124, CUDA Toolkit 12.5 y RTX 3060 (SM 8.6). El compilador señaló la diferencia menor entre versiones CUDA y el retorno ausente en la función trilineal original. PyTorch emitió su aviso beta de `index_reduce_` durante la fixture de adaptación.

Las cuatro comprobaciones automatizadas pasaron.

1. Referencia en caras, esquinas y centro. Bases positivas, suma normalizada, 22 canales, residual inicial exactamente cero y coincidencia del render inicial en color, profundidad, normales, transmitancia y pesos máximos con tres puntos de densidad.
2. Backward CUDA cotejado con autograd de PyTorch y diferencias finitas sobre una característica, una escala y una fase, con `atol=1e-4` y `rtol=1e-2`. La prueba también verificó gradientes finitos de MLP, densidad y SH0.
3. Overfit sintético con una vista de 16×16 y un vóxel, geometría y SH0 congelados, durante 80 pasos. MSE de 0,01830 a 9,9794×10⁻⁷. La imagen recargada coincidió con `atol=1e-5`, y se restauraron el scheduler y los momentos angulares. PSNR de **59,86 dB**, **0,2179 ms/render** y **4.588,5 FPS** en 100 renders con eventos CUDA, y **16,362 MiB** de pico de memoria CUDA asignada por PyTorch durante la prueba completa.
4. Activación en 1000, aparición de gradientes de codificador y primera capa tras el primer paso de la salida, proyección de escalas, subdivisión de dos a nueve vóxeles, interpolación y conservación por clave, poda de un vóxel y un backward posterior con regularizadores de concentración de color, orden y distorsión activos.

La comprobación de extremo a extremo utilizó otra fixture de 16×16 con una imagen de entrenamiento, una de evaluación y ocho vóxeles. El entrenador guardó checkpoints a ambos lados de la transición. Reanudar desde 999 arrancó en 1000 y llegó a 1002 con coincidencia de geometría y campo a `atol=1e-6`, y coincidencia del scheduler y del RNG CUDA. El reloj angular alcanzó 1000 tras el primer paso RBF4. `render.py` escribió una imagen de evaluación y `viz_rbf4.py` cargó el checkpoint e inició el servidor Viser. La interacción del visor en un navegador queda fuera de esta comprobación.

El costo neuronal se repite por intersección aceptada. Los gradientes compartidos están sujetos a contención y el backward vuelve a evaluar el campo. Las mediciones corresponden al objetivo sintético descrito. Calidad, tiempo y memoria en escenas reales quedan sin medir.

## Comandos

Desde la raíz del repositorio, con PyTorch y sus dependencias instalados en el entorno de SVRaster y con un toolkit CUDA compatible:

```bash
(cd cuda && python setup.py build_ext --inplace)
export PYTHONPATH="$PWD/cuda${PYTHONPATH:+:$PYTHONPATH}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s tests/test_rbf4_iq_sine.py

python train.py --rbf4-iq-sine --model_path output/rbf-app \
  --cfg_files cfg/synthetic_nerf.yaml --source_path /ruta/a/escena \
  --checkpoint_iterations 999 1000

python train.py --rbf4-iq-sine --model_path output/rbf-app --load_iteration 1000

python render.py output/rbf-app --rbf4-iq-sine --iteration -1
python eval.py output/rbf-app
python viz_rbf4.py output/rbf-app --iteration -1 --port 7007
```
