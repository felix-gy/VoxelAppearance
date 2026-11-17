# Vertex-MLP conjunto y visor: modificaciones, razones e implicaciones

## 1. Estado y alcance de la entrega

Integración implementada en el entorno Conda `svraster`. No se ejecutó una corrida completa sobre Lego u otra escena real: queda a cargo del usuario. Se compiló CUDA y se ejecutaron 25 pruebas, incluidas pruebas del entrenador real sobre una fixture sintética pequeña y del visor con servidor local sin navegador.

Este documento complementa las auditorías de la variante congelada. No atribuir al entrenamiento conjunto los resultados PSNR/FPS previamente publicados para apariencia sobre geometría congelada.

La integración modifica el entrenamiento existente mediante `--appearance_mode vertex_mlp`; no lo sustituye por otro entrenador con hiperparámetros distintos. El valor predeterminado sigue siendo `sh`.

## 2. Calendario exacto de apariencia

Con `cfg/synthetic_nerf.yaml`, `sh_degree_init=0` y máximo SH3:

| Iteración | Ruta original | Nueva ruta optativa |
|---|---|---|
| 1–999 | SH0 | SH0 original; todavía no existe el módulo neuronal |
| 1000 | Activa SH1 antes del render | Activa Vertex-MLP antes del render |
| 1001–1999 | SH1 | SH0 entrenable + residual neuronal |
| 2000–2999 | SH2 | SH0 entrenable + residual neuronal |
| Desde 3000 | SH3 | SH0 entrenable + residual neuronal |

La última capa MLP empieza a cero, por lo que el primer forward neuronal coincide con SH0. H y la primera capa pueden recibir gradiente cero en ese primer backward; se habilita su aprendizaje al actualizar la capa de salida.

La sustitución no congela densidad ni SH0. Si se configura un grado SH inicial mayor que cero, Vertex-MLP se activa desde la inicialización; si el máximo grado es cero, se conserva SH0 sin activación neuronal. No se reescala el evento fijo de 1000 al aplicar `sche_mult`, porque tampoco lo hace el código original.

## 3. Parámetros originales conservados

No se editaron `src/config.py` ni `cfg/synthetic_nerf.yaml`. Sus valores y los overrides del usuario siguen siendo la fuente de verdad.

| Componente | Predeterminado original conservado |
|---|---|
| Duración y semilla | 20000 pasos; 3721 |
| Bounding Synthetic | outside_level=0; fondo blanco |
| Inicialización | Nivel 6; densidad −10; SH0 RGB 0.5 |
| LR densidad / SH0 | 0.025 / 0.01 |
| LR angular | shs_lr=0.00025, ahora aplicado a H y MLP |
| Optimizador | SparseAdam, betas (0.1,0.99), eps=1e-15 |
| Scheduler | MultiStepLR: hito 19000, multiplicador 0.1 |
| Fotometría | lambda_photo y selección MSE/L1/Huber originales |
| SSIM | lambda_ssim=0.02 |
| Concentración RGB | lambda_R_concen=0.01 |
| Concentración T Synthetic | lambda_T_concen=0.1 |
| Distorsión | lambda_dist=0.1 desde 10000 |
| TV densidad | 1e-10, calendario original |
| Supersampling entrenamiento | ss=1 hasta incluir 1000; después augmentación original |
| Augmentación predeterminada | Uniforme entre 1 y ss_aug_max=1.5 |
| Supersampling modelo/evaluación | 1.5 |
| Adaptación | Desde 1000, cada 1000, condiciones originales de finalización |
| Poda | Hasta 18000; umbral 0.0001→0.05 |
| Subdivisión | Hasta 15000; proporción 0.05; máximo 10000000 |
| Preset Synthetic | Subdividir todos en primer evento; umbral de muestreo 0.5 |
| Cámaras, datos y pérdidas adicionales | Selección, lectores, opciones y calendarios originales |

Se mantiene la recreación del optimizador tras poda/subdivisión: los momentos se reinician como en el original. No se impone el límite de 2 millones propuesto en una versión anterior del plan. No se usan las tasas 0.01/0.001 ni el Adam de la variante congelada.

## 4. Inventario de código y ubicaciones

Líneas de referencia del estado implementado; localizar también por símbolo si el archivo cambia.

| Archivo | Ubicación principal | Cambio y razón |
|---|---|---|
| [train.py](../train.py) | training, líneas 82,112,128,169,212,444; CLI desde 593 | Selección optativa, grupos originales, transición angular, restauración y guardado conjunto |
| [joint.py](../src/vertex_appearance/joint.py) | JointVoxelModel, línea 15 | Modelo adaptativo que incorpora H/MLP sin reescribir el bucle original |
| [renderer.py del modelo](../src/sparse_voxel_gears/renderer.py) | render_vertex, línea 188 | Nuevo joint_training=False; habilita geometría/SH0 solo explícitamente |
| [renderer.py CUDA](../cuda/svraster_cuda/renderer.py) | validación del modo | Retira rechazo de regularizadores que ya contiene el kernel heredado |
| [vertex_backward.cu](../cuda/src/vertex_backward.cu) | grad_pack[8:11] | Aplica máscara de color al gradiente SH0 base |
| [viz.py](../viz.py) | constructor, línea 36; render_model, línea 275 | Inyección de modelo/cámaras y punto de extensión de render reutilizable |
| [viz_vertex_fourier3.py](../viz_vertex_fourier3.py) | load_viewer 19; VertexViewer 52; main 95 | Carga de formatos y visor independiente |
| [test_vertex_joint.py](../tests/test_vertex_joint.py) | Pruebas nuevas | Gradientes, activación, remapeo, checkpoint, entrenador real y reloj Adam |
| [test_vertex_viewer.py](../tests/test_vertex_viewer.py) | Prueba nueva | Carga conjunta y render de salidas con servidor Viser temporal |

Los kernels forward originales, constructor geométrico, SVAdaptive, SVInOut, lectores, cámaras, configuración y funciones de pérdidas no fueron reescritos. Se reutilizan por herencia o llamada directa.

## 5. Integración dentro de train.py

### Inicialización y configuración

`--appearance_mode` acepta `sh` y `vertex_mlp`. Solo el segundo instancia JointVoxelModel. Rechaza multimuestreo y `--save_quantized` para el formato conjunto float32.

El modo original conserva su clase, primer paso y carga de optimizador. El modo conjunto reanuda desde el paso guardado + 1, para no repetir una iteración.

Se conserva el mismo SparseAdam y sus tres grupos. Antes de la transición, el tercer grupo contiene los SH superiores originales. Al activar, se sustituye por los cinco tensores de VertexAppearance: H, pesos/bias de la primera capa y pesos/bias de salida.

### Reloj de Adam en la transición

`activate_for_optimizer` mantiene LR, betas, epsilon y contador de pasos del antiguo grupo angular. Los momentos de los parámetros nuevos empiezan en cero, porque sus formas y significado difieren de SH. Esto evita reiniciar arbitrariamente el reloj de bias correction del grupo después del warmup.

Al adaptar la topología se recrea el optimizador completo como antes, con el scheduler restaurado. No se conserva una política neuronal especial de momentos. Los pesos de la MLP sí se conservan.

### Orden del bucle preservado

Continúan en su sitio los resets de apariencia, augmentación de supersampling, condiciones de profundidad/normales, concentración RGB, TV, pérdidas, optimizer.step, estadísticas, poda/subdivisión, scheduler.step e informes.

La secuencia de cámaras sigue calculándose mediante compute_iter_idx. No se reemplaza por muestreo con reemplazo ni por otro generador. El módulo neuronal se crea dentro de torch.random.fork_rng para no consumir permanentemente el estado aleatorio del entrenamiento al activarse.

### Configuración y reanudación

El checkpoint guarda la configuración anterior al escalado `sche_mult`, leída de config.yaml cuando existe. Al reanudar, main la restaura y aplica el escalado original una sola vez. En pruebas directas del bucle sin ese archivo se usa cfg.dump.

La reanudación conjunta exige una iteración explícita y no admite overrides de configuración o --cfg_files: utiliza su snapshot. Los argumentos directos de checkpoint/logging siguen disponibles. La secuencia guardada debe cubrir la duración configurada; no se implementó ampliar la duración ni cambiar el scheduler al reanudar.

## 6. JointVoxelModel: geometría, apariencia y adaptación

### Activación y render

- `activate_vertex` crea VertexAppearance(P), mantiene active_sh_degree=0 y desactiva gradientes SH superiores.
- `angular_parameters` devuelve SHs durante warmup o parámetros neuronales después de activar.
- `render` enruta las llamadas de color predeterminado a render_vertex con joint_training=True. Las llamadas explícitas dontcare para estadísticas siguen el renderer geométrico original.
- El callback interno evita recursión del dispatch al llamar al render común.
- `reset_sh_from_cameras` conserva el reset geométrico/SH0 y pone a cero la última capa neuronal para recuperar residual nulo; no borra H ni la primera capa.

No se cambian Fourier, dirección cámara→vértice, arquitectura 26→64→3, interpolación trilineal ni punto medio original de densidad.

### Remapeo de H

`_adapt_vertex` delega geometría y SH0 en pruning/subdividing originales. Antes reúne H por las ocho esquinas viejas; elimina las filas podadas o interpola hijos con subdivide_by_interp. Después agrega a los vértices nuevos usando agg_voxel_into_grid_pts.

Las claves enteras grid_pts_key identifican los vértices supervivientes: sus valores H se copian exactamente sobre el resultado agregado. Los vértices nuevos conservan el promedio de contribuciones interpoladas. Se sustituye únicamente vertex_features por un Parameter de la nueva forma; los pesos del decoder no se reinicializan.

No se garantiza continuidad automática entre resoluciones ni conservación exacta de RGB tras interpolar H, debido a la no linealidad del decoder. Esa diferencia es propia de la representación nueva, no de un cambio del calendario geométrico.

## 7. Gradientes y regularizadores CUDA

`render_vertex(..., joint_training=False)` conserva el detach de geos, SH0 y prioridad de la variante congelada. Con True deja pasar esos gradientes. No debe emplearse freeze_vox_geo persistente durante entrenamiento conjunto.

Se conserva el transporte global [V,27]: tres canales SH0 y ocho residuales RGB. El backward mantiene [V,36], con densidad, base, prioridad y esquinas. No se compactan solo algunos buffers ni se altera su orden.

El gradiente base ahora usa `mask[channel] * dL_drgb[channel]`, donde mask corresponde a preclamp>0. Después SH_eval.backward aplica la activación SH0 original. Antes, el gradiente base no necesitaba esta corrección porque siempre se descartaba.

Se elimina el rechazo Python de lambda_R_concen, lambda_dist y lambda_ascending. El kernel especializado ya heredaba sus operaciones del backward original; ahora se utilizan con el color reconstruido por intersección. No se modificaron sus pesos ni calendarios.

Concentración RGB y distorsión conservan el mecanismo de gradientes auxiliares del rasterizador original: no se añadió un término escalar nuevo a loss.item. Las diferencias finitas de fotometría se prueban sin esos gradientes auxiliares; los regularizadores se contrastan con el backward SH0 original cuando el residual es cero. No se afirma un gradcheck exhaustivo de todos sus parámetros.

## 8. Checkpoint conjunto y memoria

JointVoxelModel.save conserva los campos geométricos existentes y añade `vertex_joint` con schema_version=1, active, module, layout, config, training y priority.

- module contiene H y decoder en CPU, o None antes de activar.
- layout usa el hash existente de topología, orden de vértices y transformación mundo.
- training contiene optimizador, scheduler, iteración, orden de cámaras, opciones de render sin gt_color y RNG Python/NumPy/PyTorch CPU/CUDA.
- priority guarda la prioridad acumulada si hay gradiente.

Se guarda primero en un archivo temporal del mismo directorio y se publica mediante os.replace. Los nombres siguen siendo los originales `iterNNNNNN_model.pt`, evitando un segundo formato de nombres para el mismo bucle.

Se liberan las referencias temporales al estado de entrenamiento después de guardar/restaurar para no retener momentos antiguos en GPU tras recrear el optimizador. Las estructuras derivadas se reconstruyen al cargar; no se serializa una densidad congelada potencialmente obsoleta.

El checkpoint conjunto es autónomo y diferente del schema v2 congelado. El visor detecta ambos. Solo se cargan archivos de confianza porque torch.load utiliza weights_only=False.

La reanudación conserva estados y primer paso dentro de tolerancia, pero no promete igualdad bit a bit tras cientos de actualizaciones: los atomics CUDA ya introducen no determinismo en el renderer original.

## 9. Visor separado y cambios mínimos de viz.py

El constructor SVRasterViewer acepta opcionalmente modelo y camera_pack; el main original pasa explícitamente model_path/iteration. Se elimina la dependencia de un args global dentro del constructor y se introduce render_model como punto de extensión. El error de render ya no continúa con render_pkg sin inicializar.

`viz_vertex_fourier3.py`:

- Detecta checkpoint conjunto o apariencia congelada v2. En v2 permite --geometry-checkpoint y valida identidad del modelo base.
- Carga cámaras sin imágenes, congela geometría para inferencia y usa eval/no_grad.
- Ofrece Vertex-MLP o Vertex-RGB, SH0 y SH3 únicamente para checkpoints congelados con coeficientes válidos.
- Un checkpoint conjunto anterior a activación muestra SH0; no presenta SH3 como entrenado.
- Reutiliza navegación, frustums train/test, resolución, supersampling, descarga y salidas RGB/alfa/profundidad/normales.
- Solicita los buffers que requiere cada salida, aunque el otro selector de render estuviera en RGB solamente.
- Un lock serializa render y descarga; ante errores conserva la última imagen y muestra el diagnóstico.

El visor mantiene la dependencia Viser existente. No ejecuta optimización ni crea automáticamente checkpoints de entrenamiento.

## 10. Pruebas y alcance real de validación

Resultado final: **25 passed** en Conda svraster, con CUDA compilado. Se conservaron las 17 pruebas previas y se añadieron ocho.

- Activación idempotente, RNG conservado y salida inicial igual a SH0.
- Gradientes de densidad, SH0 y prioridad presentes.
- Concentración RGB/distorsión: comparación de gradientes frente a SH0 con residual cero.
- Clamp de SH0 y diferencias finitas fotométricas de SH0/densidad.
- Poda/subdivisión: H superviviente exacto, forma válida y decoder conservado.
- Save/load/render y restauración de RNG/optimizador/scheduler.
- Bucle real sobre fixture 16×16: llega a 1501, guarda 999/1000/1001, activa en 1000 y poda/subdivide; comprueba LR/betas/eps y augmentación SS.
- Reanuda en 1001, compara el primer paso y termina con valores finitos.
- Compara el primer paso con el entrenador SH original y verifica topología/SHs y estado no activo hasta 999. No exige pesos idénticos tras 999 atomics acumulados.
- Prueba el reloj Adam angular heredado de 999.
- Carga un checkpoint conjunto y renderiza RGB, alfa, profundidad, normales y SH0 con un servidor Viser temporal, sin navegador.

La fixture reduce únicamente tamaño de imagen, rejilla y duración dentro del test; no cambia defaults de producción. No se entrenaron datasets reales ni se midió calidad/FPS de esta nueva modalidad. La interacción humana en navegador y los experimentos completos quedan pendientes de ejecución por el usuario.

## 11. Comandos para el usuario

Ejecutar desde la raíz del repositorio. Los datos de Lego deben estar en data/nerf_synthetic/lego con transforms_train.json, transforms_test.json e imágenes correspondientes.

### Preparación del entorno y extensión

```bash
conda activate svraster
python -m pip install -r requirements-dev.txt
CUDA_HOME=/usr/local/cuda-12.5 MAX_JOBS=2 python -m pip install -e cuda/ --no-build-isolation
python -m pytest tests -q
```

El comando de compilación usa el toolkit disponible en este equipo; la extensión ya fue compilada y verificada aquí.

### Entrenar Lego desde cero con parámetros originales

```bash
python train.py \
  --appearance_mode vertex_mlp \
  --cfg_files cfg/synthetic_nerf.yaml \
  --source_path data/nerf_synthetic/lego \
  --model_path output/vertex_joint/lego \
  --checkpoint_iterations 1000 5000 10000 15000
```

Los checkpoints intermedios son una opción original de guardado y no cambian hiperparámetros de aprendizaje. El final de 20000 se guarda automáticamente. No hace falta --save_optimizer para reanudar el modo conjunto: el estado completo ya está dentro del checkpoint.

### Reanudar desde una iteración guardada

```bash
python train.py \
  --appearance_mode vertex_mlp \
  --model_path output/vertex_joint/lego \
  --load_iteration 10000
```

No añadir --cfg_files, --source_path ni overrides de entrenamiento al reanudar. Se conserva configuración, duración total y calendario originales. Indicar una iteración positiva existente; no usar -1 como búsqueda automática en esta modalidad.

### Visualizar el resultado conjunto

```bash
python viz_vertex_fourier3.py \
  --checkpoint output/vertex_joint/lego/checkpoints/iter020000_model.pt \
  --port 7007
```

Abrir http://localhost:7007. También se puede visualizar un checkpoint intermedio, incluido el warmup anterior a activación.

### Visualizar el resultado congelado anterior

```bash
python viz_vertex_fourier3.py \
  --checkpoint output/vertex_materials_final/checkpoints/iter005000_appearance.pt \
  --port 7007
```

### Baseline SH sin modificar

```bash
python train.py \
  --cfg_files cfg/synthetic_nerf.yaml \
  --source_path data/nerf_synthetic/lego \
  --model_path output/sh_baseline/lego
```

Cambiar lego por chair, drums, ficus, hotdog, materials, mic o ship y usar directorios de salida distintos. No utilizar el batch synthetic_nerf_run.sh para este flujo: elimina checkpoints y no selecciona Vertex-MLP.

## 12. Límites, incompatibilidades y diferencias inevitables

- El cambio de representación altera gradientes y evolución geométrica después de activar, aunque los hiperparámetros sean idénticos.
- La MLP y H usan LR angular original; no se realizó búsqueda de hiperparámetros ni se afirma que sea su óptimo.
- Se conserva máximo de 10 millones, sin reducirlo automáticamente para la RTX 3060. H, activaciones y gather pueden aumentar el consumo hasta agotar VRAM según la escena.
- El formato conjunto no admite cuantización; el modo original mantiene su opción habitual.
- La reanudación exige mismo dataset y orden de cámaras; no guarda un hash de imágenes ni admite ampliar duración en este flujo.
- render.py y render_vertex_fourier3.py siguen orientados a sus formatos anteriores. Para checkpoints conjuntos usar el visor nuevo; un lector geométrico antiguo puede ignorar vertex_joint y mostrar solo SH0. Los informes del propio entrenamiento sí usan el modelo conjunto.
- La continuidad en interfaces adaptativas no está garantizada por interpolar características.
- La corrección se comprobó con pruebas pequeñas; no se ejecutó la corrida de 20000 ni se sustituyeron sus resultados por los de la variante congelada.

## 13. Trazabilidad

- eb32e6c: integración adaptativa, máscara SH0, regularizadores, checkpoints y pruebas.
- 49610ed: visor separado y punto de extensión del visor original.
- 1bed1f4: reloj Adam angular, configuración de reanudación y liberación de referencias temporales.

Este documento se añade en un commit documental posterior. Las auditorías previas describen la variante congelada en sus revisiones históricas; este archivo documenta las diferencias actuales.
