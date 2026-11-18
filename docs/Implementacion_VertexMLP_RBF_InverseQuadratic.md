# Vertex-MLP + RBF inverse quadratic: cambios, razones e implicaciones

## 1. Rama, alcance y estado

Rama: `feature/vertex-mlp-rbf-iq`, creada desde `GaussianApp` en `f43385b`.
Especificación de referencia: `Plan_SVRaster_VertexMLP_RBF_InverseQuadratic.md` de la raíz, conservada sin modificar.

Se implementaron las tres etapas: aprender solo escalas, aprender apariencia con geometría fija y entrenamiento conjunto adaptativo desde cero. La MLP sigue siendo **por vértice**, no por vóxel: `26→64→3`. La propuesta anterior de concatenar ocho esquinas dentro de una MLP no forma parte de esta rama.

La ruta RBF tiene kernels y autograd propios. Reutiliza infraestructura de modelos, datos y entrenamiento de la rama base; independencia no significa duplicar esos componentes. No se cambió la fórmula de densidad ni el algoritmo geométrico original.

Validación ejecutada: **solo cuatro pruebas nuevas RBF**, 4 passed en 2.12 s. No se ejecutó la batería trilineal anterior, ni comparaciones de entrenamiento o rendimiento con ella. No se entrenaron escenas completas ni se afirma una mejora de calidad o solución del OOM.

## 2. Qué permanece igual

- H[P,8] en vértices únicos compartidos y MLP compartida con 1923 parámetros.
- Fourier-3 de dirección cámara→vértice, sin concatenar dirección cruda.
- Color base SH0 por vóxel y RGB residual por vértice/vista.
- Una muestra en el mismo punto medio del segmento rayo-vóxel.
- Interpolación de densidad, ray–AABB, ordenamiento, alfa, T, profundidad, normales y fondos.
- Los campos de salida del renderer y clamp final [0,1].
- Para entrenamiento conjunto: hiperparámetros, regularizadores, LR, scheduler, warmup SH0, activación en 1000 y calendario de adaptación originales.
- Los archivos src/config.py y cfg/synthetic_nerf.yaml no fueron editados.

En la nueva ruta solo se sustituyen los pesos del residual RGB. Cambiar el color sí modifica el gradiente de la pérdida hacia densidad durante entrenamiento conjunto; conservar el algoritmo geométrico no significa obtener la misma trayectoria de entrenamiento.

## 3. Parámetros y fórmulas

Se añade `rbf_log_scales[P,3]`, inicializado a cero. Cada vértice conserva centro fijo y tres escalas diagonales; no existen rotaciones, centros móviles, covarianza completa ni RBF de densidad.

Para la posición local t y esquina q_i, con orden 000,001,010,011,100,101,110,111:

```text
eta = clamp(rbf_log_scales, -2, 2)
D_i  = sum_a (t_a - q_ia)^2 * exp(-2*eta_ia)
D0_i = sum_a (t_a - q_ia)^2
rho_i = (1 + D0_i) / (1 + D_i)
w_i = N_i*rho_i / sum_j(N_j*rho_j)
residual = sum_i(w_i*rgb_delta_i)
color = max(SH0 + residual, 0)
```

N_i son los pesos originales y permanecen intactos para densidad. Con eta=0, rho_i=1 y w_i recupera analíticamente N_i, salvo redondeo. No se añadió un atajo de ejecución que elimine el gradiente de escalas en cero.

Las escalas se acotan en forward CUDA y se proyectan después de optimizer.step. Backward utiliza gradiente uno dentro de [-2,2], incluidos sus extremos, y cero fuera. La proyección permite volver desde un extremo cuando cambia la dirección de actualización. El módulo rechaza log-escalas no finitas antes de proyectar.

En el dominio válido t∈[0,1]^3 y con estas cotas, el denominador es positivo y las exponenciales están acotadas. No se añade epsilon a la normalización. Las fórmulas no prometen propiedades de interpolación para posiciones fuera del vóxel.

Aunque las escalas se comparten por vértice, las distancias son locales al vóxel consultado. No son una matriz física global independiente del tamaño del vóxel. No se garantiza continuidad entre niveles diferentes del octree.

## 4. Inventario completo de código

| Ubicación | Cambio | Razón e implicación |
|---|---|---|
| cuda/src/rbf_weights.h | Nuevo helper de pesos y factores diferenciales | Misma matemática en forward/backward; N de densidad es solo lectura |
| cuda/src/rbf_forward.cu, .h | Nuevas funciones FORWARD_RBF | Evalúa RGB con pesos RBF; conserva recorrido y composición heredados |
| cuda/src/rbf_backward.cu, .h | Nuevas funciones BACKWARD_RBF | Gradiente de normalización y acumulación de escalas compartidas |
| cuda/binding.cpp | Exporta rasterize_rbf y rasterize_rbf_backward | Binario anterior no contiene la API nueva |
| cuda/setup.py | Compila las dos unidades RBF | Requiere recompilación; conserva exportaciones anteriores |
| cuda/svraster_cuda/rbf_renderer.py | Nueva función autograd RasterizeRBF | Guarda eta/claves y devuelve gradiente adicional sin cambiar salidas de imagen |
| cuda/svraster_cuda/renderer.py | Modo vertex_rbf_iq, validación y dispatch | Enruta únicamente el modo nuevo a RBF |
| src/sparse_voxel_gears/renderer.py | Selección de interpolación y entrega de eta/claves | Evita gather global extra de escalas; conserva selección explícita de compatibilidad |
| src/vertex_appearance/rbf.py | RBFAppearance, RBFJointModel y reference_weights | Parámetro nuevo, proyección, activación y remapeo |
| src/vertex_appearance/joint.py | Discriminador kernel, especificación y scope en checkpoints | Impide cargar estados de otra interpolación como si fueran compatibles |
| train_rbf.py | Entrenador nuevo de geometría fija | Separa scales_only y appearance; importa checkpoints previos |
| train.py | Opción vertex_rbf_iq y proyección de escalas | Etapa conjunta sin otro bucle ni nuevos hiperparámetros del baseline |
| viz_vertex_fourier3.py | Detecta RBF y etiqueta su apariencia | Visor compatible con el nuevo formato y selección manual de interpolación |
| tests/test_rbf_iq.py | Cuatro comprobaciones dedicadas | No importa ni ejecuta tests de la variante trilineal |

Los nuevos kernels derivan de la implementación de rasterización existente y conservan sus avisos de licencia. Duplican lógica común para aislar la ruta experimental; futuras correcciones del rasterizador deberán revisarse también en esta copia.

## 5. Contratos Python/CUDA y memoria

Los buffers de vóxel siguen usando índices **globales**, aunque el callback reciba el subconjunto visible.

| Entrada | Forma | Tipo |
|---|---|---|
| RGB base y ocho residuales | [V,27] | CUDA float32 contiguo |
| Geometría por esquinas | [V,8] | CUDA float32 contiguo |
| Log-escalas | [P,3] | CUDA float32 contiguo |
| vox_key | [V,8] | CUDA int64 contiguo |

Python comprueba formas y límites de claves. C++ valida formas, tipos, contigüidad, dispositivos y una muestra. Los bindings directos presuponen que los índices son válidos; el uso soportado pasa por el wrapper.

El kernel reúne las escalas de cada esquina directamente con vox_key. No se materializan eta[V,8,3] ni grad_eta[V,8,3]. El backward crea grad_eta[P,3] a cero y acumula con atomicAdd, sumando contribuciones de rayos, vóxeles y vértices compartidos.

Se conserva el acumulador heredado [V,36] para densidad, RGB base, prioridad y RGB por esquina. El backward C++ devuelve cuatro tensores: grad_geos, grad_rgbs, prioridad y grad_eta. El forward mantiene sus ocho retornos internos; autograd conserva color, depth, normal, T y max_w.

Se añaden 3P parámetros: el total pasa a 11P+1923. Log-escalas ocupan 12P bytes; parámetro, gradiente y dos momentos añaden aproximadamente 48P bytes cuando se optimizan con Adam. Evitar el gather de escalas no elimina activaciones MLP, gather RGB, buffers geométricos ni presión de registros del kernel. No se promete evitar el OOM del entrenamiento anterior.

## 6. Backward y compatibilidad geométrica

Para residual r=Σw_i*a_i y g igual al gradiente RGB después de máscara y composición:

```text
dL/da_i = w_i * g
dL/deta_ia = w_i * dot(g, a_i-r)
             * 2*(t_a-q_ia)^2*exp(-2*eta_ia)/(1+D_i)
             * derivative_of_clamp
```

El término a_i-r incluye la derivada del denominador. No se multiplican otra vez alfa y transmitancia. Los factores diferenciales se calculan en rbf_weights.h; la máscara del color también se aplica a SH0.

Los gradientes RGB vuelven al gather PyTorch y después a H/MLP. Las escalas reciben directamente el gradiente CUDA [P,3]. Los gradientes de densidad se conservan con el color reconstruido de esta ruta. No se implementan gradientes respecto de posiciones de centros, transformaciones de cámara o estructura discreta del octree.

Con geometría fija, cambiar eta no modifica T, profundidad o normales. Con geometría entrenable, la nueva apariencia puede alterar sus actualizaciones a través de la pérdida fotométrica.

## 7. Etapas y políticas de optimización

### Etapa 1: scales_only

Importar un checkpoint con H/MLP aprendidos. Congelar densidad, SH0, H, MLP y prioridad; aprender solo log-escalas. La apariencia no debe ser degenerada: ocho residuales iguales producen gradiente de escalas cero legítimamente.

### Etapa 2: appearance

Congelar densidad, SH0 y prioridad; aprender H, MLP y escalas. Se puede iniciar desde el resultado de etapa 1, preservando sus escalas.

Para ambas etapas, train_rbf.py crea un nuevo SparseAdam con shs_lr, betas y epsilon de la configuración fuente. El contador de la etapa empieza en uno: no se copian los momentos ni el LR instantáneo de un optimizador de otra modalidad. El scheduler usa los hitos originales relativos a esta nueva etapa.

Se respetan MSE/L1/Huber, lambda_photo y lambda_ssim del origen. Se mantiene concentración RGB mediante el backward auxiliar original. Los términos exclusivamente geométricos no se optimizan aquí porque esa geometría está congelada. El supersampling es el valor del modelo, no el warmup/augmentación del entrenador adaptativo.

### Etapa 3: entrenamiento conjunto

`train.py --appearance_mode vertex_rbf_iq` selecciona RBFJointModel. Mantiene SH0 en 1–999 y activa MLP/escalas en 1000. Densidad, SH0, H, MLP y eta se optimizan conjuntamente con grupos, LR, reloj angular, pérdidas y scheduler originales. Tras cada paso solo se proyecta el parámetro nuevo eta.

La poda/subdivisión geométrica sigue heredada. H usa su remapeo actual. Las escalas supervivientes se conservan exactamente por grid_pts_key; los vértices nuevos reciben cero, sin interpolar escalas. La MLP conserva sus pesos. Se recrea el optimizador después de adaptar igual que en la rama base.

Inicializar escalas nuevas a cero puede cambiar el campo de color tras subdividir un padre con eta no nulo; es la política indicada por la especificación, no una garantía de conservación exacta de RGB.

## 8. Checkpoints e importación

Se mantiene el contenedor conjunto con schema_version=1 y se añaden:

- kernel: vertex_rbf_iq, distinto de vertex_trilinear.
- appearance_spec.version=1, fórmula relative_inverse_quadratic, límites [-2,2], coordenadas voxel_local, H de 8, capas 26/64/3 y frecuencias [1,2,4]·π.
- optimization_scope: joint, scales_only o appearance.
- module contiene rbf_log_scales además de H/MLP.

La carga conjunta rechaza kernel o especificación incompatibles. train.py impide reanudar una etapa congelada como entrenamiento conjunto: sus grupos de optimizador son diferentes.

train_rbf.py puede importar checkpoints conjuntos antiguos o apariencia congelada schema v2. Conserva geometría/SH0/H/MLP y añade eta cero si faltaba. Comprueba layout o identidad del modelo base según el formato. No importa un control RGB sin MLP ni un checkpoint anterior a activación.

Reanudar una etapa congelada exige mismo scope y orden de cámaras guardado; no se admite ampliar el presupuesto de pasos por encima del orden ya guardado. Se restauran optimizador, scheduler y RNG. Una nueva etapa sin --resume reinicia el optimizador deliberadamente.

El guardado sigue siendo atómico mediante temporal y os.replace, con geometría completa y estado en el mismo archivo. El formato es autónomo después de importar un checkpoint anterior. Se utilizan archivos de confianza porque la carga conserva weights_only=False.

## 9. Visor y selección de compatibilidad

El visor identifica el kernel del checkpoint y construye RBFJointModel. Su primera opción es Vertex-MLP + RBF-IQ. El renderer selecciona RBF automáticamente cuando el módulo tiene log-escalas.

La opción manual Vertex-MLP trilineal se conserva porque la especificación la solicita. Equivale a render_vertex(..., interpolation='trilinear'); omite escalas y usa el kernel anterior solo si el usuario lo selecciona. interpolation='rbf_iq' exige un módulo con escalas. No se usó esta opción en la validación de la rama.

RGB, alfa, profundidad, normales, cámara libre y descarga reutilizan el visor existente. No se ejecutó una sesión interactiva de navegador ni se añadieron pruebas extensas del visor.

## 10. Comprobaciones mínimas ejecutadas

Comando único:

```bash
conda activate svraster
python -m pytest tests/test_rbf_iq.py -q
```

Resultado: **4 passed, 6 warnings, 2.12 s** con PyTorch 2.5.0+cu124, toolkit CUDA 12.5 y paquete local. Las advertencias son las existentes de MiniCam/index_reduce. La extensión exporta rasterize_rbf y rasterize_rbf_backward.

1. Identidad analítica eta=0, soporte, no negatividad y suma uno en referencia float64; no invoca el renderer anterior.
2. Diferencias finitas de una escala y un RGB, escogidos con gradiente no nulo, en la nueva ruta CUDA; gradientes finitos de SH0/densidad y T/depth/normal invariantes al cambiar escalas con geometría fija. Tolerancias de gradiente: atol=1e-5, rtol=0.03, perturbación 1e-2.
3. Save/load/render con atol=1e-5; un paso real de cada scope congelado con dataset sintético mínimo e inyección de cámaras, verificando scope guardado.
4. Subdivisión/poda pequeña: eta superviviente exacto, nuevos vértices en cero, backward válido y un paso conjunto posterior a adaptar.

Las pruebas no importan archivos test_vertex_*. No se ejecutaron las 25 pruebas anteriores, comparaciones trilineales, entrenamiento completo, búsqueda de hiperparámetros ni métricas PSNR/SSIM/LPIPS/FPS de una escena real. La etapa 3 no se validó con una corrida real hasta 1000: se comprobó su modelo adaptativo directamente y se reutilizó el calendario existente.

## 11. Comandos para el usuario

Ejecutar desde la raíz. Para recompilar esta rama:

```bash
conda activate svraster
CUDA_HOME=/usr/local/cuda-12.5 MAX_JOBS=2 python -m pip install -e cuda/ --no-build-isolation
python -m pytest tests/test_rbf_iq.py -q
```

### Etapa 1: importar el Lego ya entrenado y aprender solo escalas

```bash
python train_rbf.py \
  --checkpoint output/vertex_joint/lego/checkpoints/iter015000_model.pt \
  --scope scales_only \
  --steps 5000 \
  --output output/rbf_iq/lego_scales
```

Usar un checkpoint de apariencia aprendido y existente; el ejemplo reutiliza el que se guardó antes del OOM informado. Los checkpoints de la etapa se guardan cada 1000 y al finalizar.

### Etapa 2: H, MLP y escalas con geometría fija

```bash
python train_rbf.py \
  --checkpoint output/rbf_iq/lego_scales/checkpoints/iter005000_model.pt \
  --scope appearance \
  --steps 5000 \
  --output output/rbf_iq/lego_appearance
```

### Reanudar una etapa congelada interrumpida

```bash
python train_rbf.py \
  --checkpoint output/rbf_iq/lego_scales/checkpoints/iter001000_model.pt \
  --scope scales_only \
  --steps 5000 \
  --resume \
  --output output/rbf_iq/lego_scales
```

### Etapa 3: desde cero con entrenamiento adaptativo original

```bash
python train.py \
  --appearance_mode vertex_rbf_iq \
  --cfg_files cfg/synthetic_nerf.yaml \
  --source_path data/nerf_synthetic/lego \
  --model_path output/rbf_iq/lego_joint \
  --checkpoint_iterations 1000 5000 10000 15000
```

No se baja automáticamente el límite de vóxeles ni se cambia el protocolo para ajustarlo a la VRAM. La rama sigue pudiendo agotar memoria.

### Reanudar el entrenamiento conjunto

```bash
python train.py \
  --appearance_mode vertex_rbf_iq \
  --model_path output/rbf_iq/lego_joint \
  --load_iteration 15000
```

No añadir --cfg_files ni overrides al reanudar. No usar directamente un checkpoint trilineal con este comando: importarlo primero mediante una etapa congelada o comenzar desde cero.

### Visualizar RBF

```bash
python viz_vertex_fourier3.py \
  --checkpoint output/rbf_iq/lego_scales/checkpoints/iter005000_model.pt \
  --port 7007
```

Abrir http://localhost:7007. Para la etapa conjunta usar su iterNNNNNN_model.pt. La carga de un checkpoint RBF no requiere cambiar manualmente el modo del visor.

## 12. Límites y trazabilidad

La implementación no garantiza mejora de calidad, continuidad entre niveles, menor latencia ni solución del OOM. La anisotropía diagonal está alineada con ejes locales del octree y no representa rotaciones aprendidas. Centros, cámaras y topología discreta no reciben gradientes nuevos.

Los atomics producen posible no determinismo de redondeo. Se validó un conjunto pequeño de casos no degenerados; no un gradcheck exhaustivo ni múltiples escenas. Los límites de las escalas no se optimizan ni se seleccionan automáticamente.

Commits de implementación:

- af3c3d3: kernels, helper, bindings y fuentes de compilación RBF.
- 224d7c9: autograd, modelos, tres modalidades, checkpoints, visor y cuatro comprobaciones.

La documentación se guarda en un commit posterior. La especificación proporcionada y el log del usuario se dejaron intactos. Las rutas de color anteriores solo permanecen como compatibilidad explícita; no son una dependencia de ejecución del nuevo modo.
