# Auditoría técnica de modificaciones: Vertex-MLP Fourier-3 trilineal

## 1. Alcance, fuente de verdad y criterio de lectura

Este documento detalla **el código implementado**, no solamente el diseño
previsto. La comparación usa el commit base `74e8907` y el estado integrado
`894911776390d7ac1801ce8ee73b93d1c4c024f0`. Las ubicaciones y líneas corresponden a ese estado; después de
otros cambios deben localizarse también por nombre de símbolo.

La integración registrada comprende **22 archivos: 4 modificados y 18 nuevos**,
incluyendo el informe previo. Este documento de auditoría es un archivo
adicional. No incluye como cambios propios archivos originales sin seguimiento,
datos, checkpoints, imágenes ni binarios generados.

El informe de resultados y uso está en
[Integracion_VertexMLP_Fourier3_Trilineal.md](Integracion_VertexMLP_Fourier3_Trilineal.md).
Aquí se explican las modificaciones por capa, sus contratos, consecuencias,
restricciones y límites de la evidencia. No se modificó código al redactarlo.

### Inventario completo del cambio integrado

| Estado | Ubicación |
|---|---|
| Nuevo | [cfg/vertex_fourier3_trilinear.yaml](../cfg/vertex_fourier3_trilinear.yaml) |
| Modificado | [cuda/binding.cpp](../cuda/binding.cpp) |
| Modificado | [cuda/setup.py](../cuda/setup.py) |
| Nuevo | [cuda/src/vertex_backward.cu](../cuda/src/vertex_backward.cu) |
| Nuevo | [cuda/src/vertex_backward.h](../cuda/src/vertex_backward.h) |
| Nuevo | [cuda/src/vertex_forward.cu](../cuda/src/vertex_forward.cu) |
| Nuevo | [cuda/src/vertex_forward.h](../cuda/src/vertex_forward.h) |
| Modificado | [cuda/svraster_cuda/renderer.py](../cuda/svraster_cuda/renderer.py) |
| Nuevo | [docs/Integracion_VertexMLP_Fourier3_Trilineal.md](../docs/Integracion_VertexMLP_Fourier3_Trilineal.md) |
| Nuevo | [render_vertex_fourier3.py](../render_vertex_fourier3.py) |
| Nuevo | [requirements-dev.txt](../requirements-dev.txt) |
| Modificado | [src/sparse_voxel_gears/renderer.py](../src/sparse_voxel_gears/renderer.py) |
| Nuevo | [src/vertex_appearance/__init__.py](../src/vertex_appearance/__init__.py) |
| Nuevo | [src/vertex_appearance/checkpoint.py](../src/vertex_appearance/checkpoint.py) |
| Nuevo | [src/vertex_appearance/decoder.py](../src/vertex_appearance/decoder.py) |
| Nuevo | [src/vertex_appearance/layout.py](../src/vertex_appearance/layout.py) |
| Nuevo | [src/vertex_appearance/runtime.py](../src/vertex_appearance/runtime.py) |
| Nuevo | [tests/test_fourier3.py](../tests/test_fourier3.py) |
| Nuevo | [tests/test_trilinear_reference.py](../tests/test_trilinear_reference.py) |
| Nuevo | [tests/test_vertex_checkpoint.py](../tests/test_vertex_checkpoint.py) |
| Nuevo | [tests/test_vertex_cuda.py](../tests/test_vertex_cuda.py) |
| Nuevo | [train_vertex_fourier3.py](../train_vertex_fourier3.py) |

## 2. Arquitectura resultante y responsabilidad de cada capa

```text
train_vertex_fourier3.py / render_vertex_fourier3.py
    runtime.setup → carga base, validación, congelación, apariencia y datos
    runtime.train / evaluate / benchmark
        model.render_vertex(camera, vertex_module)
            VertexAppearance o VertexRGB
                residual único [P,3]
            indexing con vox_key → [V,8,3] → [V,24]
            callback vox_fn → densidad [V,8] y SH0 [V,3]
            detach del baseline; concatenación RGB → [V,27]
            rasterize_voxels / _RasterizeVoxels
                dispatch original_sh o vertex_trilinear
                    _C.rasterize_vertex → forward CUDA
                    _C.rasterize_vertex_backward → backward CUDA
            posprocesamiento SVRenderer original → imagen
```

En backward, el camino entrenable es imagen → buffer RGB empaquetado → gather
PyTorch → residual por vértice → características/MLP. Geometría, cámaras y SH
base no se optimizan en el experimento. El control usa RGB directo por vértice
y elimina el decoder y su dependencia angular.

Se añadieron kernels independientes; no se sustituyeron los originales. Esto
reduce el acoplamiento de comportamiento entre modos, pero duplica lógica de
rasterización que debe actualizarse de forma coordinada en futuras correcciones.

## 3. Modificaciones en el modelo de apariencia

### 3.1. `src/vertex_appearance/decoder.py`

**`fourier_direction_3bands`, línea 8.** Comprueba entrada de rango 2 y tres
componentes. Construye frecuencias π,2π,4π en el mismo dtype/device que la
entrada; calcula phase `[P,3,3]`, concatena senos y cosenos en la última dimensión
y devuelve `[P,18]`. El orden es por frecuencia, con sin(XYZ),cos(XYZ).

La función no normaliza ni exige explícitamente float32/CUDA: normalizar es
responsabilidad del módulo, y dtype/device se controlan en el pipeline. Por
eso puede probarse en CPU. No existen frecuencias aleatorias o aprendibles ni
concatenación de dirección cruda. Cambiar ese orden invalida el significado de
los pesos entrenados, aunque la forma siga siendo `[P,18]`.

**`VertexAppearance.__init__`, línea 16.** Añade `vertex_features` como Parameter
`[P,8]`, inicialmente normal aleatoria escalada por 0.01; crea Linear(26,64),
ReLU y Linear(64,3). La última capa se inicializa completamente a cero. La
primera conserva la inicialización por defecto de PyTorch. Los tensores se
crean inicialmente en CPU y setup mueve el módulo a CUDA.

Hay 1923 parámetros en la MLP y `8P+1923` en el módulo. La salida inicial es
exactamente cero; durante el primer backward puede no llegar gradiente a H ni
a la primera capa porque el peso de salida aún vale cero. Esa situación se
verifica de nuevo después del primer paso y no se interpreta como un fallo.

**`VertexAppearance.forward`, línea 23.** Usa
`normalize(vertex_xyz-camera_origin, eps=1e-8)`, concatenando H y Fourier antes
de decodificar. La dirección es cámara→vértice en mundo. La salida `[P,3]` es
residual sin sigmoid/tanh ni clamp propio. Se evalúan todos los vértices una
vez por render, no por píxel/intersección. No guarda RGB entre llamadas.

**`VertexRGB`, línea 28.** Añade el control experimental: Parameter `[P,3]`
inicializado a cero. Su forward ignora coordenadas y cámara y devuelve ese
parámetro. Tiene `3P` valores entrenables y no modela dependencia angular.

### 3.2. `src/vertex_appearance/__init__.py`

Exporta VertexAppearance, VertexRGB y fourier_direction_3bands. No instala
hooks globales ni modifica SparseVoxelModel al importar el paquete.

### 3.3. `src/vertex_appearance/layout.py`

**`validate_layout`, línea 5.** Antes de reservar apariencia comprueba:

1. P y V positivos; vox_key `[V,8]` int64.
2. Coordenadas `[P,3]`, float32 CUDA; mismo dispositivo que los índices.
3. Mínimo de índices no negativo, máximo menor que P y coordenadas finitas.
4. Densidad `[P,1]` y SH0 `[V,3]`.
5. Correspondencia geométrica entre ocho esquinas y centro/tamaño del vóxel.

El último punto usa hasta 4096 vóxeles seleccionados mediante linspace; no es
una comprobación exhaustiva de todas las esquinas. Utiliza bits con
`k=4*bx+2*by+bz`; tolerancia absoluta `max(1e-6, |scene_extent|*1e-6)` y relativa
1e-6. Los límites de índices y la finitud sí se comprueban sobre sus tensores
completos. Las reducciones `.item()` y decisiones sobre tensores sincronizan
GPU/CPU durante setup, no en cada render.

Esta validación no comprueba exhaustivamente finitud de densidad/SH, tipos de
todos los atributos del checkpoint ni integridad continua del layout mientras
se entrena. El entrenamiento evita las operaciones adaptativas; no introduce un
bloqueo global de subdivisión/poda en toda la API del modelo.

## 4. Modificaciones del renderer Python del modelo

Archivo: `src/sparse_voxel_gears/renderer.py`.

### 4.1. `SVRenderer.render`: puntos nuevos, comportamiento conservado

- Línea 112: extrae `profile_events` de other_opt antes de crear RasterSettings.
- Línea 126: extrae un callback opcional `vertex_vox_fn`; por defecto sigue
  usando self.vox_fn. No se pasa el callback como argumento desconocido al NamedTuple.
- Líneas 144–151: registra raster_start y llama a rasterize_voxels con el callback.
- Línea 157: raster_end, antes del posprocesamiento existente.
- Línea 184: post_end, después del clamp final y redimensionamiento.

El fondo, exposición opcional del renderer general, supersampling, campos
raw_* y clamp final `[0,1]` mantienen su lógica. `render_vertex` y el runtime
reutilizan esa ruta. Con profiling omitido, no se crean eventos nuevos dentro
del renderer. Si se pasa un diccionario de eventos incompleto, no existe un
validador específico: puede producir KeyError.

### 4.2. `SVRenderer.render_vertex`, línea 188

La API nueva acepta cámara, módulo de apariencia y opciones de render.
Rechaza color_mode diferente de None/sh0 y muestras por vóxel distintas de 1
antes de evaluar la MLP. Obtiene cámara de `camera.c2w[:3,3]`.

1. Decodifica residual `[P,3]`.
2. Indexa con vox_key y lo reinterpreta como `[V,24]`.
3. Crea un callback que llama a vox_fn con SH0 para los índices visibles.
4. Aplica detach a geos, subdiv_p y RGB base.
5. Concatena base y residual en `[V,27]` contiguo.
6. Llama a render con appearance_mode='vertex_trilinear'.

El detach impide entrenar SH0 y densidad por este camino aunque el llamador
externo no haya cambiado sus requires_grad. No modifica permanentemente esos
flags; setup sí los congela de forma explícita. No es una API de optimización
de cámaras o geometría. La preparación neuronal puede construir un grafo si el
llamador no usa no_grad; los scripts de evaluación sí lo hacen.

### 4.3. Índices globales frente a visibles

El preprocesamiento produce in_frusts_idx y el callback procesa esos índices,
pero los buffers entregados a CUDA mantienen tamaño **V global**. No existe
compactación. Los índices de densidad, SH0, residual y binning se refieren al
mismo vóxel global. El gather neuronal se prepara para todos los vóxeles;
freeze_vox_geo también preagrupa densidad global `[V,8]`.

No cambiar este contrato unilateralmente: compactar solo RGB o solo geometría
puede asociar colores a vóxeles incorrectos sin un error de forma evidente.

## 5. Puente Python/C++ y contrato de autograd

### 5.1. `cuda/svraster_cuda/renderer.py`

**RasterSettings, línea 37:** añade al final `appearance_mode='original_sh'`.
Los constructores existentes pueden omitirlo; aumenta la longitud del NamedTuple,
por lo que consumidores externos que lo desempaqueten por longitud exacta deben
revisarse. No se modifican las firmas posicionales originales de las funciones C++.

**rasterize_voxels, líneas 54–60:** permite original_sh y vertex_trilinear. El
segundo exige una muestra y rechaza lambda_R_concen, lambda_ascending y
lambda_dist no nulos. Se conserva el rango general de muestras del baseline.

**Líneas 115–118:** exige RGB `[V,3]` para SH o `[V,27]` para vértices, float32
y contiguo. Esta validación más estricta afecta también a llamadores SH que
antes aportaran buffers de otra forma/dtype o no contiguos. El SH habitual pasa
la comprobación. Se mantienen los controles de dispositivos y dimensiones
preexistentes de geometría y otros campos.

**_RasterizeVoxels.forward, línea 185:** selecciona _C.rasterize_vertex o la
función original. Conserva contexto con geometría, RGB empaquetado y buffers de
rasterización necesarios para backward. La función C++ devuelve ocho elementos:
num_rendered, binningBuffer, imgBuffer, color, depth, normal, T y max_w. Autograd
expone los cinco últimos; max_w continúa marcado como no diferenciable.

**_RasterizeVoxels.backward, línea 247:** selecciona el backward correspondiente.
Ambos retornan tres tensores (dL_dgeos, dL_drgbs, subdiv_p_bw). En modo nuevo,
dL_drgbs es `[V,27]`; el gradiente del cat/indexing continúa en PyTorch. No se
agregó una entrada independiente vertex_rgbs a apply ni un retorno público extra.
Las entradas no diferenciables conservan retorno None.

### 5.2. `cuda/binding.cpp`, líneas 16–17 y 27–28

Incluye las dos cabeceras nuevas y exporta:

| Nombre Python en _C | Implementación |
|---|---|
| rasterize_vertex | FORWARD_VERTEX::rasterize_voxels |
| rasterize_vertex_backward | BACKWARD_VERTEX::rasterize_voxels_backward |

Las exportaciones originales permanecen. Un binario anterior no contiene estas
funciones: copiar únicamente los archivos Python no completa la instalación.

### 5.3. `cuda/setup.py`, líneas 23–24

Añade vertex_forward.cu y vertex_backward.cu a las fuentes de CUDAExtension.
No cambia el nombre del paquete ni el módulo binario svraster_cuda._C. Requiere
recompilar/reinstalar; dos checkouts con igual nombre de extensión pueden
confundirse si la instalación editable apunta al lugar equivocado.

## 6. Forward CUDA: disposición, interpolación y composición

### 6.1. `cuda/src/vertex_forward.h`

Nueva guarda y namespace FORWARD_VERTEX. Declara el mismo esquema de argumentos
y tupla de retorno del forward original; `rgbs` conserva el nombre pero pasa a
significar `[V,27]` en esta función. Incluye cámara, geometría, flags, buffers y
debug; no se introduce un selector de modo dentro de esta firma.

### 6.2. `cuda/src/vertex_forward.cu`

Archivo derivado del forward original bajo namespace e include propios. Las
partes de ray–AABB, organización de tiles, ordenamiento y composición se
conservan. Las diferencias semánticas principales son:

- Línea 175: base RGB de cada vóxel se carga como rgbs[vox_id*9]. Cada float3
  ocupa tres float32; un registro de nueve float3 equivale a 27 floats.
- Líneas 257–263: color se inicia en la base y suma ocho residuales desde
  rgbs[vox_id*9+1+corner], ponderados por interp_w.
- Cada canal usa fmaxf(valor,0), sin saturación superior a uno por muestra.
- La contribución continúa siendo T*alpha*color; el clamp superior permanece
  únicamente en el posprocesamiento Python original.

Con n_samp=1, interp_w conserva los pesos de la muestra media original de
densidad. No se crea otro punto de muestreo, ni se evalúa Fourier/MLP en CUDA.
Cambiar n_samp invalidaría esa reutilización cromática y está rechazado.

### 6.3. Controles de la entrada CUDA, líneas 775–780

Se exige RGB contiguo CUDA float32 `[V,27]`, geometría contigua CUDA float32
`[V,8]`, coincidencia de dispositivos con centros y una muestra por vóxel.
Se conserva el resto de las comprobaciones/punteros heredados. No son un
validador exhaustivo de todas las precondiciones posibles del binding directo.
El primer mensaje AT_ERROR de forma RGB conserva el texto heredado `(num_points, 3)`
aunque su condición exige 27 canales: el diagnóstico puede resultar confuso;
la comprobación posterior TORCH_CHECK sí describe `[V,27]`.

## 7. Backward CUDA: composición, máscaras y acumulación

### 7.1. `cuda/src/vertex_backward.h`

Declara el namespace BACKWARD_VERTEX con la firma de argumentos y los tres
retornos del backward original. El significado ampliado de RGB es parte de
este contrato aunque el nombre del tensor siga siendo rgbs.

### 7.2. `cuda/src/vertex_backward.cu`

- Línea 235: recupera RGB base con stride de nueve float3.
- Líneas 341–344: reconstruye preclamp por intersección con los pesos originales
  y aplica el mismo clamp mínimo para usar el color correcto en la composición inversa.
- Líneas 518–522: construye el paquete de 36 floats y distribuye a cada esquina
  `interp_w[corner] * (pre[channel]>0) * dL_drgb[channel]`.
- Líneas 525–529: mantiene también geometría, gradiente RGB base y prioridad.
- Línea 534: atomicAdd para las 36 posiciones; evita carreras entre píxeles.
- Líneas 702–707: controles de formas, tipos, dispositivos y n_samp.
- Líneas 711–715: caso V=0 devuelve ceros con las formas de sus entradas RGB/geometría.
- Línea 718: reserva buffer cero `[V,36]`.
- Líneas 776–779: split 8/3/1/24 y cat de RGB base con esquinas → `[V,27]`.

### Disposición exacta

| Buffer | Intervalo de columnas | Significado |
|---|---|---|
| rgbs `[V,27]` | 0:3 | SH0 evaluado y activado |
| rgbs `[V,27]` | 3:27 | Ocho residuales RGB, orden de esquinas original |
| dL_dvox `[V,36]` | 0:8 | Gradientes de densidad por esquina |
| dL_dvox `[V,36]` | 8:11 | Gradiente RGB base heredado |
| dL_dvox `[V,36]` | 11:12 | Prioridad de subdivisión heredada |
| dL_dvox `[V,36]` | 12:36 | Gradientes RGB residuales por esquina |

dL_drgb ya contiene el factor T*alpha de la composición; no se multiplica de
nuevo al distribuirlo. Los atomics pueden introducir pequeñas diferencias de
redondeo entre ejecuciones de backward. No se promete reproducibilidad bit a
bit de todo el entrenamiento.

**Límite importante:** el gradiente base heredado no se redefine como una nueva
API diferenciable del SH0 entrenable; el wrapper lo desacopla. La ruta soportada
es apariencia con base congelada. Los gradientes de geometría/prioridad siguen
calculándose por herencia, pero no alcanzan parámetros del modelo en esta ruta;
esto implica trabajo y memoria que todavía no se han optimizado.

## 8. Configuración, inicialización y ciclo del experimento

### 8.1. `cfg/vertex_fourier3_trilinear.yaml`

Declara checkpoint y YAML base, modo neuronal, semilla 3721, 5000 pasos,
validación cada 500, checkpoint cada 1000 y las dos tasas de aprendizaje.
appearance registra dimensión 8, frecuencias [1,2,4] multiplicadas por π,
orden, dirección, exclusión de dirección cruda, capas y dtype.

No es el YAML jerárquico ilustrativo completo de la especificación ni se mezcla
con el CfgNode global existente. Los valores estructurales se validan contra
una arquitectura fija; escribir otras capas no selecciona una arquitectura
arbitraria. En el control RGB se conserva el bloque descriptivo de la variante,
pero se instancia VertexRGB en función de mode.

### 8.2. `src/vertex_appearance/runtime.py`: parser, línea 13

CLI compartida por ambos scripts:

| Opción | Efecto real |
|---|---|
| --config | YAML propio; por defecto cfg/vertex_fourier3_trilinear.yaml |
| --output | Directorio requerido de escritura |
| --steps | Total objetivo, no cantidad adicional al reanudar |
| --resume | Carga apariencia, Adam y RNG al entrenar |
| --appearance-checkpoint | Selecciona estado para renderizado |
| --geometry-checkpoint | Reubica/sustituye archivo base; identidad debe coincidir al cargar apariencia |
| --mode | vertex_trilinear_fourier3 o vertex_rgb_trilinear |
| --benchmark | En render, mide tiempos en lugar de exportar comparación |

Es un parser compartido, no dos interfaces totalmente separadas: algunas
opciones solo tienen efecto en una ruta. Véanse los límites de uso de la sección 13.

### 8.3. setup, línea 26

Comprueba que el paquete importado svraster_cuda.__file__ está dentro de este
checkout. No comprueba por separado la ruta resuelta de _C ni una versión de
sus símbolos; ambas rutas se verificaron manualmente durante la integración.

Lee YAML propio; si hay apariencia/reanudación, sustituye esa configuración
por la guardada. Después aplica overrides explícitos de steps, mode y ruta
geométrica. Prefiere resolved_base si está guardado y, si no, lee base_config.
Valida modo, pasos/intervalos positivos, arquitectura exacta, una muestra,
exposición desactivada y que no se use L1/Huber.

Fija semilla torch, crea/carga SparseVoxelModel, valida layout, desactiva gradiente
en densidad, SH0, SHs y prioridad; llama a freeze_vox_geo. Después crea el módulo
CUDA, adapta data.eval a use_test y obtiene DataPack. Finalmente construye la
identidad con hash del archivo geométrico, hash del layout, P y V.

No copia todos los hiperparámetros del entrenamiento adaptativo: no crea su
optimizador, scheduler, pérdidas geométricas ni actualizaciones de exposición.
Para materials_t, el protocolo usado es MSE + 0.02 SSIM y ss=1.5 fijo; no se
reproduce la posible augmentación de supersampling del entrenador general.

### 8.4. train, línea 97

Escribe configuración resuelta, crea Adam solo con apariencia y separa LR de H
y decoder. El control usa un solo grupo con lr_vertex_features. Betas, eps y
weight decay efectivo se fijan en el código (0.9/0.99,1e-8,0), sin scheduler.

La secuencia de cámaras usa random.Random(seed) y baraja épocas completas.
Se reconstruye desde el principio para el número total solicitado y se indexa
por step-1, lo que preserva el prefijo al reanudar con más pasos. El generador
torch guardado no es actualmente el generador usado para ese orden.

Cada paso: train(), zero_grad, render_vertex, GT a CUDA, MSE + lambda_ssim *
fast_ssim_loss, control de pérdida finita, backward, control de gradientes
finitos y optimizer.step. El control acepta grad=None en ese chequeo; no exige
que todos los parámetros tengan gradiente no nulo en cada vista. Las pruebas
específicas sí comprueban propagación después de la inicialización.

La pérdida usa directamente l2_loss con peso uno: no aplica lambda_photo del
YAML base ni incorpora todos sus regularizadores. En materials_t ese peso es
uno, de modo que coincide con el protocolo elegido. No debe interpretarse como
un ejecutor genérico de cualquier configuración del entrenador original.

Cada 100 pasos imprime progreso. Guarda estado cada intervalo y al final;
evalúa todas las vistas test cada intervalo y al final. memory.json contiene
pico **allocated** de PyTorch después del reset, no memoria reservada ni uso
total que muestra nvidia-smi.

### 8.5. evaluate, línea 75

Usa eval y no_grad; renderiza SH solicitado o variante, calcula PSNR, SSIM,
LPIPS y LPIPS corregido con funciones existentes. Métricas sobre float32 antes
de cuantización PNG; PNG RGB de 8 bits redondeados para inspección. Exporta
renders/gt con índices de vista de cuatro dígitos y metrics.json con filas y media.
Si no hay cámaras devuelve per_view vacío; no considera eso un experimento válido.

No restaura automáticamente el modo train: el bucle lo hace en el siguiente
paso. La evaluación incorpora el coste de redes de métricas y escritura de
imágenes; ese coste no forma parte del FPS de render reportado.

### 8.6. render, línea 143

Carga apariencia si --appearance-checkpoint está presente. Luego ejecuta
benchmark o compara sh0, sh3 y variante en carpetas separadas. Los baselines
comparten geometría, cámaras, configuración y resolución; no se reentrenan.

### 8.7. benchmark, línea 154

Usa eventos CUDA, 10 iteraciones de calentamiento y 50 mediciones; rota por
las cámaras disponibles con índice i % número_de_cámaras. Son 50 mediciones,
no todas las 200 vistas del cálculo de calidad. Sincroniza el evento final.

- Total: entrada a llamada de render hasta su salida; incluye apariencia.
- Apariencia: llamada completa del módulo.
- Gather: indexing/reshape, antes de SH y cat.
- Raster: incluye preprocesamiento, callback, SH, cat y raster CUDA.
- Post: fondo, resize y clamp posteriores a raster.
- Medición aislada neuronal: direcciones/Fourier y concatenación/MLP.

Estos intervalos no son todos componentes disjuntos: los tiempos aislados no
se suman otra vez al total. FPS es 1000 dividido por media de latencia, no
media de FPS individuales. Eventos GPU no caracterizan exhaustivamente I/O ni
coste de arranque CPU. No se reportan percentiles, desviación ni intervalo de confianza.

Mueve el módulo a CPU durante SH0/SH3 y a CUDA para variante, antes del reset
del pico. El pico aún incluye el modelo base y temporales; parameters_bytes
cuenta solamente apariencia. No incluye estados de Adam de entrenamiento.
Guarda medias, bytes, P,V,P/V y número de parámetros en benchmark.json.

## 9. Persistencia y reanudación detalladas

Archivo: `src/vertex_appearance/checkpoint.py`.

### 9.1. file_digest, línea 7

SHA-256 del archivo geométrico en bloques de 1 MiB. Comprueba identidad de
archivo, no solo equivalencia aproximada de geometría. Recuantizar o volver a
guardar un checkpoint puede cambiar ese hash aunque las imágenes sean similares.

### 9.2. layout_fingerprint, línea 15

Concatena en SHA-256 forma, dtype y bytes CPU contiguos de octpath, octlevel,
grid_pts_key, scene_center y scene_extent. El orden de vértices queda protegido
por grid_pts_key. No hashea por separado vox_key o grid_pts_xyz; se derivan de
esos atributos y el layout se valida al cargar. La identidad del archivo base
protege además el contenido geométrico y SH.

### 9.3. save_checkpoint, línea 25

| Campo | Contenido |
|---|---|
| schema_version | Entero 2 |
| mode | Modo de config |
| config | Configuración propia y resolved_base |
| identity | SHA-256 base, hash layout, P,V |
| step | Paso completado |
| vertex_module | state_dict del módulo |
| optimizer | Estado Adam |
| torch_rng | RNG CPU |
| cuda_rng | Estados RNG CUDA |
| camera_rng | Estado del generador torch auxiliar |

Se crea el directorio y se usa torch.save directamente. Los tensores del
state_dict no se convierten explícitamente a CPU al guardar; la carga usa
map_location='cpu' y load_state_dict copia a los parámetros del módulo CUDA.
No hay escritura atómica con archivo temporal, verificación posterior ni
checkpoint reducido exclusivo de inferencia; interrumpir una escritura puede
dejar un archivo incompleto.

### 9.4. load_checkpoint, línea 34

Rechaza schema distinto de 2 o identity diferente; carga estado neuronal.
Si se proporciona optimizador, carga Adam, restaura RNG CPU/CUDA y generador.
No compara explícitamente mode contra la clase del módulo: el setup lo elige
por configuración y load_state_dict detecta muchas incompatibilidades de claves/formas.

La reanudación determinista de cámaras presupone el mismo dataset, orden y
número de cámaras. No se guarda un hash de imágenes o de la lista de cámaras.
No se promete continuidad bit a bit en backward CUDA con atomics. Los archivos
se cargan con weights_only=False y deben tratarse como checkpoints de confianza.
No se añadió una política de carga segura para archivos externos.

## 10. Entradas ejecutables y dependencias

**train_vertex_fourier3.py y render_vertex_fourier3.py, línea 1:** wrappers de
cuatro líneas que importan parser y función correspondiente y ejecutan bajo
__main__. No duplican lógica del experimento.

**requirements-dev.txt, línea 1:** pytest>=8,<10. No modifica requirements.txt,
versiones de PyTorch ni instala automáticamente toolkit CUDA. PyYAML/Pillow y
las dependencias de métricas deben estar disponibles en el entorno. Las pruebas
sintéticas no requieren descargar datos, pero la evaluación LPIPS necesita sus pesos.

Instalación probada en Conda svraster con PyTorch 2.5.0+cu124 y toolkit 12.5:

```bash
conda activate svraster
python -m pip install -r requirements-dev.txt
CUDA_HOME=/usr/local/cuda-12.5 MAX_JOBS=2 python -m pip install -e cuda/ --no-build-isolation
python -m pytest tests -q
```

Las rutas relativas de configuración, datos y salidas se interpretan desde el
directorio de ejecución, no desde el directorio del YAML; ejecutar desde la raíz.
Comprobar las rutas de svraster_cuda y svraster_cuda._C después de instalar.
Cambios C++/CUDA requieren recompilar y un proceso nuevo para cargar el binario.

## 11. Pruebas añadidas y alcance de la evidencia

Se añadieron cuatro archivos con **17 tests**; el helper scene no es un test.
Última ejecución de integración: 17 passed, 13 warnings, 1.73 s en svraster.
Las advertencias observadas proceden de plugins instalados y del MiniCam
existente; la presencia de warnings no se ocultó.

| Archivo / test | Qué demuestra |
|---|---|
| test_fourier3: test_fourier_order | Orden y expresión Fourier, incluyendo dirección cero |
| test_initialization_and_two_steps | 1923 parámetros MLP, salida cero y gradientes tras dos pasos |
| test_normalized_directions_and_camera_dependence | Invariancia a escala de dirección y dependencia de cámara |
| test_trilinear_reference: test_partition_corners_center_and_gradients | Esquinas, centro, suma de pesos y derivada analítica |
| test_shared_vertex_accumulation | Indexing PyTorch suma un vértice reutilizado |
| test_vertex_checkpoint: test_checkpoint_roundtrip_and_rejection | Estado neuronal, paso e identidad incompatible |
| test_vertex_cuda: test_zero_regression_and_geometry_invariance | SH0 inicial, T/depth/normal invariantes y rechazo multimuestreo |
| test_backward_finite_difference | Backward RGB en tres vértices frente a diferencias finitas |
| test_decoder_overfit | Descenso de pérdida de una vista y gradientes finitos |
| test_constant_residual_matches_constant_sh_color | Color constante compatible con SH evaluado |
| test_color_clamp_has_zero_gradient | Gradiente bloqueado bajo preclamp negativo |
| test_checkpoint_preserves_render | Imagen conservada al guardar/cargar |
| test_empty_frustum_and_partial_visibility | No contribuciones y visibilidad parcial |
| test_layout_matches_kernel_corner_order | Coordenadas de esquinas compatibles con el kernel |
| test_two_view_overfit | Descenso conjunto y salida angular diferente |
| test_wrapper_never_trains_base_sh | SH base sin gradiente aunque tenga requires_grad |
| test_layout_validation_rejects_invalid_keys | Rechazo de índices fuera de rango |

Regresión/persistencia: atol 1e-5. Diferencias finitas float32: atol 1e-4,
rtol 1e-2 y perturbación 1e-3. No es gradcheck exhaustivo de todos los parámetros,
ni cobertura formal de todas las geometrías o umbrales. Las pruebas CUDA se
omiten cuando torch.cuda.is_available es falso; una corrida solo CPU no certifica
los kernels. Si CUDA está disponible pero falta la extensión, los tests no la compilan.

Evidencia adicional real:

- materials_t: P=4.338.721 y V=2.937.583; layout validado.
- Entrenamiento neuronal y control de 5000 pasos cada uno, 200 vistas test.
- Reanudar desde checkpoint real funcionó; el run definitivo se reinició desde
  cero con el muestreo corregido para que ambos métodos compartieran protocolo.
- Tras cargar el checkpoint neuronal final, las cuatro medias de calidad
  coincidieron exactamente con las guardadas en evaluación de entrenamiento.
- Una vista SH3 comparada con la extensión del checkout anterior dio error
  máximo RGB=0 y T=0. Esto no equivale a una comparación exhaustiva de todas
  las vistas o del backward de ambos binarios.
- git diff --check sin errores al cerrar la implementación.

## 12. Implicaciones medibles de memoria, velocidad y calidad

Para materials_t:

| Estructura | Fórmula float32 | Bytes |
|---|---|---:|
| H y MLP entrenables | 4*(8P+1923) | 138846764 |
| Residual único RGB | 12P | 52064652 |
| Gather RGB por esquinas | 96V | 282007968 |
| RGB empaquetado | 108V | 317258964 |
| Buffer backward dL_dvox | 144V | 423011952 |

El cat del buffer empaquetado reserva memoria adicional: no es una vista gratis
del gather. El backward añade temporales de cat/split contiguos. Hay además
activaciones `[P,18]`, `[P,26]`, `[P,64]`, almacenamiento de Adam, geometría,
SH base y buffers de raster. No inferir el pico total sumando solo parámetros.

| Método | PSNR | SSIM | LPIPS corregido | FPS total |
|---|---:|---:|---:|---:|
| SH0 | 23.65449 | 0.901724 | 0.091297 | 99.76 |
| SH3 | 29.96786 | 0.954652 | 0.043374 | 81.26 |
| Vertex-MLP | 27.89134 | 0.935391 | 0.072690 | 16.80 |
| RGB por vértice | 24.94174 | 0.921024 | 0.077455 | 58.89 |

Neuronal: 59.52 ms totales, 43.50 ms apariencia, 2.26 ms gather, 13.49 ms
raster/preprocesamiento y 0.25 ms post. Fourier/direcciones y concatenación/MLP
medidos aisladamente: 7.30 y 36.19 ms. Control: 16.98 ms totales.

Picos allocated de entrenamiento: 5531939328 bytes neuronal y 3270554112 control.
Inferencia: 4117459968 neuronal, 2175752704 control, 1508395008 SH0/SH3.

La variante mejora SH0 y RGB fijo, pero pierde calidad y velocidad frente a SH3.
Son resultados de una escena/protocolo, no una conclusión universal sobre todos
los datasets. No hay assertions de rendimiento/PSNR en pytest, ni metas de FPS
que conviertan una mejora de calidad en un criterio de corrección del código.

## 13. Limitaciones y precauciones de uso observadas en el código

Estas son propiedades del estado auditado, no funcionalidades implementadas:

1. **Solo geometría fija y una muestra.** No hay remapeo de H tras adaptación;
   modificar el layout desde otra API no está globalmente bloqueado.
2. **Todos los vértices por vista.** No hay compactación visible de apariencia,
   caché entre cámaras, partición de MLP por bloques, AMP ni precisión reducida.
3. **Continuidad limitada.** El residual es continuo en interfaces conformes
   con vértices compartidos; no se garantiza en interfaces adaptativas ni en
   color total si el SH0 de los vóxeles cambia.
4. **CLI compartida.** --resume solo carga pesos/optimizador en train; en render
   selecciona configuración pero no activa load_checkpoint neuronal. Para
   inferencia usar --appearance-checkpoint. Si se omite, render utiliza un
   módulo recién inicializado; no busca automáticamente el último checkpoint.
5. **Flags no mutuamente excluyentes.** Pasar --resume y --appearance-checkpoint
   juntos puede elegir configuración y pesos desde rutas distintas; no es una
   combinación validada. --mode puede sobrescribir el modo guardado y provocar
   incompatibilidad de state_dict. No se implementó migración de arquitectura.
6. **Dataset vacío.** train no valida que haya cámaras; su construcción de épocas
   no termina si la lista de entrenamiento está vacía. benchmark requiere
   cámaras para el módulo del índice. Usar un dataset válido y el split esperado.
7. **Paso objetivo ya alcanzado.** Si steps<=paso cargado no se ejecutan pasos,
   y el bucle no genera una nueva evaluación/checkpoint final automáticamente.
8. **No validación completa de hiperparámetros.** El YAML se carga como dict,
   no mediante un schema exhaustivo; faltantes pueden producir KeyError y
   tasas inválidas errores de Adam. No hay un validador de todas las claves extra.
9. **Persistencia no transaccional.** No hay guardado atómico, bloqueo del
   directorio ni protección frente a dos ejecuciones que escriban igual ruta.
10. **Dependencia del entorno.** El paquete se valida por ruta Python, no por un
    identificador ABI/schema del binario. Se requiere extensión recompilada.
11. **Cómputo geométrico heredado.** Aunque se descarten esos gradientes, el
    backward aún realiza parte de su trabajo. No se implementó un kernel mínimo
    exclusivo para apariencia congelada.
12. **Compatibilidad del binding directo.** Los kernels nuevos asumen el mismo
    contrato de buffers y orden que el wrapper. Su gradiente base no es una API
    soportada para entrenar SH0. El uso recomendado pasa por render_vertex.
13. **Coste de validación y almacenamiento.** Evaluar/exportar 200 imágenes por
    intervalo agrega I/O y espacio; las medidas de FPS lo excluyen deliberadamente.
14. **Sin garantía de determinismo bit a bit.** Atomics, bibliotecas y GPU pueden
    introducir diferencias pequeñas; la evidencia usa tolerancias numéricas.

## 14. Diferencias concretas frente al diseño inicial

| Diseño o intención inicial | Implementación efectiva / consecuencia |
|---|---|
| Entradas vertex_rgbs/base_rgbs separadas | Empaquetado `[V,27]`, retorno RGB único |
| Buffer independiente de esquinas | Acumulador `[V,36]` y reempaquetado `[V,27]` |
| YAML ilustrativo L1+0.2 SSIM | Protocolo acordado MSE+0.02 SSIM de materials_t |
| Extender esquema YACS | YAML propio con validaciones explícitas y resolved_base |
| Etapa provisional de promediar esquinas | Sustituida por pruebas directas; no se publica un modo promedio |
| Preservar ruta SH | Kernels originales sin modificaciones; dispatch adicional Python |
| Guardar apariencia por separado | Schema v2 propio; SVInOut no conoce el módulo |
| Comprobar layout por hash octpath/octlevel | También grid_pts_key, transformación mundo y SHA-256 del archivo base |
| Verificar geometría al cargar | Rangos completos y probe geométrico limitado a 4096 vóxeles |

## 15. Qué componentes base se reutilizan sin modificarlos

SparseVoxelModel, SVProperties (vox_key/grid_pts_xyz y cachés), SVInOut,
constructores/adaptación, lectores de datos, cámaras, evaluación/métricas,
train.py, render.py, src/config.py, kernels forward/backward SH, SH_eval,
GatherGeoParams y preprocessing. Reutilizar no significa duplicar sus fuentes
en el nuevo paquete. La excepción es la copia especializada de kernels de
rasterización en los cuatro archivos nuevos vertex_*.

La compatibilidad preservada concierne a esta ruta y al protocolo probado;
las nuevas validaciones de RGB y longitud del NamedTuple son cambios públicos
que deben considerarse si existen consumidores externos personalizados.

## 16. Ubicaciones de símbolos Python

Líneas obtenidas mediante AST del código auditado. Se incluyen también símbolos
preexistentes de los archivos modificados para facilitar navegación; no todos
fueron reescritos. Las secciones anteriores identifican qué cambió en cada uno.

| Archivo | Símbolo | Línea inicial |
|---|---|---:|
| `cuda/svraster_cuda/renderer.py` | `RasterSettings` | 15 |
| `cuda/svraster_cuda/renderer.py` | `rasterize_voxels` | 40 |
| `cuda/svraster_cuda/renderer.py` | `_RasterizeVoxels` | 142 |
| `cuda/svraster_cuda/renderer.py` | `_RasterizeVoxels.forward` | 144 |
| `cuda/svraster_cuda/renderer.py` | `_RasterizeVoxels.backward` | 198 |
| `cuda/svraster_cuda/renderer.py` | `SH_eval` | 263 |
| `cuda/svraster_cuda/renderer.py` | `SH_eval.forward` | 265 |
| `cuda/svraster_cuda/renderer.py` | `SH_eval.backward` | 305 |
| `cuda/svraster_cuda/renderer.py` | `GatherGeoParams` | 331 |
| `cuda/svraster_cuda/renderer.py` | `GatherGeoParams.forward` | 333 |
| `cuda/svraster_cuda/renderer.py` | `GatherGeoParams.backward` | 350 |
| `cuda/svraster_cuda/renderer.py` | `GatherFeatParams` | 360 |
| `cuda/svraster_cuda/renderer.py` | `GatherFeatParams.forward` | 362 |
| `cuda/svraster_cuda/renderer.py` | `GatherFeatParams.backward` | 379 |
| `cuda/svraster_cuda/renderer.py` | `mark_n_duplicates` | 389 |
| `cuda/svraster_cuda/renderer.py` | `mark_max_samp_rate` | 420 |
| `cuda/svraster_cuda/renderer.py` | `mark_near` | 438 |
| `src/sparse_voxel_gears/renderer.py` | `SVRenderer` | 14 |
| `src/sparse_voxel_gears/renderer.py` | `SVRenderer.freeze_vox_geo` | 16 |
| `src/sparse_voxel_gears/renderer.py` | `SVRenderer.unfreeze_vox_geo` | 28 |
| `src/sparse_voxel_gears/renderer.py` | `SVRenderer.vox_fn` | 35 |
| `src/sparse_voxel_gears/renderer.py` | `SVRenderer.render` | 96 |
| `src/sparse_voxel_gears/renderer.py` | `SVRenderer.render_vertex` | 188 |
| `src/vertex_appearance/checkpoint.py` | `file_digest` | 7 |
| `src/vertex_appearance/checkpoint.py` | `layout_fingerprint` | 15 |
| `src/vertex_appearance/checkpoint.py` | `save_checkpoint` | 25 |
| `src/vertex_appearance/checkpoint.py` | `load_checkpoint` | 34 |
| `src/vertex_appearance/decoder.py` | `fourier_direction_3bands` | 8 |
| `src/vertex_appearance/decoder.py` | `VertexAppearance` | 15 |
| `src/vertex_appearance/decoder.py` | `VertexAppearance.__init__` | 16 |
| `src/vertex_appearance/decoder.py` | `VertexAppearance.forward` | 23 |
| `src/vertex_appearance/decoder.py` | `VertexRGB` | 28 |
| `src/vertex_appearance/decoder.py` | `VertexRGB.__init__` | 29 |
| `src/vertex_appearance/decoder.py` | `VertexRGB.forward` | 33 |
| `src/vertex_appearance/layout.py` | `validate_layout` | 5 |
| `src/vertex_appearance/runtime.py` | `parser` | 13 |
| `src/vertex_appearance/runtime.py` | `setup` | 26 |
| `src/vertex_appearance/runtime.py` | `evaluate` | 75 |
| `src/vertex_appearance/runtime.py` | `train` | 97 |
| `src/vertex_appearance/runtime.py` | `render` | 143 |
| `src/vertex_appearance/runtime.py` | `benchmark` | 154 |
| `tests/test_fourier3.py` | `test_fourier_order` | 6 |
| `tests/test_fourier3.py` | `test_initialization_and_two_steps` | 12 |
| `tests/test_fourier3.py` | `test_normalized_directions_and_camera_dependence` | 26 |
| `tests/test_trilinear_reference.py` | `weights` | 4 |
| `tests/test_trilinear_reference.py` | `test_partition_corners_center_and_gradients` | 9 |
| `tests/test_trilinear_reference.py` | `test_shared_vertex_accumulation` | 22 |
| `tests/test_vertex_checkpoint.py` | `test_checkpoint_roundtrip_and_rejection` | 7 |
| `tests/test_vertex_cuda.py` | `scene` | 9 |
| `tests/test_vertex_cuda.py` | `test_zero_regression_and_geometry_invariance` | 29 |
| `tests/test_vertex_cuda.py` | `test_backward_finite_difference` | 46 |
| `tests/test_vertex_cuda.py` | `test_decoder_overfit` | 65 |
| `tests/test_vertex_cuda.py` | `test_constant_residual_matches_constant_sh_color` | 80 |
| `tests/test_vertex_cuda.py` | `test_color_clamp_has_zero_gradient` | 90 |
| `tests/test_vertex_cuda.py` | `test_checkpoint_preserves_render` | 98 |
| `tests/test_vertex_cuda.py` | `test_empty_frustum_and_partial_visibility` | 114 |
| `tests/test_vertex_cuda.py` | `test_layout_matches_kernel_corner_order` | 131 |
| `tests/test_vertex_cuda.py` | `test_two_view_overfit` | 138 |
| `tests/test_vertex_cuda.py` | `test_wrapper_never_trains_base_sh` | 157 |
| `tests/test_vertex_cuda.py` | `test_layout_validation_rejects_invalid_keys` | 167 |

## 17. Trazabilidad por commits

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
- `8949117`: Consolidate integration report with completed RGB control and reproducible results

El presente informe se añade en un commit documental posterior. AGENTS.md y la
especificación original no forman parte de ese diff integrado de 22 archivos.

## 18. Anexo: diferencias semánticas exhaustivas de los kernels nuevos

Para evitar presentar las 1600 líneas heredadas como algoritmo nuevo, los
diffs siguientes comparan cada archivo nuevo contra su original **normalizando
solo namespace, guarda e include propios**. Las renombradas son:
FORWARD→FORWARD_VERTEX, BACKWARD→BACKWARD_VERTEX y las cabeceras vertex_*.
Los diff muestran las demás diferencias efectivas del código CUDA auditado.
No representan cambios aplicados a los kernels SH originales.

### forward.h

Tras normalizar namespace/guard/include, la firma coincide con la original.

### forward.cu

```diff
--- original/forward.cu
+++ vertex_normalizado/forward.cu
@@ -172,7 +172,7 @@
             collected_vox_l[thread_id] = vox_lengths[vox_id];
             for (int k=0; k<8; ++k)
                 collected_geo_params[thread_id*8 + k] = geos[vox_id*8 + k];
-            collected_rgb[thread_id] = rgbs[vox_id];
+            collected_rgb[thread_id] = rgbs[vox_id * 9];
         }
         block.sync();
 
@@ -254,7 +254,13 @@
 
             // Accumulate to the pixel.
             float pt_w = T * alpha;
-            C = C + pt_w * collected_rgb[j];
+            float3 color = collected_rgb[j];
+            for (int corner=0; corner<8; ++corner)
+                color = color + interp_w[corner] * rgbs[vox_id * 9 + 1 + corner];
+            color.x = fmaxf(color.x, 0.f);
+            color.y = fmaxf(color.y, 0.f);
+            color.z = fmaxf(color.z, 0.f);
+            C = C + pt_w * color;
 
             if (need_depth)
             {
@@ -761,11 +767,17 @@
 {
     if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
         AT_ERROR("vox_centers must have dimensions (num_points, 3)");
-    if (rgbs.ndimension() != 2 || rgbs.size(1) != 3)
+    if (rgbs.ndimension() != 2 || rgbs.size(1) != 27)
         AT_ERROR("rgbs should be either (num_points, 3)");
     if (vox_centers.size(0) != rgbs.size(0))
         AT_ERROR("size mismatch");
 
+    TORCH_CHECK(rgbs.is_cuda() && rgbs.scalar_type() == torch::kFloat32 && rgbs.is_contiguous(), "Expected contiguous float32 CUDA RGB buffer");
+    TORCH_CHECK(rgbs.dim() == 2 && rgbs.size(0) == vox_centers.size(0) && rgbs.size(1) == 27, "Expected RGB [V,27]");
+    TORCH_CHECK(geos.is_cuda() && geos.scalar_type() == torch::kFloat32 && geos.is_contiguous(), "Expected contiguous float32 CUDA geometry");
+    TORCH_CHECK(geos.dim() == 2 && geos.size(0) == vox_centers.size(0) && geos.size(1) == 8, "Expected geometry [V,8]");
+    TORCH_CHECK(rgbs.device() == geos.device() && rgbs.device() == vox_centers.device(), "Device mismatch");
+    TORCH_CHECK(n_samp_per_vox == 1, "Vertex appearance requires one sample");
     const int P = vox_centers.size(0);
     const int H = image_height;
     const int W = image_width;
```

### backward.h

Tras normalizar namespace/guard/include, la firma coincide con la original.

### backward.cu

```diff
--- original/backward.cu
+++ vertex_normalizado/backward.cu
@@ -232,7 +232,7 @@
             collected_vox_l[thread_id] = vox_lengths[vox_id];
             for (int k=0; k<8; ++k)
                 collected_geo_params[thread_id*8 + k] = geos[vox_id*8 + k];
-            collected_rgb[thread_id] = rgbs[vox_id];
+            collected_rgb[thread_id] = rgbs[vox_id * 9];
         }
         block.sync();
 
@@ -338,7 +338,10 @@
             // Load from share memory.
             const int vox_id = collected_vox_id[j];
 
-            const float3 c = collected_rgb[j];
+            float3 pre = collected_rgb[j];
+            for (int corner=0; corner<8; ++corner)
+                pre = pre + interp_w[corner] * rgbs[vox_id * 9 + 1 + corner];
+            const float3 c = make_float3(fmaxf(pre.x,0.f), fmaxf(pre.y,0.f), fmaxf(pre.z,0.f));
 
             // The gradients w.r.t. voxel alpha.
             float dL_dpt_w = dot(dL_dpix, c);
@@ -512,7 +515,11 @@
             /**************************
             Write back the gradient below.
             **************************/
-            float grad_pack[12];
+            float grad_pack[36] = {0.f};
+            const float mask[3] = {float(pre.x > 0.f),float(pre.y > 0.f),float(pre.z > 0.f)};
+            for (int corner=0; corner<8; ++corner)
+                for (int ch=0; ch<3; ++ch)
+                    grad_pack[12 + corner*3 + ch] = interp_w[corner] * mask[ch] * dL_drgb[ch];
             #pragma unroll
             for (int iii=0; iii<8; ++iii)
                 grad_pack[iii] = dL_dgeo_params[iii];
@@ -523,8 +530,8 @@
 
             const int base_id = cg::this_grid().thread_rank();
             #pragma unroll
-            for (int iii=0; iii<12; ++iii)
-                atomicAdd(dL_dvox + vox_id * 12 + (base_id+iii)%12, grad_pack[(base_id+iii)%12]);
+            for (int iii=0; iii<36; ++iii)
+                atomicAdd(dL_dvox + vox_id * 36 + (base_id+iii)%36, grad_pack[(base_id+iii)%36]);
         }
     }
 }
@@ -692,17 +699,23 @@
     if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
         AT_ERROR("vox_centers must have dimensions (num_points, 3)");
 
+    TORCH_CHECK(rgbs.is_cuda() && rgbs.scalar_type() == torch::kFloat32 && rgbs.is_contiguous(), "Expected contiguous float32 CUDA RGB buffer");
+    TORCH_CHECK(rgbs.dim() == 2 && rgbs.size(0) == vox_centers.size(0) && rgbs.size(1) == 27, "Expected RGB [V,27]");
+    TORCH_CHECK(geos.is_cuda() && geos.scalar_type() == torch::kFloat32 && geos.is_contiguous(), "Expected contiguous float32 CUDA geometry");
+    TORCH_CHECK(geos.dim() == 2 && geos.size(0) == vox_centers.size(0) && geos.size(1) == 8, "Expected geometry [V,8]");
+    TORCH_CHECK(rgbs.device() == geos.device() && rgbs.device() == vox_centers.device(), "Device mismatch");
+    TORCH_CHECK(n_samp_per_vox == 1, "Vertex appearance requires one sample");
     const int P = vox_centers.size(0);
 
     if (P == 0)
     {
-        torch::Tensor dL_dgeos = torch::empty({0});
-        torch::Tensor dL_drgbs = torch::empty({0});
-        torch::Tensor subdiv_p_bw = torch::empty({0});
+        torch::Tensor dL_dgeos = torch::zeros_like(geos);
+        torch::Tensor dL_drgbs = torch::zeros_like(rgbs);
+        torch::Tensor subdiv_p_bw = torch::zeros({0,1}, geos.options());
         return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
     }
 
-    torch::Tensor dL_dvox = torch::zeros({P, geos.size(1)+3+1}, vox_centers.options());
+    torch::Tensor dL_dvox = torch::zeros({P, geos.size(1)+3+1+24}, vox_centers.options());
     dim3 tile_grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);
     dim3 block(BLOCK_X, BLOCK_Y, 1);
 
@@ -760,9 +773,9 @@
         dL_dvox.contiguous().data_ptr<float>());
     CHECK_CUDA(debug);
 
-    std::vector<torch::Tensor> gradient_lst = dL_dvox.split({geos.size(1), 3, 1}, 1);
+    std::vector<torch::Tensor> gradient_lst = dL_dvox.split({geos.size(1), 3, 1, 24}, 1);
     torch::Tensor dL_dgeos = gradient_lst[0].contiguous();
-    torch::Tensor dL_drgbs = gradient_lst[1].contiguous();
+    torch::Tensor dL_drgbs = torch::cat({gradient_lst[1], gradient_lst[3]}, 1).contiguous();
     torch::Tensor subdiv_p_bw = gradient_lst[2].contiguous();
 
     return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
```

