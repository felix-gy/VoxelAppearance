# Especificación de implementación — SVRaster Vertex-MLP + Fourier-3 + color trilineal

**Versión:** 2.0  
**Estado:** especificación de desarrollo; no implica código implementado ni compilado.  
**Interpolación de apariencia:** trilineal (sin RBF).  
**Decodificador:** MLP compartida de una capa oculta con 64 neuronas.  
**Codificación angular:** Fourier Encoding de 3 bandas.

## 1. Configuración obligatoria

| Componente | Valor |
|---|---|
| Modelo base | Checkpoint previamente entrenado con SVRaster |
| Geometría, densidad, posiciones de vértices | Congeladas |
| Layout de vóxeles | Congelado; sin subdivisión ni poda |
| Color base | SH grado 0 del checkpoint, congelado |
| Coeficientes SH de grado superior | No utilizados por el nuevo modo |
| Características de apariencia | 8 valores aprendibles por vértice único |
| Dirección de vista | Cámara → vértice, en coordenadas mundo y normalizada |
| Codificación angular | Fourier, tres frecuencias: `π`, `2π`, `4π` |
| Canales Fourier | 18; no concatenar la dirección cruda |
| Entrada MLP | 26 = 8 + 18 |
| MLP | `Linear(26,64) → ReLU → Linear(64,3)` |
| Salida MLP | Un RGB residual por vértice |
| Evaluación MLP | Vectorizada, una por vértice por vista |
| Reunión por vóxel | Ocho RGB mediante `vox_key` |
| Color de muestra | SH0 base + interpolación trilineal de los ocho RGB residuales |
| Muestras por rayo-vóxel | `n_samp_per_vox=1` |
| Posición de color | Exactamente el punto medio empleado por la densidad en el segmento rayo-vóxel |
| Composición, ordenamiento, profundidad y fondos | Lógica original de SVRaster |
| Optimización | Adam sobre características de vértices y MLP |
| Dtype de la primera versión | `torch.float32` |
| Dispositivo | CUDA |

**Parámetros entrenables:** `H ∈ float32[P,8]` y los parámetros de la MLP.  
**Parámetros MLP:** `(26×64+64)+(64×3+3)=1,923`.  
**Total de nuevos parámetros aprendibles:** `8P+1,923`, siendo `P` el número de vértices únicos.

### Contrato de formas

| Identificador | Dimensiones | Tipo |
|---|---|---|
| `P` | número de vértices únicos | `int` |
| `V` | número de vóxeles hoja | `int` |
| `grid_pts_xyz` | `[P,3]` | `float32` |
| `vox_key` | `[V,8]` | enteros, según el tipo real de SVRaster |
| `vertex_features` | `[P,8]` | `float32`, `Parameter` |
| `view_dirs` | `[P,3]` | `float32` |
| `view_fourier` | `[P,18]` | `float32` |
| `decoder_input` | `[P,26]` | `float32` |
| `vertex_rgb_delta` | `[P,3]` | `float32` |
| `voxel_vertex_rgb_delta` | `[V,8,3]` o subconjunto activo equivalente | `float32` |
| `base_rgb` | `[V,3]` o subconjunto activo equivalente | `float32` |
| `geo` | `[V,8]` o subconjunto activo equivalente | `float32` |
| `rendered_rgb` | formato de imagen que devuelve SVRaster | `float32` |

Todas las estructuras indexadas por vóxel que entren en CUDA deben compartir **el mismo orden de índices** (global o compactado). No mezclar una matriz global `[V,...]` con índices de un subconjunto activo sin transformar ambas.

## 2. Fourier Encoding: definición exacta

Dado el centro de cámara `o ∈ R³` y la posición de un vértice `p_i ∈ R³`:

\[
\mathbf d_i=\frac{\mathbf p_i-\mathbf o}{\max(\|\mathbf p_i-\mathbf o\|_2,10^{-8})}.
\]

Frecuencias (radianes):

\[
\boldsymbol\omega=[\pi,2\pi,4\pi].
\]

Para cada frecuencia `ω_k`, el bloque de seis valores es:

\[
B_k(\mathbf d)=
[\sin(\omega_k d_x),\sin(\omega_k d_y),\sin(\omega_k d_z),
 \cos(\omega_k d_x),\cos(\omega_k d_y),\cos(\omega_k d_z)].
\]

Codificación completa, en este orden exacto:

\[
\boxed{\gamma_F(\mathbf d)=[B_0(\mathbf d),B_1(\mathbf d),B_2(\mathbf d)]\in\mathbb R^{18}.}
\]

**No** añadir `d_x,d_y,d_z` como componentes extra: la MLP recibe 26 canales, no 29. **No** utilizar bandas aleatorias ni una matriz de proyección aprendible.

### Implementación PyTorch

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_direction_3bands(d: torch.Tensor) -> torch.Tensor:
    """
    d: [N,3] float32, direcciones unitarias en espacio mundo.
    Salida: [N,18], orden por frecuencia: sin(x,y,z), cos(x,y,z).
    """
    assert d.ndim == 2 and d.shape[1] == 3
    freq = d.new_tensor([1.0, 2.0, 4.0]) * math.pi   # [3]
    phase = d[:, None, :] * freq[None, :, None]       # [N,3,3]
    bands = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    return bands.reshape(d.shape[0], 18)             # [N,18]


class VertexAppearance(nn.Module):
    def __init__(self, num_vertices: int):
        super().__init__()
        self.vertex_features = nn.Parameter(
            0.01 * torch.randn(num_vertices, 8, dtype=torch.float32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        # Inicialmente produce residual exactamente cero.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, vertex_xyz: torch.Tensor,
                camera_origin: torch.Tensor) -> torch.Tensor:
        assert vertex_xyz.ndim == 2 and vertex_xyz.shape[1] == 3
        assert vertex_xyz.shape[0] == self.vertex_features.shape[0]
        camera_origin = camera_origin.reshape(1, 3).to(
            device=vertex_xyz.device, dtype=vertex_xyz.dtype
        )
        d = F.normalize(vertex_xyz - camera_origin, dim=-1, eps=1e-8)
        enc = fourier_direction_3bands(d)
        inputs = torch.cat([self.vertex_features, enc], dim=-1)
        return self.decoder(inputs)  # [P,3], sin sigmoid ni tanh
```

Instanciar `VertexAppearance(P).cuda()` **después** de cargar el checkpoint SVRaster y calcular `P = model.num_grid_pts`.

**Comportamiento de inicialización:** durante el primer backward, el peso de la última capa recibe gradiente, pero la primera capa y las características pueden recibir gradiente cero porque la última capa está inicializada exactamente a cero. Verificar gradientes otra vez tras el primer `optimizer.step()`. No tratar esa condición inicial como un fallo.

## 3. Modelo de color y densidad

### 3.1. Color residual por vértice

\[
\boxed{\mathbf a_i=F_\theta([\mathbf h_i,\gamma_F(\mathbf d_i)])\in\mathbb R^3.}
\]

- Se ejecuta la **misma MLP** para todos los vértices; no existe una MLP por vértice.
- No concatenar las características de los ocho vértices de un vóxel para producir 24 canales en una sola evaluación.
- Reunir los ocho RGB **después** de ejecutar la MLP:

```python
vertex_rgb_delta = vertex_appearance(
    model.grid_pts_xyz, camera_origin
)  # [P,3]
voxel_vertex_rgb_delta = vertex_rgb_delta[model.vox_key].contiguous()
# [V,8,3] si vox_key contiene todos los vóxeles.
```

### 3.2. Interpolación trilineal RGB por muestra

Para la posición de muestra `x`, vóxel `v` y posición local `u`:

\[
\mathbf u=\frac{\mathbf x-\mathbf m_v}{\mathbf s_v},
\qquad \mathbf m_v=\mathbf c_v-\frac{\mathbf s_v}{2}.
\]

Para cada esquina `k`, con bits `(b_x,b_y,b_z) ∈ {0,1}³`:

\[
N_k(\mathbf u)=
\prod_{a\in\{x,y,z\}}
\begin{cases}u_a,&b_a=1,\\1-u_a,&b_a=0.\end{cases}
\]

Forma equivalente para implementar los pesos:

\[
N_{b_xb_yb_z}=u_x^{b_x}(1-u_x)^{1-b_x}
\,u_y^{b_y}(1-u_y)^{1-b_y}
\,u_z^{b_z}(1-u_z)^{1-b_z}.
\]

\[
\boxed{\Delta\mathbf c_v(\mathbf x)=\sum_{k=0}^{7}N_k(\mathbf u)\,\mathbf a_{\mathrm{vox\_key}[v,k]}.}
\]

**El orden de las ocho esquinas debe verificarse y coincidir con la interpolación de densidad del kernel original.** No asumir un orden binario convencional.

### 3.3. Color base y salida

El color base `b_v` es el **RGB obtenido de evaluar SH de grado 0** de SVRaster, usando la misma implementación, activación y convenciones del baseline. No tomar el tensor `_sh0` como si ya fuese RGB.

\[
\boxed{\mathbf c_v(\mathbf x)=\max(0,\,\mathbf b_v+\Delta\mathbf c_v(\mathbf x)).}
\]

- `b_v` es fijo durante el entrenamiento inicial.
- Los SH de grado superior no intervienen en este modo.
- No introducir `sigmoid` en la salida de la MLP.
- Con la MLP inicializada a cero, el color debe ser igual a SVRaster renderizado en modo **SH0**, dentro de la tolerancia numérica.
- Conservar el `clamp(0,1)` final de `SVRenderer.render()`; no añadir otro `clamp(0,1)` por muestra.

### 3.4. Densidad y opacidad

No cambiar:

\[
\sigma(\mathbf x)=\operatorname{explin}\left(\sum_{k=0}^{7}N_k(\mathbf u)\rho_{v,k}\right),
\qquad
\alpha_v=1-\exp[-\ell_v\sigma(\mathbf x_m)]
\]

para `n_samp_per_vox=1` y punto medio `x_m` del segmento rayo-vóxel de longitud `ℓ_v`.

La **misma posición `x_m`** debe utilizarse para la interpolación trilineal RGB. No crear un muestreador de apariencia separado. Mantener ray-AABB, ordenamiento, transmitancia, supresión de contribuciones pequeñas, depth/normal y fondo exactamente como están en SVRaster.

## 4. Pipeline de inferencia

```text
Cargar checkpoint geométrico SVRaster
  ├─ octpath / octlevel / centros / tamaños
  ├─ vox_key [V,8] / grid_pts_xyz [P,3]
  ├─ densidad [P,1], congelada
  └─ SH0 por vóxel, congelado

Cargar checkpoint VertexAppearance [P,8] + MLP 26→64→3
                 │
Cámara nueva → centro de cámara o [3]
                 │
Direcciones cámara→vértice [P,3]
                 │
Fourier-3 [P,18]
                 │
Concatenar características [P,8] → [P,26]
                 │
MLP vectorizada [P,26] → [P,3]
                 │
Gather con vox_key → [V,8,3]
                 │
Preprocesamiento SVRaster (geometría + SH0)
                 │
Rasterizador CUDA extendido
  ├─ recorrido original y ray–AABB
  ├─ punto medio original y opacidad original
  ├─ N_k(x_m): ocho pesos trilineales
  ├─ RGB_sample = clamp_min(SH0 + Σ N_k RGB_vertex[k], 0)
  └─ composición volumétrica original
                 │
Imagen RGB / depth / T / normal según flags originales
```

La MLP se vuelve a evaluar si cambia la vista o si cambian sus parámetros. No reutilizar el RGB entre cámaras diferentes ni entre pasos de optimización.

## 5. Integración Python con SVRaster

### 5.1. Archivos existentes a intervenir

| Archivo | Cambio concreto |
|---|---|
| `src/sparse_voxel_gears/renderer.py` | Añadir una ruta explícita `appearance_mode='vertex_trilinear'` y preparar los RGB por vértice antes de llamar a CUDA |
| `src/sparse_voxel_gears/properties.py` | Reutilizar `vox_key`, `grid_pts_xyz`, `num_grid_pts` sin alterar su cómputo |
| `src/sparse_voxel_model.py` | No introducir cambios para la v1 si la apariencia se guarda en un módulo separado |
| `src/sparse_voxel_gears/io.py` | No modificar el formato original; serializar la nueva apariencia por separado |
| `train.py` | No mezclar el entrenamiento híbrido con el entrenamiento original; crear script separado |
| `cuda/` | Extender binding, forward y backward de rasterización sin cambiar la ruta SH existente |

### 5.2. Archivos nuevos

```text
src/vertex_appearance/
    __init__.py
    decoder.py               # VertexAppearance y fourier_direction_3bands
    renderer_wrapper.py      # Orquestación de la vista y tensors de apariencia
    checkpoint.py            # Guardar/cargar estado híbrido

cfg/vertex_fourier3_trilinear.yaml
train_vertex_fourier3.py
render_vertex_fourier3.py

tests/
    test_fourier3.py
    test_vertex_layout.py
    test_trilinear_reference.py
    test_vertex_forward.py
    test_vertex_backward.py
    test_vertex_checkpoint.py
    test_render_regression.py
```

### 5.3. Contrato de preparación por vista

```python
def prepare_vertex_appearance(model, vertex_module, camera_origin):
    # Requiere geometría fija y todos los tensores sobre la misma GPU.
    vertex_delta = vertex_module(
        model.grid_pts_xyz, camera_origin
    )  # [P,3]

    vox_key = model.vox_key
    assert vox_key.shape[1] == 8
    assert vox_key.min() >= 0
    assert vox_key.max() < vertex_delta.shape[0]

    voxel_delta = vertex_delta[vox_key].contiguous()  # [V,8,3]
    return voxel_delta
```

`prepare_vertex_appearance` es código nuevo que deberá incorporarse al wrapper. Si el rasterizador compacta vóxeles activos, debe aplicarse **la misma selección y el mismo orden** a `voxel_delta`, densidades y SH0.

### 5.4. Ruta del renderizador

En `SVRenderer` conservar el `render()` actual y una ruta de regresión a `color_mode='sh'` o `color_mode='sh0'`.

Implementación recomendada:

1. Añadir un wrapper o un método nuevo `render_vertex(camera, vertex_module, ...)`, sin cambiar el significado de `color_mode` del SVRaster original.
2. Obtener el centro de cámara del objeto de cámara real (`c2w` o campo equivalente; no asumir `camera.center`).
3. Ejecutar `prepare_vertex_appearance(...)` **una vez por llamada a render**.
4. Calcular SH0 de los vóxeles con el mecanismo original y en el mismo orden de índices utilizado por el rasterizador.
5. Invocar una extensión CUDA con nuevo modo `vertex_trilinear`.
6. Mantener el procesamiento posterior original: fondos, supersampling, autoexposición si corresponde y `clamp(0,1)`.

**Comprobación de compatibilidad obligatoria:** `vox_fn(idx, cam_pos, ...)` recibe índices de vóxeles activos; antes de modificar bindings, determinar qué entradas están en orden global y cuáles en orden compactado. Si se utiliza `freeze_vox_geo()`, comprobar especialmente si `frozen_vox_geo` contiene todos los vóxeles. No añadir solo una nueva clave a un diccionario Python suponiendo que CUDA la leerá automáticamente.

## 6. Extensión C++/CUDA

### 6.1. Interfaz mínima requerida

La extensión debe aceptar, además de las propiedades habituales:

```text
appearance_mode    : original_sh | vertex_trilinear
vertex_rgbs        : float32 contiguous [N,8,3]
base_rgbs          : float32 contiguous [N,3]
```

`N` es el número de vóxeles realmente procesados por esa etapa (global `V` o compactado, según la interfaz constatada). Las matrices `vertex_rgbs`, `base_rgbs` y `geos` deben usar el mismo índice para el mismo vóxel.

**Alternativa de memoria (no v1):** entregar `vertex_rgb_delta[P,3]` y `vox_key[V,8]` al kernel y reunir valores de vértice al vuelo. La versión inicial entrega `[N,8,3]` ya reunido mediante PyTorch, de manera que su backward se resuelve mediante el autograd del *gather*.

### 6.2. Forward por intersección

```cpp
// PSEUDOCÓDIGO. Adaptar nombres, índices y estructuras al CUDA real.
// Campos geo, rgb_base y vertex_rgb pertenecen al mismo voxel local.

RayVoxelSegment seg = intersect_ray_voxel(...);
if (!seg.valid) continue;

float3 x_mid = origin + 0.5f * (seg.t_enter + seg.t_exit) * ray_dir;
float3 u = (x_mid - voxel_min) / voxel_size;

// Mantener fórmula exacta de densidad/opacidad del renderer original.
float alpha = original_alpha_from_density(..., x_mid, ...);

float N[8];
compute_original_corner_order_trilinear_weights(u, N);

float3 delta = make_float3(0.f, 0.f, 0.f);
#pragma unroll
for (int k = 0; k < 8; ++k) {
    delta += N[k] * vertex_rgbs[voxel_id][k];
}

float3 color_preclamp = base_rgbs[voxel_id] + delta;
float3 color = max_per_channel(color_preclamp, 0.f);

// Conservar composición y umbrales originales.
compose_original_volume_color(alpha, color, ...);
```

No crear una posición de muestra distinta para apariencia. No modificar la ruta de obtención de `alpha`, ordenamiento de vóxeles, transmitancia ni premultiplicación.

### 6.3. Backward respecto al RGB por esquina

Sea `g_c = ∂L/∂color` la derivada obtenida al retropropagar **a través de la composición completa**. Para cada canal RGB:

\[
M_j=\mathbf 1[(b_{v,j}+\Delta c_{v,j})>0],
\qquad
\frac{\partial L}{\partial\mathbf a_{v,k}}
\mathrel{+}= N_k(\mathbf u)\,(\mathbf M\odot\mathbf g_c).
\]

`g_c` ya debe contener todos los factores de transmitancia y alfa aportados por el backward del rasterizador. **No multiplicarlos dos veces.**

El backward CUDA devuelve:

```text
grad_vertex_rgbs: float32 contiguous [N,8,3]
```

Reutilizar el mecanismo de acumulación de gradientes compatible con las estructuras de hilos de SVRaster. Cuando varias contribuciones escriban la misma posición, usar reducción segura/operaciones atómicas según corresponda; no permitir carreras de escritura.

Cadena de autograd:

```text
loss
  → backward CUDA: grad_voxel_vertex_delta [N,8,3]
  → PyTorch indexing/gather: grad_vertex_delta [P,3]
  → MLP: grad(vertex_features), grad(Linear26x64), grad(Linear64x3)
```

En la versión con geometría fija **no** se necesitan gradientes de la posición de muestra, tamaños de vóxel, transformaciones de cámara o `vox_key` para entrenar apariencia. No se deben borrar gradientes de `vertex_rgbs` aunque la densidad esté congelada.

### 6.4. Consideraciones de integridad de la extensión

- Mantener una ruta separada para el renderizado SH original y comprobar que no cambia sus imágenes.
- Modificar **toda** la cadena: binding Python/C++ → estructura de entradas → forward → estado guardado para backward → backward → retornos de gradiente.
- No cambiar la firma de retornos `color, depth, normal, T, max_w` a menos que se adapte cada consumidor.
- Asegurar alineación, `contiguous`, `float32`, tamaños, device CUDA e índices válidos.
- Definir cómo se maneja `color_mode` del código original cuando aparece `appearance_mode='vertex_trilinear'`.
- Rechazar explícitamente `n_samp_per_vox != 1` en este modo hasta implementar la composición cromática multimuestreo.

## 7. Entrenamiento

### 7.1. Secuencia de inicialización

1. Cargar el checkpoint SVRaster original con sus funciones de carga existentes.
2. Comprobar `V`, `P`, `vox_key.min()`, `vox_key.max()` y coordenadas de ocho esquinas.
3. Congelar `octpath`, `octlevel`, `_geo_grid_pts`, `_sh0`, `_shs` y cualquier otro parámetro geométrico; deshabilitar poda y subdivisión.
4. Si corresponde, llamar `freeze_vox_geo()` y comprobar el contrato de indexación de su resultado.
5. Crear `VertexAppearance(P).cuda()`.
6. Usar únicamente SH0 para `base_rgbs` en la rama nueva; conservar SH3 intacto para el baseline.
7. Inicializar y registrar únicamente los parámetros de `VertexAppearance` en Adam.
8. Ejecutar pruebas de forward y backward antes de entrenar con datos reales.

### 7.2. Configuración inicial

Crear `cfg/vertex_fourier3_trilinear.yaml`:

```yaml
experiment: vertex_fourier3_trilinear_v2
geometry_checkpoint: "REEMPLAZAR_RUTA"
seed: 42

appearance:
  mode: vertex_trilinear
  feature_dim: 8
  direction: camera_to_vertex_world
  direction_encoding: fourier
  fourier_bands: 3
  fourier_frequencies: [1.0, 2.0, 4.0]  # multiplicadas por pi
  include_raw_direction: false
  fourier_output_dim: 18
  mlp_layers: [26, 64, 3]
  activation: relu
  output: rgb_residual
  last_layer_init: zero
  base: pretrained_sh0_frozen
  interpolation: trilinear
  output_activation: clamp_min_zero

geometry:
  freeze_density: true
  freeze_layout: true
  freeze_sh0: true
  disable_pruning: true
  disable_subdivision: true

renderer:
  n_samp_per_vox: 1
  appearance_sample: density_midpoint
  sampling: existing_svraster
  dtype: float32
  supersampling: 1.0

optimizer:
  name: Adam
  lr_vertex_features: 0.01
  lr_mlp: 0.001
  betas: [0.9, 0.99]
  eps: 1.0e-8
  weight_decay: 0.0

training:
  steps_debug: 1000
  steps_initial: 5000
  photometric_loss: l1
  lambda_ssim: 0.2
  validation_interval: 500
  checkpoint_interval: 1000

metrics:
  psnr: true
  ssim: true
  lpips: true
  total_fps: true
  memory_mb: true
```

Son valores iniciales de configuración, no parámetros medidos ni validados. Mantener `supersampling`, resolución y preprocesado de imágenes iguales entre el baseline y la variante al comparar.

### 7.3. Grupos de parámetros

```python
optimizer = torch.optim.Adam(
    [
        {"params": [vertex_module.vertex_features], "lr": 1e-2},
        {"params": vertex_module.decoder.parameters(), "lr": 1e-3},
    ],
    betas=(0.9, 0.99),
    eps=1e-8,
    weight_decay=0.0,
)
```

### 7.4. Iteración de entrenamiento

```python
# PSEUDOCÓDIGO: render_vertex debe implementarse sobre la API real.
vertex_module.train()
optimizer.zero_grad(set_to_none=True)

camera, gt = next_training_camera_and_image()
model_base = frozen_svraster_model

vertex_rgb = vertex_module(model_base.grid_pts_xyz, camera_origin(camera))
voxel_rgb = vertex_rgb[model_base.vox_key].contiguous()

out = render_vertex(
    model_base, camera,
    voxel_rgb_delta=voxel_rgb,
    appearance_mode="vertex_trilinear",
    n_samp_per_vox=1,
)
loss = photometric_l1(out["color"], gt) + 0.2 * (1 - ssim(out["color"], gt))
loss.backward()

# La primera iteración con última capa a cero puede no dar gradiente a H.
assert all(torch.isfinite(p.grad).all()
           for p in vertex_module.parameters() if p.grad is not None)
optimizer.step()
```

Usar exactamente el mismo protocolo de cámaras, escalado de imagen, pérdidas y evaluación que el baseline elegido cuando se realice una comparación cuantitativa. Los hiperparámetros iniciales son configurables, pero cualquier modificación debe registrarse junto al checkpoint.

## 8. Plan de integración por etapas

### E0. Baseline inalterado

- [ ] Guardar commit, entorno CUDA/PyTorch, ruta de checkpoint y configuración.
- [ ] Renderizar con SVRaster original en `color_mode='sh0'` y `'sh'`.
- [ ] Guardar imágenes, PSNR/SSIM/LPIPS, tiempos y VRAM.
- [ ] Confirmar `n_samp_per_vox=1` y `ss=1.0` para la primera comparación.

**Salida:** baseline reproducible.

### E1. MLP sin extensión CUDA

- [ ] Implementar `fourier_direction_3bands()` y `VertexAppearance`.
- [ ] Validar tensores `[P,18]`, `[P,26]`, `[P,3]`, `[V,8,3]`.
- [ ] Crear una ruta provisional que promedie los ocho RGB por vóxel: `delta_vox = vertex_rgb[vox_key].mean(dim=1)`.
- [ ] Integrar ese residual en una salida RGB constante por vóxel utilizando la ruta existente.
- [ ] Validar que el render con MLP inicializada a cero coincide con SH0.
- [ ] Confirmar que la pérdida disminuye y se propagan gradientes hacia las características.

**Salida:** el decodificador y el flujo PyTorch funcionan. **Esta etapa todavía no implementa interpolación espacial por intersección.**

### E2. Forward CUDA trilineal

- [ ] Verificar el orden de los ocho vértices.
- [ ] Hacer que el kernel reciba RGB de ocho esquinas.
- [ ] Evaluar color en el punto medio del segmento rayo-vóxel.
- [ ] Mantener opacidad y composición originales.
- [ ] Comparar con referencia de interpolación implementada en Python.
- [ ] Validar independencia de profundidad/opacidad respecto a cambios de color.

**Salida:** render correcto con color espacial variable, sin requerir backward completo.

### E3. Backward CUDA

- [ ] Añadir gradientes de `[N,8,3]`.
- [ ] Comprobar múltiples contribuciones al mismo vóxel.
- [ ] Verificar acumulación hacia vértices compartidos vía `indexing` de PyTorch.
- [ ] Comparar gradientes CUDA frente a referencia analítica y diferencias finitas.
- [ ] Ejecutar overfit de una vista y comprobar descenso de pérdida.
- [ ] Ejecutar overfit de dos vistas para verificar dependencia angular.

**Salida:** entrenamiento diferencial completo.

### E4. Experimentos y empaquetado

- [ ] Entrenamiento controlado con topología fija.
- [ ] Guardar checkpoints del módulo neuronal y config.
- [ ] Renderizar prueba y medir métricas.
- [ ] Medir tiempos de codificación Fourier, MLP, `gather` y rasterizador.
- [ ] Comparar con SH0 y SH3 originales en geometría y resolución idénticas.

**Salida:** resultados y ejecución reproducibles.

## 9. Tests obligatorios

### 9.1. Fourier

1. `d=[0,0,0]` devuelve senos cero y cosenos uno; ningún NaN.
2. `d=[1,0,0]` genera 18 canales en el orden definido; validar contra expresión calculada manualmente.
3. Normalizar dirección antes de codificar; comprobar `||d||≈1` para vectores no nulos.
4. Mismo resultado en entrenamiento e inferencia para una misma cámara y vértice.
5. `vertex_features.shape=[P,8]`, Fourier `[P,18]`, entrada `[P,26]`, salida `[P,3]`.

### 9.2. Trilineal

1. Los ocho pesos suman uno con error máximo `1e-6` en muestras dentro del vóxel.
2. En cada una de las ocho esquinas, un peso es uno y los demás cero.
3. En el centro del vóxel, los pesos son `1/8`.
4. Si los ocho RGB son idénticos, la salida es ese RGB para cualquier posición.
5. A través de caras de vóxeles conformes con vértices compartidos, el **residual** interpolado es continuo a tolerancia numérica.
6. En interfaces de distintas resoluciones, registrar diferencias; no asumir continuidad automática.
7. Confirmar que `vox_key` coincide geométricamente con las esquinas del kernel de densidad.

### 9.3. Gradientes

1. Referencia de un vóxel: `∂delta/∂a_k = N_k` por canal.
2. Comparar `grad_vertex_rgbs` con diferenciación finita sobre un caso sintético pequeño, lejos de umbrales y puntos de saturación.
3. Reutilización del mismo vértice por varios vóxeles suma gradientes correctamente.
4. Con `SH0+delta > 0` no debe perderse gradiente por la activación de color.
5. Al terminar el segundo paso de optimización, comprobar gradientes finitos en `vertex_features` y ambas capas de la MLP.
6. Para densidad congelada, el RGB no modifica alfa, `T` ni profundidad.

### 9.4. Regresión y persistencia

1. `vertex_rgb_delta=0` produce la misma imagen que SVRaster SH0, tolerancia inicial `max_abs_error ≤ 1e-5` con la misma configuración de cámara/fondo.
2. El modo SH3 original sigue reproduciendo su imagen sin diferencias relevantes.
3. `save → load → render` conserva los parámetros e imagen a tolerancia `1e-5`.
4. Cambiar solamente la cámara altera la entrada angular, pero no las características aprendibles.
5. Cambiar los parámetros de la MLP invalida la caché de RGB de la vista anterior.
6. Rechazar `n_samp_per_vox > 1` en el modo nuevo hasta soportarlo.

## 10. Guardado y carga

No depender de que `SVInOut.save()` incluya la nueva MLP: ese método guarda un conjunto explícito de atributos de SVRaster. Mantener la apariencia en un checkpoint independiente.

```python
# PSEUDOCÓDIGO de guardado
payload = {
    "schema_version": 2,
    "mode": "vertex_trilinear_fourier3",
    "svraster_checkpoint": str(original_checkpoint),
    "num_vertices": int(model.num_grid_pts),
    "num_voxels": int(model.num_voxels),
    "vertex_module": {k: v.detach().cpu()
                      for k, v in vertex_module.state_dict().items()},
    "optimizer": optimizer.state_dict(),  # opcional para inferencia
    "step": int(step),
    "config": config,
    "layout_fingerprint": layout_fingerprint(model.octpath, model.octlevel),
}
torch.save(payload, out_path)
```

`layout_fingerprint()` debe implementarse sobre bytes y formas de `octpath` y `octlevel`, con un hash estable. Durante carga:

- Verificar `num_vertices`, `num_voxels` y el *fingerprint* antes de usar `vertex_features`.
- Reconstruir `VertexAppearance(P)` y cargar su `state_dict`.
- No admitir un nuevo layout de vértices con las características de otro checkpoint sin remapeo explícito.
- Registrar las frecuencias `[π,2π,4π]`, el orden Fourier, la arquitectura `26→64→3` y la convención de dirección en el config.
- Para inferencia, usar `vertex_module.eval()` y `torch.no_grad()`, no guardar gráficos de autograd.

## 11. Medición de rendimiento

### Variables a registrar

```text
P, V, P/V
num_trainable = 8*P + 1923
memory_model_parameters_bytes
memory_vertex_rgb_bytes = 12*P
memory_voxel_vertex_rgb_bytes = 96*V   # si [V,8,3] float32 completo
GPU_peak_allocated_bytes

latency_fourier_ms
latency_mlp_ms
latency_vertex_gather_ms
latency_raster_ms
latency_postprocess_ms
latency_total_ms
fps_total = 1000 / latency_total_ms

PSNR / SSIM / LPIPS por vista y promedios
```

Para tiempos GPU usar eventos CUDA, calentamiento previo, sincronización adecuada y el mismo número de mediciones en todos los modelos. Medir **latencia total por vista**, incluyendo Fourier, MLP y reunión de vértices; no reportar solo la velocidad del kernel rasterizador.

Medir al menos las siguientes rutas:

- SVRaster original SH0.
- SVRaster original SH3.
- Vertex-MLP Fourier-3 + trilineal.

Para aislar el aporte de la MLP, implementar como control adicional *RGB aprendible por vértice + interpolación trilineal* con la misma geometría y protocolo de entrenamiento.

## 12. Restricciones de la versión

- No implementar RBF ni anisotropía.
- No evaluar la MLP por intersección, píxel o muestra de densidad.
- No utilizar una MLP `73→64→24`; la única MLP es `26→64→3` por vértice.
- No recalcular Fourier ni MLP dentro del rasterizador CUDA.
- No permitir subdivisión, poda ni cambios de índices durante entrenamiento.
- No entrenar densidad ni posiciones de vértices.
- No incluir componentes SH de grado 1–3 en el color final de la variante.
- No modificar el muestreo geométrico original ni el orden de composición.
- No activar AMP ni optimizaciones de precisión reducida antes de pasar las pruebas de regresión y gradientes en float32.
- No dar por finalizada la implementación si el nuevo forward CUDA existe pero no propaga gradientes hasta `vertex_features`.

## 13. Criterios de implementación completada

- [ ] `fourier_direction_3bands`: entrada `[P,3]`, salida `[P,18]`, orden verificado.
- [ ] `VertexAppearance`: arquitectura exacta `26→64→3` y 1,923 parámetros de MLP.
- [ ] Características `H[P,8]` asociadas a los vértices únicos reales de SVRaster.
- [ ] RGB residual decodificado `[P,3]` una vez por vista.
- [ ] RGB agrupado por vóxel `[N,8,3]` con indexación consistente.
- [ ] RGB interpolado **en cada punto medio del segmento rayo-vóxel** con ocho pesos trilineales.
- [ ] Densidad, opacidad, ordenamiento, fondos y salida de regresión conservados.
- [ ] Backward correcto desde imagen a colores de esquinas, a vértices y a MLP.
- [ ] Entrenamiento sobre al menos una escena funciona sin pérdida de gradientes.
- [ ] Checkpoint nuevo guarda/carga apariencia sin modificar el formato original.
- [ ] Render de apariencia cero coincide con SVRaster SH0.
- [ ] Métricas y FPS de *pipeline completo* reproducibles.

**Resultado requerido:** una implementación funcional del modo `vertex_trilinear_fourier3` integrada en SVRaster, con ruta SH original conservada, entrenamiento diferencial de apariencia y pruebas automáticas de forward/backward.
