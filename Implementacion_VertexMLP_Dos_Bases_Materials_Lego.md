# Implementación: Vertex-MLP con dos bases RGB y mezcla por vóxel

## Objetivo

Extender **Vertex-MLP Fourier-3 trilineal** de SVRaster para que una MLP compartida produzca **dos bases RGB residuales por vértice**, combinadas mediante **dos coeficientes aprendibles por vóxel**. Realizar pruebas simples sobre los checkpoints de **`materials_t`** y **`lego`**, con **geometría fija** en esta primera fase.

## 1. Arquitectura

Conservar la estructura actual: `grid_pts_xyz[P,3]`, `vox_key[V,8]`, características `H[P,8]`, dirección cámara→vértice normalizada, Fourier-3 de 18 dimensiones, SH0 base e interpolación trilineal en el punto medio de la intersección.

Cambios:

- MLP compartida: **`Linear(26,64) → ReLU → Linear(64,6)`** (antes `26→64→3`).
- Salida por vértice: **`bases[P,2,3]`**; dos residuales RGB.
- Parámetros nuevos: **`mix[V,2]`**; dos coeficientes independientes por vóxel, sin `softmax` en esta primera prueba.
- Para cada vóxel `v` y esquina `j`, combinar **antes** del rasterizador:

  `delta[v,j,:] = mix[v,0] * bases[vox_key[v,j],0,:] + mix[v,1] * bases[vox_key[v,j],1,:]`

- Conservar el buffer CUDA **`rgbs[V,27] = [SH0[V,3], delta[V,8,3]]`**, el kernel actual y el color `max(0, SH0 + trilineal(delta))`. No añadir nuevas consultas MLP por intersección.

Ecuación:

\[
\Delta c_v(x,d)=\sum_{j=1}^{8}N_j(x)\sum_{m=1}^{2}\lambda_{v,m}\,F_{\theta,m}([h_{k(v,j)},\gamma_3(d_{k(v,j)})]).
\]

**Nota:** el número de evaluaciones neuronales permanece en **una por vértice único y vista**. Aumentan los canales de salida y el costo de reunir/combinar bases por vóxel.

## 2. Integración mínima

1. Añadir un modo **`vertex_two_bases`** sin reemplazar `vertex_trilinear_fourier3` ni el renderizado SH original.
2. Crear un módulo con `H[P,8]`, MLP `26→64→6` y `mix[V,2]`. Mantener el cálculo Fourier actual.
3. Formar el residual `[V,8,3]` en PyTorch con `vox_key` y `mix`. Evitar materializar simultáneamente tensores grandes `[V,8,2,3]` si es posible; liberar temporales y medir VRAM.
4. Reutilizar **sin modificar** el forward/backward CUDA `vertex_trilinear` y su contrato `[V,27]`; asegurar que autograd propague a `mix`, `H` y MLP.
5. En la primera prueba, cargar un checkpoint geométrico completo de SVRaster, **congelar geometría, densidad, SH0 y topología**; entrenar únicamente `H`, MLP y `mix` mediante el protocolo de apariencia congelada ya existente.
6. Guardar `mix`, MLP, H, configuración y la identidad/layout del checkpoint en un formato de apariencia diferenciado. No aplicar poda/subdivisión en esta fase.

### Inicialización desde el checkpoint Vertex-MLP anterior (preferida si es compatible)

- Copiar `H` y la primera capa del decodificador existente.
- Copiar la última capa RGB anterior a los **primeros tres canales** (base 1).
- Inicializar los **últimos tres canales** (base 2) con pesos pequeños **no nulos**.
- Inicializar `mix[:,0]=1` y `mix[:,1]=0`.
- Comprobar que el render inicial coincide con la Vertex-MLP original. La base 2 recibe gradiente al activarse `mix[:,1]`; evitar inicializar **simultáneamente** base 2 y `mix[:,1]` a cero.

Si no existe un checkpoint Vertex-MLP compatible para esa escena, entrenar el modo desde cero, usando la misma partición/cámaras que el control Vertex-MLP; registrar explícitamente que la comparación ya no es una continuación de pesos idénticos.

## 3. Pruebas mínimas

- **Forma y finitud:** `bases[P,2,3]`, `mix[V,2]`, `delta[V,8,3]`, buffer `[V,27]`; sin NaN/Inf.
- **Equivalencia inicial:** con `mix=(1,0)` y la base 1 copiada, comparación de imágenes con Vertex-MLP previa (`max_abs_error ≤ 1e-5`, misma cámara/configuración).
- **Gradientes:** pérdida sobre imagen de prueba; verificar gradientes finitos en `mix`, H y ambas capas de MLP tras unos pasos. Confirmar que al activar la base 2 también recibe gradientes.
- **Regresión:** el modo SH0/SH3 y la Vertex-MLP anterior siguen funcionando sin cambios.
- **Prueba corta de entrenamiento:** unas pocas iteraciones en una cámara; comprobar descenso de pérdida y guardar/cargar sin cambiar el render.

## 4. Experimento simple: Materials y Lego

Ejecutar **dos pruebas separadas** usando los checkpoints disponibles:

| Escena | Checkpoint base | Comparación |
|---|---|---|
| **Materials (`materials_t`)** | `out/materials_t/checkpoints/iter020000_model.pt` (ruta histórica; verificar existencia) | SH0, SH3, Vertex-MLP de una base y dos bases |
| **Lego (`lego`)** | checkpoint SVRaster entrenado de Lego, ruta a indicar según archivo existente | SH0, SH3 si están disponibles, Vertex-MLP de una base y dos bases |

- Empezar con un **smoke test de 10–20 pasos** y luego una prueba corta de **300–500 pasos** por escena, sin pretender concluir convergencia o superioridad.
- Para comparar de forma justa una y dos bases, mantener **misma geometría congelada, cámaras/split, resolución, pérdida, optimizador, pasos y semilla**. Si se usa warm-start, incluir también el control de una base continuado por el mismo número de pasos.
- Reportar: **PSNR, SSIM, LPIPS, FPS total, memoria pico GPU, número de parámetros y latencia** de preparación neuronal, mezcla y rasterizador.
- Guardar una imagen comparativa por vista representativa, configuración y checkpoint de apariencia.
- Registrar si la mejora de calidad compensa el costo de combinación de bases y si aparecen discontinuidades en fronteras entre vóxeles.

## 5. Restricciones y criterio de decisión

No introducir Top-K, filtrado por frustum, nuevas bases SH, normales, FiLM, cambios de densidad, modificaciones de CUDA ni entrenamiento conjunto en este experimento. Mantener las optimizaciones de velocidad como trabajo posterior independiente.

**Éxito inicial:** implementación correcta y reproducible; mejora de calidad frente a la variante de una base en las mismas condiciones, sin aumento desproporcionado de VRAM o latencia. **No afirmar mejora sobre SH3** sin resultados comparables y completos.
