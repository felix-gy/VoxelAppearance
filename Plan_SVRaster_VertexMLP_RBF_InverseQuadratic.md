# SVRaster + Vertex-MLP: interpolación RBF anisotrópica

**Estado:** especificación mínima de implementación. Extensión de la versión Vertex-MLP con interpolación trilineal existente.

## 1. Objetivo

Incorporar una interpolación espacial aprendible para el **RGB residual** de Vertex-MLP usando kernels **inverse quadratic** anisotrópicos, sin cambiar la densidad, la composición volumétrica ni la arquitectura neuronal.

## 2. Modelo que se conserva

- Vértices únicos compartidos entre vóxeles, cada uno con característica latente `H[P,8]`.
- Codificación de dirección Fourier-3 (18 dimensiones) y MLP compartida `26 → 64 → 3`, que produce RGB residual por vértice y vista.
- Color base **SH0 por vóxel**.
- Ocho vértices por vóxel; una muestra por intersección inicialmente.
- Interpolación **trilineal original para la densidad**, sin modificaciones.

## 3. RBF propuesta

Cada vértice único `i` tiene:

- **Centro fijo:** posición del vértice (esquina de los vóxeles incidentes).
- **Matriz de forma anisotrópica diagonal, aprendible:** tres log-escalas `η_i = (η_ix, η_iy, η_iz)`; no se aprenden rotaciones ni centros.

Para una muestra dentro del vóxel, usar coordenadas locales `t ∈ [0,1]^3` y esquinas locales `q_i ∈ {0,1}^3`:

\[
D_i(\mathbf t)=\sum_{a\in\{x,y,z\}}(t_a-q_{ia})^2e^{-2\eta_{ia}},\qquad
D_i^0(\mathbf t)=\sum_a(t_a-q_{ia})^2.
\]

Kernel **inverse quadratic** y factor relativo:

\[
\phi_i=\frac{1}{1+D_i},\qquad
\rho_i=\frac{\phi_i}{\phi_i^0}=\frac{1+D_i^0}{1+D_i}.
\]

Pesos de interpolación (con `N_i` trilineal original):

\[
\boxed{w_i(\mathbf t)=\frac{N_i(\mathbf t)\rho_i(\mathbf t)}{\sum_{j=1}^{8}N_j(\mathbf t)\rho_j(\mathbf t)}}
\]

Color de la muestra:

\[
\boxed{\mathbf c_v=\max\!\left(0,\mathbf c_v^{SH0}+\sum_{i=1}^{8}w_i\,\Delta\mathbf c_i\right)}
\]

**Inicialización:** `η = 0` para todos los vértices, por lo que `ρ_i = 1` y `w_i = N_i`: la ruta nueva reproduce la interpolación trilineal existente al inicio.

**Alcance:** las log-escalas se comparten por vértice único, pero la distancia está normalizada en el vóxel consultado. Por tanto, es una **modulación RBF local**, no una RBF con una única matriz física global por vértice. La continuidad del residual en fronteras de diferente nivel de octree no se garantiza.

## 4. Cambios mínimos en el código

1. Añadir el parámetro aprendible `rbf_log_scales[P,3]`, inicializado a cero, asociado a las mismas claves de vértices que `H`.
2. Extender el **forward CUDA** de apariencia para calcular los ocho pesos `w_i`; dejar intactos densidad, alfa y composición.
3. Extender el **backward CUDA** para acumular gradientes a RGB residual y `rbf_log_scales`, incluyendo la derivada de la normalización de pesos. Mantener gradientes existentes de SH0 y densidad.
4. Guardar/cargar las log-escalas en checkpoints e incluirlas en el optimizador; mantener una opción para renderizar con interpolación trilineal original.
5. Si se usa adaptación del octree: conservar parámetros de vértices supervivientes e inicializar nuevas log-escalas a cero, reutilizando el remapeo por claves de vértice.

Para estabilidad numérica, parametrizar las escalas con exponencial y acotar `η` a un intervalo razonable (p. ej., `[-2,2]`), usando el mismo comportamiento en forward y backward.

## 5. Implementación por etapas

**Etapa 1 (prioritaria):** cargar un checkpoint Vertex-MLP trilineal existente con geometría fija; añadir RBF y validar equivalencia inicial. Entrenar solo log-escalas para aislar el efecto de la interpolación.

**Etapa 2:** permitir optimización conjunta de `H`, MLP y log-escalas con geometría fija; comparar con trilineal bajo igual presupuesto de entrenamiento.

**Etapa 3 (posterior):** integrar en el entrenamiento conjunto y la poda/subdivisión originales de SVRaster, una vez validada la etapa anterior.

## 6. Pruebas mínimas

- **Identidad:** con `η = 0`, RGB renderizado equivalente al trilineal dentro de tolerancia numérica.
- **Pesos:** `w_i ≥ 0`, `Σ_i w_i = 1`, y `N_i = 0 ⇒ w_i = 0`.
- **Backward:** diferencias finitas sobre una log-escala y un RGB de vértice; verificar gradientes finitos y no nulos en un caso no degenerado.
- **Compatibilidad:** densidad y SH0 conservan sus rutas de gradiente; guardar/cargar produce el mismo render.
- **Adaptación (solo etapa 3):** remapeo correcto de log-escalas y ausencia de índices obsoletos después de una subdivisión.

## 7. Evaluación mínima

Comparar **Vertex-MLP trilineal** frente a **Vertex-MLP + RBF inverse quadratic**, partiendo del mismo checkpoint, con la misma geometría, cámaras y cantidad de pasos adicionales. Medir **PSNR, SSIM, LPIPS, FPS y VRAM**. La mejora no está garantizada: debe demostrarse experimentalmente.

**Fuera de alcance en esta versión:** covarianza completa, centros móviles, kernel gaussiano como alternativa, RBF para densidad, MLP por vóxel y cambios en el algoritmo de subdivisión.
