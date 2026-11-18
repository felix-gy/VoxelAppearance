# Plan mínimo: MLP diferida para SVRaster

**Estado:** especificación de implementación; no implica que esté implementada.

## 1. Objetivo y alcance

Implementar una variante de **apariencia diferida** en SVRaster, independiente de Vertex-MLP y RBF. El rasterizador acumulará un color base y características latentes por píxel; una MLP compartida producirá **una corrección RGB por píxel**, no por vóxel ni por intersección.

**Primera fase:** utilizar un checkpoint SVRaster existente, **congelar densidad, geometría y topología**, y entrenar solo la apariencia (SH0, características y MLP). No modificar poda, subdivisión, muestreo de densidad, orden de intersecciones ni composición alfa. No ejecutar entrenamientos completos desde cero en esta fase.

## 2. Modelo matemático

Cada vóxel activo `v` almacena:

- Densidad: parámetros originales, congelados en la primera fase.
- Color base: `SH0`, conservando exactamente la convención y transformación del renderer original.
- **Nuevo:** vector latente aprendible `features[v] ∈ R^8`, inicialmente pequeño (p. ej., normal con desviación `0.01`).

Para cada píxel `p`, reutilizar los pesos de composición originales `w_{p,v} = T_{p,v} α_{p,v}`:

\[
A_p=\sum_v w_{p,v},\qquad
\mathbf F_p=\sum_vw_{p,v}\mathbf f_v,\qquad
\mathbf B_p=\sum_vw_{p,v}\mathbf c_v^{SH0}+T_{p,\mathrm{final}}\mathbf c_{bg}.
\]

**Importante:** `B_p` incluye el fondo exactamente una vez. `F_p` no incluye características de fondo. Usar el mismo tratamiento de alfa, rayos y fondo blanco del modo original.

La dirección `d_p` es el rayo normalizado **cámara → muestra/píxel** en coordenadas mundiales; para un rayo recto es constante durante sus intersecciones. Su codificación Fourier-3 tiene 18 componentes, con la convención de orden ya utilizada en el proyecto.

MLP compartida con ReLU y salida lineal:

\[
D_\theta:\mathbb R^{29}\rightarrow\mathbb R^{32}\rightarrow\mathbb R^{32}\rightarrow\mathbb R^3,\quad
\Delta\mathbf C_p=D_\theta([\mathbf F_p,\mathbf B_p,\gamma_3(\mathbf d_p)]).
\]

Color final:

\[
\boxed{\mathbf C_p=\operatorname{clamp}_{[0,1]}\bigl(\mathbf B_p+A_p\,\Delta\mathbf C_p\bigr)}
\]

Inicializar **pesos y sesgo de la última capa en cero**, de modo que el primer render reproduzca el color base SH0 (salvo redondeo y el tratamiento final de clamp). No aplicar otro clamp a la salida de la MLP. El factor `A_p` impide una corrección neural sobre píxeles vacíos.

> **Decisión explícita:** utilizar `F_p` acumulado **sin dividir por `A_p`** para el primer prototipo. No introducir simultáneamente variantes normalizadas, nuevos regularizadores ni redes por píxel adicionales.

## 3. Cambios mínimos requeridos

1. **Modo opcional:** `appearance_mode=deferred_mlp`; mantener el modo SH original intacto.
2. **Parámetros:** añadir `features[V,8]` y MLP `29→32→32→3` (2,115 parámetros). Mantener SH0 por vóxel; no entrenar ni renderizar grados SH superiores en este modo.
3. **Rasterización:** producir `B[H,W,3]`, `F[H,W,8]` y `A[H,W,1]` usando las contribuciones volumétricas originales. Evitar una segunda pasada de rayos si es posible.
4. **Backward:** propagar gradientes a `features` y SH0; mantener correcta la acumulación de gradientes de vértices/vóxeles compartidos según la representación. La fase inicial congela densidad; antes de habilitar geometría entrenable, implementar y validar también la derivada de `F` y `B` respecto a densidad/transmitancia.
5. **Entrenamiento:** reutilizar cámaras, pérdida fotométrica, evaluación y configuración originales donde sean compatibles. Crear un optimizador para los parámetros nuevos sin modificar silenciosamente los grupos del baseline. No asumir que todos los regularizadores de color del renderer original son equivalentes cuando la corrección se aplica después de la composición.
6. **Inferencia:** evaluar la MLP **una vez por píxel** (en batch). Conservar fondo, alfa, profundidad y normales originales; profundidad y normales no pasan por la MLP.
7. **Persistencia:** guardar `appearance_mode`, SH0, características, MLP, geometría de referencia, configuración y estados necesarios para reanudar. Rechazar checkpoints incompatibles; conservar lectura de checkpoints SH originales.
8. **Medición:** reportar calidad y costo completo por fotograma, incluyendo acumulación de características, Fourier y MLP; medir VRAM pico, FPS sincronizado y número de vóxeles.

## 4. Pruebas mínimas (sin entrenamiento completo)

- **Identidad inicial:** última capa cero ⇒ la salida diferida coincide con SH0 renderizado por la ruta original, dentro de tolerancia numérica.
- **Composición:** escena sintética pequeña con 1–2 vóxeles; verificar `A`, `F`, `B`, transparencia y fondo, incluido un píxel vacío (`A=0`).
- **Gradientes:** diferencias finitas para latentes, SH0 y MLP en un caso pequeño, evitando los puntos no diferenciables del clamp. Confirmar que latentes y capas anteriores reciben gradientes después de actualizar la última capa.
- **Entrenamiento breve:** 10–30 pasos con geometría fija; pérdida finita, actualización de parámetros y descenso de pérdida en un conjunto reducido de rayos.
- **Guardar/cargar:** restaurar exactamente los parámetros y reproducir la imagen dentro de tolerancia `float32` (inicialmente `atol=rtol=1e-5`).
- **Compatibilidad:** el modo SH original conserva su salida; medir un render diferido sin NaN/Inf y registrar FPS/VRAM.

Para diferencias finitas `float32`, usar inicialmente `atol=1e-4`, `rtol=1e-2` y probar perturbaciones `1e-3` y `1e-2`; ajustar solo tras inspeccionar el error numérico.

## 5. Criterio para continuar

Una vez validados forward, backward y persistencia, ejecutar **un experimento corto de apariencia con geometría fija**. Registrar PSNR/SSIM, FPS, tiempo de la MLP y VRAM; usar SH0 y SH3 del mismo checkpoint como referencias de contexto, sin afirmar equivalencia de presupuestos. **No iniciar entrenamiento conjunto desde cero ni adaptación del octree** hasta comprobar estabilidad y una relación calidad/costo razonable.

**Fuera de alcance:** Vertex-MLP, RBF, SIREN, MSC, U-Net/posprocesamiento, reflejos físicos explícitos, características por vértice y cambios en la geometría durante esta primera integración.
