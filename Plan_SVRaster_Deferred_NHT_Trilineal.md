# Plan mínimo — SVRaster diferido con interpolación trilineal y codificación armónica

**Estado:** propuesta de siguiente variante, **no implementada**. Se apoya en `deferred_mlp` de la rama `MLP-Diferida` (commit `7f44699`). No usa Vertex-MLP, RBF ni MLP por intersección.

## 1. Evidencia que motiva el experimento

Prueba en `out/materials_t`, geometría de SVRaster congelada desde SH iteración 20 000, 2 937 583 vóxeles, 4 cámaras de entrenamiento y 4 reservadas, 256×256, `ss=1`, una muestra de densidad por vóxel, fondo blanco y semilla 3721.

| Modo/paso | PSNR entrenamiento | PSNR reservado | SSIM reservado |
|---|---:|---:|---:|
| SH0 | — | 22,50 dB | 0,8813 |
| SH3 | — | 26,16 dB | 0,9483 |
| Diferido, paso 30 | 25,60 dB | 22,63 dB | 0,8789 |
| Diferido, paso 300 | 34,53 dB | 22,60 dB | 0,8601 |

En 300 pasos el PSNR de entrenamiento subió **8,93 dB**, el PSNR reservado prácticamente no mejoró y el SSIM reservado disminuyó. **Es compatible con sobreajuste; no demuestra por sí solo que falte expresividad intravóxel.** Velocidad medida al paso 300: 16,54 ms/fotograma (60,5 FPS), de los cuales 0,52 ms corresponden a la MLP; pico CUDA asignado por PyTorch: 1 562 MiB. Estas mediciones no son directamente comparables con otros FPS sin igualar resolución, sincronización y protocolo.

**Hipótesis de la nueva variante:** las características que varían con la posición dentro del vóxel y su codificación armónica pueden aportar información espacial útil al sombreador diferido. **Riesgo:** aumentan parámetros y cálculo y pueden empeorar el sobreajuste con solo cuatro vistas.

## 2. Alcance cerrado

- Conservar la geometría, densidad, topología, pesos volumétricos, fondo, SH0, dirección de vista, rasterización ordenada y MLP diferida actuales. SH sigue siendo el modo predeterminado.
- Añadir un modo opcional, por ejemplo `deferred_nht_trilinear`, **sin sustituir** `deferred_mlp`.
- Mantener la primera fase con **geometría y topología congeladas**; optimizar solo SH0, características nuevas y MLP.
- No implementar por ahora RBF, adaptación del octree, entrenamiento conjunto desde cero, nuevas pérdidas, redes por intersección ni normalización adicional de las características por opacidad.

## 3. Definición matemática exacta

**Parámetros de apariencia.** Cada vóxel activo `v` tiene ocho vectores nuevos de **4 canales**, uno por esquina local:

\[
\mathbf h_{v,j}\in\mathbb R^4,\qquad j=1,\dots,8.
\]

En esta **primera implementación son independientes por vóxel**, no vértices globales compartidos. Esto simplifica la integración con la topología congelada; no garantiza continuidad entre vóxeles. Los parámetros nuevos ocupan `features[V,8,4]` (= **32V** escalares), en vez de `features[V,8]` (= 8V) de la variante diferida constante.

**Consulta local por contribución rayo-vóxel.** Para el mismo punto representativo `x_{p,v}` ya usado en el cálculo de apariencia/densidad con una muestra por vóxel, obtener su posición local `q∈[0,1]^3`. No cambiar el muestreo de densidad. Calcular pesos trilineales originales `N_j(q)`, con `Σ_j N_j=1`:

\[
\mathbf z_{p,v}=\sum_{j=1}^{8}N_j(\mathbf q)\,\mathbf h_{v,j}\in\mathbb R^4.
\]

**Codificación armónica local (inspirada en NHT):**

\[
\boxed{\mathbf e_{p,v}=[\sin(\mathbf z_{p,v}),\cos(\mathbf z_{p,v})]\in\mathbb R^8.}
\]

Seno y coseno se aplican **después de interpolar las características de una intersección y antes de la composición del píxel**. No se evalúa ninguna MLP por intersección. No introducir de momento frecuencias o fases adicionales.

**Composición diferida.** Reutilizar exactamente los pesos existentes `w_{p,v}=T_{p,v} α_{p,v}`:

\[
\mathbf F_p=\sum_v w_{p,v}\mathbf e_{p,v},\qquad
A_p=\sum_vw_{p,v},
\]
\[
\mathbf B_p=\sum_vw_{p,v}\mathbf c_v^{SH0}+T_{p,\mathrm{final}}\mathbf c_{bg}.
\]

`B_p` incluye el fondo exactamente una vez. `F_p` no incluye características de fondo ni se divide por `A_p`.

**Decodificación, una sola MLP por píxel:** mantener Fourier-3 de la dirección normalizada cámara→rayo (18 canales) y el decodificador existente `29→32→32→3` con entrada de 8 latentes + 3 RGB base + 18 de dirección:

\[
\Delta\mathbf C_p=D_\theta([\mathbf F_p,\mathbf B_p,\gamma_3(\mathbf d_p)]),
\]
\[
\boxed{\mathbf C_p=\operatorname{clamp}_{[0,1]}(\mathbf B_p+A_p\Delta\mathbf C_p).}
\]

Inicializar **la última capa de la MLP a cero** para que el primer render reproduzca SH0; inicializar las características con valores pequeños y registrar su escala. Con salida nula, las características y capas anteriores no recibirán gradientes útiles en el primer backward, lo cual es esperado.

## 4. Cambios mínimos de implementación

1. Añadir parámetros `features[V,8,4]` y el modo `deferred_nht_trilinear`; conservar intacta la interfaz y la salida de los modos SH y `deferred_mlp`.
2. En la misma pasada CUDA que compone `B` y `A`, calcular la interpolación trilineal, `sin/cos` y acumular los ocho canales `F`. **No** usar una segunda pasada de rayos ni ejecutar una MLP dentro del rasterizador.
3. Implementar backward desde `F` a las ocho características de cada vóxel; mantener gradientes a SH0 y MLP. La geometría permanece congelada. No activar gradientes geométricos nuevos sin validación posterior.
4. Guardar/cargar modo, tensor de características, SH0, MLP, configuración, optimizador y referencia a geometría; permitir inferencia desde `render.py` y reanudación sin modificar el modo SH original.
5. Mantener el protocolo de resolución, cámara, fondo y temporización del experimento actual. Reportar tiempo de rasterización/codificación, tiempo de MLP, FPS y memoria pico. **Medir previamente la memoria**, pues 32V parámetros más gradientes y estados Adam puede ser costoso a ~2,94 M vóxeles.

## 5. Pruebas mínimas, antes de una corrida larga

- **Identidad SH0:** salida diferida = SH0 inicial (tolerancia float32) cuando la última capa es cero, incluido fondo y píxeles vacíos.
- **Forward local:** caso sintético con uno o dos vóxeles; verificar pesos `Σ N_j=1`, dimensión 8 del descriptor, `F`, `A` y `B`, y los valores sin/cos frente a PyTorch de referencia.
- **Backward:** diferencias finitas de unas pocas entradas de `features`, SH0 y MLP, evitando el clamp activo y comprobando que los latentes reciban gradiente después del primer paso. Valores iniciales orientativos: `atol=1e-4`, `rtol=1e-2` en float32, con pasos `1e-3` y `1e-2`.
- **Smoke train:** 10–30 pasos, geometría fija; pérdida finita, parámetros actualizados y sin NaN/Inf; controlar memoria antes de escalar.
- **Checkpoint y compatibilidad:** guardar/cargar y reanudar sin alterar el render; los modos SH y diferido original deben seguir funcionando.

## 6. Evaluación mínima y criterio de decisión

**No repetir como única evaluación cuatro vistas de entrenamiento.** La divergencia observada aconseja ampliar las vistas de entrenamiento, conservando un conjunto independiente de validación, y evaluar sin seleccionar vistas de validación para entrenar. Si por razones prácticas solo se dispone de cuatro vistas, describir el resultado explícitamente como *prueba de sobreajuste/capacidad*, no como mejora de generalización.

Comparar, con **el mismo checkpoint de geometría**, mismas vistas, mismos pasos, pérdida, resolución y presupuesto de entrenamiento:

1. `deferred_mlp` actual (8 canales constantes por vóxel).
2. **Interpolación trilineal sin sin/cos**, usando 8 canales acumulados (control para separar la interpolación de la codificación armónica; requiere 8 canales por esquina y cambia el presupuesto de parámetros, por lo que debe reportarse).
3. `deferred_nht_trilinear` propuesto (4 canales por esquina → 8 canales sin/cos).
4. SH0 y SH3 congelados como referencias de calidad, no como entrenamientos equivalentes.

Reportar PSNR/SSIM de entrenamiento **y** vistas reservadas, FPS, costo por etapa, VRAM, número de parámetros y, cuando corresponda, LPIPS. El principal criterio para continuar es **mejorar la calidad reservada sin un deterioro de rendimiento desproporcionado**. Si el PSNR reservado sigue estancado y el SSIM cae, revisar generalización, cantidad de cámaras y regularización antes de añadir RBF, más canales o redes mayores.

**Fuera de alcance inmediato:** RBF anisotrópicas, vértices globales compartidos, adaptación de topología y entrenamiento conjunto desde cero.
