# Plan mínimo: SVRaster con RBF adaptativas y MLP diferida

**Estado:** propuesta experimental; no implementado. **Base:** rama `Deferred-NHT-Trilineal` (commit `38b0471`), reutilizando su rasterizador de características, codificación armónica y MLP diferida. **Alcance:** apariencia únicamente, con geometría y topología congeladas.

## 1. Objetivo y límites

Comprobar si una **combinación radial adaptativa** de características locales mejora la apariencia de SVRaster frente a la interpolación trilineal, sin ejecutar una MLP por intersección. Inspiración en **NeuRBF (ICCV 2023)**: kernel *inverse quadratic*, normalización de bases y formas espaciales adaptables. Se mantiene la codificación `sin/cos` de la variante NHT existente para aislar el efecto de sustituir los pesos espaciales; **no** se implementa todavía la composición sinusoidal multicanal de NeuRBF sobre cada RBF.

No modificar densidad, intersecciones, transmisión/alfa, SH0, cámaras, fondo, poda, subdivisión ni entrenamiento de geometría. SH sigue como modo predeterminado. No usar Vertex-MLP ni RBF seguida de MLP por intersección.

## 2. Representación y ecuaciones

Para cada vóxel activo `v`, almacenar **ocho vectores de 4 canales** `h[v,j] ∈ R⁴` y **ocho log-escalas anisotrópicas diagonales** `η[v,j] ∈ R³`, `j=1,…,8`. En esta primera prueba, los parámetros son **por esquina local del vóxel**, sin compartirlos entre vóxeles vecinos; el objetivo es reutilizar el layout de la variante NHT. Los centros son las ocho esquinas **fijas** del vóxel, no centros móviles.

En el punto de consulta `x*` ya usado para apariencia durante la intersección rayo-vóxel, calcular coordenadas locales `t ∈ [0,1]³`; cada esquina tiene `q_j ∈ {0,1}³`. Definir:

\[
D_j(t)=\sum_{a\in\{x,y,z\}}(t_a-q_{ja})^2\exp(-2\eta_{ja}),
\qquad \phi_j(t)=\frac{1}{1+D_j(t)}.
\]

Normalizar **solo las RBF**, sin multiplicar por pesos trilineales:

\[
\alpha_j(t)=\frac{\phi_j(t)}{\sum_{k=1}^{8}\phi_k(t)},
\qquad f_v(t)=\sum_{j=1}^{8}\alpha_j(t)h[v,j]\in\mathbb R^4.
\]

Mantener la codificación armónica **después de agregar características**, tal como en la variante NHT existente:

\[
e_v(t)=[\sin(f_v(t)),\cos(f_v(t))]\in\mathbb R^8.
\]

La composición diferida permanece:

\[
F_p=\sum_{v\in\mathcal R_p}T_{p,v}\,\alpha^{\mathrm{vol}}_{p,v}\,e_v(t_{p,v}),
\qquad A_p=\sum_v T_{p,v}\alpha^{\mathrm{vol}}_{p,v}.
\]

Aquí **`α_j` es el peso RBF local** y **`α^{vol}` es la opacidad volumétrica original**; no confundirlos. Conservar exactamente la fórmula de color base SH0, fondo y residual diferido de la implementación actual, incluida su máscara/factor de opacidad, con la **misma MLP `29 → 32 → 32 → 3` evaluada una vez por píxel** y dirección Fourier-3.

**Restricción relevante:** la combinación RBF pura no reproduce la interpolación trilineal cuando `η=0`, no interpola exactamente el valor de una esquina y no garantiza continuidad en caras compartidas. Son propiedades del diseño, no fallos de implementación.

## 3. Inicialización y entrenamiento

- Inicializar `η = 0` (escalas unitarias respecto al tamaño local del vóxel), usar `s = exp(η)` y restringir `η` al intervalo `[-2,2]` mediante proyección consistente tras la actualización.
- Reutilizar la inicialización de características, MLP y fases/frecuencias **ya existente** en NHT; no introducir nuevos cambios de capacidad simultáneos.
- Cargar el checkpoint SH de 20 000 pasos para geometría; congelar densidad, topología, geometría y SH superiores. Optimizar SH0, características, escalas RBF y MLP.
- Conservar el protocolo de Materials T: 32 vistas de entrenamiento, 4 reservadas, 256×256, fondo blanco, `ss=1`, una muestra de densidad por vóxel, MSE y semillas `3721` y `3722`. Mantener `lr_SH0=0.01`, `lr_features=lr_MLP=0.00025`, Adam `betas=(0.1,0.99)`, `eps=1e-15`. **Nuevo hiperparámetro explícito:** `lr_RBF=0.00025` inicialmente, sin afirmar que sea óptimo.
- Evaluar en pasos **30, 100 y 300**; no entrenar escenas completas ni ampliar el programa experimental en esta etapa.
- No introducir pérdida adicional de regularización en la primera ejecución; registrar distribución y saturación de `η` para decidir después si hace falta.

## 4. Integración mínima

1. Crear modo opcional `deferred_neurbf_iq` sin afectar `sh`, `deferred_mlp`, `deferred_trilinear` ni `deferred_nht_trilinear`.
2. Reutilizar los parámetros de cuatro características por esquina de NHT; añadir `[V,8,3]` log-escalas entrenables y su persistencia.
3. En **forward CUDA**, sustituir únicamente los pesos trilineales de apariencia por los ocho pesos RBF normalizados; conservar la codificación `sin/cos`, la acumulación volumétrica y la MLP diferida.
4. En **backward CUDA**, propagar gradientes hacia características y log-escalas, incluyendo la derivada de la normalización RBF, así como los gradientes existentes hacia SH0 y el decodificador. La geometría permanece congelada.
5. Guardar/cargar modo, parámetros RBF, optimizadores, iteración y metadatos necesarios. `render.py` debe reconocer el nuevo modo. Preservar los formatos anteriores.
6. No adaptar poda/subdivisión ni implementar centros móviles en esta fase.

**Presupuesto de memoria:** el nuevo modo añade `24V` parámetros de escala a los `32V` latentes de NHT, sin contar gradientes/Adam. Para `V=2 937 583`, las escalas solas ocupan aproximadamente **282 MB** en float32 decimal, y aproximadamente cuatro veces ese tamaño si se cuentan parámetros, gradientes y los dos momentos Adam. Supervisar VRAM antes de una ejecución larga.

## 5. Pruebas mínimas

- **Forward:** ocho pesos RBF positivos, finitos y cuya suma es 1, en centro, esquina y cara del vóxel; salida con forma correcta.
- **Gradientes:** diferencias finitas en un ejemplo pequeño para `h` y `η` (lejos de límites de proyección y clamps); verificar actualización no nula de las escalas cuando exista señal.
- **Regresión:** `sh` y los modos diferidos existentes producen los mismos resultados con sus checkpoints previos.
- **Persistencia:** guardar, cargar y reanudar 1 paso reproduce resultados dentro de tolerancia float32.
- **Smoke test:** 10–30 pasos con cámara/escena pequeña, sin `NaN`, parámetros acotados y densidad/topología congeladas.

Usar como criterio inicial `atol=1e-5, rtol=1e-5` para persistencia y `atol=1e-4, rtol=1e-2` para gradientes float32, con perturbaciones `1e-3` y `1e-2`; ajustar si el análisis numérico lo justifica.

## 6. Evaluación acotada y decisión

Medir PSNR y SSIM por vista y promedio en las cuatro vistas reservadas, PSNR de entrenamiento, FPS sincronizados, tiempo de rasterizador/MLP, memoria pico y distribución de log-escalas. Utilizar los resultados **ya registrados** de NHT trilineal y diferido base como contexto bajo el mismo protocolo; no comparar con Vertex-MLP. Repetir la nueva variante con ambas semillas antes de atribuir significado a cambios pequeños de PSNR.

**Criterio de continuación:** solo estudiar RBF con centros móviles, covarianza completa, regularización, parámetros compartidos entre vóxeles o MSC por RBF si esta versión muestra una mejora reproducible de calidad que compense la memoria y el tiempo. No confundir una adaptación de NeuRBF con una reproducción completa del paper.

## Referencias

- NeuRBF: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_NeuRBF_A_Neural_Fields_Representation_with_Adaptive_Radial_Basis_Functions_ICCV_2023_paper.html
- Implementación oficial: https://github.com/oppo-us-research/NeuRBF
