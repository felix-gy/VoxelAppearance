# Plan mínimo: integración de RBF de 4 canales en SVRaster

**Estado:** especificación de implementación, no funcionalidad ya implementada.

## 1. Objetivo y alcance

Integrar una **codificación de apariencia mediante RBF de cuatro canales**, inspirada en NeuRBF, en SVRaster. La codificación radial se construirá **antes de la MLP** y se evaluará en el punto de muestra de cada intersección rayo–vóxel.

**No se realizará una comparación con Vertex-MLP trilineal** en este trabajo de integración. La meta inmediata es obtener una variante RBF funcional y verificable.

Se conservan la geometría, el campo de densidad trilineal, el SH0 por vóxel, el muestreo y la composición volumétrica del SVRaster existente. No se incorporan centros móviles, matrices completas, RBF para densidad ni cambios en la política de poda/subdivisión.

## 2. Bases espaciales y parámetros

Para cada **vértice único** `i` de SVRaster:

- Centro fijo `p_i`: coordenada espacial del vértice.
- Vector aprendible `a_i ∈ R⁴`: contenido de los cuatro canales RBF (reemplaza, en esta variante, al latente de 8 canales de Vertex-MLP; no se decodifican RGB por vértice).
- Log-escalas aprendibles `η_i ∈ R³`: control anisotrópico **diagonal**, sin rotación.

Usaremos una RBF *inverse quadratic*. En coordenadas locales de cada vóxel `t ∈ [0,1]³` y con las esquinas `q_i ∈ {0,1}³`:

\[
D_i(t)=\sum_{k\in\{x,y,z\}}(t_k-q_{ik})^2\exp(-2\eta_{ik}),
\qquad
\phi_i(t)=\frac{1}{1+D_i(t)}.
\]

**Importante:** las log-escalas son compartidas por vértice, pero su interpretación espacial es relativa al tamaño de cada vóxel. Esta versión no representa una única RBF de escala física global en regiones de distinto nivel del octree.

## 3. Codificación RBF de cuatro canales

Para cada consulta, utilizar las ocho RBF de las esquinas del vóxel interceptado. **No multiplicar por pesos trilineales** en la rama de apariencia.

Normalización:

\[
\alpha_i(t)=\frac{\phi_i(t)}{\sum_{j=1}^{8}\phi_j(t)}.
\]

Composición sinusoidal multicanal, inspirada en NeuRBF:

\[
\mathbf b_i(t)=\sin\big(\alpha_i(t)\mathbf m+\boldsymbol\beta\big)\in\mathbb R^4,
\qquad
\mathbf f(t)=\sum_{i=1}^{8}\big(\mathbf a_i\odot\mathbf b_i(t)\big)\in\mathbb R^4.
\]

- `m ∈ R⁴`: frecuencias **globales fijas**; usar inicialmente `[1, 2, 4, 8]` (valor de implementación inicial, no atribuirlo al paper como configuración universal).
- `β ∈ R⁴`: fases **globales aprendibles**, inicializadas a cero.
- `a_i`: cuatro coeficientes aprendibles por vértice.

Aquí **«cuatro canales RBF» significa que el codificador produce un vector espacial de dimensión 4**; no significa cuatro centros por vértice. Existen ocho centros por vóxel, uno en cada esquina.

## 4. Decodificación y renderizado

Para el punto de muestra `x` y la dirección de vista `d` usada por el renderer:

\[
\Delta\mathbf c(x,d)=F_\theta\big([\mathbf f(t),\gamma_3(d)]\big),
\quad F_\theta:\mathbb R^{22}\to\mathbb R^3.
\]

- Mantener codificación direccional **Fourier-3 de 18 dimensiones**.
- MLP compartida **22 → 64 → 3**, con una capa oculta y activación igual a la versión actual cuando sea compatible.
- Inicializar su última capa a cero para empezar con residual RGB nulo.
- Color de la muestra: `c = max(0, c_SH0 + Δc)` con la misma convención de SH0 y activación que el renderer actual.
- Conservar densidad, alfa, profundidad, fondos y composición originales.



## 5. Cambios mínimos necesarios

1. Añadir los parámetros `rbf_features[P,4]`, `rbf_log_scales[P,3]`, `rbf_phase[4]` y la MLP `22→64→3`.
2. Implementar la obtención de ocho bases, normalización, codificación multicanal y decodificación en la ruta de renderizado. Evitar materializar un tensor global de todas las intersecciones si excede VRAM; implementar la evaluación en lotes o una ruta CUDA compatible.
3. Propagar gradientes hacia características, escalas, fases y MLP, además de mantener los de densidad y SH0. Respetar la derivada del clamp RGB y los regularizadores activos.
4. Guardar y cargar los nuevos parámetros con un identificador de modo de apariencia independiente.
5. En poda/subdivisión, preservar parámetros de vértices supervivientes y **inicializar en cero las log-escalas nuevas**; definir un remapeo consistente para las cuatro características nuevas. No modificar los criterios de adaptación originales.
6. Habilitar la ruta de evaluación y visualización para que utilice el mismo modelo RBF que el entrenamiento.

## 6. Pruebas mínimas

- **Kernel y normalización:** `φ_i > 0`, `sum_i α_i ≈ 1`; sin NaN/Inf, incluidas esquinas y caras.
- **Dimensiones y estado inicial:** salida del codificador `[4]`, entrada MLP `[22]`, salida RGB `[3]`; residual inicial cero.
- **Backward:** gradientes finitos y diferencias finitas sobre al menos una característica RBF, una log-escala y una fase; comprobar que densidad y SH0 siguen recibiendo gradientes.
- **Entrenamiento corto:** sobreajustar 1–2 vistas con geometría congelada y confirmar descenso de pérdida.
- **Persistencia:** guardar/cargar y reproducir la misma imagen dentro de tolerancia numérica.
- **Adaptación (si se habilita):** cruzar al menos un evento de subdivisión sin romper índices de vértices ni introducir NaN.

## 7. Criterio de aceptación

La integración se considera funcional cuando renderiza y entrena con el codificador RBF de cuatro canales, pasa las pruebas anteriores y reporta **PSNR, FPS, VRAM y tiempo por render** de su propia ejecución. **No forma parte de esta etapa compararla con Vertex-MLP trilineal.**

