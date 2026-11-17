# Integración mínima de Vertex-MLP conservando el entrenamiento original

Estado: plan revisado y acordado; no representa funcionalidades ya implementadas. Sustituye la versión anterior de este archivo.

## 1. Objetivo y momento de activación

Conservar el entrenamiento de SVRaster para NeRF Synthetic y sustituir únicamente la apariencia SH de grados superiores por Vertex-MLP.

El calendario comprobado en `train.py:178` es:

| Iteraciones | Original con preset Synthetic | Variante propuesta |
|---|---|---|
| 1–999 | SH0 | SH0, misma ruta original |
| 1000–1999 | SH1 | SH0 + residual Vertex-MLP |
| 2000–2999 | SH2 | SH0 + residual Vertex-MLP |
| Desde 3000 | SH3 | SH0 + residual Vertex-MLP |

La activación será antes del render de la iteración 1000, donde actualmente se incrementa SH0→SH1. No será después de completar 1000 pasos.

Vertex-MLP conservará arquitectura `H[P,8]`, Fourier-3 y MLP `26→64→3`. La última capa se inicializará a cero: la primera imagen de la nueva ruta deberá coincidir con SH0. Densidad y SH0 continuarán entrenándose.

## 2. Parámetros y comportamiento originales que se conservan

La configuración seguirá procediendo de `src/config.py`, `cfg/synthetic_nerf.yaml` y los overrides originales. No se duplicará con valores distintos en otro entrenador.

| Componente | Valor o comportamiento original conservado |
|---|---|
| Duración y semilla | 20.000 iteraciones; semilla 3721 |
| Inicialización | Rejilla de nivel 6; densidad −10; SH0 RGB inicial 0,5 |
| Datos y bounding | Lectores, particiones, resolución, bounding y filtros originales |
| Synthetic | Fondo blanco; outside_level=0; SH inicial grado 0 |
| Densidad y SH0 | LR 0.025 y 0.01 |
| Optimizador | SparseAdam, betas (0.1,0.99), epsilon 1e-15 |
| Scheduler | Multiplicar LR por 0.1 en iteración 19000 |
| Pérdida fotométrica | Respetar lambda_photo, MSE/L1/Huber y parámetros existentes |
| SSIM | Peso original 0.02 |
| Concentración RGB | lambda_R_concen=0.01, con activación original |
| Concentración T | lambda_T_concen=0.1 del preset Synthetic |
| Distorsión | lambda_dist=0.1 desde iteración 10000 |
| TV de densidad | 1e-10, calendario original |
| Otros regularizadores | Configuración, condiciones y calendarios existentes |
| Supersampling | ss=1 hasta incluir iteración 1000; después augmentación original |
| Augmentación SS predeterminada | Uniforme entre 1 y ss_aug_max=1.5 |
| Supersampling de evaluación | Valor original del modelo: 1,5 |
| Adaptación | Desde 1000, cada 1000; condiciones de finalización originales |
| Poda | Hasta 18000; umbral de 0.0001 a 0.05 |
| Subdivisión | Hasta 15000; proporción 0.05; máximo original 10.000.000 |
| Preset Synthetic | Subdividir todos al primer evento; umbral de muestreo 0.5 |
| Cámaras | Misma secuencia original de épocas barajadas |
| Estadísticas y evaluación | Mismos puntos del bucle, métricas, informes y opciones |
| Optimizador tras adaptación | Recrearlo después de poda/subdivisión, como el original |

Conservar también `sche_mult` con su interpretación actual. No alterar valores automáticamente para ajustarlos a VRAM.

Correcciones al plan anterior: eliminar límite nuevo de 2 millones, supersampling fijo durante entrenamiento, nuevos valores de Adam y desactivación de regularizadores.

## 3. Cambios estrictamente necesarios para Vertex-MLP

### Integración en el entrenador existente

- Añadir un modo de apariencia optativo al flujo de train.py; el predeterminado seguirá siendo SH.
- Reutilizar el mismo bucle, configuración, pérdidas, inicialización y adaptación.
- Sustituir el primer evento que activaría SH angular por la activación neuronal. No ejecutar después la progresión SH1→SH2→SH3 en esta modalidad.
- Antes de activar, no introducir operaciones aleatorias nuevas que alteren el warmup original.
- Mantener intacto el entrenamiento SH cuando no se seleccione Vertex-MLP.

### Parámetros nuevos y correspondencia con los originales

H y pesos MLP son parámetros nuevos inevitables. No introducir nuevas tasas o políticas:

| Original | Variante |
|---|---|
| Densidad | Mismo parámetro, LR y optimizador |
| SH0 | Mismo parámetro, LR y optimizador |
| SH de grados superiores | Sustituidos por H y MLP |
| shs_lr=0.00025 | Aplicado a H y MLP |
| Betas, epsilon y scheduler de SHs | Aplicados a los parámetros sustitutos |

No usar por defecto las tasas 0.01/0.001 ni Adam (0.9,0.99) de la variante congelada.

### Render y backward

- Mantener ruta congelada existente y añadir ruta conjunta explícita.
- En entrenamiento conjunto, propagar gradientes hacia densidad, SH0 y prioridad de subdivisión.
- Corregir máscara del gradiente RGB base del kernel; conservar activación SH0 original.
- Soportar concentración RGB, distorsión y demás opciones CUDA activadas por el entrenador original. No resolver incompatibilidades desactivando sus pesos.
- Calcular concentración RGB con el color interpolado de cada intersección, aplicando gradientes a apariencia y términos geométricos correspondientes.
- Conservar rayos, pesos de densidad, composición, profundidad, normales, fondos y clamp final.

### Adaptación

- Mantener máscaras, umbrales, orden de operaciones y criterios originales de poda/subdivisión.
- Reutilizar sin redefinir el remapeo geométrico y de SH0 existente.
- Añadir solamente remapeo de H: conservar vértices supervivientes por clave e interpolar características nuevas; resolver contribuciones compartidas por agregación.
- Conservar pesos aprendidos de la MLP.
- Recrear optimizador y restaurar scheduler en el mismo punto que el original. No preservar momentos de densidad/SH0 ni introducir una política especial de conservación de momentos neuronales.
- Invalidar buffers derivados afectados por el nuevo layout.

La interpolación de H inicializa nuevos parámetros; por la no linealidad de la MLP no garantiza conservar exactamente el campo RGB tras subdividir. Documentar y medir esa diferencia.

## 4. Persistencia, visor y documentación de diferencias

- Guardar conjuntamente geometría, apariencia, estado de activación, paso, optimizador, scheduler y estados aleatorios.
- Reanudar respetando si Vertex-MLP ya estaba activo, sin repetir transición ni iteración completada.
- Conservar lectura del formato congelado v2.
- Crear visor separado viz_vertex_fourier3.py, con carga de ambos formatos, navegación y comparación Vertex-MLP/SH0.
- Ofrecer SH3 únicamente en checkpoints que realmente lo contengan; no presentarlo como entrenado en el nuevo modo.
- Mantener RGB, alfa, profundidad, normales y descarga de imágenes.

Al implementar, actualizar la auditoría con una matriz explícita:

1. Original conservado: configuración, símbolo y ubicación.
2. Original sustituido: progresión SH angular y grupo de parámetros correspondiente.
3. Nuevo necesario: decoder, remapeo de H, backward, persistencia y visor.
4. Diferencia matemática: apariencia trilineal por intersección y regularización aplicada al nuevo color.

No describir como idéntico el entrenamiento después de activar Vertex-MLP: se conservan protocolo e hiperparámetros, pero cambian representación, gradientes y evolución de geometría.

## 5. Validación y commits

- Comparar configuraciones resueltas y comprobar que ninguna clave original cambia al seleccionar la variante.
- Verificar warmup original: cámaras, supersampling, pérdidas y actualizaciones hasta iteración 999.
- Probar frontera 999→1000→1001: transición exacta, residual inicial cero y propagación posterior hacia H/MLP.
- Verificar que el modo original sigue activando SH1/SH2/SH3 en sus iteraciones habituales.
- Comprobar betas, epsilon, tasas, scheduler y recreación de optimizador después de adaptar.
- Añadir diferencias finitas de densidad, SH0 y apariencia, incluyendo concentración RGB y distorsión.
- Probar remapeo de características y reanudación antes/después de activación y adaptación.
- Mantener las 17 pruebas existentes; añadir pruebas del visor y checkpoints.
- Ejecutar smoke que cruce activación y al menos un evento adaptativo; después comparar corrida completa contra baseline con exactamente la misma configuración original.
- Crear commits por hitos y documentar cualquier diferencia adicional antes de introducirla.
