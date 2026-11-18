# Decodificación Vertex-MLP según el frustum

En `GaussianApp`, el rasterizador ya calculaba `in_frusts_idx` mediante `n_duplicates > 0` y lo entregaba al callback. La ruta anterior calculaba los RGB neuronales de todos los vértices globales y reunía las esquinas de todos los vóxeles antes del preprocesamiento.

El callback ahora selecciona `vox_key[in_frusts_idx]`, deduplica sus índices globales y usa la correspondencia inversa de `torch.unique` para reconstruir las ocho esquinas en su orden original. `VertexAppearance.forward_subset` calcula Fourier-3 y la MLP solo sobre esas filas. `VertexRGB.forward_subset` indexa sus parámetros directamente. Cuando todos los vóxeles son candidatos, se conserva el cálculo denso y se omite la deduplicación. Los módulos que ofrecen solo la interfaz original también conservan la ruta densa.

El buffer contiguo `float32 [V,27]` conserva las filas globales que esperan los kernels. La ruta compacta rellena sus filas candidatas mediante `index_copy_` diferenciable. Una cámara sin candidatos usa cero filas del decoder. La reducción se rige por los candidatos del preprocesamiento y puede incluir vértices de vóxeles ocluidos.

El perfil registra por separado preprocesamiento, selección, decodificación, empaquetado y rasterización. En la ruta compacta, el intervalo de empaquetado incluye la preparación de SH0. El benchmark aislado del decoder sigue midiendo el layout completo como referencia. Las firmas CUDA y la estructura de los checkpoints se conservan.

## Validación

Pasaron 23 pruebas CUDA de Vertex-MLP, entrenamiento conjunto, checkpoint y visor. Las nuevas pruebas contrastaron salida neuronal no nula y gradientes de parámetros frente a la ruta densa, con frustum total, parcial y vacío. Verificaron color, profundidad, normales y transmitancia con `atol=1e-5`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_vertex_cuda.py tests/test_vertex_joint.py \
  tests/test_vertex_checkpoint.py tests/test_vertex_viewer.py
```

## Lego, checkpoint 20.000

RTX 3060, 256×256, supersampling 1. Se usaron las primeras ocho vistas de test por nombre y dos encuadres de diagnóstico de la primera cámara, con su FOV multiplicado por 0,35 y 0,55. Cada modo tuvo tres renders de calentamiento por cámara y doce renders medidos con sincronización CUDA. La comparación llamó al método denso original de `f43385b` con el mismo rasterizador y el mismo checkpoint. En las diez cámaras, el máximo error absoluto RGB fue cero. Profundidad, normales y transmitancia coincidieron con `atol=1e-5`. El visor Viser también renderizó un fotograma del checkpoint real.

| Vistas | Vértices decodificados, antes → después | ms por fotograma, antes → después | FPS, antes → después | Pico CUDA, antes → después (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Ocho vistas originales, promedio | 6.481.767 → 6.481.767 | 101,04 → 101,39 | 9,90 → 9,86 | 5.685 → 5.807 |
| Recorte con FOV ×0,35 | 6.481.767 → 3.192.958 | 85,60 → 67,52 | 11,68 → 14,81 | Ver JSON |
| Recorte con FOV ×0,55 | 6.481.767 → 4.572.441 | 87,45 → 98,73 | 11,44 → 10,13 | 5.685 → 5.362 |
| Dos recortes, promedio de tiempo | — | 86,53 → 83,13 | 11,56 → 12,03 | 5.685 → 5.362 |

Los ocho encuadres originales incluyeron los 3.979.317 vóxeles y usaron la ruta densa. En los dos recortes, el decoder bajó de 69,21 a 43,09 ms en promedio y la selección añadió 22,37 ms. Con un recorte moderado, la selección y el empaquetado costaron más que el ahorro de decodificación. Las cifras por cámara y etapa, y el script reproducible del benchmark, están en `output/vertex-visible-lego/`.
