# Integración Vertex-MLP Fourier-3 trilineal

## Protocolo

Base: `out/materials_t/checkpoints/iter020000_model.pt`; configuración de
`out/materials_t/config.yaml`. MSE + 0.02 SSIM, ss=1.5, fondo blanco,
semilla 3721. Geometría fija. El formato base permanece independiente.

## Estado y trazabilidad

Implementación en curso. No se han validado todavía kernels, entrenamiento ni
rendimiento de la variante. No interpretar los hiperparámetros como resultados.

## Cambios

- `src/vertex_appearance/decoder.py`: Fourier de tres bandas y MLP compartida
  26→64→3; última capa cero. Incluye control RGB por vértice.
- Los buffers del rasterizador permanecen globales; el orden de esquinas es
  000,001,010,011,100,101,110,111.

## Implicaciones

4.338.721 vértices, 2.937.583 vóxeles. Gather RGB: aproximadamente 269 MiB.
Las activaciones y Adam requieren memoria adicional. CUDA debe compilarse desde
este checkout: el entorno Conda svraster apuntaba inicialmente a otro proyecto.

## Validación inicial

Compilación con Conda `svraster`, PyTorch 2.5.0/cu124 y toolkit 12.5.
Paquete y binario cargados desde este checkout. `python -m pytest tests -q`:
5 pruebas aprobadas (Fourier, inicialización, regresión SH0, invariancia de
geometría, diferencias finitas y overfit sintético).

## Cambios CUDA

Se añaden `vertex_forward` y `vertex_backward` como especializaciones separadas
basadas en los kernels originales; no se modifican sus implementaciones SH.
El transporte interno empaqueta base RGB y 8 residuales en `[V,27]`. El buffer
backward usa 36 valores (8 geometría, 3 base, 1 prioridad, 24 residuales).
Esta desviación del buffer independiente propuesto preserva la interfaz de
retornos existente, pero duplica código CUDA que debe mantenerse sincronizado.
La MLP no se ejecuta en CUDA personalizado y se usa el mismo `interp_w` de densidad.

## Uso

```bash
conda activate svraster
CUDA_HOME=/usr/local/cuda-12.5 MAX_JOBS=2 python -m pip install -e cuda/ --no-build-isolation
python -m pytest tests -q
python train_vertex_fourier3.py --output output/vertex_materials --steps 5000
python render_vertex_fourier3.py --output output/vertex_materials/eval --appearance-checkpoint output/vertex_materials/checkpoints/iter005000_appearance.pt
```

El control usa `--mode vertex_rgb_trilinear`. La ruta original sigue disponible
mediante `render.py`; los checkpoints geométricos no se reescriben. No hay push.

## Entrenamiento y persistencia implementados

- `runtime.py`: configuración YAML específica, carga del protocolo base,
  congelación, Adam, selección reproducible de cámaras, evaluación y medición.
  Se usa un diccionario YAML en lugar de ampliar el estado global YACS, para
  aislar completamente el experimento del entrenamiento adaptativo original.
- `checkpoint.py`: SHA-256 del archivo base, hash de layout/orden/coordenadas,
  estado neuronal, Adam y estados aleatorios CPU/CUDA/cámaras. La carga rechaza
  identidades incompatibles y permite reubicar el checkpoint base.
- Scripts raíz separados para entrenar y renderizar; control con RGB libre.
- El instrumento de tiempos distingue raster (incluye preprocesamiento y
  empaquetado SH0), posprocesamiento, apariencia y gather. Fourier y MLP se
  miden además aisladamente. El total siempre incluye todas esas operaciones.

Prueba real de dos pasos: 200 vistas test evaluadas, PSNR 23.48459,
SSIM 0.900107, LPIPS 0.077448 y LPIPS corregido 0.091256.
Pico asignado durante esta prueba: 5.470.391.296 bytes. Estos valores son una
prueba de funcionamiento, no el resultado final del experimento.

Se produjo OOM al solapar accidentalmente evaluación y debug en dos procesos;
se repitió el debug con GPU libre. No se cambió precisión ni algoritmo.
