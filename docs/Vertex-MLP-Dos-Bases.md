# Vertex-MLP con dos bases RGB

La rama `Vertex-MLP-Dos-Bases` parte de `GaussianApp` en `5d13250`. El modo de apariencia congelada `vertex_two_bases` usa una MLP `26→64→6` compartida y dos coeficientes libres por vóxel. Mantiene ocho latentes por vértice global y la dirección cámara→vértice Fourier-3. Los coeficientes combinan las bases por esquina antes de la interpolación trilineal.

```text
bases = MLP([H, Fourier3(d)]).reshape(P, 2, 3)
delta[v,j] = mix[v,0] * bases[key[v,j],0] + mix[v,1] * bases[key[v,j],1]
rgb = max(0, SH0 + Σ_j peso_trilineal_j * delta[v,j])
```

La ampliación copia los latentes y la MLP de una base a los primeros tres canales. La segunda salida tiene pesos con desviación estándar 0,01 y sesgo cero. La mezcla empieza en (1,0), por lo que el render inicial coincide. El primer backward puede entrenar el segundo coeficiente y la segunda salida adquiere gradiente después de activarse la mezcla. Una base compartida da un residual continuo entre caras del mismo nivel que comparten vértices. La mezcla independiente por vóxel puede introducir saltos incluso en esas caras.

## Cambios de código

| Archivo | Cambio y razón |
| --- | --- |
| `src/vertex_appearance/decoder.py`, `__init__.py` | Añade y exporta `VertexTwoBases`, la inicialización desde una base y la mezcla. Reúne una base por vez y evita un intermedio único `[V,8,2,3]`. Autograd conserva los valores requeridos para derivar los coeficientes. |
| `src/sparse_voxel_gears/renderer.py` | Añade la combinación opcional de residuales en PyTorch y permite fijar la decodificación densa para la comparación. Preserva el buffer `float32[V,27]` y reutiliza los kernels existentes sin cambios. |
| `src/vertex_appearance/checkpoint.py`, `runtime.py` | Valida el modo y el layout, publica atómicamente los checkpoints v2 y permite conservar el SH0 de referencia. Reutiliza el bucle congelado, extraído a `fit`, y registra un grupo de coeficientes en el entrenador ordinario. La configuración de comparación mantiene cámara, pérdida y optimizador al reanudar. |
| `viz_vertex_fourier3.py` | Reconstruye el decoder según el modo y carga el SH0 congelado del experimento. |
| `scripts/compare_vertex_two_bases.py` | Reproduce los dos brazos, valida el warm-start, mide métricas, tiempos, memoria y saltos de residual, y crea comparativas visuales. |
| `tests/test_vertex_two_bases.py` | Comprueba forma, identidad inicial, gradientes de las dos bases, descenso de pérdida, compatibilidad y recarga. |

## Protocolo

Materials T comienza desde el Vertex-MLP de una base al paso 300 de la semilla 3721 sobre SH 20.000. Ambos brazos congelan su SH0 guardado. SH0 original y SH3 se evalúan antes de sustituir SH0. Lego usa SH 20.000 de `out/lego-original`; ambos brazos empiezan desde la misma inicialización de una base. En cada escena se congelan densidad, geometría y topología. Cada brazo recorre 32 cámaras de entrenamiento en ciclo y se evalúa en cuatro reservadas del mismo split, a 256×256, con fondo blanco, supersampling 1 y una muestra de densidad. Adam empieza sin momentos, con LR 0,00025, betas (0,1, 0,99) y epsilon 1e−15. La pérdida es MSE.

```bash
export PYTHONPATH="$PWD:$PWD/cuda${PYTHONPATH:+:$PYTHONPATH}"
python scripts/compare_vertex_two_bases.py materials_t --output output/vertex-two-bases/materials_t
python scripts/compare_vertex_two_bases.py lego --output output/vertex-two-bases/lego
python viz_vertex_fourier3.py --checkpoint output/vertex-two-bases/lego/two-step300.pt
```

## Resultados

RTX 3060. El tiempo se midió en 32 fotogramas reservados por brazo, tras calentamiento, con sincronización CUDA por fotograma. La etapa de preparación incluye Fourier y MLP. Reunión/mezcla incluye el empaquetado y la preparación SH0. LPIPS-VGG se evaluó sobre entradas normalizadas a [-1,1].

| Escena | Modo | PSNR (dB) | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| materials_t | sh0 | 22.4954 | 0.881316 | 0.113880 |
| materials_t | sh0_frozen | 22.5413 | 0.867706 | 0.153418 |
| materials_t | sh3 | 26.1609 | 0.948318 | 0.067560 |
| materials_t | one | 23.0103 | 0.868850 | 0.153395 |
| materials_t | two | 23.0335 | 0.868138 | 0.155281 |
| lego | sh0 | 26.5590 | 0.949486 | 0.051542 |
| lego | sh3 | 27.7887 | 0.962576 | 0.039621 |
| lego | one | 26.8659 | 0.951364 | 0.050105 |
| lego | two | 26.9991 | 0.953573 | 0.048777 |

| Escena | Bases | Parámetros | FPS | ms decoder | ms reunión/mezcla | ms raster | MiB inf. | MiB paso 15 | MiB entren. | RMS de salto RGB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| materials_t | 1 | 34711691 | 15.90 | 43.47 | 5.68 | 13.13 | 4016 | 5204 | 5204 | 0 |
| materials_t | 2 | 40587052 | 14.35 | 42.93 | 12.92 | 13.18 | 4040 | 5295 | 5295 | 0.00858 |
| lego | 1 | 53378923 | 10.00 | 67.33 | 8.76 | 22.93 | 6021 | 7853 | 7853 | 0 |
| lego | 2 | 61943762 | 9.14 | 66.24 | 19.75 | 22.55 | 6055 | 7984 | 7984 | 0.0167 |

Se midió el residual RGB en centros de caras del mismo nivel usando una cámara reservada. El cálculo compara campos antes de sumar SH0 y antes del clamp. Las métricas por vista y los máximos de salto se guardan en los JSON.

- **materials_t.** Diferencias dos menos una base: +0.0232 dB, -0.000712 SSIM y +0.001886 LPIPS. El error RGB inicial máximo fue 0. Se compararon 3072 caras.
- **lego.** Diferencias dos menos una base: +0.1332 dB, +0.002209 SSIM y -0.001328 LPIPS. El error RGB inicial máximo fue 0. Se compararon 3072 caras.

Ambos checkpoints finales de cada escena se recargaron y reprodujeron el render de la primera vista reservada con tolerancia 1e−5. Se ejecutaron 24 pruebas CUDA, incluidas las regresiones de los modos previos. El experimento usa una sola semilla y cuatro vistas reservadas. El directorio `output/vertex-two-bases/` contiene la configuración, los cuatro checkpoints, los resultados por cámara y las imágenes comparativas.

En Materials T, el aumento de PSNR fue pequeño y SSIM y LPIPS empeoraron. En Lego, las tres métricas mejoraron frente a una base a los 300 pasos. SH3 conservó mayor PSNR y SSIM y menor LPIPS en ambas escenas. La mezcla añadió latencia y produjo saltos de residual donde el campo de una sola base era continuo. Estos datos miden magnitud de salto en RGB lineal por cara y no establecen por sí solos que el artefacto sea visible en la imagen final. El lector de Viser cargó los dos checkpoints de dos bases y renderizó `r_0.png` a 256×256.
