# Vertex-MLP trilineal en Materials T

## Protocolo

Se ejecutó la rama `feature/vertex-mlp-rbf-iq` en `6ec06e2`, en un worktree separado con su propia extensión CUDA. Se usó su Vertex-MLP trilineal, con ocho características por vértice global, dirección cámara→vértice Fourier-3 en radianes con multiplicador π y MLP 26→64→3. La red genera un residual por vértice que se interpola en el vóxel, antes de la composición volumétrica. El decoder conserva su inicialización de salida cero.

El script de comparación empleó el renderer de la rama y adaptó el entrenamiento a las condiciones de la prueba diferida. Usó SH0 entrenable, densidad y topología congeladas, checkpoint SH 20.000 de Materials T, 32 cámaras de entrenamiento en ciclo y las mismas cuatro reservadas, 256×256, `ss=1`, fondo blanco y una muestra de densidad. Adam usó MSE sin regularizadores, LR 0,01 para SH0 y 0,00025 para características y decoder, betas (0,1, 0,99) y epsilon 1e−15. Semillas 3721 y 3722, 300 actualizaciones. Se comprobó identidad inicial con SH0, gradientes de SH0 y latentes, igualdad de los tensores congelados y contadores de Adam. Los promedios SH0 y SH3 coincidieron con los registros comparables dentro de 1e−4 dB. Los presupuestos de parámetros y las arquitecturas difieren entre métodos.

## Resultados

| Semilla | Paso | PSNR train | SSIM train | PSNR reservado | SSIM reservado | Δ PSNR frente a base | Δ PSNR frente a NHT | Δ PSNR frente a IQ × trilineal armónico |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3721 | 30 | 25.091 | 0.92867 | 22.740 | 0.88230 | +0.093 | +0.089 | +0.089 |
| 3721 | 100 | 26.325 | 0.94569 | 22.897 | 0.87996 | +0.056 | +0.046 | +0.046 |
| 3721 | 300 | 29.073 | 0.96752 | 22.968 | 0.86862 | -0.165 | -0.190 | -0.191 |
| 3722 | 30 | 25.094 | 0.92874 | 22.730 | 0.88232 | +0.066 | +0.054 | +0.054 |
| 3722 | 100 | 26.335 | 0.94545 | 22.935 | 0.88029 | +0.053 | +0.019 | +0.019 |
| 3722 | 300 | 29.074 | 0.96743 | 23.047 | 0.86940 | -0.049 | -0.056 | -0.057 |

### Comparación por cámara al paso 300 frente a NHT

| Semilla | Vista | Δ PSNR (dB) | Δ SSIM |
| --- | --- | ---: | ---: |
| 3721 | r_0.png | -0.146 | -0.00150 |
| 3721 | r_1.png | -0.154 | -0.00119 |
| 3721 | r_10.png | -0.233 | -0.00117 |
| 3721 | r_100.png | -0.229 | -0.00254 |
| 3722 | r_0.png | -0.012 | -0.00093 |
| 3722 | r_1.png | -0.009 | -0.00081 |
| 3722 | r_10.png | -0.059 | -0.00103 |
| 3722 | r_100.png | -0.146 | -0.00166 |

### Rendimiento

El layout contiene 4,338,721 vértices y 2,937,583 vóxeles. Se optimizaron 34,709,768 características, 1,923 parámetros del decoder y 8,812,749 escalares SH0. Se midieron 48 fotogramas en la RTX 3060, con calentamiento y sincronización individual. Apariencia incluye Fourier y la MLP por vértice.

| Semilla | ms completos | FPS | ms apariencia | ms gather | ms raster | Pico inferencia (MiB) | Pico entrenamiento (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3721 | 64.45 | 15.52 | 44.40 | 2.30 | 17.67 | 3927 | 5304 |
| 3722 | 64.63 | 15.47 | 44.53 | 2.30 | 17.74 | 3927 | 5304 |

Referencias registradas con el mismo protocolo: SH3 26,161 dB, 0,94832 SSIM y 60,44 FPS. Diferido base 58,23 FPS, NHT 47,74 FPS y NeuRBF entre aproximadamente 37 y 39 FPS.

Los resultados reproducibles, incluyendo métricas por cámara, checkpoints, diferencias emparejadas y el script de entrenamiento, están en `output/vertex-mlp-materials-t/`.
