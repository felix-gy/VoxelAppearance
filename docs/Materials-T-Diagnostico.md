# Diagnóstico de Materials T

Se evaluaron las mismas 32 cámaras de entrenamiento y cuatro reservadas a 256×256, con una muestra por vóxel, fondo blanco y `ss=1`. Los 15 checkpoints coincidieron con el layout y los tensores congelados del SH 20.000 original. Se comprobó la lista ordenada de cámaras, la configuración y el contador de Adam en sus ocho tensores entrenables. Las tres variantes realizaron respectivamente 30, 100 y 300 actualizaciones con MSE de peso 1 y tasas 0,01 para SH0 y 0,00025 para características y MLP. Los presupuestos de características fueron 8V, 32V y 64V.

## Calidad

Las columnas indican PSNR de entrenamiento, PSNR reservado y SSIM reservado.

| Modo | Semilla | Paso | PSNR train | PSNR reservado | SSIM reservado |
| --- | ---: | ---: | ---: | ---: | ---: |
| SH0 | — | — | 24,551 | 22,495 | 0,88132 |
| SH3 | — | — | 30,086 | 26,161 | 0,94832 |
| Base | 3721 | 30 | 25,023 | 22,648 | 0,88227 |
| Base | 3721 | 100 | 26,252 | 22,841 | 0,87902 |
| Base | 3721 | 300 | 29,341 | 23,133 | 0,87000 |
| NHT | 3721 | 30 | 25,024 | 22,651 | 0,88229 |
| NHT | 3721 | 100 | 26,269 | 22,851 | 0,87909 |
| NHT | 3721 | 300 | 29,282 | 23,158 | 0,87022 |
| Control | 3721 | 30 | 25,023 | 22,648 | 0,88227 |
| Control | 3721 | 100 | 26,252 | 22,841 | 0,87902 |
| Control | 3721 | 300 | 29,358 | 23,135 | 0,87008 |
| Base | 3722 | 30 | 25,046 | 22,664 | 0,88237 |
| Base | 3722 | 100 | 26,308 | 22,883 | 0,87992 |
| Base | 3722 | 300 | 29,280 | 23,096 | 0,87052 |
| NHT | 3722 | 30 | 25,057 | 22,676 | 0,88237 |
| NHT | 3722 | 100 | 26,324 | 22,916 | 0,87983 |
| NHT | 3722 | 300 | 29,218 | 23,103 | 0,87051 |

En la semilla 3721, la brecha de PSNR entre entrenamiento y validación pasó de unos 2,38 dB en el paso 30 a 6,12–6,22 dB en el 300. A la vez mejoró el PSNR reservado y cayó su SSIM.

## Diferencias NHT por cámara al paso 300

Cada celda muestra el cambio de PSNR en dB y de SSIM frente al diferido base con la misma semilla.

| Vista | Semilla 3721 | Semilla 3722 |
| --- | ---: | ---: |
| r_0.png | +0,032 / +0,000247 | +0,022 / +0,000080 |
| r_1.png | +0,039 / +0,000235 | +0,025 / +0,000007 |
| r_10.png | +0,058 / −0,000127 | +0,042 / +0,000152 |
| r_100.png | −0,029 / +0,000534 | −0,060 / −0,000294 |

La mejora media de PSNR fue +0,0251 dB con 3721 y +0,0073 dB con 3722. La mejora de SSIM de 0,00022 pasó a aproximadamente cero. El patrón por vista del PSNR se repitió con ambas semillas. SH3 mantuvo el mayor PSNR reservado en las cuatro vistas.

## Costo por etapas

Mediciones sincronizadas en 48 renders por modo después del calentamiento.

| Modo | ms completos | FPS | ms raster | ms MLP |
| --- | ---: | ---: | ---: | ---: |
| SH0 | 14,011 | 71,37 | 13,855 | — |
| SH3 | 16,546 | 60,44 | 16,409 | — |
| Base | 17,173 | 58,23 | 16,397 | 0,457 |
| NHT | 20,948 | 47,74 | 20,204 | 0,478 |
| Control | 25,487 | 39,24 | 24,773 | 0,472 |

Una compilación temporal midió por separado ciclos de densidad, consulta, codificación y acumulación, con un hilo muestreado por warp. La consulta incluye lectura de características y, cuando aplica, coordenadas locales, pesos e interpolación. NHT registró aproximadamente 2.783 ciclos de consulta y 460 de seno/coseno por contribución aceptada. El control registró aproximadamente 4.427 ciclos de consulta, pese a omitir la codificación armónica. La acumulación instrumentada midió unos 550 ciclos en la base, 45 en NHT y 68 en el control. El consumo del dato puede trasladar esperas por memoria a la sección de acumulación. Los ciclos de densidad rondaron 296–325 por intersección candidata. Los ciclos son diagnósticos del kernel instrumentado y no se suman para estimar milisegundos por fotograma. El costo por fotograma se midió con la extensión normal.

Los datos completos por cámara y split, el resumen, el evaluador, los checkpoints nuevos y el parche y log de perfilado están en `output/materials-t-diagnostic/`.
