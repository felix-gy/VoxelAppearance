# Vertex-MLP con base principal y residual acotado

El modo `vertex_two_bases_bounded` reutiliza las características por vértice
`[P,8]`, Fourier-3 y el decoder `26→64→6`. Su residual por esquina es
`B1 + eta * tanh(voxel_gate[v] * B2)`, con `eta = 0.25` por defecto.
Autograd de PyTorch deriva la mezcla y el rasterizador recibe las mismas 27
componentes por vóxel que antes. La primera base conserva su amplitud completa.
Los pesos trilineales conservan la cota por canal de la segunda contribución.
Un salto entre vóxeles sigue siendo posible porque el gate es independiente.

El checkpoint guarda `voxel_gate[V]`, `eta` como buffer persistente, el
decoder, las características y la identidad geométrica. El modo se distingue
del checkpoint de dos bases libres. El lector del visor selecciona el
decoder correspondiente. La penalización de coeficientes libres se mantiene
separada de esta arquitectura.

## Inicialización y protocolo

En cada semilla, el control de una base y el residual acotado arrancan desde su
checkpoint de Lego de una base al paso 300. Se copian características, capa
oculta y primera salida. La segunda salida usa pesos normales con desviación
estándar 0,01 y sesgo cero. El gate parte de cero. El primer render coincide con
el control, el gate recibe gradiente en el primer backward y la segunda
salida recibe gradiente tras el primer cambio del gate.

Se usan las mismas 86 cámaras de entrenamiento y cuatro reservadas, a
256×256, supersampling 1, fondo blanco y una muestra de densidad. Densidad,
geometría, SH0 y topología quedan congelados. Cada brazo usa Adam nuevo a
0,00025, betas (0,1, 0,99), epsilon 1e−15 y MSE. Los hitos 300, 500 y 1000
son pasos adicionales al checkpoint de origen. El control se vuelve a
ejecutar con ese mismo origen y presupuesto. Los resultados existentes de
dos bases libres nacieron de otra inicialización.

## Reproducción

```bash
export PYTHONPATH="$PWD/cuda:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -u scripts/evaluate_vertex_bounded_lego.py
python viz_vertex_fourier3.py \
  --checkpoint output/vertex-bounded-residual-lego/seed3721/bounded/step1000.pt
```

El script guarda checkpoints y JSON por hito y un informe que reúne PSNR de
entrenamiento y de reserva, SSIM, LPIPS-VGG, RMS de salto, RMS de corrección
y saturación. Las dos últimas métricas evalúan todas las esquinas RGB por
vista reservada, antes de la interpolación. La saturación cuenta las
componentes con `abs(voxel_gate * B2) > 2`.

## Resultados en Lego

Media de las semillas 3721 y 3722. Los deltas se calculan como residual
acotado menos control de una base. Las métricas de imagen corresponden a las
cuatro vistas reservadas.

| Pasos adicionales | Δ PSNR (dB) | Δ SSIM | Δ LPIPS | RMS de corrección | RMS de salto | Saturación |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | +0,0166 | +0,000275 | −0,000287 | 0,002066 | 0,002020 | 0 % |
| 500 | +0,0232 | +0,000277 | −0,000196 | 0,004569 | 0,005012 | 0 % |
| 1000 | +0,0152 | −0,000266 | +0,000708 | 0,013237 | 0,016343 | 0 % |

Al paso 1000 el PSNR reservado subió +0,0184 dB con la semilla 3721 y
+0,0119 dB con la 3722. El PSNR de entrenamiento subió +0,3933 dB en
promedio. La ventaja en vistas reservadas alcanzó su máximo entre los hitos
medidos al paso 500. A 1000 pasos ambas semillas perdieron SSIM y aumentaron
LPIPS frente al control. El RMS de salto creció mientras la saturación
observada se mantuvo en cero. La cota limita la amplitud del residual por
canal y permite saltos entre vóxeles vecinos.

Pasaron 26 pruebas. El informe generado en
`output/vertex-bounded-residual-lego/report.md` contiene el desglose por
semilla y los JSON junto a los checkpoints contienen los resultados por vista.
