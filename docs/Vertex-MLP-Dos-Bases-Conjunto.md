# Vertex-MLP conjunto con dos bases libres

`train.py --appearance_mode vertex_two_bases` usa la inicialización geométrica
del entrenamiento original. Con el preset sintético, comienza con SH0 y
activa la apariencia neuronal en el paso 1000. El módulo usa Fourier-3 y una
MLP 26 → 64 → 6. Cada vóxel aprende dos coeficientes sin restricciones. La
primera salida comienza en cero, la segunda con pesos de desviación 0,01 y
los coeficientes comienzan en (1, 0). Esto reproduce SH0 al activarse y
permite que el segundo coeficiente reciba gradiente desde el primer paso.

El callback existente mezcla las bases por esquina para los vóxeles visibles.
La mezcla diferenciable alimenta el rasterizador trilineal existente y el
entrenador optimiza también densidad y SH0. La poda conserva los coeficientes
de los supervivientes y cada hijo hereda los del padre al subdividir.
Las características por vértice y el optimizador siguen el proceso de
adaptación del entrenamiento conjunto.

Los checkpoints guardan el modo, el estado de apariencia y el del
entrenamiento. Los checkpoints conjuntos anteriores se interpretan como
Vertex-MLP de una base. Para reanudar, conserva el modo y el directorio del
entrenamiento y añade `--load_iteration 5000`, omitiendo el preset y los
argumentos que sobrescriben la configuración guardada. El visor
`viz_vertex_fourier3.py --checkpoint RUTA` restaura la variante guardada.
