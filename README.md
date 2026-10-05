Hola buenas tardes

## Fotos de gestos

Las fotos originales estan en `images/`, renombradas como
`<gesto>_<id>.jpeg`. La clase `otros` agrupa gestos distintos de OK, stop,
paz y pulgar arriba. Las fotos descartadas (gesto ambiguo) se llaman
`descarte_<id>.jpeg` y no se usan para entrenar ni evaluar.

`images/etiquetas.csv` registra el nombre original, el nombre nuevo, la
etiqueta y el SHA-256 de cada foto. El renombrado no modifica la
resolucion ni el contenido de las imagenes.

## Pasos

1. `uv sync` instala las dependencias.
2. `uv run python preparar_datos.py` reparte las fotos en
   `data/train/<gesto>/` y `data/test/<gesto>/` como copias reducidas
   (lado mas largo de 512 px). Las fotos consecutivas del mismo gesto
   (una "tanda", de hasta 15 fotos) van completas a train o a test. El detalle queda en
   `data/particion.csv`. Si se agregan fotos, basta con volver a ejecutarlo.
3. `uv run jupyter lab` y ejecutar `clasificador.ipynb`: hace fine-tuning de
   una ResNet-18 preentrenada en ImageNet, con data augmentation solo en
   train, y mide exactitud y matriz de confusion en `data/test`.

El Lab 14 exige al menos 50 fotos de train y 10 de test por clase.
`preparar_datos.py` avisa que clases no llegan a ese minimo.

El notebook `entrenamiento_mnist.ipynb` ensena el ciclo de entrenamiento
de una red densa sobre digitos MNIST, y `desafio_lab12.ipynb` es el desafio
de Fashion-MNIST; ninguno usa una red preentrenada.