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

1. `uv sync` instala las dependencias. En Windows instala PyTorch y
   torchvision con CUDA 13.0 desde el indice oficial de PyTorch.
   Requiere una GPU NVIDIA y un controlador compatible; no hace falta
   instalar el CUDA Toolkit por separado. En otros sistemas se conserva
   el indice predeterminado.
2. `uv run python preparar_datos.py` reparte las fotos en
   `data/train/<gesto>/` y `data/test/<gesto>/` como copias reducidas
   (lado mas largo de 512 px). Las fotos consecutivas del mismo gesto
   (una "tanda", de hasta 15 fotos) van completas a train o a test. El detalle queda en
   `data/particion.csv`. Si se agregan fotos, basta con volver a ejecutarlo.
3. `uv run jupyter lab` y ejecutar `clasificador.ipynb`: hace fine-tuning de
   una ResNet-18 preentrenada en ImageNet, con data augmentation solo en
   train, y mide exactitud y matriz de confusion en `data/test`.
   La primera celda comprueba CUDA con una operacion real e imprime la GPU.
   Si CUDA no esta disponible, avisa explicitamente antes de usar CPU.
   El entrenamiento actual conserva precision completa (float32) en GPU.
   Despues de ejecutar todo, guarda el modelo en
   `resultados/gestos_resnet18.pt` y las metricas, historial de entrenamiento
   y predicciones de test en `resultados/evaluacion.json`.
   Para reutilizar el modelo hay que aplicar el mismo relleno cuadrado,
   resize y normalizacion del notebook; el checkpoint incluye las clases,
   el tamano y los parametros de normalizacion.
   Se puede cargar con `torch.load(..., map_location="cpu", weights_only=True)`
   y reconstruir una ResNet-18 con una capa final de cinco salidas.

El Lab 14 exige al menos 50 fotos de train y 10 de test por clase.
`preparar_datos.py` avisa que clases no llegan a ese minimo.

Las dos tandas agregadas el 5 de octubre contienen 47 fotos utilizables de
`otros` y una descartada por desenfoque de la mano. Los originales conservan
sus bytes y su SHA-256 en `images/etiquetas.csv`. Hay 74 fotos utilizables de
`otros`; el reparto actualizado se registra en `data/particion.csv`.
El video adjunto se conserva como original, pero no se extraen fotogramas
para evitar inflar el dataset con imagenes casi identicas.

El notebook `entrenamiento_mnist.ipynb` ensena el ciclo de entrenamiento
de una red densa sobre digitos MNIST, y `desafio_lab12.ipynb` es el desafio
de Fashion-MNIST; ninguno usa una red preentrenada.