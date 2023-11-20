# Proyecto final de Estructuras de Datos Avanzados - dLSHBF

División de datos 80-20 para MNIST como SPEECH. De los datos de MNIST solo se extrajo la parte de entrenamiento y se subidivió de acuerdo al paper es decir 60_000 imágenes de 10 categorías considerando las etiquetas diferente de 1 como normales y con valor 1 anormales. Para el conjunto de datos SPEECH se dividio a partir de los datos considerando la totalidad del conjunto, es decir 3686 segmentos de habla en inglés con diferentes acentos, aqui solo hay 2 categorías 1 y 0 (anómalo y normal) por lo que no fue necesario considerar el paso que se adiciono para el MNIST.

<div align="center">

| Dataset         | DR       | FAR      |
|-----------------|----------|----------|
| SPEECH DATASET  | 0.983718 | 0.016282 |
| MNIST DATASET   | 0.612395 | 0.387605 |

#### Datos MNIST
| Tipo de Dato     | Cantidad |
|------------------|----------|
| Total Normal     | 53258    |
| Total Anormal    | 6742     |

#### Datos SPEECH
| Tipo de Dato     | Cantidad |
|------------------|----------|
| Total Normal     | 3625     |
| Total Anormal    | 61       |

<img src="img/speech_desbalanceo.png" width="500" height="300">

</div>
