---
title: "CNN : convolution, Deep Dream, Grad Cam"
date: 2022-01-01T17:10:10-02:00 
classes: wide
layout: single
categories:
  - blog
tags :
  - machine learning
  - deep learning
  - convolution
  - deep dream
header:
  teaser: /assets/images/teaser_deepdream.jpg
---

**écriture en cours ...**

Les réseaux de neurones convolutionnels (CNN) sont les modèles qui ont permis de faire un bon en avant dans les problèmes de reconnaissance d'image. Ils sont au coeur de nombreuses applications allant des systèmes de sécurité par identification faciale à la classification de vos photos de vacances en passant par la génération synthétique de visage et les filtres snapchat. L'un des fondateurs de ce modèle est Yann Le Cun (un français !) qui, en 1989, applique la backpropagation du gradient pour apprendre des filtres de convolution et permet à un réseau de neurone à reconnaître des chiffres manuscrits. Cependant, c'est seulement en 2012 que les CNN se répandent largement dans la communauté scientifique de la vision par ordinateur avec Alex Krizhevsky qui conçoit l'architecture *AlexNet* et remporte la compétition *ImageNet Large Scale Visual Recognition Challenge* (1 million d'images de 1000 classes différentes) en implémentant son algorithme sur des GPUs ce qui permet au modèle d'apprendre rapidement d'une grande quantité d'image. Ce modèle atteint des performances 10% plus élevées que tous les autres à cette époque et il est désormais l'un des papiers publiés les plus influents en Computer Vision (en 2021, plus de 80 000 citations selon Google Scholar).

<p align="center">
   <img src="/assets/images/cnn_header.png" width="80%"/>
</p>

## Convolutions et Réseaux de neurones

Les modèles de réseaux de neurones complètements connectés (cf [post précédent](https://julienguegan.github.io/posts/2021-09-10-reseau_de_neurone/)) ne sont pas adaptés pour résoudre des problèmes de traitement d'image. En effet, les MLP ont chaque neurone d'une couche connecté à chaque unité d'entrée : le nombre de paramètre à apprendre devient vite élevé et une forte redondance dans les poids du réseau peut exister. De plus, pour utiliser une image dans un tel réseau, tous les pixels devrait être transformée en vecteur et aucune information sur la structure locale des pixels serait alors prise en compte. 

Le produit de convolution, noté $\ast$, est un opérateur qui généralise l'idée de moyenne glissante. Il s'applique aussi bien à des données temporelles (en traitement du signal par exemple) qu'à des données spatiales (en traitement d'image). Pour le cas des images, c'est-à-dire discret et en 2 dimensions, la convolution entre une image $I$ et un noyau  $w$ (ou kernel) peut se calculer comme suit :

$$I(i,j) * \omega =\sum_{x=-a}^a{\sum_{y=-b}^b{ I(i+x,j+y)} \ \omega(x,y)}$$

L'idée est de faire glisser le noyau spatialement sur toute l'image et à chaque fois de faire une moyenne pondérée des pixels de l'image se retrouvant dans la fenêtre concernée par les éléments du noyau. Selon la valeur des éléments du noyau de convolution $w$, l'opération peut mettre en avant des caractéristiques particulières se trouvant dans l'image comme des contours, des textures, des formes.

<p align="center">
   <img src="/assets/images/image_convolution.gif" width="40%"/>
</p>

**Remarque:** Il existe plusieurs paramètres associés à l'opération de convolution comme la taille du noyau utilisé, la taille du pas lorsqu'on fait glissé la fenêtre sur l'image, la façon dont on gère les bords de l'image, le taux de dilatation du noyau ([plus d'infos ici](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215))
{: .notice--warning}

On peut par exemple mettre en avant les pixels d'une image correspondants aux contours horizontaux en appliquant une convolution avec un noyau de taille $3 \times 3$ avec des $-1$ dans la 1ère ligne, des $0$ dans la 2ème ligne et des $+1$ dans la 3ème ligne de la matrice.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
# read image
image = np.array(Image.open("path/to/file.jpg").convert('L'))
# apply convolution
kernel = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [+1, +1, +1]])
conv_output = convolve2d(image, kernel, mode='same')
# display
plt.figure(figsize=(15,5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.axis('off')
plt.subplot(122), plt.imshow(np.abs(conv_output), cmap='gray'), plt.axis('off')
plt.tight_layout()
plt.show()
```

<p align="center">
   <img src="/assets/images/convolution_exemple.png" width="80%"/>
</p>

L'idée de l'architecture des modèles CNN est de garder des couches complètement connectées pour la classification. Cependant, en entrées de ces couches, l'image n'est pas directement utilisée, mais la sortie de plusieurs opérations de convolution qui ont pour but de mettre en avant les différentes caractéristiques d'une image en encodant d'une certaine façon les objets qui sont présents ou non. On utilise notamment des convolutions multi-canaux qui consistent à appliquer une convolution standard à chaque canaux de l'entrée puis sommer chaque produits de convolution obtenus pour obtenir une unique matrice 2D. Par exemple pour une image couleur les canaux sont le rouge, vert et bleu, on a alors 3 kernels à convoluer avec les canaux associés puis les 3 produits obtenus sont sommés.

<p align="center">
   <img src="/assets/images/multichannel_convolution.png" width="100%"/>
</p>

**Note:** En 2D (1 seul canal), on utilise le terme *kernel* pour parler du noyau. En 3D (plus d'un canal), on utilise le terme *filtre* qui a alors le même nombre de canaux que le volume d'entrée. 
{: .notice--info}

Plus précisément dans les CNN, une couche convolutionnelle est composée un ensemble de $N_f$ filtres de taille $N_W$ x $N_H$ x $N_C$ plus un biais par filtre suivi d'une fonction d'activation non linéaire. Ici, $N_W$ et $N_H$ désigne les tailles spatiales du filtre alors que $N_C$ est le nombre de canaux (parfois appelé *feature map*). Chaque filtres réalisent une convolution multi-canaux, on obtient alors $N_f$ produits de convolution qui sont concaténés dans un volume de sortie. Ces $N_f$ produits deviennent alors les canaux du prochain volume qui passera dans la prochaine couche convolutionnelle. Notez que la profondeur des filtres doit nécessairement correspondre au nombre de canaux du volume d'entrée de chaque couche mais le nombre de filtres est un hyperparamètre d'architecture du modèle. Au final, l'enchaînement de ces convolutions multicanaux crée en sortie un volume de caractéristiques (*features*) de l'image d'entrée, ces features sont alors passées au réseau complètement connecté pour la classification.

<p align="center">
   <img src="/assets/images/architecture_cnn.png" width="100%"/>
</p>

**Important:** Une couche convolutionnelle est généralement composée (en plus de la convolution) d'une fonction d'activation non linéaire et parfois d'autres types d'opérations comme le pooling, la batch-normalization, le dropout ... 
{: .notice--success}

```python
# todo : insert code pytorch of model

```

Le plus intéressant avec ces opérations de convolutions est qu'elles peuvent écrites comme un produit matricielle et donc les poids des filtres peuvent appris lors de l'optimisation par rétropropogation du gradient. 
backpropagation pour CNN ?

```python
# todo : insert code pytorch of training loop (sans les dataloaders)

```


## Deep Dream

calculating the gradient of the image with respect to the activations of a particular layer. The image is then modified to increase these activations, enhancing the patterns seen by the network, and resulting in a dream-like image.

calcule le gradient de l'image par rapport aux activations d'une couche particulière. L'image est ensuite modifiée pour augmenter les activations de cette couche, renforçant les motifs vus par le réseau et résultant en une image onirique.

## Grad Cam

