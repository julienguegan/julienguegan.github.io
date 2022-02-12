---
title: "CNN : convolution, Pytorch, Deep Dream"
date: 2022-01-01T17:10:10-02:00 
lang: fr
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

**Remarque:** Il existe plusieurs paramètres associés à l'opération de convolution comme la taille du noyau utilisé, la taille du pas lorsqu'on fait glisser la fenêtre sur l'image, la façon dont on gère les bords de l'image, le taux de dilatation du noyau ... [plus d'infos ici](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)
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

**Note:** En 2D (1 seul canal), on utilise le terme *kernel* pour parler du noyau. En 3D (plus d'un canal), on utilise le terme *filtre* qui est constitué d'autant de kernel que le nombre de canaux du volume d'entrée. 
{: .notice--info}

Plus précisément dans les CNN, une couche convolutionnelle est composée un ensemble de $N_f$ filtres de taille $N_W$ x $N_H$ x $N_C$ plus un biais par filtre suivi d'une fonction d'activation non linéaire. Ici, $N_W$ et $N_H$ désigne les tailles spatiales du filtre alors que $N_C$ est le nombre de canaux (parfois appelé *feature map*). Chaque filtres réalisent une convolution multi-canaux, on obtient alors $N_f$ produits de convolution qui sont concaténés dans un volume de sortie. Ces $N_f$ produits deviennent alors les canaux du prochain volume qui passera dans la prochaine couche convolutionnelle. Notez que la profondeur des filtres doit nécessairement correspondre au nombre de canaux du volume d'entrée de chaque couche mais le nombre de filtres est un hyperparamètre d'architecture du modèle. Au final, l'enchaînement de ces convolutions multicanaux crée en sortie un volume de caractéristiques (*features*) de l'image d'entrée, ces features sont alors passées au réseau complètement connecté pour la classification.

<p align="center">
   <img src="/assets/images/architecture_cnn.png" width="100%"/>
</p>

**Important:** Une couche convolutionnelle est généralement composée (en plus de la convolution) d'une fonction d'activation non linéaire et parfois d'autres types d'opérations (pooling, batch-normalization, dropout ...).
{: .notice--success}

Dans le post précédent, on a défini un MLP et son entraînement de zéro. Ici, la librairie **PyTorch** est utilisée. Elle permet de facilement construire des réseaux de neurones en profitant de son [moteur de différentiation automatique](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/#what-is-autograd) pour l'entraînement ainsi que ses nombreuses fonctions spécialisées (comme la [convolution](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
```

```python
class My_Custom_Model(nn.Module):

    def __init__(self):
        ''' define some layers '''
        super().__init__()
        # feature learning
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # classification
        self.fc1 = nn.Linear(16*5* 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ''' create model architecture - how operations are linked '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Comme vous l'aurez peut être compris, ce qui est intéressant avec ces opérations de convolutions est que le poids des filtres peuvent être appris lors de l'optimisation par rétropropogation du gradient puisqu'il est possible de calculer de façon exacte la valeur de $\frac{\partial\mathcal{L}}{\partial W}$ par dérivation en chaîne. 


```python
# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_cnn_model.parameters(), lr=0.001)
# training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # optimization step
        optimizer.step()
```

**Note:** Pour des données d'entrées volumineuses, on utilise souvent comme algorithme d'optimisation une *descente de gradient stochastique* où la loss est approchée en utilisant un batch de quelques données (par exemple, 8, 16 ou 32 images).  
{: .notice--info}


## Deep Dream

L'un des challenges des réseaux de neurones est de comprendre ce qu'il se passe exactement à chaque couche. En effet, leur architecture en cascade ainsi que leurs nombreuses interconnexions font qu'il n'est pas évident d'interpréter le rôle de chaque filtre. La visualisation des *features* est un axe de recherches s'étant développé ces dernières années qui consiste à trouver des méthodes pour comprendre comment les CNNs voient un image.

DeepDream est le nom d'une de ces techniques créée en 2015 par une équipe d'ingénieur de Google, l'idée est d'utiliser un réseau déjà entraîné à reconnaître des formes pour modifier une image afin qu'un neurone donné renvoie une sortie plus élevée que les autres. L'algorithme ressemble à la backpropagation classique mais au lieu de modifier les poids du réseau on ajuste les pixels de l'image d'entrée. De plus, le critère d'optimisation n'est pas une cross entropie mais directement la norme de la sortie du neurone à visualiser (ça peut être la couche entière ou un filtre) qu'on va chercher à maximiser, on fait alors une montée de gradient (on pourrait également minimiser l'opposée).

```python
# Parameters
iterations   = 25   # number of gradient ascent steps per octave
at_layer     = 26   # layer at which we modify image to maximize outputs
lr           = 0.02 # learning rate
octave_scale = 2    # image scale between octaves
num_octaves  = 4    # number of octaves
```

```python
# Load Model pretrained
network = models.vgg19(pretrained=True)
# Use only needed layers
layers = list(network.features.children())
model = nn.Sequential(*layers[: (at_layer + 1)])
# Use GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

Une astuce supplémentaire pour obtenir une visualisation intéressante est d'opérer à des résolutions spatiales différentes, ici on parle d'*octave*. De plus, la loss est normalisée à toutes les couches pour que la contribution des grandes couches ne l'emporte pas sur celle des petites couches.

```python
# loop on different resolution scale
detail = np.zeros_like(octaves[-1])
for k, octave_base in enumerate(tqdm(octaves[::-1], desc="Octaves : ")):
    # Upsample detail to new octave dimension
    if k > 0: 
      detail = nd.zoom(detail, np.array(octave_base.shape)/np.array(detail.shape), order=1) 
    # Add detail from previous octave to new base
    input_image = octave_base + detail
    # Updates the image to maximize outputs for n iterations
    input_image = Variable(torch.FloatTensor(input_image).to(device), requires_grad=True)
    for i in trange(iterations, desc="Iterations : ", leave=False):
        model.zero_grad()
        out  = model(input_image)
        loss = out.norm()
        loss.backward()
        # gradient ascent
        avg_grad = np.abs(input_image.grad.data.cpu().numpy()).mean()
        norm_lr  = lr/avg_grad
        input_image.data = input_image.data + norm_lr * input_image.grad.data
        input_image.data = clip(input_image.data)
        input_image.grad.data.zero_()
        # Extract deep dream details
        detail = input_image.cpu().data.numpy() - octave_base
```

On obtient, selon le nombre d'itération, des images de plus en plus abstraites avec des formes psychédéliques qui apparaissent au fur et à mesure d'où le nom de *DeepDream*. En fait, ces formes abstraites sont présentes surtout pour les couches les plus profondes, les premières couches accentuent généralement des *features* simples comme des arêtes, des coins, des textures ...

<p align="center">
   <img src="/assets/images/deepdream_exemple.gif" width="80%"/>
</p>

Avec cet outil, on peut créer des effets artistiques très avancées comme sur [l'instagram de DeepDreamGenerator](https://www.instagram.com/deepdreamgenerator/). Mais on peut également accentuer l'effet pscychédélique en faisant beaucoup d'itérations ou en alimentant plusieurs fois la sortie de l'algorithme en entrée. Et avec un peu d'effort, on peut parvenir à visualiser à quoi ça ressemble d'aller au supermarché dans ces rêves à partir d'images bien réelles. 

{% include video id="DgPaCWJL7XI" provider="youtube" %}

Tel que présenté ci-dessus, Deep Dream présente un inconvénient si on veut le lancer sur une image de bruit blanc en entrée pour visualiser ce qui pourrait en émerger et ainsi avoir une représentation plus exact des *features* du CNN. En effet, on voit que l'image reste dominée par des motifs hautes-fréquences. 

<p align="center">
   <img src="/assets/images/deepdream_noise.png" width="80%"/>
</p>

Généralement, pour contrer cet effet, ce qui marche le mieux est d'introduire une régularisation d'une façon ou d'une autre dans le modèle. Par exemple, la robustesse à la transformation essaie de trouver des exemples qui activent toujours fortement la fonction d'optimisation lorsqu'on les transforment très faiblement. Concrètement, cela signifie qu'on tremble, tourne, diminue ou augmente l'image de façon aléatoire avant d'appliquer l'étape d'optimisation. Les librairies [lucid](https://github.com/tensorflow/lucid) (tensorflow) et [lucent](https://github.com/greentfrapp/lucent) (pytorch) sont des packages open-source qui implémentent toutes sortes de méthodes de visualisation. 

```python
# load librairies
from lucent.optvis import render
from lucent.modelzoo import vgg19
# load model
model = vgg19(pretrained=True)
model = model.to(device)
model.eval()
# run optimisation
image = render.render_vis(model, "features:30",thresholds=[100],show_inline=True)
```

Un article bien plus complète sur les techniques de visualisation de features est disponible [ici](https://distill.pub/2017/feature-visualization/)

<p align="center">
   <img src="/assets/images/lucid_viz.png" width="100%"/>
</p>

---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/visualisation_CNN.ipynb) [![Generic badge](https://img.shields.io/badge/execute_le_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/visualisation_CNN.ipynb)