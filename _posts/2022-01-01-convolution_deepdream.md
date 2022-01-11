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

Les réseaux de neurones convolutionnels (CNN) sont les modèles qui ont permis de faire un bon en avant dans les problèmes de reconnaissance d'image. Ils sont au coeur de nombreuses applications allant des systèmes de sécurité par identification faciale à la classification de vos photos de vacances en passant par la génération synthétique de visage et les filtres snapchat. L'un des fondateurs de ce modèle est Yann Le Cun qui, en 1989, applique la backpropagation du gradient pour apprendre des filtres de convolution et permet à un réseau de neurone à reconnaître des chiffres manuscrits. Cependant, c'est seulement en 2012 que les CNN se répandent largement dans la communauté scientifique de la vision par ordinateur avec Alex Krizhevsky qui conçoit l'architecture *AlexNet* et remporte la compétition *ImageNet Large Scale Visual Recognition Challenge* (1 million d'images de 1000 classes différentes) en implémentant son algorithme sur des GPUs ce qui permet au modèle d'apprendre rapidement d'une grande quantité d'image. Ce modèle atteint des performances 10% plus élevées que tous les autres à cette époque et il est désormais l'un des papiers publiés les plus influents en Computer Vision (en 2021, plus de 80 000 citations selon Google Scholar).

## CNN

Les modèles de réseaux de neurones complètements connectés (cf [post précédent](https://julienguegan.github.io/posts/2021-09-10-reseau_de_neurone/)) ne sont pas adaptés pour résoudre des problèmes de traitement d'image. En effet, les MLP ont pour une couche chaque neurone connecté à chaque unité d'entrée : le nombre de paramètre à apprendre devient vite élevé et une forte redondance dans les poids du réseau peut exister. De plus, pour utiliser une image dans un tel réseau, tous les pixels devrait être transformée en
vecteur et aucune information sur la structure locale des pixels serait alors prise en compte. 



intro convolution
architecture
backpropagation pour CNN

## Deep Dream

 calculating the gradient of the image with respect to the activations of a particular layer. The image is then modified to increase these activations, enhancing the patterns seen by the network, and resulting in a dream-like image.

 calcule le gradient de l'image par rapport aux activations d'une couche particulière. L'image est ensuite modifiée pour augmenter les activations de cette couche, renforçant les motifs vus par le réseau et résultant en une image onirique.

## Grad Cam

