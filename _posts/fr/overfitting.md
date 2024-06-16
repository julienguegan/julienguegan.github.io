---
title: "Overfitting"
---

https://lilianweng.github.io/posts/2019-03-14-overfit/#intrinsic-dimension

Les réseaux de neurones ont généralement beaucoup de paramètres et des erreurs d'entrainement souvent proche de zéro, ils devraient normalement souffrir de surapprentissage. 

Comment arrivent-ils à généraliser à des nouveaux points ?

# Les théorêmes sur la sélection de modèles

## le rasoir d'Occam

concept philosophique déclarant que "les solutions les plus simples sont souvent les meilleurs"

## la longueur de description minimale 

on regarde la notion d'apprentissage comme de la compression de donnée. En compressant les données, on doit trouver des motifs dans les données avec le potentiel de généraliser à des exemples jamais vu. Donc le modèle ne peut pas être arbitrairement grand.

## la complexité de Kolmogorov

C'est la longueur du plus court programme informatique binaire qui décrit un objet

# La puissance expressive des modèles DNN

## théorême d'approximation universelle

un réseau de neurones avec une couche de sortie, une fonctiond d'activation et une couche cachée peut approximer n'importe quelle fonction continue. 

Mais la taille du réseau peut être exponentiellement large et rien n'est dit sur l'apprentissage et la capacité à généraliser

## les DNN peuvent apprendre du bruit aléatoire

on peut mélanger tout les labels associés un jeu de donnée de classification d'image, les DNN sont toujours capables d'atteindre une erreur d'entrainement quasi-nulle

# Le surapprentissage des DNN

## courbe d'erreur moderne

Traditionnellement, on cherche un modèle qui minimise l'erreur de test en étant ni trop complexe ni pas assez. Mais ça ne s'applique pas au DNN qui, lorsque le nombre de paramètre augmente, l'erreur de test se remet à descendre

## la régularisation n'est pas forcément la clé

## dimension intrinsèque

## hétérogénéité des couches

# Expériences