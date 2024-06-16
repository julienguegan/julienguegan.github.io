---
title: "GAN"
---

https://lilianweng.github.io/posts/2017-08-20-gan/#improved-gan-training


# KB and JS Divergence

quantifier la similarité entre distribution de probabilité

1) Kullback-Leibler divergence mesure à quel point une distribution p(x) diverges d'une autre distribution q(x)

$$ D_{KL}(p \| q) = \int_x p(x) \log \frac{p(x)}{q(x)} dx $$

atteint un minimum de zéro quand p(x) == q(x)
 partout. N'est pas symmétrique.

2) Jensen-Shannon divergence appartient à [0,1], est symmétrique et plus lisse

$$ D_{JS}(p \| q) = \frac{1}{2} D_{KL}(p \| \frac{p + q}{2}) + \frac{1}{2} D_{KL}(q \| \frac{p + q}{2}) $$

# GAN

On a 2 modèles : 

- un générateur $G$ qui prend un entrée une variable $z$ tiré d'une loi aléatoire et qui la transforme en un échantillon synthétique qui suit la distribution à apprendre
- un discriminateur $D$ qui classifie si une donnée appartient bien au jeu de données réelles ou bien si la donnée est synthétique. 

On parle de réseaux adversariaux puisque les 2 modèles sont en compétition durant le processus d'entrainement : $G$ essaie de tromper $D$ en générant une donnée la plus réelle possible alors que $D$ apprend à reconnaître le mieux possible une vraie donnée d'une donnée synthétique. Ce jeu à somme nulle oblige les 2 modèles à s'améliorer tout les deux en même temps.

D'une part, le discriminateur $D$ doit maximiser $\mathbb{E}_x [\log D(x)]$ avec $x$ nos données réelles et il doit également reconnaître les faux exemples $G(z)$ qui sortent du générateur en maximisant $\mathbb{E}_{z} [\log (1 - D(G(z)))]$. D'autre part, le générateur est entrainé de telle sorte que $D$ se trompe, c'est-à-dire minimiser $\mathbb{E}_{z} [\log (1 - D(G(z)))]$. On peut rassembler ces différents aspects en un jeu de minimax dans lequel on optimise la fonction : 

$$ \min_G \max_D L(D, G) = \mathbb{E}_{x \sim p_{r}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))] $$


## valeur optimale pour D

D*(x) est 1/2 quand p_g = p_r

## optimal global

$$ \begin{aligned}
L(G, D^*) 
&= \int_x \bigg( p_{r}(x) \log(D^*(x)) + p_g (x) \log(1 - D^*(x)) \bigg) dx \\
&= \log \frac{1}{2} \int_x p_{r}(x) dx + \log \frac{1}{2} \int_x p_g(x) dx \\
&= -2\log2
\end{aligned} $$

## la fonction cout

# Les problemes dans les GANs

## Difficile d'atteindre l'équilibre de Nash

## Faible support dimensionnel

## Disparition du gradient

quand le discriminateur classifie parfaitement on a que $D(x) = 1, \forall x \in p_r$ et $D(x) = 0, \forall x \in p_g$. Dans ce cas la fonction $L$ vaut 0 est aucune mise à jour du gradient n'est alors possible.

L'entrainement d'un GAN fait face à un dilemme : 
- si le discriminateur se comporte mal alors le générateur n'a pas un feedback précis pour se mettre à jour et la fonction de cout ne peut pas représenter la réalité.
- si le discriminateur est très bon, le gradient de la loss est proche de zéro et l'apprentissage devient très lent.

## Mode effondrement (*mode collapse*)

Durant l'entrainement, le générateur peut s'effondrer dans une configuration où il produit toujours la même sortie. C'est un cas courant appelé *mode collapse* où il arrive à tromper le discriminateur mais sans pour autant représenter la vraie distribution des données réelles et est bloqué dans un espace restreint avec peu de complexité

# Amélioration de l'apprentissage

**feature matching** : optimiser le discriminateur de telle sorte que les sorties du générateur suivent des statistiques identiques aux données réelles (comme la moyenne ou la médiane)

**minibatch discrimination** : plusieurs points au lieu d'un seul

**moyenne historique** : on ajoute $| \Theta - \frac{1}{t} \sum_{i=1}^t \Theta_i |^2$ pour chaque modèle dans la fonction de coût où $\Theta_i$ est les paramètres du modèle à l'itération $i$. Pénalise la vitesse d'entrainement quand $ \Theta $ change de façon trop importante dans le temps.

**lissage des labels unilateral** : au lieu de fournir des labels 0 et 1 au discriminateur, on utilise des valeurs adoucies telles que 0.9 et 0.1

**normalisation des batch virtuel** : Chaque échantillon de données est normalisé sur la base d'un batch fixe («de référence») de données plutôt que dans son mini-batch. Le batch de référence est choisi une seule fois au début et reste le même tout au long de l'apprentissage.


# Wasserstein GAN (WGAN)




