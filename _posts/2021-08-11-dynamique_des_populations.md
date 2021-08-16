---
title: "Dynamique des populations"
date: 2021-08-11T22:18:30-04:00
classes: wide
layout: single
categories:
  - blog
tags:
  - équations différentielles
---

Parmis les enjeux du 21<sup>ème</sup> siècle, l'écologie a un rôle majeure puisqu'elle est la science qui étudie les interactions des êtres vivants entre eux et avec leur milieu. Pour modéliser ces interactions, la dynamique des populations est la branche qui s'intéresse aux fluctuations démographiques des espèces. Ses applications sont nombreuses puisqu'elle peut permettre de répondre à des problèmes variés comme la gestion d'espèces menacées, la protection des cultures contre des nuisibles, le contrôle de bioréacteurs ou la prédiction des épidémies.

## Modèle de Verhulst

A la fin du 18<sup>ème</sup> siècle, le modèle de **Malthus** décrit la variation d'une taille de population $y$ au cours du temps $t$ par l'équation différentielle :

$$ y'(t) = (n-m) y(t) = r y(t) $$

avec les constantes : $n$ le taux de natalité, $m$ le taux de mortalité et $r$ le taux de croissance. Ce modèle nous dit que, selon le taux de croissance $r$, la taille des populations peut soit diminuer, rester constante ou augmenter de manière exponentielle. Ce modèle ne reflète pas la réalité puisque une population n'augmentera jamais à l'infini.

En 1840, **Verlhust** propose un modèle de croissance plus adapté en partant de l'hypothèse que le taux de croissance $r$ n'est pas une constante mais est fonction affine de la taille de population $y$ :

$$ y'(t) = \big(n(y) - m(y)\big) y(t)$$

Verlhust part notamment de l'hypothèse que plus la taille d'une population augmente alors plus son taux de natalité $n$ diminue et plus son taux de mortalité $m$ augmente. En partant de cette hypothèse et en appliquant quelques manipulations algébriques astucieuses, on peut montrer que l'équation précédente peut se réécrire :

$$ y'(t) = r y(t) \big(1 - \frac{y(t)}{K}\big) $$

avec $K$ une constante appelée *capacité d'accueil*, elle correspond au nombre d’individus maximal que le milieu peut accueillir (selon l'espace, les ressources ...)

## Modèle de Lotka-Volterra

Les modèles de Lotka-Volterra sont des sytèmes d'équations simples qui sont apparus au début du 20<sup>ème</sup> siècle. Ils portent le nom de deux mathématiciens qui ont publié en même temps mais indépendamment sur le sujet : Volterra, en 1926, pour modéliser les populations de sardines et de leurs prédateurs et Lotka, en 1924, dans son livre _Elements of Physical Biology_.

<p align="center">
   <img src="/assets/images/lotka_volterra_photos.png" width="50%"/>
</p>

### *Proie-prédateur*

Le modèle proie-prédateur de Lotka-Volterra a permis d'expliquer des données collectées de certaines populations d'animaux comme le lynx et lièvre ainsi que le loup et l'élan aux Etats-Unis. On y représente l'évolution du nombre proies $x$ et de prédateurs $y$ au cours du temps $t$ selon le modèle suivant :

$$
\left\{
  \begin{array}{ccc}
    x'(t) = x(t)\ \big(\alpha - \beta y(t)\big) \\
    y'(t) = y(t) \ \big( \delta x(t) - \gamma\big)
  \end{array}
\right.
$$

où:

- $\alpha$ et $\delta$ sont les taux de reproduction, respectivement, des proies et des prédateurs
- $\beta$ et $\gamma$ sont les taux de mortalité, respectivement, des proies et des prédateurs.

**Note:** On parle de système **autonome** : le temps $t$ n'apparaît pas explicitement dans les équations.
{: .notice--primary}

<p align="center">
   <img src="/assets/images/lotka_volterra.png" width="100%"/>
</p>

**Attention:** Les unités de la simulation ne reflète pas la réalité, il faut des populations suffisamment grandes pour que la modélisation soit correcte.
{: .notice--danger}

### *Compétition*

## runge kutta

methode numerique d'approximation de solutions d'equations différentielles, elle calcule itérativement des estimations de plus en plus précise

animation courbe qui approche petit à petit une solution theorique
