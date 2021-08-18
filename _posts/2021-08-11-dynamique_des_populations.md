---
title: "Dynamique des populations"
date: 2021-08-11T22:18:30-04:00
classes: wide
layout: single
categories:
  - blog
tags:
  - équations différentielles ordinaires
  - écologie
  - modélisation
  - mathématiques appliquées
  - équation logistique
header:
  teaser: /assets/images/teaser_dynamique_population.jpg
---

Parmis les enjeux du 21<sup>ème</sup> siècle, l'écologie a un rôle majeure puisqu'elle est la science qui étudie les interactions des êtres vivants entre eux et avec leur milieu. Pour modéliser ces interactions, la dynamique des populations est la branche qui s'intéresse aux fluctuations démographiques des espèces. Ses applications sont nombreuses puisqu'elle peut permettre de répondre à des problèmes variés comme la gestion d'espèces menacées, la protection des cultures contre des nuisibles, le contrôle de bioréacteurs ou la prédiction des épidémies.
{: .text-justify}

## Modèle de Verhulst

A la fin du 18<sup>ème</sup> siècle, le modèle de **Malthus** décrit la variation d'une taille de population $y$ au cours du temps $t$ par l'équation différentielle :

$$ y'(t) = (n-m) y(t) = r y(t) $$

avec les constantes : $n$ le taux de natalité, $m$ le taux de mortalité et $r$ le taux de croissance. Ce modèle nous dit que, selon le taux de croissance $r$, la taille des populations peut soit diminuer, rester constante ou augmenter de manière exponentielle. Ce modèle ne reflète pas la réalité puisque une population n'augmentera jamais à l'infini.

<p align="center">
   <img src="/assets/images/malthus_verlhust_photos.png" width="50%"/>
</p>

En 1840, **Verlhust** propose un modèle de croissance plus adapté en partant de l'hypothèse que le taux de croissance $r$ n'est pas une constante mais est fonction affine de la taille de population $y$ :

$$ y'(t) = \big(n(y) - m(y)\big) y(t) $$

Verlhust part notamment de l'hypothèse que plus la taille d'une population augmente alors plus son taux de natalité $n$ diminue et plus son taux de mortalité $m$ augmente. En partant de cette hypothèse et en appliquant quelques manipulations algébriques astucieuses, on peut montrer que l'équation différentielle précédente peut se réécrire sous la forme :

$$ y'(t) = r y(t) \left(1 - \frac{y(t)}{K}\right) $$

avec $K$ une constante appelée *capacité d'accueil*. On peut résoudre analytiquement cette équation avec la condition initiale $y(t=0)=y_0$, on obtient la **solution logistique** :

$$ y(t) = \frac{K}{1+\left(\frac{K}{y_0}-1\right)e^{-rt}} $$

<p align="center">
   <img src="/assets/images/verlhust_graph.png" width="70%"/>
</p>

<details>
  <summary>Résolution détaillée de l'équation différentielle logistique par séparation de variable</summary>

  $$
  \begin{align*}
    \int_{y_0}^{y(t)} \frac{1}{y(1-y/K)}dy &= \int_0^t r \ d\tau \\
    \int_{y_0}^{y(t)} \frac{K}{y(K-y)}dy &= \int_0^t r \ d\tau \\
    \int_{y_0}^{y(t)} \frac{1}{y}dy +  \int_{y_0}^{y(t)} \frac{1}{K-1}dy &= \int_0^t r \ d\tau \\
    \ln \left| \frac{y(t)}{y_0} \right| - \ln \left| \frac{K-y(t)}{K-y_0} \right| &= r \ t \\
    \ln \left( \frac{y(t)\big(K-y_0\big)}{y_0\big(K-y(t)\big)} \right) &= r \ t \\
    \frac{y(t)}{K-y(t)} &= \frac{y_0}{K-y_0}e^{rt} \\
    y(t)\left(1+\frac{y_0}{K-y_0}e^{rt} \right) &= \frac{K y_0 e^{rt}}{K-y_0} \\
    y(t) &= \frac{Ky_0e^{rt}}{K-y_0+y_0e^{rt}} \\
    y(t) &= \frac{K y_0}{(K-y_0)e^{-rt}+y_0} \\
  \end{align*} \\
  \square
  $$
</details> {: .notice--primary}

On remarque que $ \lim\limits_{t\to\infty} y(t) = K $. Ce qui signifie que peut importe la taille de la population initiale $y_0$, la population finira toujours par tendre vers $K$ la capacité d'accueil qu'on qualifie souvent comme le nombre d’individus maximal que le milieu peut accueillir (selon l'espace, les ressources ...). Cette [fonction dite logistique](https://fr.wikipedia.org/wiki/Fonction_logistique_(Verhulst)) introduite pour la première fois par Verlhust pour modéliser la croissance des populations trouvera par la suite plein d'application dans des domaines variés comme l'économie, la chimie, les statistiques et plus récemment les réseaux de neurones artificielles.
{: .text-justify}

## Modèle de Lotka-Volterra

Les modèles de Lotka-Volterra sont des sytèmes d'équations simples qui sont apparus au début du 20<sup>ème</sup> siècle. Ils portent le nom de deux mathématiciens qui ont publié en même temps mais indépendamment sur le sujet : Volterra, en 1926, pour modéliser les populations de sardines et de leurs prédateurs et Lotka, en 1924, dans son livre _Elements of Physical Biology_. Contrairement au modèle de Verlhust qui s'intéresse à une seule population, les modèles de Lotka-Volterra modélisent les interactions entre plusieurs espèces, chacune ayant un impact sur le développement de l'autres.
{: .text-justify}

<p align="center">
   <img src="/assets/images/lotka_volterra_photos.png" width="50%"/>
</p>

### *Proie-prédateur*

Le modèle proie-prédateur de Lotka-Volterra a permis d'expliquer des données collectées de certaines populations d'animaux comme le lynx et lièvre ainsi que le loup et l'élan aux Etats-Unis. On y représente l'évolution du nombre proies $x$ et de prédateurs $y$ au cours du temps $t$ selon le modèle suivant :
{: .text-justify}

$$
\left\{
  \begin{array}{ccc}
    x'(t) = x(t)\ \big(\alpha - \beta y(t)\big) \\
    y'(t) = y(t)\ \big( \delta x(t) - \gamma\big)
  \end{array}
\right.
$$

avec les paramètres $\alpha$ et $\delta$ sont les taux de reproduction respectivement des proies et des prédateurs et $\beta$ et $\gamma$ sont les taux de mortalité, respectivement, des proies et des prédateurs. 
{: .text-justify}

**Note:** On parle de système autonome : le temps $t$ n'apparaît pas explicitement dans les équations.
{: .notice--primary}

Si on développe chacune des équations, on peut plus facilement donner une interprétation. Pour les proies, on a d'une part le terme $\alpha x(t)$ qui modélise la croissance exponentielle avec une source illimitée de nourriture et d'autre part $- \beta x(t) y(t)$ qui représente la prédation proportionnelle à la fréquence de rencontre entre prédateurs et proies. L'équation des prédateurs est très semblable à celle des proies, $\delta x(t)y(t)$ est la croissance des prédateurs proportionnelle à la quantité de nourriture disponible (les proies) et $- \gamma y(t)$ représente la mort naturelle des prédateurs.
{: .text-justify}

animation pixellique de lapin et renard

On peut caculer les équilibres de ce système d'équations différentielles et également en déduire un comportement mais les solutions n'ont pas d'expression analytique simple. Néanmoins, il est possible de calculer une solution approchée numériquement (plus de détails dans la [`section suivante`](#méthode-numérique-pour-les-EDO)).
{: .text-justify}

```python
# define ODE to resolve
r, c, m, b = 3, 4, 1, 2
def lotka_volterra(XY, t=0):
    dX = r*XY[0] - c*XY[0]*XY[1]
    dY = b*XY[0]*XY[1] - m*XY[1]
    return [dX, dY]
```
```python
# discretization
T0   = 0
Tmax = 12
n    = 200
T    = np.linspace(T0, Tmax, n) 
```
```python
# TEMPORAL DYNAMIC
solution = integrate.odeint(lotka_volterra, X0, T) # use scipy solver
```
<p align="center">
   <img src="/assets/images/lotka_volterra_graph2.png" width="70%"/>
</p>
```python
# PHASES SPACE
# some trajectories
orbits = []
for i in range(5):
    X0    = [0.2+i*0.1, 0.2+i*0.1]
    orbit = integrate.odeint(lotka_volterra, X0, T)
    orbits.append(orbit) 
# vector field
x, y             = np.linspace(0, 2.5, 20), np.linspace(0, 2, 20)
X_grid, Y_grid   = np.meshgrid(x, y)                      
DX_grid, DY_grid = lotka_volterra([X_grid, Y_grid])
N                = np.sqrt(DX_grid ** 2 + DY_grid ** 2) 
N[N==0]          = 1
DX_grid, DY_grid = DX_grid/N, DY_grid/N
```

<p align="center">
   <img src="/assets/images/lotka_volterra_graph1.png" width="70%"/>
</p>

**Attention:** Les unités des simulations ne reflète pas la réalité, il faut des populations suffisamment grandes pour que la modélisation soit correcte.
{: .notice--danger}

Dans le modèle utilisé, les prédateurs prospèrent lorsque les proies sont nombreuses, mais finissent par épuiser leurs ressources et déclinent. Lorsque la population de prédateurs a suffisamment diminué, les proies profitant du répit se reproduisent et leur population augmente de nouveau. Cette dynamique se poursuit en un cycle de croissance et déclin. Il existe 2 équilibres : le point $(0,0)$ est un point de selle instable qui montre que l'extinction des 2 espèce est difficile à obtenir et le point $(\frac{\gamma}{\delta}, \frac{\alpha}{\beta})$ est un centre stable, les populations oscillent autour cet état.

**Note:** Cette modélisation reste assez simple, un grande nombre de variante existe. On peut rajouter des termes de disparition des 2 espèces (dus à la pêche, chasse ...), tenir compte de la capacité d'accueil du milieu en ajoutant un terme logistique.
{: .notice--info}

### *Compétition*

$$
\left\{
  \begin{array}{ccc}
    x_1'(t) = r_1x_1(t)\left(1- \frac{x_1(t)+\alpha_{12}x_2(t)}{K_1}\right) \\
    x_2'(t) = r_2x_2(t)\left(1- \frac{x_2(t)+\alpha_{21}x_1(t)}{K_2}\right)
  \end{array}
\right.
$$

## Méthode numérique pour les EDO

methode numerique d'approximation de solutions d'equations différentielles, elle calcule itérativement des estimations de plus en plus précise

animation courbe qui approche petit à petit une solution theorique
