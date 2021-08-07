---
title: "Objet Fractal : Dimension, Auto-similarité, Infini"
date: 2021-08-02T15:34:30-04:00
classes: wide
layout: single
categories:
  - blog
tags:
  - Fractal
  - Python
  - Julia
  - Mandelbrot
  - Récursivité
  - Infini
---

> *Elles sont présentes dans les forêts tropicales, à la pointe de la recherche médicale, dans les films et partout où reigne la communication sans film. Ce mystère de la nature a enfin été percé à jour. "Bon sang ! Mais c'est bien sûr !". Peut-être n'avez vous jamais entendu parler de ces formes étranges, pourtant elles sont partout autour de vous. Leur nom : les fractales.*
>
> --<cite> reportage ARTE - à la recherche de la dimension cachée </cite>

{% include video id="Tpsu2uz9rCE" provider="youtube" %}

## Introduction

Comme vous l'aurez compris si vous avez regardé l'excellent documentaire d'ARTE ci-dessus, les fractales sont des objets géométriques *infiniment morcelés* qui ont la particularité de présenter des structures similaires à toutes les échelles. Ce type de géométrie permet de modéliser avec de simples formules récursives des figures infiniment complexes mais aussi de décrire des phénomènes naturels comme (les motifs des flocons, le chemin pris par la foudre, la forme d'un choux de romanesco, la structure des poumons ...) et de trouver des applications dans des domaines technologiques (antennes, transistors, génération graphique de paysages ...).

<p align="center">
   <img src="/assets/images/fractals_in_nature.png" width="100%"/>
</p>

**Note:** Les fractales qu'on trouve dans la nature sont des approximations finies des vrais objets mathématiques.
{: .notice--primary}

Les fractales sont notamment caractérisées par la notion contre intuitive de **dimension non entière**. En effet, on peut définir une régle générale de mise à l'échelle qui relie la mesure $N$, un facteur d'échelle $\varepsilon$ et la dimension $D$ :

$$ N = \varepsilon^{-D} $$

Par exemple, pour une figure géométrique usuelle comme le carré, sa dimension est $D=2$ et si on le subdivise en $3$ son aire est $N=9$, on a bien $9=\frac{1}{3}^{-2}$. On peut appliquer ce même raisonnement pour un cube ou même une ligne.

<p align="center">
  <img src="/assets/images/scaling_rule.png" width="60%"/>
</p>

Maintenant, on cherche à trouver la dimension d'une figure fractale simple. La formule précédente nous donne :

$$ D = -\frac{\log N}{\log \varepsilon} $$

Si on s'intéresse à une figure telle que la courbe de Von Koch qui consiste, à partir d'un segment, construire récursivement des triangles équilatéraux sur chaque sous-segment (cf animation ci-dessous). En comptant les segments à chaque nouvelle mise à l'échelle, on comprends que la longueur de la courbe de Koch est multipliée par $4$ pour chaque mise à l'échelle $\varepsilon=\frac{1}{3}$ (on divise les segments par 3). On trouve donc que sa dimension est $D = \frac{\log 4}{\log 3} \approx 1.26$. Il ne s'agit pas d'une simple courbe unidimensionelle, ni d'une surface mais quelque chose "entre les deux".

<p align="center">
  <img src="/assets/images/von_koch.gif" width="60%"/>
</p>

**Note:** L'approche présentée précédemment est conceptuelle et didactique. Une définition rigoureuse et définie pour tout ensemble est la [dimension de Hausdorff](https://fr.wikipedia.org/wiki/Dimension_de_Hausdorff). Elle est peu aisée à mettre en oeuvre...
{: .notice--primary}

## julia

fractale par recurrence (escape-time fractal)
julia : equations

## mandelbrot

représente les fractales de julia connexes ou pas

## generation software

optimisé les temps de calcul sur gpu en c++, des softwares existent

mettre des screenshot avec legende = link to website
- kalles fraktaler
- superFractalThing
- mandelbulber
- fragmentarium
- maxima

| ![image](/assets/images/mandelbulb3d.png)     | ![image](/assets/images/mandelbulber.png) | ![image](/assets/images/fragmentarium.png) |
|:---------------------------------------------:| :----------------------------------------:| :-----------------------------------------: |
| [Mandelbul3D](https://www.mandelbulb.com/2014/mandelbulb-3d-mb3d-fractal-rendering-software/)| [Mandelbuler](https://mandelbulber.com/) |  [Fragmentarium](https://syntopia.github.io/Fragmentarium/get.html) |
