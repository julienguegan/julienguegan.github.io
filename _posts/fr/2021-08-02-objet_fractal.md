---
title: "Objet Fractal : Dimension, Auto-similarité, Infini"
date: 2021-08-02T15:34:30-04:00
lang: fr
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
gallery:
  - image_path: /assets/images/fractal_1.png
  - image_path: /assets/images/fractal_2.png
  - image_path: /assets/images/fractal_3.png
header:
  teaser: /assets/images/teaser_fractal.jfif
---

> *Elles sont présentes dans les forêts tropicales, à la pointe de la recherche médicale, dans les films et partout où reigne la communication sans fil. Ce mystère de la nature a enfin été percé à jour. "Bon sang ! Mais c'est bien sûr !". Peut-être n'avez vous jamais entendu parler de ces formes étranges, pourtant elles sont partout autour de vous. Leur nom : les fractales.*

<cite> reportage ARTE </cite> -- à la recherche de la dimension cachée
{: .small}

{% include video id="Tpsu2uz9rCE" provider="youtube" %}

## Introduction

Comme vous l'aurez compris si vous avez regardé l'excellent documentaire d'ARTE présenté ci-dessus, les fractales sont des objets géométriques *infiniment morcelés* qui ont la particularité de présenter des structures similaires à toutes les échelles. Ce type de géométrie permet de modéliser avec de simples formules récursives des figures infiniment complexes mais aussi de décrire des phénomènes naturels (motifs des flocons,  chemin pris par la foudre, forme d'un choux de romanesco, structure des poumons ...) et de trouver des applications dans des domaines technologiques (antennes, transistors, génération graphique de paysages ...).

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
{: .text-justify}

<p align="center">
  <img src="/assets/images/von_koch.gif" width="60%"/>
</p>

**Note:** L'approche présentée précédemment est conceptuelle. Une définition rigoureuse et définie pour tout ensemble est la [dimension de Hausdorff](https://fr.wikipedia.org/wiki/Dimension_de_Hausdorff). Elle est peu aisée à mettre en oeuvre...
{: .notice--primary}

On peut différencier 3 catégories principales de fractale :

- les systèmes de **fonctions itérées**. Ils ont une règle géométrique fixe comme le flocon de Von Koch, le tapis de Sierpinski, la courbe de Peano.
- les fractales **aléatoires**. Elles sont générées par un processus stochastiques comme dans la nature ou les paysages fractales.
- les ensembles définies par une **relation de récurrence** en chaque point d'un espace. On peut citer l'ensemble de Julia, de mandelbrot, de lyapunov. On les appelle parfois en anglais des *Escape-time fractals*.

## Ensemble de Julia

L'ensemble de Julia associé à un nombre complexe $c$ fixé est l'ensemble des valeurs initiales $z_0$ pour lesquelles la suite suivante est bornée :

$$
\left\{
  \begin{array}{ll}
    z_0 \in \mathbb{C} \\
    z_{n+1} = z_n^2 + c
  \end{array}
\right.
$$

Pour générer un ensemble de Julia informatiquement, l'idée est de discrétiser l'espace dans un intervalle fixé pour avoir un nombre fini de valeur $z_0$ pour lesquelle on va tester la convergence de la suite.

```python
# INITIALIZATION
# value of c fixed
c_reel, c_imag = 0.3, 0.5 
# interval limit
x_min, x_max = -1, 1
y_min, y_max = -1, 1
# discretization
size = 5000 
x_step = (x_max - x_min)/size
y_step = (y_max - y_min)/size
M = np.zeros((size,size))
```

Pour pouvoir travailler avec des nombres complexes, j'ai choisi de décomposer la partie réelle `z_reel` et la partie imaginaire `z_image`. Ensuite, on teste la convergence pour un point donné en regardant si on a pas dépassé un nombre fini d'itération `n_iter < itermax`. On peut également, en plus, vérifier que la suite $(z_n)$ est divergente si son module est strictement supérieur à $2$, `z_reel**2 + z_imag**2 < 4` (cf [demonstration](https://fr.wikipedia.org/wiki/Ensemble_de_Mandelbrot#Barri%C3%A8re_du_module_%C3%A9gal_%C3%A0_2)). Finalement, on peut remplir une matrice `M` de $0$ ou de $1$ selon le test de convergence. Mais, pour un rendu visuelle final plus estéthique on peut également remplir la matrice `M` selon le taux de convergence estimé avec `n_iter/itermax`.
{: .text-justify}

```python
# LOOP ON ALL PIXEL = COMPLEX PLANE
for i in (range(size)):
    for j in range(size):
        n_iter = 0
        # convert pixel position to a complex number
        z_reel = i * x_step + x_min
        z_imag = j * y_step + y_min
        # update sequence until convergence
        while (z_reel**2 + z_imag**2 < 4) and (n_iter < itermax):
            z_reel, z_imag = z_reel**2 - z_imag**2 + c_reel, 2*z_imag*z_reel + c_imag
            n_iter = n_iter + 1
        # color image according to convergence rate
        M[j,i] = 255*n_iter/itermax
```

**Astuce:** En python, on aurait pu directement utiliser la fonction `complex()` pour avoir un objet complexe. Dans ce cas, les variables `z_reel` et `z_imag` seraient inutiles et on pourrait directement récupérer la valeur absolue et mettre au carré une unique variable complexe `z`.
{: .notice--info}

Finalement, on peut générer des ensembles de Julia pour différentes valeurs de $c$ fixées et pour changer le visuel on peut s'amuser à tester différentes *colormap*. Ci-dessous quelques résultats que j'ai généré.

{% include gallery %}

On remarque que les figures obtenues varient grandement en fonction de la valeur du complexe $c$ choisie. En fait, on peut générer les ensembles de julia pour une suite de complexes consécutifs pour voir comment les figures évoluent et en faire une animation.

<p align="center">
  <img src="/assets/images/fractal_julia.gif" width="40%"/>
</p>

## Ensemble de Mandelbrot

L'ensemble de Mandelbrot est fortement lié aux ensembles de Julia, en effet on peut définir l'ensemble de Mandelbrot $M$ comme l'ensemble des complexes $c$ pour lesquels l'ensemble de Julia $J_c$ correspondant est **connexe**, c'est-à-dire qu'il est fait d'un seul morceau. On peut dire que l'ensemble de Mandelbrot représente une carte des ensembles de Julia. Et, contrairement au nom qu'il porte, c'est les mathématiciens Julia et Fatou qui l'ont découvert et qui ont montré que la définition précédente est équivalente à l'ensemble des points $c$ du plan complexe $\mathbb{C}$ pour lesquels la suite suivante est bornée :
{: .text-justify}

$$
\left\{
  \begin{array}{ll}
    z_0 = 0 \\
    z_{n+1} = z_n^2 + c
  \end{array}
\right.
$$

Cette définition est très similaire à celle de l'ensemble de Julia à la différence qu'on s'intéresse à la variable $c$. Dans le code précédent, il faudrait modifier la ligne `z_reel = i * x_step + x_min` par `c_reel = i * x_step + x_min` et fixé `z_reel = 0` (idem pour la partie imaginaire). On obtient la figure suivante :

<p align="center">
  <img src="/assets/images/mandelbrot.png" width="40%"/>
</p>

**Note:** Benoît Mandelbrot est le fondateur de la théorie fractale avec notamment son article _"How Long Is the Coast of Britain ? Statistical Self-Similarity and Fractional Dimension"_ en 1967. C'est également lui qui obtient pour la première fois, une visualisation par ordinateur de cet ensemble.
{: .notice--primary}

## Logiciels

La génération de fractale n'est pas une tâche facile : beaucoup de paramètres peuvent être à prendre en compte et les temps de calcul sont souvent long. Dans les figures que j'ai généré, on ne voit pas au premiers abords le caractère auto-similaire des fractales, il faudrait changer d'échelle en zoomant de plus en plus profond sur un point du plan.
{: .text-justify}

Il existe de nombreux logiciels générateur de fractal gratuits disponibles. Ils sont souvent optimisés pour faire du multi-processing ou du calcul sur GPU, possèdent une interface graphique pour gérer les nombreux paramètres et sont parfois capables de créer des objets 3D (comme les 3 affichés ci-dessous). Une liste assez complète est disponible [ici](https://en.wikipedia.org/wiki/Fractal-generating_software#Programs).
{: .text-justify}

| ![image](/assets/images/mandelbulb3d.png)     | ![image](/assets/images/mandelbulber.png) | ![image](/assets/images/fragmentarium.png) |
|:---------------------------------------------:| :----------------------------------------:| :-----------------------------------------: |
| [Mandelbul3D](https://www.mandelbulb.com/2014/mandelbulb-3d-mb3d-fractal-rendering-software/)| [Mandelbuler](https://mandelbulber.com/) |  [Fragmentarium](https://syntopia.github.io/Fragmentarium/get.html) |

Et pour ceux qui ne veulent pas se compliquer et juste se laisser porter par la géométrie psychadélique des fractales sans effort, vous pourrez trouver sur internet des gens qui ont déjà fait le travail à votre place. On trouve sur youtube une floppée de vidéos comme par exemple *The Hardest Trip - Mandelbrot Fractal Zoom* qui zoome pendant 2h30 sur un point précis du plan complexe.

{% include video id="LhOSM6uCWxk" provider="youtube" %}

---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/fractale.ipynb) [![Generic badge](https://img.shields.io/badge/execute_le_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/fractale.ipynb)