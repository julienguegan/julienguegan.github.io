---
title: "Simulation de Fluide : Navier-Stokes, grille, JavaScript"
lang: fr
classes: wide
header:
  teaser: /assets/images/teaser_fluid_simulation.png
read_time: true
published: true
---

La mécanique des fluides est une branche de la physique étudiée depuis longtemps par de nombreux scientifiques, que ce soit par Archimède, Newton, Bernoulli ou encore Lagrange. Elle permet de comprendre comment les fluides (les gaz et les liquides) se comportent et aide les ingénieurs à construire des turbines, prédire la météo et améliorer l'aérodynamisme des avions. Généralement, dans la vie de tous les jours il est difficiel de s'imaginer les trajectoire d'un fluide si ce n'est regarder l'eau qui coule de notre robinet. En cherchant un peu, on se rend compte que les trajectoires des fluides peuvent être assez chaotiques avec de nombreuses turbulences et vortex. Dans ce post, je vais simuler via les équations de la physique et visualiser comment un fluide réagit au clic de ma souris avec un petit code en Javascript. Pour se faire, je me suis grandement inspiré du travail de Matthias Müller et de sa [vidéo](https://www.youtube.com/watch?v=iKAVRgIrUOU) ainsi que de son document [17-fluidSim.pdf](https://matthias-research.github.io/pages/tenMinutePhysics/17-fluidSim.pdf) qui décortique la magie derrière la mécanique des fluides numérique (CFD).

<p align="center">
   <img src="/assets/images/fluid_simulation_header.png" width="90%"/>
</p>

## Les Équations de Navier-Stokes

Les [équations de Navier-Stokes](https://fr.wikipedia.org/wiki/Équations_de_Navier-Stokes) régissent le comportement des fluides et décrivent leur comportements avec des termes aux dérivées partielles non linéaire. De nombreuses déclinaisons existe selon le cas qu'on étudie, ici j'utiliserais 2 équations princiaples : la conservation de la quantité de mouvement et la condition d'incompressibilité. La première correspond en fait à la 2ème loi de Newton $F = ma$ appliquée à un fluide. Si on note $u$ le champ de vitesse du fluide, $p$ la pression et les coefficients de densité $\rho$ et de viscosité $\nu$ alors on peut l'écrire de la façon suivante:

$$
\underbrace{\frac{\partial \mathbf{u}}{\partial t}}_{\text{Accélération}} +
\underbrace{(\mathbf{u} \cdot \nabla) \mathbf{u}}_{\text{Advection}} =
\underbrace{-\frac{1}{\rho} \nabla p}_{\text{Gradient de Pression}} +
\underbrace{\nu \nabla^2 \mathbf{u}}_{\text{Viscosité}} +
\underbrace{\mathbf{f}}_{\text{Forces}}
$$

- L'accélération $\frac{\partial u}{\partial t}$ : C'est l'accélération locale. Elle décrit comment la vitesse en un point précis de la grille change d'un instant à l'autre. Si vous ne touchez à rien et que le fluide finit par s'immobiliser, c'est ce terme qui traduit cette évolution vers le repos.

- L'advection $(u \cdot \nabla) u$ : C'est le terme responsable du comportement chaotique et non-linéaire. Il décrit comment la vitesse du fluide se transporte elle-même à travers l'espace. C'est ce qui crée les tourbillons (vortex).

- Le gradient de pression $-\frac{1}{\rho} \nabla p$ : Il garantit que le fluide est poussé des zones de haute pression vers les zones de basse pression. C'est le moteur qui "répare" le champ de vitesse pour le maintenir incompressible.

- La viscosité $\nu \nabla^2 u$ : Elle représente le "frottement" interne ou la diffusion de la quantité de mouvement. Un fluide très visqueux résiste au glissement de ses couches internes (comme le miel).

- Les forces externes $f$ : Généralement la gravité. Dans notre démo, ça sera l'interaction avec l'utilisateur via la souris.

La seconde équation modélise la condition d'incompressibilité. Pour la fumée ou l'eau, on considère souvent que le fluide ne peut pas être compressé (son volume reste constant) :

$$
\nabla \cdot \mathbf{u} = 0
$$

Cette équation de divergence nulle signifie mathématiquement que tout ce qui entre dans une cellule de notre grille doit obligatoirement en ressortir. C'est la contrainte qui force le fluide à créer des courbes et des tourbillons plutôt que de simplement s'écraser et disparaitre contre un mur. 

## Modéliser le mouvement

Lors de la simulation de fluides, deux points de vue existent, chacun ayant sa propre manière d'interpréter le mouvement :

1. <u>Approche Lagrangienne</u> : on traite le fluide comme une collection de particules, on suit un ensemble de points individuellement s'entrechoquer au cours du temps. C'est intuitif, comme simuler un tas de billes.

    | Points Forts | Points Faibles |
    | :----------- | :------------- |
    | les frontières complexes et les éclaboussures sont gérées de façon naturelle et efficace. Puisque la masse est portée par les particules, elle est conservée par défaut. | garantir que le fluide ne se "comprime" pas (l'incompressibilité) nécessite des calculs de voisinage très coûteux, car il faut savoir quelles particules sont proches les unes des autres à chaque instant. |

   <div style="display: flex; justify-content: center; gap: 10%; margin-top: 1rem; margin-bottom: 2rem;">
   <img src="/assets/images/fluid_simulation_lagrange.png" style="width: 70%; height: auto;" alt="Description 2">
   </div>

2. <u>Approche Eulérienne</u> : au lieu de suivre le mouvement, on fixe notre regard sur des points précis de l'espace. On divise le domaine en une grille (un peu comme les pixels d'une image) et on observe comment les propriétés (vitesse, pression, densité) évoluent dans chaque cellule.

   | Points Forts | Points Faibles |
   | :----------- | :------------- |
   | les calculs sont plus stable numériquement. Le calcul des gradients de pression et de la diffusion est beaucoup plus simple sur une structure fixe, ce qui permet de simuler facilement des fluides qui conservent un volume constant. | un phénomène de "dissipation numérique" (le fluide a tendance à devenir un peu trop visqueux ou flou avec le temps) peut avoir lieu si la résolution de la grille n'est pas assez fine. |

   <div style="display: flex; justify-content: center; gap: 10%;margin-top: 1rem; margin-bottom: 2rem;">
   <img src="/assets/images/fluid_simulation_euler.png" style="width: 70%; height: auto;" alt="Description 1">
   </div>

Pour cette simulation, j'utiliserais l'approche Eulérienne car elle est souvent plus facile à gérer pour l'incompressibilité et produit de meilleurs résultats simuler la fumée. On peut obtenir des mouvements tourbillonnants comme les vortex si caractéristiques des gaz, tout en maintenant un rendu visuel lisse et continu, là où une approche par particules pourrait paraître "granuleuse".

## Résolution Numérique

Dans une simulation Eulérienne, la manière dont on organise les données sur la grille est cruciale. Une approche naïve consisterait à stocker toutes nos variables au centre de chaque cellule, on parle de grille colocalisée. Cependant, elle ne permet de calculer correctement les gradients d'une cellule à l'autre. On utilise généralement une **grille décalée** (ou [*Marker-and-Cell grid*](https://en.wikipedia.org/wiki/Marker-and-cell_method)) en séparant les composants : la pression $p$ reste au centre de la cellule $(i, j)$, la vitesse horizontale $u$ est décalée sur les faces verticales $(i \pm \frac{1}{2}, j)$, la vitesse verticale $v$ est décalée sur les faces horizontale $(i, j \pm \frac{1}{2})$. Cette disposition permet de calculer plus facilement la pression et assure une stabilité numérique.

<p align="center">
  <img src="/assets/images/fluid_simulation_grid.png" alt="Diagramme de Grille Décalée" width="25%">
  <br>
  <i>L'arrangement de la Grille Décalée</i>
</p>

Ensuite, pour la résolution, on utilise la technique de [**splitting d'opérateur**](https://en.wikipedia.org/wiki/Splitting_method) où on décompose l'équation en plusieurs sous-parties et on résout chacune séquentiellement. À chaque pas de temps, la boucle exécute trois étapes dans l'ordre :

<div style="display:flex;align-items:center;justify-content:center;gap:0.75rem;margin:2rem 0;flex-wrap:wrap;">
  <div style="border:2px solid #ff9d00;border-radius:8px;padding:0.9rem 1.2rem;text-align:center;min-width:140px;">
    <div style="color:#ff9d00;font-size:0.65rem;font-weight:bold;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">Étape 1</div>
    <div style="font-weight:bold;margin-bottom:0.4rem;">Viscosité</div>
    <div style="font-size:0.85rem;opacity:0.75;">$\partial \mathbf{u}/\partial t = \nu \nabla^2 \mathbf{u}$</div>
    <div style="font-size:0.72rem;opacity:0.5;margin-top:0.35rem;">diffuse la vitesse</div>
  </div>
  <div style="color:#ff9d00;font-size:1.5rem;font-weight:bold;">→</div>
  <div style="border:2px solid #ff9d00;border-radius:8px;padding:0.9rem 1.2rem;text-align:center;min-width:140px;">
    <div style="color:#ff9d00;font-size:0.65rem;font-weight:bold;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">Étape 2</div>
    <div style="font-weight:bold;margin-bottom:0.4rem;">Projection</div>
    <div style="font-size:0.85rem;opacity:0.75;">$\nabla \cdot \mathbf{u} = 0$</div>
    <div style="font-size:0.72rem;opacity:0.5;margin-top:0.35rem;">corrige la pression</div>
  </div>
  <div style="color:#ff9d00;font-size:1.5rem;font-weight:bold;">→</div>
  <div style="border:2px solid #ff9d00;border-radius:8px;padding:0.9rem 1.2rem;text-align:center;min-width:140px;">
    <div style="color:#ff9d00;font-size:0.65rem;font-weight:bold;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">Étape 3</div>
    <div style="font-weight:bold;margin-bottom:0.4rem;">Advection</div>
    <div style="font-size:0.85rem;opacity:0.75;">$\mathbf{x}' = \mathbf{x} - \mathbf{u} \cdot \Delta t$</div>
    <div style="font-size:0.72rem;opacity:0.5;margin-top:0.35rem;">transporte le fluide</div>
  </div>
</div>

Cette technique permet de résoudre plus simplement que de tout résoudre simultanément au prix d'une légère approximation qui est suffisante pour une simulation en temps réel.

### Viscosité

Le premier terme qu'on traite est la viscosité, elle traduit le « frottement » interne : un fluide très visqueux s'écoule lentement comme le miel. On résout l'équation de diffusion : 

$$ \frac{\partial \mathbf{u}}{\partial t} = \nu \nabla^2 \mathbf{u} $$

Le Laplacien $\nabla^2 \mathbf{u}$ mesure à quel point la vitesse d'une cellule diffère de ses voisines — si elle est plus rapide, la viscosité la freine en la ramenant vers la moyenne. Sur grille discrète, on l'approche par [différences finies](https://fr.wikipedia.org/wiki/Méthode_des_différences_finies) — chaque cellule est comparée à ses quatre voisines.

<details style="margin:0.5rem 0 1rem;padding:0.6rem 1rem;border-left:3px solid #ff9d00;background:rgba(255,157,0,0.04);">
<summary style="cursor:pointer;font-weight:bold;color:#ff9d00;">Formule de discrétisation du Laplacien</summary>

$$\nabla^2 u_{i,j} \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}$$

</details>

La question suivante est : comment avancer dans le temps ? Le réflexe le plus simple serait de calculer la dérivée à l'instant $n$ et de l'utiliser directement pour prédire $n+1$. C'est le **schéma explicite**, mais il est instable : si le pas de temps ou la viscosité sont trop grands, les erreurs s'amplifient à chaque itération et la simulation explose. On utilise donc un **schéma implicite**, où le Laplacien est évalué à l'instant $n+1$ (l'inconnu). C'est comme corriger la trajectoire d'une voiture en anticipant le virage plutôt qu'en regardant où on était avant — on ne peut jamais sur-corriger, quelle que soit la taille du pas de temps.

Le prix à payer est qu'on ne peut plus calculer directement : $u^{n+1}$ apparaît des deux côtés de l'équation. En posant $\alpha = h^2 / (\nu \cdot dt)$, on obtient un système linéaire que [Gauss-Seidel](https://fr.wikipedia.org/wiki/Méthode_de_Gauss-Seidel) résout par itérations successives.

<details style="margin:0.5rem 0 1rem;padding:0.6rem 1rem;border-left:3px solid #ff9d00;background:rgba(255,157,0,0.04);">
<summary style="cursor:pointer;font-weight:bold;color:#ff9d00;">Équation à résoudre pour chaque cellule</summary>

$$u^{n+1}_{i,j} = \frac{u^n_{i,j} \cdot \alpha \;+\; u^{n+1}_{i+1,j} + u^{n+1}_{i-1,j} + u^{n+1}_{i,j+1} + u^{n+1}_{i,j-1}}{4 + \alpha}$$

</details>

Inverser ce système exactement coûterait trop cher pour une grille de $10\,000$ cellules. On utilise à la place Gauss-Seidel : on boucle sur toutes les cellules et on applique cette formule en réutilisant immédiatement les voisins déjà mis à jour dans l'itération courante. C'est comme une rumeur qui se propage : après quelques tours, tout le monde s'est aligné. Le paramètre $\alpha$ contrôle l'intensité de la diffusion — grand $\alpha$ (faible viscosité) : peu de changement ; petit $\alpha$ (forte viscosité) : la vitesse s'homogénéise fortement entre voisins. Le code en javascript s'écrit de la façon suivante :

```javascript
applyViscosity(dt, viscosity) {
    const alpha = (this.h * this.h) / (viscosity * dt);
    // valeur initiale du solveur
    this.bufferX.set(this.velocityX);   
    this.bufferY.set(this.velocityY);
    // boucle jusqu'a convergence (10 est assez)
    for (let iter = 0; iter < 10; iter++) {
        // pour chaque cellule de la grille
        for (let i = 1; i < this.nx - 1; i++) {
            for (let j = 1; j < this.ny - 1; j++) {
                const c = this.getIdx(i, j);
                const l = this.getIdx(i-1,j), r = this.getIdx(i+1,j);
                const b = this.getIdx(i,j-1), t = this.getIdx(i,j+1);
                this.bufferX[c] = (this.velocityX[c]*alpha + this.bufferX[l] + this.bufferX[r] + this.bufferX[b] + this.bufferX[t]) / (4 + alpha);
                this.bufferY[c] = (this.velocityY[c]*alpha + this.bufferY[l] + this.bufferY[r] + this.bufferY[b] + this.bufferY[t]) / (4 + alpha);
            }
        }
    }
    this.velocityX.set(this.bufferX);
    this.velocityY.set(this.bufferY);
}
```

**À retenir —** schéma implicite : stable quelle que soit la viscosité. Gauss-Seidel : 10 itérations suffisent, sans jamais inverser de matrice.
{: .notice--info}

### Incompressibilité

Un fluide incompressible ne peut pas se comprimer : tout ce qui entre dans une cellule doit en ressortir ($\nabla \cdot \mathbf{u} = 0$). Sur la grille MAC, la divergence d'une cellule est simplement la somme des flux sortants sur ses quatre faces : 

$$\text{div} = (u_x[i{+}1,j] - u_x[i,j]) + (u_y[i,j{+}1] - u_y[i,j])$$

Si elle est non nulle, il faut corriger les vitesses.

L'approche est locale et directe : on calcule de combien modifier les quatre faces pour annuler la divergence, en répartissant la correction symétriquement. Chaque face reçoit $p = -\text{div}/4$, ce qui ramène la divergence exactement à zéro pour cette cellule. Mais corriger une cellule modifie les faces partagées avec ses voisines, qui se retrouvent à leur tour légèrement divergentes. On ne peut donc pas tout régler en un seul passage — on boucle et chaque itération réduit l'erreur résiduelle jusqu'à ce qu'elle soit négligeable. Le facteur `overRelaxation = 1.9` accélère cette convergence : au lieu d'appliquer exactement la correction calculée, on en applique 1.9×. On "sur-corrige" légèrement, ce qui permet d'atteindre la même précision avec deux fois moins d'itérations. Au-delà de 2.0, le système divergerait — 1.9 est empiriquement le meilleur compromis pour ce type de problème.

```javascript
projectIncompressibility(iterations, overRelaxation) {
    for (let iter = 0; iter < iterations; iter++) {
        for (let i = 1; i < this.nx - 1; i++) {
            for (let j = 1; j < this.ny - 1; j++) {
                const c = this.getIdx(i,j), r = this.getIdx(i+1,j), t = this.getIdx(i,j+1);

                const divergence = this.velocityX[r] - this.velocityX[c] + this.velocityY[t] - this.velocityY[c];
                const pressure = (-divergence / 4) * overRelaxation;

                this.velocityX[c] -= pressure;   this.velocityX[r] += pressure;
                this.velocityY[c] -= pressure;   this.velocityY[t] += pressure;
            }
        }
    }
}
```

**À retenir —** on corrige la divergence cellule par cellule en itérant. La sur-relaxation (×1.9) divise par deux le nombre d'itérations nécessaires.
{: .notice--info}

### Advection

L'advection transporte les propriétés (densité, vitesse) le long du flux. Ce terme est le plus délicat : il est non-linéaire (la vitesse se transporte elle-même), et une discrétisation directe par différences finies serait instable — les erreurs s'amplifient exponentiellement au fil des pas de temps. On utilise à la place un schéma **Semi-Lagrangien**, qui exploite une propriété fondamentale de l'équation d'advection : chaque grandeur est conservée le long des trajectoires du fluide. Plutôt que de pousser les valeurs vers l'avant, on remonte en arrière dans le temps pour trouver d'où venait le fluide arrivant en chaque cellule.

$$
\mathbf{x}_{prev} = \mathbf{x} - \mathbf{u}(\mathbf{x}) \cdot \Delta t
\qquad\Rightarrow\qquad
q_{new}(\mathbf{x}) = q_{old}(\mathbf{x}_{prev})
$$

<p align="center">
  <img src="/assets/images/advection_diagram.png" alt="Advection Semi-Lagrangienne" width="50%">
  <br>
  <i>Schéma d'Advection Semi-Lagrangienne</i>
</p>

Ce schéma est inconditionnellement stable : on ne fait que lire des valeurs existantes, jamais les extrapoler. Son seul défaut est une légère **dissipation numérique** — l'[interpolation bilinéaire](https://fr.wikipedia.org/wiki/Interpolation_bilinéaire) nécessaire pour lire une valeur entre deux cellules introduit un léger flou, comme si le fluide était un tout petit peu plus visqueux que prévu. C'est un compromis acceptable pour la stabilité. On notera aussi que les calculs se font dans un buffer temporaire plutôt que directement dans les tableaux de vitesse : si on écrasait les valeurs au fur et à mesure, les premières cellules mises à jour pollueraient les lectures des suivantes — on ne lirait plus $q^n$ mais un mélange de $q^n$ et $q^{n+1}$.

```javascript
applyAdvection(dt, dissipation) {
    for (let i = 1; i < this.nx - 1; i++) {
        for (let j = 1; j < this.ny - 1; j++) {
            const idx = this.getIdx(i, j);

            // vitesse au centre de la cellule (moyenne des faces de la grille MAC)
            const u = (this.velocityX[idx] + this.velocityX[this.getIdx(i+1,j)]) * 0.5;
            const v = (this.velocityY[idx] + this.velocityY[this.getIdx(i,j+1)]) * 0.5;
le faire mais c'est plus de la bidouille qu'autre chose et Obsidian n'est pas adaptés a de la prod
            // remonter en arrière dans le temps
            const prevX = (i + 0.5) * this.h - dt * u;
            const prevY = (j + 0.5) * this.h - dt * v;

            this.bufferD[idx] = this.interpolate(prevX, prevY, FIELD_TYPE.DENSITY)    * dissipation;
            this.bufferX[idx] = this.interpolate(prevX, prevY, FIELD_TYPE.VELOCITY_X);
            this.bufferY[idx] = this.interpolate(prevX, prevY, FIELD_TYPE.VELOCITY_Y);
        }
    }
    this.density.set(this.bufferD);
    this.velocityX.set(this.bufferX);
    this.velocityY.set(this.bufferY);
}
```

**À retenir —** schéma semi-Lagrangien : inconditionnellement stable car on lit toujours des valeurs passées. Le léger flou introduit par l'interpolation est le prix à payer.
{: .notice--info}

## Démo Interactive

Ces trois étapes suffisent à produire un comportement étonnamment réaliste. Essayez de faire varier la viscosité ou la dissipation pour observer leur effet sur le fluide :

Voici une implémentation complète en JavaScript que vous pouvez essayer directement. Cliquez et maintenez pour ajouter de la fumée :

<style>
    #sim-container { position: relative; width: 100%; height: 600px; background: #050505; border-radius: 8px; overflow: hidden; border: 1px solid #333; }
    #simCanvas { width: 100%; height: 100%; display: block; }
    .interface-labo { position: absolute; top: 10px; left: 10px; background: rgba(10, 10, 10, 0.85); padding: 8px 10px; border-radius: 8px; border: 1px solid #444; width: 160px; z-index: 10; font-family: sans-serif; }
    .param-group { margin-bottom: 7px; }
    .param-group label { display: block; color: #ff9d00; font-size: 9px; text-transform: uppercase; margin-bottom: 2px; }
    .valeur-bulle { float: right; color: #fff; background: #222; padding: 0 4px; border-radius: 3px; }
    input[type=range] { width: 100%; accent-color: #ff9d00; cursor: pointer; }
    .btn-reset { width: 100%; padding: 5px; cursor: pointer; background: #222; border: 1px solid #444; color: #fff; border-radius: 4px; font-size: 11px; transition: 0.3s; margin-top: 2px; }
    .btn-reset:hover { background: #333; border-color: #ff9d00; }
</style>

<div id="sim-container">
    <div class="interface-labo">
        <div class="param-group">
            <label>Viscosité <span id="label-visc" class="valeur-bulle">0.00</span></label>
            <input type="range" id="input-visc" min="0" max="0.5" step="0.01" value="0">
        </div>
        <div class="param-group">
            <label>Rayon <span id="label-radius" class="valeur-bulle">0.05</span></label>
            <input type="range" id="input-radius" min="0.01" max="0.15" step="0.01" value="0.05">
        </div>
        <div class="param-group">
            <label>Dissipation <span id="label-diss" class="valeur-bulle">0.99</span></label>
            <input type="range" id="input-diss" min="0.90" max="1.0" step="0.001" value="0.99">
        </div>
        <button class="btn-reset" onclick="location.reload()">Réinitialiser</button>
    </div>
    <canvas id="simCanvas"></canvas>
</div>


<script src="/assets/js/fluid-simulation.js"></script>



Dans ce post, je me suis concentré sur les parties essentielles du calcul (diffusion, advection, résolution de pression), mais le code complet — boucle de rendu, initialisation, constructeur, conditions limites — est disponible [ici](https://github.com/julienguegan/notebooks_blog/blob/main/fluid-simulation.js). Il reste quelques instabilités : en jouant un peu avec la souris et le canvas, j'obtiens de temps en temps des pressions qui explosent ... Mais le résultat actuel me convient, si vous voulez allez plus loin vous pouvez explorer l'internet. Il existe de nombreuses autres manières de faire, différents phénomènes à modéliser plus ou moins précisément, avec des effets graphiques variés. Par exemple, un simulateur web plus poussé utilisant webgl peut être trouver à l'adresse : https://paveldogreat.github.io/WebGL-Fluid-Simulation/.

---

[![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/fluid-simulation.js) ![Generic badge](https://img.shields.io/badge/ecrit_avec-Javascript-yellow.svg?style=plastic&logo=javascript)