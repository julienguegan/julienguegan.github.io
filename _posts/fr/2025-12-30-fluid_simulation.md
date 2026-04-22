---
title: "Simulation de Fluide en 200 Lignes de Code"
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

Les équations de Navier-Stokes régissent le comportement des fluides et décrivent leur comportements avec des termes aux dérivées partielles non linéaire. De nombreuses déclinaisons existe selon le cas qu'on étudie, ici j'utiliserais 2 équations princiaples : la conservation de la quantité de mouvement et la condition d'incompressibilité. La première correspond en fait à la 2ème loi de Newton $F = ma$ appliquée à un fluide. Si on note $u$ le champ de vitesse du fluide, $p$ la pression et les coefficients de densité $\rho$ et de viscosité $\nu$ alors on peut l'écrire de la façon suivante:

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

Pour passer des équations à un programme, on utilise le **splitting d'opérateur** : chaque terme de Navier-Stokes est traité séquentiellement. À chaque image, la boucle exécute trois étapes dans l'ordre :

```javascript
simulator.applyViscosity(1/60, config.viscosity);       // diffuser la vitesse
simulator.projectIncompressibility(40, 1.9);            // garantir ∇·u = 0
simulator.applyAdvection(1/60, config.dissipation);     // transporter le fluide
```

Avant de détailler chaque étape, il faut comprendre comment les données sont organisées sur la grille.

### La Grille Décalée (Staggered Grid)

Dans une simulation Eulérienne, la manière dont nous organisons nos données sur la grille est cruciale. Une approche naïve consisterait à stocker toutes nos variables (pression et vecteurs de vitesse) au centre de chaque cellule. C'est ce qu'on appelle une **grille colocalisée**. Cependant, cette simplicité cache un piège redoutable : le **couplage vitesse-pression**. Si la pression et la vitesse sont au même endroit, le calcul des gradients de pression (qui font bouger le fluide) utilise souvent des cellules adjacentes. Mathématiquement, une pression qui oscille fortement d'une cellule à l'autre (haut, bas, haut, bas) pourrait avoir un gradient calculé de **zéro**, laissant le solveur "aveugle" à des variations de pression absurdes. Le fluide se fige alors dans un motif en damier instable.

Pour résoudre cela, on utilise une **grille décalée** (ou *Marker-and-Cell grid*). On sépare physiquement les composants :

* **Pression ($p$) :** Reste au centre de la cellule $(i, j)$. Elle représente le "poids" ou l'énergie interne de la zone.
* **Vitesse Horizontale ($u$) :** Décalée sur les faces **verticales** $(i \pm 1/2, j)$. Elle mesure le flux entrant et sortant par les côtés.
* **Vitesse Verticale ($v$) :** Décalée sur les faces **horizontales** $(i, j \pm 1/2)$. Elle mesure le flux par le haut et le bas.

Cette disposition offre trois avantages majeurs pour la précision numérique :

1. **Différences Centrales Naturelles :** Pour calculer comment la pression pousse le fluide horizontalement ($\frac{\partial p}{\partial x}$), on soustrait les pressions des deux cellules voisines. Le résultat tombe **exactement** là où se trouve la variable $u$. Pas d'interpolation nécessaire, pas de perte de précision.
2. **Conservation de la Masse :** La divergence du fluide ($\nabla \cdot \mathbf{u}$), qui garantit que notre fluide est incompressible, se calcule naturellement au centre de la cellule en utilisant les vitesses sur ses quatre faces.

$$\frac{u_{i+1/2, j} - u_{i-1/2, j}}{\Delta x} + \frac{v_{i, j+1/2} - v_{i, j-1/2}}{\Delta y} = 0$$

3. **Stabilité :** Elle élimine mathématiquement les modes de pression "fantômes" (le damier), agissant comme un filtre naturel pour les erreurs numériques.

<p align="center">
  <img src="/assets/images/fluid_simulation_grid.png" alt="Diagramme de Grille Décalée" width="50%">
  <br>
  <i>L'arrangement de la Grille Décalée</i>
</p>

### Viscosité

La viscosité traduit le « frottement » interne : un fluide très visqueux lisse sa propre vitesse au fil du temps. On résout l'équation de diffusion $\frac{\partial \mathbf{u}}{\partial t} = \nu \nabla^2 \mathbf{u}$ avec la méthode de Gauss-Seidel, où $\alpha = h^2 / (\nu \cdot dt)$ contrôle l'intensité de la diffusion.

```javascript
applyViscosity(dt, viscosity) {
    const alpha = (this.h * this.h) / (viscosity * dt);

    this.bufferX.set(this.velocityX);   // valeur initiale du solveur
    this.bufferY.set(this.velocityY);

    for (let iter = 0; iter < 10; iter++) {
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

### Projection (Incompressibilité)

Un fluide incompressible ne peut pas se comprimer : tout ce qui entre dans une cellule doit en ressortir ($\nabla \cdot \mathbf{u} = 0$). Pour chaque cellule, on mesure la divergence et on redistribue une correction de pression vers les quatre voisins jusqu'à convergence.

```javascript
projectIncompressibility(iterations, overRelaxation) {
    for (let iter = 0; iter < iterations; iter++) {
        for (let i = 1; i < this.nx - 1; i++) {
            for (let j = 1; j < this.ny - 1; j++) {
                const c = this.getIdx(i,j), r = this.getIdx(i+1,j), t = this.getIdx(i,j+1);

                const divergence = this.velocityX[r] - this.velocityX[c]
                                 + this.velocityY[t] - this.velocityY[c];
                const pressure = (-divergence / 4) * overRelaxation;

                this.velocityX[c] -= pressure;   this.velocityX[r] += pressure;
                this.velocityY[c] -= pressure;   this.velocityY[t] += pressure;
            }
        }
    }
}
```

### Advection

L'advection transporte les propriétés (densité, vitesse) le long du flux. On utilise un schéma **Semi-Lagrangien** : au lieu de suivre des particules vers l'avant, on remonte en arrière dans le temps pour trouver d'où venait le fluide arrivant en chaque cellule.

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

```javascript
applyAdvection(dt, dissipation) {
    for (let i = 1; i < this.nx - 1; i++) {
        for (let j = 1; j < this.ny - 1; j++) {
            const idx = this.getIdx(i, j);

            // vitesse au centre de la cellule (moyenne des faces de la grille MAC)
            const u = (this.velocityX[idx] + this.velocityX[this.getIdx(i+1,j)]) * 0.5;
            const v = (this.velocityY[idx] + this.velocityY[this.getIdx(i,j+1)]) * 0.5;

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
