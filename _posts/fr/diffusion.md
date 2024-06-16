---
title: "Modèles de diffusion"
---

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#what-are-diffusion-models


# Diffusion vers l'avant

on ajoute un bruit gaussien à l'échantillon en $T$ étapes, ce qui produit une séquence d'échantillon bruité $x_1$, ..., $x_T$. La taille des pas est controllé par $\{\beta_t \in (0, 1)\}_{t=1}^T$

$$ q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) $$

Petit à petit la donnée $x_0$ perd ses caractéristiques au fur et à mesure que $t$ augmente, et quand $T \to \infty$, $x_T$ est équivalent à une distribution gaussienne.

Une propriété cool est qu'on a pas besoin de stocker toutes ces séquences de données bruitées puisqu'il existe une formule pour échantillonner n'importe quel $x_t$ à partir de $x_0$ et $\beta_t$