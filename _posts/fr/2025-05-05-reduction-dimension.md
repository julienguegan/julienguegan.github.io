---
title: "Réduction de Dimension : PCA, t-SNE et UMAP démystifiés"
date: 2025-05-05T19:30:00+02:00
lang: fr
classes: wide
layout: single
categories:
  - blog
tags :
  - machine learning
  - data science
  - réduction de dimension
  - visualisation de données
header:
  teaser: /assets/images/teaser_dimension_reduction.png 
---

Dans le monde de la data science, nous sommes souvent confrontés à des jeux de données possédant un grand nombre de caractéristiques (features). Si cette richesse d'information peut être bénéfique, elle apporte aussi son lot de défis, connus sous le nom de **"fléau de la dimensionnalité"** (*curse of dimensionality*). Plus le nombre de dimensions augmente, plus l'espace des données devient vaste et épars, rendant l'analyse, la visualisation et même l'entraînement de modèles de machine learning complexes et coûteux en ressources.

<p align="center">
   <img src="/assets/images/dimension_curse_dimensionnality.png" width="70%"/>
</p>

Heureusement, des techniques de **réduction de dimension** existent pour nous aider à projeter ces données de haute dimension dans un espace de plus faible dimension (souvent 2D ou 3D pour la visualisation) tout en préservant au maximum l'information pertinente ou la structure intrinsèque des données.

Parmi les méthodes les plus populaires, trois se distinguent particulièrement :
1.  **PCA (Principal Component Analysis)** : Une approche linéaire classique, rapide et interprétable.
2.  **t-SNE (t-Distributed Stochastic Neighbor Embedding)** : Une méthode non linéaire très efficace pour visualiser des clusters locaux.
3.  **UMAP (Uniform Manifold Approximation and Projection)** : Une technique non linéaire plus récente, souvent plus rapide que t-SNE et offrant un bon équilibre entre structure locale et globale.

Dans cet article, nous allons explorer ces trois algorithmes, comprendre leurs principes fondamentaux, leurs avantages et inconvénients, et voir comment les appliquer en Python avec des exemples concrets. J'utiliserais le jeu de donnée MNIST qui est très simple mais montre bien les difficultés de visualisation pour un problème à haute dimension. Il est composé de 1797 petites images de taille 8 par 8, chaque pixel étant une dimension de notre problème on a alors 64 dimensions.

```python
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
n_samples, n_features = X.shape
print(f"Dataset shape: {X.shape}")
# Dataset shape: (1797, 64)
```

<p align="center">
   <img src="/assets/images/dimension_mnist.png" width="70%"/>
</p>

## PCA : L'analyse en Composantes Principales

L'Analyse en Composantes Principales (PCA) est sans doute la technique de réduction de dimension la plus connue et utilisée. C'est une méthode **linéaire** qui vise à transformer les données originales en un nouvel ensemble de variables, appelées **composantes principales**, qui sont non corrélées entre elles et ordonnées selon la quantité de variance des données originales qu'elles expliquent.

L'idée maîtresse est de trouver les directions (axes) dans l'espace multi-dimensionnel le long desquelles les données varient le plus. La première composante principale est l'axe qui capture la plus grande variance. La deuxième composante principale est l'axe orthogonal au premier qui capture la plus grande partie de la variance restante, et ainsi de suite.

Mathématiquement, si $X$ est notre matrice de données centrées (chaque feature a une moyenne de 0), la matrice de covariance est donnée par $C = \frac{1}{N-1} X^T X$, où $N$ est le nombre d'échantillons. PCA cherche alors les vecteurs propres $v$ et les valeurs propres $\lambda$ de cette matrice $C$ qui satisfont l'équation :

$$C v = \lambda v$$

Les vecteurs propres $v$ (ordonnés par les valeurs propres $\lambda$ décroissantes) forment les directions des composantes principales. Les valeurs propres $\lambda$ indiquent la quantité de variance expliquée par chaque composante principale correspondante. La projection des données $X$ sur les $k$ premiers vecteurs propres $V_k = [v_1, v_2, ..., v_k]$ donne la représentation réduite $X_{pca} = X V_k$.

```python
# Standardize data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA  to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variance expliquée par composante: {pca.explained_variance_ratio_}")
print(f"Variance totale expliquée: {np.sum(pca.explained_variance_ratio_):.2f}")
```

<p align="center">
   <img src="/assets/images/dimension_pca.png" width="70%"/>
</p>

**Avantages de PCA :**
*   Simple, rapide et facile à calculer.
*   Interprétable : la variance expliquée par chaque composante est claire.
*   Utile pour le débruitage et la compression de données.

**Inconvénients de PCA :**
*   Suppose des relations linéaires entre les variables.
*   Sensible à l'échelle des données (standardisation nécessaire).
*   Peut mal capturer des structures non linéaires complexes (ex: variétés courbes).
*   Les composantes principales ne correspondent pas nécessairement aux features originales.

**Note:** PCA maximise la variance globale. Si la structure intéressante des données ne se trouve pas dans les directions de plus grande variance, PCA peut la manquer.
{: .notice--info}

## t-SNE : Plongement Stochastique Distribué en t

Contrairement à PCA, t-SNE (t-Distributed Stochastic Neighbor Embedding) est une technique **non linéaire** particulièrement conçue pour la **visualisation** de données de haute dimension en basse dimension (typiquement 2D ou 3D). Son objectif principal est de préserver la **structure locale** des données : les points qui sont proches dans l'espace de haute dimension devraient rester proches dans l'espace de basse dimension.

t-SNE modélise la similarité entre deux points $x_i$ et $x_j$ dans l'espace de haute dimension comme une probabilité conditionnelle $p_{j|i}$ qu' $x_i$ choisirait $x_j$ comme son voisin si les voisins étaient choisis en proportion de leur densité de probabilité sous une Gaussienne centrée sur $x_i$. Ensuite, il définit une probabilité de similarité jointe $p_{ij}$.

Dans l'espace de basse dimension, il modélise la similarité entre les points correspondants $y_i$ et $y_j$ avec une probabilité jointe $q_{ij}$ utilisant une distribution t de Student à un degré de liberté (qui est équivalente à une distribution de Cauchy). Cette distribution à queues lourdes permet de mieux séparer les points dissemblables dans la carte de basse dimension.

Plus formellement :
*   La probabilité conditionnelle $p_{j|i}$ est calculée comme :
    $$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$
    où $\sigma_i$ est la variance de la Gaussienne centrée sur $x_i$, déterminée de manière à correspondre à une perplexité fixée (liée au nombre effectif de voisins).
*   La probabilité jointe symétrique dans l'espace de haute dimension est :
    $$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
    où $N$ est le nombre total de points.
*   La probabilité jointe dans l'espace de basse dimension $y_i, y_j$ utilise une distribution t-Student à 1 degré de liberté :
    $$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$
*   L'algorithme minimise ensuite la divergence de Kullback-Leibler (KL) entre les distributions $P = \{p_{ij}\}$ et $Q = \{q_{ij}\}$ :
    $$KL(P\|Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
    Cette minimisation est généralement effectuée par descente de gradient sur les positions des points $y_i$ dans l'espace de basse dimension.

```python
# Apply t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), alpha=0.7)
plt.title('t-SNE des données Digits (perplexity=30)')
plt.xlabel('Composante t-SNE 1'), plt.ylabel('Composante t-SNE 2')
plt.legend(handles=scatter.legend_elements()[0], labels=digits.target_names)
plt.grid(True)
plt.show()
```

<p align="center">
   <img src="/assets/images/dimension_tsne.png" width="70%"/> <!-- Image résultat t-SNE à générer -->
</p>

**Avantages de t-SNE :**
*   Excellent pour révéler la structure locale et les clusters dans les données.
*   Capable de capturer des structures non linéaires complexes.
*   Largement utilisé pour la visualisation exploratoire.

**Inconvénients de t-SNE :**
*   **Coûteux en calcul :** La complexité est typiquement $O(N^2)$ ou $O(N \log N)$ avec des approximations, ce qui peut être lent sur de très grands datasets.
*   **Stochastique :** Différentes exécutions peuvent donner des visualisations légèrement différentes (fixer `random_state` pour la reproductibilité).
*   **Sensible aux hyperparamètres :** Notamment la `perplexity` (liée au nombre de voisins considérés, typiquement entre 5 et 50) et le nombre d'itérations `n_iter`. Il faut souvent expérimenter.
*   **Ne préserve pas (bien) la structure globale :** La taille et la distance *entre* les clusters dans la visualisation t-SNE ne sont généralement pas significatives. On ne peut pas interpréter ces distances comme dans PCA.
*   Principalement une technique de visualisation, pas de réduction de dimension pour l'entraînement de modèles (car non défini sur de nouveaux points).

**Attention:** N'interprétez pas trop la taille relative des clusters ou les distances entre eux dans un graphique t-SNE. Concentrez-vous sur les groupements de points similaires.
{: .notice--warning}

## UMAP : Approximation et Projection Uniforme de Variétés

UMAP (Uniform Manifold Approximation and Projection) est une technique de réduction de dimension non linéaire plus récente qui gagne rapidement en popularité. Comme t-SNE, elle est efficace pour la visualisation, mais elle est souvent plus rapide et prétend mieux préserver la structure globale des données.

UMAP est basé sur des fondements mathématiques solides issus de la topologie algébrique et de la théorie des variétés riemanniennes. L'algorithme fonctionne en trois étapes principales :
1.  **Construction d'un graphe de voisinage pondéré :** Pour chaque point, UMAP trouve ses $k$ plus proches voisins et construit une représentation pondérée de la structure topologique locale des données (une "variété floue").
2.  **Calcul d'une représentation bas-dimensionnelle similaire :** UMAP répète le processus de construction de graphe dans l'espace de basse dimension cible.
3.  **Optimisation :** UMAP minimise la différence (entropie croisée) entre les représentations topologiques de haute et basse dimension, cherchant ainsi une projection qui préserve au mieux la structure topologique des données originales.

UMAP est disponible en python en l'installant simplement avec la commande suivante:
```bash
pip install umap-learn
```

```python
# Apply UMAP
import umap
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), alpha=0.7)
plt.title('UMAP des données Digits (n_neighbors=15, min_dist=0.1)')
plt.xlabel('Composante UMAP 1'), plt.ylabel('Composante UMAP 2')
plt.legend(handles=scatter.legend_elements()[0], labels=digits.target_names)
plt.grid(True)
plt.show()
```

<p align="center">
   <img src="/assets/images/dimension_umap.png" width="70%"/> 
</p>

De plus, le package umap intègre des outils de visualisation très simples permettant de génère des graphes pour mieux comprendre les structures locales et globales de nos données. 

```python
import umap.plot
umap.plot.points(reducer, labels=y, theme='fire')
umap.plot.connectivity(reducer, show_points=True, background="black", values=X_umap.mean(axis=1), edge_cmap="jet")
umap.plot.connectivity(reducer, edge_bundling='hammer', background="black", edge_cmap="plasma")
```

<p align="center">
   <img src="/assets/images/dimension_umap_plot.png" width="90%"/> 
</p>

**Avantages d'UMAP :**
*   **Rapidité :** Souvent significativement plus rapide que t-SNE, surtout sur de grands datasets.
*   **Bon équilibre local/global :** Tendance à mieux préserver la structure globale des données que t-SNE, tout en étant excellent pour la structure locale.
*   **Moins sensible aux hyperparamètres ?** Les paramètres par défaut (`n_neighbors=15`, `min_dist=0.1`) fonctionnent souvent bien, bien qu'il soit toujours bon d'expérimenter.
*   **Déterministe (par défaut) :** Les résultats sont reproductibles avec le même `random_state`.
*   Peut être utilisé pour la réduction de dimension au-delà de la simple visualisation (la transformation `transform` est définie pour de nouveaux points).

**Inconvénients d'UMAP :**
*   Plus récent, la théorie sous-jacente est plus complexe à appréhender que PCA.
*   L'interprétation des distances reste délicate, bien que potentiellement plus significative que pour t-SNE.
*   Comme t-SNE, sensible au choix des métriques de distance si les données ne sont pas numériques standard.

**Conseil:** UMAP est souvent un excellent point de départ pour la visualisation non linéaire, offrant un bon compromis entre vitesse, qualité de la séparation des clusters et préservation de la structure globale.
{: .notice--success}

## PCA vs t-SNE vs UMAP : Lequel choisir ?

Il n'y a pas de "meilleure" méthode universelle ; le choix dépend de vos données et de votre objectif :

| Caractéristique        | PCA                                  | t-SNE                                      | UMAP                                           |
| :--------------------- | :----------------------------------- | :----------------------------------------- | :--------------------------------------------- |
| **Type**               | Linéaire                             | Non linéaire                               | Non linéaire                                   |
| **Objectif principal** | Max variance, compression, débruitage | Visualisation structure locale (clusters)  | Visualisation équilibre local/global           |
| **Structure globale**  | Préservée (si linéaire)              | Généralement non préservée                 | Mieux préservée que t-SNE                      |
| **Structure locale**   | Peut être perdue                     | Très bien préservée                        | Très bien préservée                            |
| **Vitesse**            | Très rapide                          | Lente (surtout N grand)                    | Rapide (souvent > t-SNE)                       |
| **Interprétabilité**   | Élevée (variance expliquée)          | Faible (distances inter-clusters non fiables) | Modérée (distances potentiellement + fiables) |
| **Stochasticité**      | Déterministe                         | Stochastique                               | Déterministe (par défaut)                      |
| **Hyperparamètres**    | `n_components`                       | `perplexity`, `n_iter`, `learning_rate`    | `n_neighbors`, `min_dist`                      |
| **Usage pré-processing**| Oui                                  | Non (généralement)                         | Oui (possible)                                 |

**En résumé :**
*   Commencez par **PCA** si vous suspectez des relations linéaires, si vous avez besoin d'interprétabilité ou si la vitesse est critique. C'est aussi une bonne étape de pré-processing pour réduire le bruit avant d'appliquer t-SNE ou UMAP.
*   Utilisez **t-SNE** si votre objectif principal est de visualiser des groupements fins et la structure locale dans vos données, et si le temps de calcul n'est pas une contrainte majeure. Soyez prudent avec l'interprétation des distances globales.
*   Essayez **UMAP** comme alternative moderne à t-SNE. Il est souvent plus rapide, gère mieux les grands datasets et offre un meilleur équilibre entre la préservation des structures locales et globales. C'est un excellent choix par défaut pour la visualisation non linéaire.

Il est souvent instructif d'appliquer plusieurs de ces méthodes et de comparer les résultats pour obtenir une compréhension plus complète de la structure de vos données. Le site https://projector.tensorflow.org/ offre un playground interactif pour visualiser en 3D des données images et textuelles sur les 3 algorithmes décrits dans ce post. Amusez-vous bien !


---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Python-blue.svg?style=plastic&logo=Python)](https://www.python.org/) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/reduction_dimension_pca_tsne_umap.ipynb) [![Generic badge](https://img.shields.io/badge/execute_le_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/reduction_dimension_pca_tsne_umap.ipynb)