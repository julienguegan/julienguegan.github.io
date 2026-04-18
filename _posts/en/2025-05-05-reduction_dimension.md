---
title: "Dimensionality Reduction: PCA, t-SNE, and UMAP"
date: 2025-05-05T19:30:00+02:00
lang: en
classes: wide
layout: single
categories:
  - blog
tags :
  - machine learning
  - data science
  - dimensionality reduction
  - data visualization
header:
  teaser: /assets/images/teaser_dimension_reduction.PNG
---

In the world of data science, we often encounter datasets with a large number of features. While this wealth of information can be beneficial, it also brings its share of challenges, known as the **"curse of dimensionality"**. As the number of dimensions increases, the data space becomes vast and sparse, making analysis, visualization, and even training machine learning models complex and resource-intensive.

<p align="center">
   <img src="/assets/images/dimension_curse_dimensionnality.png" width="70%"/>
</p>

Fortunately, **dimensionality reduction** techniques exist to help us project this high-dimensional data into a lower-dimensional space (often 2D or 3D for visualization) while preserving as much relevant information or intrinsic data structure as possible.

Among the most popular methods, three stand out:
1.  **PCA (Principal Component Analysis)**: A classic, fast, and interpretable linear approach.
2.  **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear method highly effective for visualizing local clusters.
3.  **UMAP (Uniform Manifold Approximation and Projection)**: A more recent non-linear technique, often faster than t-SNE and offering a good balance between local and global structure.

In this article, we will explore these three algorithms, understand their fundamental principles, their pros and cons, and see how to apply them in Python with concrete examples. I will use the MNIST dataset, which is very simple but clearly shows the visualization difficulties for a high-dimensional problem. It consists of 1797 small 8x8 images; since each pixel is a dimension of our problem, we have 64 dimensions.

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

## PCA: Principal Component Analysis

Principal Component Analysis (PCA) is arguably the most well-known and widely used dimensionality reduction technique. It is a **linear** method that aims to transform the original data into a new set of variables, called **principal components**, which are uncorrelated with each other and ordered by the amount of variance in the original data they explain.

The main idea is to find the directions (axes) in the multi-dimensional space along which the data varies the most. The first principal component is the axis that captures the most variance. The second principal component is the axis orthogonal to the first that captures the most remaining variance, and so on.

Mathematically, if $X$ is our centered data matrix (each feature has a mean of 0), the covariance matrix is given by $C = \frac{1}{N-1} X^T X$, where $N$ is the number of samples. PCA then seeks the eigenvectors $v$ and eigenvalues $\lambda$ of this matrix $C$ that satisfy the equation:

$$C v = \lambda v$$

The eigenvectors $v$ (ordered by decreasing eigenvalues $\lambda$) form the directions of the principal components. The eigenvalues $\lambda$ indicate the amount of variance explained by each corresponding principal component. Projecting the data $X$ onto the first $k$ eigenvectors $V_k = [v_1, v_2, ..., v_k]$ gives the reduced representation $X_{pca} = X V_k$.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Standardize data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variance explained per component: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.2f}")
```

<p align="center">
   <img src="/assets/images/dimension_pca.png" width="70%"/>
</p>

**Advantages of PCA:**
*   Simple, fast, and easy to compute.
*   Interpretable: the variance explained by each component is clear.
*   Useful for denoising and data compression.

**Disadvantages of PCA:**
*   Assumes linear relationships between variables.
*   Sensitive to data scaling (standardization necessary).
*   May poorly capture complex non-linear structures (e.g., curved manifolds).
*   Principal components do not necessarily correspond to original features.

**Note:** PCA maximizes global variance. If the interesting structure of the data is not in the directions of greatest variance, PCA may miss it.
{: .notice--info}

## t-SNE: t-Distributed Stochastic Neighbor Embedding

Unlike PCA, t-SNE (t-Distributed Stochastic Neighbor Embedding) is a **non-linear** technique particularly designed for **visualizing** high-dimensional data in low dimensions (typically 2D or 3D). Its main goal is to preserve the **local structure** of the data: points that are close in the high-dimensional space should remain close in the low-dimensional space.

t-SNE models the similarity between two points $x_i$ and $x_j$ in the high-dimensional space as a conditional probability $p_{j|i}$ that $x_i$ would pick $x_j$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $x_i$. Then, it defines a joint similarity probability $p_{ij}$.

In the low-dimensional space, it models the similarity between the corresponding points $y_i$ and $y_j$ with a joint probability $q_{ij}$ using a Student's t-distribution with one degree of freedom (which is equivalent to a Cauchy distribution). This heavy-tailed distribution allows dissimilar points to be better separated in the low-dimensional map.

More formally:
*   The conditional probability $p_{j|i}$ is calculated as:
    $$ p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)} $$
    where $\sigma_i$ is the variance of the Gaussian centered on $x_i$, determined to match a fixed perplexity (related to the effective number of neighbors).
*   The symmetric joint probability in the high-dimensional space is:
    $$ p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N} $$
    where $N$ is the total number of points.
*   The joint probability in the low-dimensional space $y_i, y_j$ uses a t-Student distribution with 1 degree of freedom:
    $$ q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}} $$
*   The algorithm then minimizes the Kullback-Leibler (KL) divergence between the distributions $P = \{p_{ij}\}$ and $Q = \{q_{ij}\}$:
    $$ KL(P\|Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}} $$
    This minimization is usually performed by gradient descent on the positions of the points $y_i$ in the low-dimensional space.

```python
# Apply t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10), alpha=0.7)
plt.title('t-SNE of Digits data (perplexity=30)')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=digits.target_names)
plt.grid(True)
plt.show()
```

<p align="center">
   <img src="/assets/images/dimension_tsne.png" width="70%"/> <!-- t-SNE result image to be generated -->
</p>

**Advantages of t-SNE:**
*   Excellent for revealing local structure and clusters in data.
*   Capable of capturing complex non-linear structures.
*   Widely used for exploratory visualization.

**Disadvantages of t-SNE:**
*   **Computationally expensive:** Complexity is typically $O(N^2)$ or $O(N \log N)$ with approximations, which can be slow on very large datasets.
*   **Stochastic:** Different runs can yield slightly different visualizations (fix `random_state` for reproducibility).
*   **Sensitive to hyperparameters:** Notably `perplexity` (related to the number of neighbors considered, typically between 5 and 50) and the number of iterations `n_iter`. Experimentation is often needed.
*   **Does not (well) preserve global structure:** The size and distance *between* clusters in the t-SNE visualization are generally not meaningful. These distances cannot be interpreted as in PCA.
*   Primarily a visualization technique, not for dimensionality reduction for model training (as it's not defined for new points).

**Caution:** Do not over-interpret the relative sizes of clusters or the distances between them in a t-SNE plot. Focus on the groupings of similar points.
{: .notice--warning}

## UMAP: Uniform Manifold Approximation and Projection

UMAP (Uniform Manifold Approximation and Projection) is a more recent non-linear dimensionality reduction technique rapidly gaining popularity. Like t-SNE, it is effective for visualization, but it is often faster and claims to better preserve the global structure of the data.

UMAP is based on solid mathematical foundations from algebraic topology and Riemannian manifold theory. The algorithm works in three main steps:
1.  **Constructing a weighted neighbor graph:** For each point, UMAP finds its $k$ nearest neighbors and builds a weighted representation of the local topological structure of the data (a "fuzzy manifold").
2.  **Calculating a similar low-dimensional representation:** UMAP repeats the graph construction process in the target low-dimensional space.
3.  **Optimization:** UMAP minimizes the difference (cross-entropy) between the high- and low-dimensional topological representations, thus seeking a projection that best preserves the topological structure of the original data.

UMAP is available in Python by simply installing it with the following command:
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
plt.title('UMAP of Digits data (n_neighbors=15, min_dist=0.1)')
plt.xlabel('UMAP component 1')
plt.ylabel('UMAP component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=digits.target_names)
plt.grid(True)
plt.show()
```

<p align="center">
   <img src="/assets/images/dimension_umap.png" width="70%"/>
</p>

Furthermore, the umap package includes very simple visualization tools to generate graphs for better understanding the local and global structures of our data.

```python
import umap.plot
umap.plot.points(reducer, labels=y, theme='fire')
umap.plot.connectivity(reducer, show_points=True, background="black", values=X_umap.mean(axis=1), edge_cmap="jet")
umap.plot.connectivity(reducer, edge_bundling='hammer', background="black", edge_cmap="plasma")
```

<p align="center">
   <img src="/assets/images/dimension_umap_plot.png" width="100%"/>
</p>

**Advantages of UMAP:**
*   **Speed:** Often significantly faster than t-SNE, especially on large datasets.
*   **Good local/global balance:** Tends to preserve global data structure better than t-SNE, while being excellent for local structure.
*   **Less sensitive to hyperparameters?** Default parameters (`n_neighbors=15`, `min_dist=0.1`) often work well, although experimentation is always good.
*   **Deterministic (by default):** Results are reproducible with the same `random_state`.
*   Can be used for dimensionality reduction beyond simple visualization (the `transform` method is defined for new points).

**Disadvantages of UMAP:**
*   More recent, the underlying theory is more complex to grasp than PCA.
*   Interpretation of distances remains tricky, though potentially more meaningful than for t-SNE.
*   Like t-SNE, sensitive to the choice of distance metrics if the data is not standard numerical.

**Tip:** UMAP is often an excellent starting point for non-linear visualization, offering a good compromise between speed, cluster separation quality, and global structure preservation.
{: .notice--success}

## PCA vs t-SNE vs UMAP: Which to choose?

There is no universal "best" method; the choice depends on your data and your objective:

| Feature                | PCA                                  | t-SNE                                      | UMAP                                           |
| :--------------------- | :----------------------------------- | :----------------------------------------- | :--------------------------------------------- |
| **Type**               | Linear                               | Non-linear                                 | Non-linear                                     |
| **Main objective**     | Max variance, compression, denoising | Local structure visualization (clusters)   | Local/global balance visualization             |
| **Global structure**   | Preserved (if linear)                | Generally not preserved                    | Better preserved than t-SNE                    |
| **Local structure**    | May be lost                          | Very well preserved                        | Very well preserved                            |
| **Speed**              | Very fast                            | Slow (especially large N)                  | Fast (often > t-SNE)                         |
| **Interpretability**   | High (explained variance)            | Low (inter-cluster distances unreliable)   | Moderate (distances potentially more reliable) |
| **Stochasticity**      | Deterministic                        | Stochastic                                 | Deterministic (by default)                     |
| **Hyperparameters**    | `n_components`                       | `perplexity`, `n_iter`, `learning_rate`    | `n_neighbors`, `min_dist`                      |
| **Preprocessing use**  | Yes                                  | No (generally)                             | Yes (possible)                                 |

**In summary:**
*   Start with **PCA** if you suspect linear relationships, need interpretability, or if speed is critical. It's also a good preprocessing step to reduce noise before applying t-SNE or UMAP.
*   Use **t-SNE** if your main goal is to visualize fine groupings and local structure in your data, and if computation time is not a major constraint. Be cautious with interpreting global distances.
*   Try **UMAP** as a modern alternative to t-SNE. It is often faster, handles large datasets better, and offers a better balance between preserving local and global structures. It's an excellent default choice for non-linear visualization.

It is often instructive to apply several of these methods and compare the results to gain a more complete understanding of your data's structure. The website [https://projector.tensorflow.org/](https://projector.tensorflow.org/) offers an interactive playground to visualize image and text data in 3D using the 3 algorithms described in this post. Have fun!


---

[![Generic badge](https://img.shields.io/badge/written_with-Python-blue.svg?style=plastic&logo=Python)](https://www.python.org/) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/access_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/reduction_dimension_pca_tsne_umap.ipynb) [![Generic badge](https://img.shields.io/badge/execute_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/dimension_reduction.ipynb)