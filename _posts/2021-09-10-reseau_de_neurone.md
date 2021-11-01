---
title: "Réseau de Neurone : statistique, gradient, perceptron"
date: 2021-09-10T19:25:30-04:00 
classes: wide
layout: single
categories:
  - blog
header:
  teaser: /assets/images/teaser_neural_network.jpg
---

**écriture en cours ...**

Ces dernières années, on entend de plus en plus dans les médias les mots : *intelligence artificielle*, *réseau de neurone*, *Deep Learning* ... En effet, de nombreuses innovations ont emmergées grâce à ces technologies mais que ce cache-t-il vraiment derrière cette terminologie ? Depuis que les premiers ordinateurs programmables ont été conçus, les gens ont été étonnés de voir ces ordinateurs résoudre des tâches impossibles pour tout être humain. Cependant, ces problèmes étaient en fait faciles à décrire par une liste formelle de règles mathématiques. Le vrai challenge pour les ordinateurs est d'effectuer des tâches que les humains réalisent facilement et intuitivement mais qu'ils ont beaucoup plus de mal à décrire formellement comme, par exemple, reconnaître un langage ou des visages. De nos jours, on appelle intelligence artificielle (IA) toute technique permettant de résoudre un problème plus ou moins complexe par le biais d’une machine. Le Machine Learning et le Deep Learning sont des champs d'étude de l'IA fondés sur des théories statistiques.

<p align="center">
   <img src="/assets/images/IA_classification.png" width="70%"/>
</p>

## Apprentissage Statistique

L'apprentissage statistique (ou *Machine Learning* en anglais) se concentre sur l'étude des algorithmes permettant aux ordinateurs d'*apprendre* à partir des données et d'améliorer leur performances à résoudre des tâches sans avoir explicitement programmée chacune d'entre elles. Soit un ensemble $(x_i,y_i)$ de $n$ données d'apprentissage (avec $x_i$ une donnée et $y_i$ sa classe). Le but de l'apprentissage supervisé est de déterminer une estimation $h$ de $f$ en utilisant les données $(x_i,y_i)$ à disposition. On distingue alors deux types de prédictions selon la nature de $Y$ : si $Y \subset \mathbb{R}$, on parle de problème de **régression** et si $Y = \{1,...,I\}$ on parle de problème de **classification**. Pour estimer la loi $f$, une approche classique est la minimisation du risque empirique. Étant donné une fonction loss $L(\hat{y},y)$ (aussi appelé fonction de coût, de perte, objectif) qui mesure à quel point la prédiction $\hat{y}$ d'un modèle $h$ est différente de la vraie classe $y$ alors le risque empirique est : 

$$ R_{emp}(h) = \frac{1}{n} \sum_{i=1}^n L\left(h(x_i),y_i\right) $$

Le principe de minimisation du risque empirique dit que l'algorithme d'apprentissage doit choisir un modèle $\hat{h}$ qui minimise ce risque empirique : $ \hat{h} = \arg \min_{h \in \mathcal{H}} R_{emp}(h) $. L'algorithme d'apprentissage consiste donc à résoudre un problème d'optimisation. A propos de la fonction loss, on peut en théorie utiliser la fonction *0-1 loss* : 

$$ L(\hat{y},y) = \left\{
    \begin{array}{ll}
        1 & \text{si } \hat{y} \neq y \\
        0 & \text{si } \hat{y} = y
    \end{array}
\right.$$

Mais, en pratique, il est préférable d'avoir une fonction loss continue et différentiable pour l'algorithme d'optimisation. Par exemple, la *mean squared error* (MSE) qui est souvent choisie pour des problèmes de regression, par exemple pour un cas linéaire où l'on veut trouver la meilleure droite passant par des points. Cependant, la MSE a le désavantage d'être dominée par les *outliers* qui peuvent faire tendre la somme vers une valeur élevée. Une loss fréquemment utilisée en classification et plus adaptée est la *cross-entropie* (CE) :

$$ MSE =  (y - \hat{y})^2 \quad \quad \quad CE = - y \log(\hat{y}) $$

```python
def compute_loss(y_true, y_pred, name):
    if name == 'MSE':
        loss = (y_true - y_pred)**2
    elif name == 'CE':
        loss = -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    return loss.mean()
```

**Note:** Pour résoudre ce problème de minimisation, le plus efficace est d'utiliser un algorithme de descente de gradient (cf [post précédent](https://julienguegan.github.io/posts/2021-08-21-optimisation_profil_aile/#descente-de-gradient)) mais nécessite de connaître la dérivée exacte de la fonction Loss. Les librairies récente de Deep Learning (Pytorch, Tensorflow, Caffe ...) implémentent ces algorithmes d'optimisation ainsi que des frameworks d'[auto-différentation](https://fr.wikipedia.org/wiki/D%C3%A9rivation_automatique)
{: .notice--info}

Maintenant que la notion d'apprentissage a été définie, l'élément le plus important est de construire un modèle $h$ capable de classifier ou régresser des données. L'un des modèles parmis les plus connues de nos jours est celui des réseaux de neurones.

## Réseaux de Neurones

### Perceptron 

Le premier modèle à l’origine des réseaux de neurones est le Perceptron (F. Rosenblatt,1957), il peut être vu comme un unique et simple neurone qui résout un problème de classification linéaire. Le Perceptron transforme un vecteur $X=(x_1,...,x_d) \in \mathbb{R}^d$ en une sortie $Y \in [0,1]$, l’étiquette de classification. L'idée est qu'on cherche à séparer notre espace d’entrée en 2 régions par un hyperplan $\mathcal{H}$ définit simplement par : 

$$\mathcal{H} : w^T X + b = 0 \iff \mathcal{H} : \sum_{i=1}^d w_i x_i + b = 0 $$

où $w=(w_1,...,w_d) \in \mathbb{R}^d$ et $b \in \mathbb{R}$ sont les paramètres de l'hyperplan $\mathcal{H}$ qu'on appelle communément les poids du neurone.

```python
class Perceptron:
    
    def __init__(self, n_inputs, n_iter=30, alpha=2e-4):
        self.w      = np.zeros((n_iter, n_inputs+1))
        self.w[0,:] = np.random.randn(n_inputs+1) # weights and bias 
        self.n_iter = n_iter # number of iterations during training
        self.alpha  = alpha # learning rate

    def predict(self, X, i=-1): 
        return np.sign(X @ self.w[i,1:] + self.w[i,0])
```

to do : inserer graphique

Cet hyperplan sépare donc l'espace en 2 et est donc capable de classifier un donnée $X$ en créant une règle comme $ f(X) = 1 \ \text{si } w^T X + b<0  \ ; 0 \text{ sinon} $. Notre modèle de neurone se résume donc à des paramètres modélisant un hyperplan et une règle de classification. Dans notre paradigme du machine learning, on suppose qu'on a, à notre disposition, un ensemble de $n$ données étiquetées $(X,Y) \in \mathbb{R}^{n\times d} \times [0,1]^n$. Comme expliqué dans la section précédentes, la phase d'apprentissage consiste à minimiser l'erreur de classification que fait le modèle sur ces données et ajuster les paramètres $w$ et $b$ qui permettent au modèle du Perceptron de séparer correctement ces données. 

```python
    def gradient_descent(self, i, y, y_pred):
        # construct input X and 1 to have bias
        inputs = np.hstack((np.ones((X.shape[0], 1)), X))
        # compute gradient of the MSE loss
        gradient_loss = -2 * np.dot(y_true - y_pred, inputs)
        # apply a step of gradient descent
        self.w[i+1,:] = self.w[i, :] - self.alpha * gradient_loss
```

<p align="center">
   <img src="/assets/images/perceptron_training.jpg" width="100%"/>
</p>



### Perceptron Multi-Couches 

Pour approximer des comportements non linéaires, il est possible d'ajouter des couches de neurones les unes à la suite des autres au modèle du Perceptron. Chaque couche est composée de plusieurs neurones (poids et biais) suivis d'une fonction non linéaire appelée fonction d'activation $a$ \footnote{Il existe de nombreuses fonctions d'activation différentes (cf \ref{fig:Activation}) : sigmoïde, tanh, ReLU, Leaky ReLU ...}. L'entrée $X$ subit alors successivement plusieurs transformations linéaires et leur fonction d'activation non linéaire jusqu'à la dernière couche qui permet, elle, d'associer les sorties des neurones à la classe finale. Une couche est désormais modélisée par une matrice de poids $W \in \mathbb{R}^{d\times m}$ et un vecteur de biais $b \in \mathbb{R}^{m}$ avec $m$ le nombre de couches, ses paramètres modélisent toutes les connexions entre les neurones de la couche actuelle et la couche précédente.

Cet enchaînement de couche de neurones pose un problème pour la phase d'entraînement puisque le calcul de $\frac{\partial\mathcal{L}}{\partial W}$ n'est pas trivial. En 1986, E. Hinton and al démocratisent la technique de la rétro-propagation du gradient qui permet d'entraîner des réseaux de neurones multi-couches en se basant sur la règle de dérivation en chaîne. Le gradient de la loss est calculé en utilisant les dérivées des poids des neurones et leur fonction d'activation en partant de la dernière couche jusqu'à la première couche. Il faut donc parcourir le réseau vers l'avant (*forward pass*) pour obtenir la valeur de la loss puis vers l'arrière (*backward pass*) pour obtenir la valeur de la dérivée de la loss nécessaire à l'algorithme d'optimisation (cf. figure \ref{fig:backpropagation} ci-dessous).

{% include mlp_training.html %}


De plus, en pratique, la fonction de classification $f$ est remplacée par une fonction différentiable $\sigma : \mathbb{R} \rightarrow [0,1]$. Pour un problème binaire, la fonction sigmoïde $\sigma(z)=\frac{1}{1+e^{-z}}$ est utilisée puis la classe est choisie selon un seuil (ex : $\sigma(z) > 0.5$). Le modèle peut également s'étendre à des problèmes multi-classes où la sortie du modèle $Y$ n'est plus un scalaire mais un vecteur de $\mathbb{R}^C$ avec $C$ le nombre de classes à prédire. La fonction de classification est alors une fonction softmax $\sigma(z_j)=\frac{e^{z_j}}{\sum_{k=1}^Ke^{z_k}}$ et la classe retenue est choisie en gardant la position de la valeur maximum du vecteur $Y$.

### Aller plus loin

des architectures multiples
des modeles de representation avec des convolutions, 

