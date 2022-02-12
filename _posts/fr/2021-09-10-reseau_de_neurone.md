---
title: "Réseau de Neurone : statistique, gradient, perceptron"
date: 2021-09-10T19:25:30-04:00 
lang: fr
classes: wide
layout: single
categories:
  - blog
tags :
  - machine learning
  - deep learning
header:
  teaser: /assets/images/teaser_neural_network.jpg
---

Ces dernières années, on entend de plus en plus dans les médias les mots : *intelligence artificielle*, *réseau de neurone*, *Deep Learning* ... En effet, de nombreuses innovations ont emmergées grâce à ces technologies mais que ce cache-t-il vraiment derrière cette terminologie ? Depuis que les premiers ordinateurs programmables ont été conçus, les gens ont été étonnés de voir ces ordinateurs résoudre des tâches impossibles pour tout être humain. Cependant, ces problèmes étaient en fait faciles à décrire par une liste formelle de règles mathématiques. Le vrai challenge pour les ordinateurs est d'effectuer des tâches que les humains réalisent facilement et intuitivement mais qu'ils ont beaucoup plus de mal à décrire formellement comme, par exemple, reconnaître un langage ou des visages. De nos jours, on appelle intelligence artificielle (IA) toute technique permettant de résoudre un problème plus ou moins complexe par le biais d’une machine. Le Machine Learning et le Deep Learning sont des champs d'étude de l'IA fondés sur des théories statistiques.

<p align="center">
   <img src="/assets/images/IA_classification.png" width="70%"/>
</p>

## Apprentissage Statistique

L'apprentissage statistique (ou *Machine Learning* en anglais) se concentre sur l'étude des algorithmes permettant aux ordinateurs d'*apprendre* à partir des données et d'améliorer leur performances à résoudre des tâches sans avoir explicitement programmée chacune d'entre elles. Soit un ensemble $(x_i,y_i)$ de $n$ données d'apprentissage (avec $x_i$ une donnée et $y_i$ sa classe). Le but de l'apprentissage supervisé est de déterminer une estimation $h$ de $f$ en utilisant les données $(x_i,y_i)$ à disposition. On distingue alors deux types de prédictions selon la nature de $Y$ : si $Y \subset \mathbb{R}$, on parle de problème de **régression** et si $Y = \{1,...,I\}$ on parle de problème de **classification**. Pour estimer la loi $f$, une approche classique est la minimisation du risque empirique. Étant donné une fonction loss $L(\hat{y},y)$ (aussi appelé fonction de coût, de perte, objectif) qui mesure à quel point la prédiction $\hat{y}$ d'un modèle $h$ est différente de la vraie classe $y$ alors le risque empirique est : 

$$ R_{emp}(h) = \frac{1}{n} \sum_{i=1}^n L\left(h(x_i),y_i\right) $$

Le principe de minimisation du risque empirique dit que l'algorithme d'apprentissage doit choisir un modèle $\hat{h}$ qui minimise ce risque empirique : $ \hat{h} = \arg \min_{h \in \mathcal{H}} R_{emp}(h) $. L'algorithme d'apprentissage consiste donc à résoudre un problème d'optimisation. La fonction à minimiser est une approximation d'une probabilité inconnue, elle est faite en moyennant les données à disposition et plus ces données seront nombreuses plus elle l'approximation sera juste. À propos de la fonction *loss* $L(\hat{y},y)$, on peut en théorie utiliser la fonction *0-1 loss* pour pénaliser les erreurs et ne rien faire sinon : 

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

Le premier modèle à l’origine des réseaux de neurones est le Perceptron (F. Rosenblatt,1957), il peut être vu comme un unique et simple neurone qui résout un problème de classification linéaire. Le Perceptron transforme un vecteur d'entrée $X=(x_1,...,x_d) \in \mathbb{R}^d$ en une sortie $Y \in [0,1]$, l’étiquette de classification. L'idée est qu'on cherche à séparer notre espace d’entrée en 2 régions par un hyperplan $\mathcal{H}$ définit simplement par : 

$$\mathcal{H} : w^T X + b = 0 \iff \mathcal{H} : \sum_{i=1}^d w_i x_i + b = 0 $$

où $w=(w_1,...,w_d) \in \mathbb{R}^d$ et $b \in \mathbb{R}$ sont les paramètres de l'hyperplan $\mathcal{H}$ qu'on appelle communément les poids du neurone.

```python
class Perceptron:
    
    def __init__(self, n_inputs, n_iter=30, alpha=2e-4):
        self.n_iter = n_iter # number of iterations during training
        self.alpha  = alpha # learning rate
        self.w      = np.zeros((n_iter, n_inputs+1))
        self.w[0,:] = np.random.randn(n_inputs+1) # weights and bias parameters

    def predict(self, X, i=-1): 
        return np.sign(X @ self.w[i,1:] + self.w[i,0]) 
```

<p align="center">
   <img src="/assets/images/perceptron.png" width="100%"/>
</p>

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

 <details> <summary> Ci-dessus, la loss MSE a été utilisée. Sa dérivée exacte en fonction de $w$ peut se calculer facilement (cf ci-dessous). </summary>

 $$
    L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 =  \frac{1}{n}\sum_{i=1}^n (y_i - (w_i x_i + b))^2 \\[5pt]
    \left\{
        \begin{array}{ccc}
            \dfrac{\partial L}{\partial w} &=& \dfrac{\partial}{\partial w} \dfrac{1}{n} \sum_{i=1}^n (y_i - (w_i x_i + b))^2  \\[10pt]
            \dfrac{\partial L}{\partial b} &=& \dfrac{\partial}{\partial b} \dfrac{1}{n} \sum_{i=1}^n (y_i - (w_i x_i + b))^2  \\
        \end{array}
    \right. \\[10pt]
    \left\{
        \begin{array}{cc}
            \dfrac{\partial L}{\partial w} &=& -\dfrac{2}{m}\sum_{i=1}^{m}(y_i - w_i x_i - b)x_i \\[10pt]
            \dfrac{\partial L}{\partial b} &=& -\dfrac{2}{m}\sum_{i=1}^{m}(y_i - w_i x_i - b)
        \end{array}
    \right. \\[10pt]
    \left\{
        \begin{array}{cc}
            \dfrac{\partial L}{\partial w} &=& -\dfrac{2}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)x_i \\[10pt]
            \dfrac{\partial L}{\partial b} &=& -\dfrac{2}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)
        \end{array}
    \right. \\
 $$
 En python, on a rajouté 1 à l'entrée X pour prendre en compte l'équation du biais. Le gradient s'écrit bien `gradient_loss = -2 * np.dot(y_true - y_pred, inputs)`
</details> {: .notice--primary}

Notez que ce modèle est très proche d'une simple régression linéaire mais présenté ici pour un problème de classification, on peut facilement l'adapter à une régression en supprimant la fonction `np.sign()` dans le code python. D'autres part, le problème résolu ici est binaire (on a 2 classes) mais on peut facilement étendre le modèle pour qu'il prédise plusieurs classes en remplaçant le vecteur de poids $w \in \mathbb{R}^d$ par une matrice $W \in \mathbb{R}^{d \times c}$ représentant donc plusieurs hyperplans séparateurs où $c$ est le nombre de classes possible, la sortie du modèle $Y$ n'est plus un scalaire mais alors un vecteur de $\mathbb{R}^c$. Enfin, on notera qu'en pratique, la fonction de classification $f$ est remplacée par une fonction différentiable $\sigma : \mathbb{R} \rightarrow [0,1]$. Pour un problème binaire, la fonction sigmoïde $\sigma(z)=\frac{1}{1+e^{-z}}$ est utilisée puis la classe est choisie selon un seuil (ex : $\sigma(z) > 0.5$). Pour un cas multi-classe, la fonction de classification est une fonction softmax $\sigma(z_j)=\frac{e^{z_j}}{\sum_{k=1}^Ke^{z_k}}$ qui retourne un vecteur de *pseudo-probabilité* (somme à 1) et la classe retenue est choisie en gardant la position de la valeur maximum du vecteur.

### Perceptron Multi-Couches 

Comme décrit ci-dessus, un neurone unique peut résoudre un problème linéairement séparable mais échoue lorsque les données ne sont pas linéairement séparable. Pour approximer des comportements non linéaires, l'idée est d'ajouter notamment des fonctions non linéaires dans le modèle, elles sont appelées *fonctions d'activation* (cf exemples ci-dessous). 

<p align="center">
   <img src="/assets/images/activation_function.png" width="100%"/>
</p>

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

def tanh(x):
    return np.tanh(x)
```

Le modèle du réseau de neurones ou Perceptron Multi-Couches consiste à enchaîner successivement plusieurs transformations linéaires effectuées par des neurones simples et des transformations non linéaires réalisées par ces fonctions d'activations jusqu'à la dernière opération qui retournera la classe prédite par le modèle. Une couche $l$ du réseau de neurone est alors composée de $m$ neurones modélisés par une matrice de poids $W^l \in \mathbb{R}^{(d+1)\times m}$ (par simplicité on intègre le biais $b$ dans $W$) ainsi que d'une fonction d'activation $A^l$. Au final, le réseau de neurone complet peut être décrit par une combinaison de **composition de fonctions et multiplications matricielles** en allant de la 1ère couche à la dernière : 

$$ h(X) = \hat{Y} = A^l(W^l A^{l-1}(W^{l-1} \cdots A^0(W^0 X)\cdots)) $$

```python
class MultiLayerPerceptron:
    ''' MLP model with 2 layers '''

    def __init__(self, n_0, n_1, n_2):
        # initialize weights of 1st layer (hidden) and 2nd layer (output)
        self.W1 = np.random.randn(n_0, n_1)
        self.b1 = np.zeros(shape=(1, n_1))
        self.W2 = np.random.randn(n_1, n_2)
        self.b2 = np.zeros(shape=(1, n_2))        

    def forward(self, X):
        # input
        self.A0 = X
        # first layer
        self.Z1 = np.dot(self.A0, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        # second layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        # output
        y_pred = self.A2
        return y_pred
```

**Note:** Notez qu'on peut construire un réseau avec autant de couches cachées que l'on veut et que chacunes de ces couches peuvent être constituées également d'un nombre arbitraire de neurones et ce peu importe la dimension d'entrée et de sortie du problème.
{: .notice--danger}

<p align="center">
   <img src="/assets/images/multi_layer_perceptron.png" width="70%"/>
</p>

Cet enchaînement de couches de neurones pose problème pour la phase d'entraînement : le calcul de $\frac{\partial\mathcal{L}}{\partial W}$ est moins trivial que pour le modèle du neurone formel puisqu'il faut prendre en compte les poids $W^l$ de chaque couche $l$. La technique de la **rétropropagation du gradient** permet d'entraîner des réseaux de neurones multi-couches en se basant sur la règle de dérivation en chaîne. Le gradient de la loss est calculé en utilisant les dérivées des poids des neurones et leur fonction d'activation en partant de la dernière couche jusqu'à la première couche. Il faut donc parcourir le réseau vers l'avant (*<span style="color:green">forward pass</span>*) pour obtenir la valeur de la loss puis vers l'arrière (*<span style="color:red">backward pass</span>*) pour obtenir la valeur de la dérivée de la loss nécessaire à l'algorithme d'optimisation. Si on s'intéresse au neurone $j$ de la couche $l$ vers le neurone $i$ de la couche $l+1$, on note $a$ la valeur du produit vectoriel, $o$ la sortie du neurone (après activation) et qu'on garde le reste des notations précédentes, le calcul du gradient de $L$ en fonction de $W$ est :

$$ 
\begin{align*}
    \dfrac{\partial L}{\partial w_{ij}^l} &= \underbrace{\quad \dfrac{\partial L}{\partial a_{j}^l} \ \ } \ \ \underbrace{\quad \dfrac{\partial a_j^l}{\partial w_{ij}^l} \quad} \\ 
                                      & \qquad \ \ \ \delta_j^l \quad \ \ \ \dfrac{\partial}{\partial w_{ij}^l} \sum_{n=0}^{N_{l-1}} w_{nj}^l o_n^{l-1} = o_i^{l-1}
\end{align*} 
$$

On a donc que la dérivée $\dfrac{\partial L}{\partial w_{ij}^l}$ dépend du terme $\delta_j^l$ de la couche $l$ et de la sortie $o_i^{l-1}$ de la couche $l-1$. Ça fait sens puisque le poids $w_{ij}^l$ connecte la sortie du neurone $i$ dans la couche $l-1$ à l'entrée du neurone $j$ dans la couche $l$. On développe maintenant le terme $\delta_j^l$ : 

$$ \delta_j^l = \dfrac{\partial L}{\partial a_{j}^l} =  \sum_{n=1}^{N_{l+1}} \dfrac{\partial L}{\partial a_n^{l+1}}\dfrac{\partial  a_n^{l+1}}{\partial  a_j^l} = \sum_{n=1}^{N_{l+1}} \delta_n^{l+1} \dfrac{\partial  a_n^{l+1}}{\partial  a_j^l} $$

or, on a : 

$$ 
\begin{align*}
    & a_n^{l+1} &=& \sum_{n=1}^{N_{l}} w_{jn}^{l+1}A(a_j^l) \\
    \Rightarrow & \dfrac{\partial a_n^{l+1}}{\partial  a_j^l} &=& \ w_{jn}^{l+1}A'(a_j^l)
\end{align*}
$$

et donc :

$$ \dfrac{\partial L}{\partial w_{ij}^l} = \delta_j^l o_i^{l-1} = A'(a_j^l) o_i^{l-1} \sum_{n=1}^{N_{l+1}} w_{jn}^{l+1} \delta_n^{l+1} $$ 

On obtient donc que la dérivée partielle de $L$ par rapport à $w_{ij}$ à la couche $l$ dépend également de la dérivée à la couche $l+1$. Pour calculer la dérivée de tout le réseau en utilisant la règle de la dérivation en chaîne, il est donc nécessaire de commencer par la dernière couche pour finir par la première d'où le terme de *backpropagation de l'erreur*.

**Attention:** Comme les calculs de la phase de backpropagation dépendent également de $a_j^l$ et $o_i^{l-1}$, il faut faire d'abord une *pass forward* avant la *pass backward* pour stocker ces valeurs en mémoires.
{: .notice--warning}

```python
''' training methods for 2 layer MLP and cross-entropy loss '''
    def backward(self, X, y):
        m = y.shape[0]
        self.dZ2 = self.A2 - y
        self.dW2 = 1/m * np.dot(self.A1.T, self.dZ2)
        self.db2 = 1/m * np.sum(self.dZ2, axis=0, keepdims=True)
        self.dA1 = np.dot(self.dZ2, self.W2.T)
        self.dZ1 = np.multiply(self.dA1, d_relu(self.Z1))
        self.dW1 = 1/m * np.dot(X.T, self.dZ1)
        self.db1 = 1/m * np.sum(self.dZ1, axis=0, keepdims=True)

    def gradient_descent(self, alpha):
        self.W1 = self.W1 - alpha * self.dW1
        self.b1 = self.b1 - alpha * self.db1
        self.W2 = self.W2 - alpha * self.dW2
        self.b2 = self.b2 - alpha * self.db2
```

{% include mlp_training.html %}

**Note:** Lors de l'apprentissage, on optimise seulement les poids $W$ du modèle. Le nombre de couches cachées ainsi que le nombre de neurones par couche sont fixes et ne changent pas. On parle d'*hyperparamètres*, il faut les choisir lors de la conception du modèle. Des techniques de recherches d'hyperparamètres optimaux existent mais sont complexes et gourmandes en temps de calcul.
{: .notice--danger}

### Aller plus loin

La popularité des réseaux de neurones ces dernières années n'est en réalité pas due au modèle du MLP  présenté jusqu'à présent. En effet, l'inconvénient principal du MLP est le grand nombre de connexion existant entre chaque neurone qui entraîne une forte redondance et une difficulté à l'entrainement lorsque le nombre de neurone et de dimension d'entrée sont élevés. 

<p align="center">
   <img src="/assets/images/advanced_neural_network.png" width="100%"/>
</p>

Sur des problèmes complexes comme l'analyse d'image, le traitement de texte ou le traitement de la parole, l'efficacité des réseaux de neurones actuels est, en majorité, due à des opérations et des connexions plus avancées qui permettent de modéliser et représenter efficacement ces problèmes. Par exemple, pour les images, des opérateurs de convolution sont exploités, ils tirent parti de la structure locale des pixels pour comprendre les informations présentes, allant de formes simples (lignes, cercles, couleur ...) à plus complexes (animaux, bâtiment, paysages ...). Pour des données séquentielles, les connexions LSTM (*long short-term memory*) sont capables de mémoriser des données importantes passées en évitant les problèmes de disparition de gradient. D'autres parts, de nombreuses techniques existent pour améliorer la phase d'apprentissage et avoir des performances qui généralisent le modèle à des données dites de test jamais vues par le modèle lors de l'entraînement (augmentation de données, dropout, early stopping ...).

---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/reseau_de_neurone.ipynb) [![Generic badge](https://img.shields.io/badge/execute_le_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/reseau_de_neurone.ipynb)