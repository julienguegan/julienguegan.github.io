---
title: "Neural Network: statistics, gradient, perceptron"
date: 2021-09-10T19:25:30-04:00 
lang: en
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

In recent years, we hear more and more in the media the words: *artificial intelligence*, *neural network*, *Deep Learning* ... Indeed, many innovations have emerged thanks to these technologies but that this hides really behind this terminology? Ever since the first programmable computers were designed, people have been amazed to see these computers solving tasks impossible for any human being. However, these problems were actually easy to describe by a formal list of mathematical rules. The real challenge for computers is to perform tasks that humans perform easily and intuitively but have much more difficulty in formally describing, such as recognizing language or faces. Nowadays, we call artificial intelligence (AI) any technique allowing to solve a more or less complex problem by means of a machine. Machine Learning and Deep Learning are fields of study of AI based on statistical theories.

<p align="center">
   <img src="/assets/images/IA_classification.png" width="70%"/>
</p>

## Statistical Learning

Statistical learning (or *Machine Learning*) focuses on the study of algorithms that allow computers to *learn* from data and improve their performance at solving tasks without having explicitly programmed each of them. between them. Consider a set $(x_i,y_i)$ of $n$ training data (with $x_i$ a data and $y_i$ its class). The goal of supervised learning is to determine an estimate $h$ of $f$ using the available data $(x_i,y_i)$. We then distinguish two types of predictions depending on the nature of $Y$: if $Y \subset \mathbb{R}$, we speak of a **regression** problem and if $Y = \{1,..., I\}$ we are talking about a **classification** problem. To estimate the law $f$, a classical approach is the minimization of the empirical risk. Given a loss function $L(\hat{y},y)$ (also called cost, loss, goal function) that measures how well the prediction $\hat{y}$ of a model $h$ is different from the true class $y$ then the empirical risk is:

$$ R_{emp}(h) = \frac{1}{n} \sum_{i=1}^n L\left(h(x_i),y_i\right) $$

The principle of empirical risk minimization says that the learning algorithm must choose a model $\hat{h}$ which minimizes this empirical risk: $ \hat{h} = \arg \min_{h \in \mathcal{ H}} R_{emp}(h) $. The learning algorithm therefore consists in solving an optimization problem. The function to be minimized is an approximation of an unknown probability, it is made by averaging the data available and the more data there is, the more accurate the approximation will be. About the *loss* $L(\hat{y},y)$ function, we can theoretically use the *0-1 loss* function to penalize errors and do nothing otherwise:

$$ L(\hat{y},y) = \left\{
    \begin{array}{ll}
        1 & \text{if } \hat{y} \neq y \\
        0 & \text{if } \hat{y} = y
    \end{array}
\right.$$

But, in practice, it is better to have a continuous and differentiable loss function for the optimization algorithm. For example, the *mean squared error* (MSE) which is often chosen for regression problems, for example for a linear case where we want to find the best straight line passing through points. However, the MSE has the disadvantage of being dominated by *outliers* which can cause the sum to tend towards a high value. A loss frequently used in classification and more suitable is the *cross-entropy* (CE):

$$ MSE = (y - \hat{y})^2 \quad \quad \quad CE = - y \log(\hat{y}) $$

```python
def compute_loss(y_true, y_pred, name):
    if name == 'MSE':
        loss = (y_true - y_pred)**2
    elif name == 'CE':
        loss = -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    return loss.mean()
```

**Note:** To solve this minimization problem, the most efficient way is to use a gradient descent algorithm (see [previous post](https://julienguegan.github.io/posts/2021-08-21-optimisation_profil_aile/#descente-de-gradient)) but requires knowing the exact derivative of the Loss function. Recent Deep Learning libraries (Pytorch, Tensorflow, Caffe ...) implement these optimization algorithms as well as [self-differentiation](https://fr.wikipedia.org/wiki/D%C3%A9rivation_automatic) frameworks 
{: .notice--info}

Now that the notion of learning has been defined, the most important element is to build a $h$ model capable of classifying or regressing data. One of the most well-known models today is that of neural networks.

## Neural Networks

### Perceptron

The first model at the origin of neural networks is the Perceptron (F. Rosenblatt, 1957), it can be seen as a single and simple neuron which solves a linear classification problem. The Perceptron transforms an input vector $X=(x_1,...,x_d) \in \mathbb{R}^d$ into an output $Y \in [0,1]$, the classification label. The idea is that we seek to separate our input space into 2 regions by a hyperplane $\mathcal{H}$ defined simply by:

$$\mathcal{H}: w^T X + b = 0 \iff \mathcal{H}: \sum_{i=1}^d w_i x_i + b = 0 $$

where $w=(w_1,...,w_d) \in \mathbb{R}^d$ and $b \in \mathbb{R}$ are the parameters of the hyperplane $\mathcal{H}$ that commonly called the weights of the neuron.

```python
class Perceptron:
    
    def __init__(self, n_inputs, n_iter=30, alpha=2e-4):
        self.n_iter = n_iter # number of iterations during training
        self.alpha = alpha # learning rate
        self.w = np.zeros((n_iter, n_inputs+1))
        self.w[0,:] = np.random.randn(n_inputs+1) # weights and bias parameters

    def predict(self, X, i=-1):
        return np.sign(X @ self.w[i,1:] + self.w[i,0])
```

<p align="center">
   <img src="/assets/images/perceptron.png" width="100%"/>
</p>

This hyperplane therefore separates the space into 2 and is therefore able to classify a datum $X$ by creating a rule like $ f(X) = 1 \ \text{si } w^T X + b<0 \ ; 0 \text{ otherwise} $. Our neuron model therefore boils down to parameters modeling a hyperplane and a classification rule. In our machine learning paradigm, we assume that we have, at our disposal, a set of $n$ data labeled $(X,Y) \in \mathbb{R}^{n\times d} \times [0 ,1]^n$. As explained in the previous section, the learning phase (or training phase) consists in minimizing the classification error made by the model on these data and adjusting the parameters $w$ and $b$ which allow the Perceptron model to correctly separate these data.

```python
    def gradient_descent(self, i, y, y_pred):
        # construct input X and 1 to have bias
        inputs = np.hstack((np.ones((X.shape[0], 1)), X))
        # compute gradient of the MSE loss
        gradient_loss = -2 * np.dot(y_true - y_pred, inputs)
        # apply a step of gradient descent
        self.w[i+1,:] = self.w[i,:] - self.alpha * gradient_loss
```

<p align="center">
   <img src="/assets/images/perceptron_training.jpg" width="100%"/>
</p>

 <details> <summary> Above, the MSE loss was used. Its exact derivative as a function of $w$ can be calculated easily (see below). </summary>

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
In python, we added 1 to the input X to take into account the bias equation. The gradient is well written `gradient_loss = -2 * np.dot(y_true - y_pred, inputs)`
</details> {: .notice--primary}

Note that this model is very close to a simple linear regression but presented here for a classification problem, it can easily be adapted to a regression by removing the `np.sign()` function in the python code. On the other hand, the problem solved here is binary (we have 2 classes) but we can easily extend the model so that it predicts several classes by replacing the weight vector $w \in \mathbb{R}^d$ by a matrix $W \in \mathbb{R}^{d \times c}$ therefore representing several separating hyperplanes where $c$ is the number of possible classes, the output of the model $Y$ is no longer a scalar but then a vector of $\mathbb{R}^c$. Finally, note that in practice, the classification function $f$ is replaced by a differentiable function $\sigma: \mathbb{R} \rightarrow [0,1]$. For a binary problem, the sigmoid function $\sigma(z)=\frac{1}{1+e^{-z}}$ is used then the class is chosen according to a threshold (ex: $\sigma(z) > $0.5). For a multi-class case, the classification function is a softmax function $\sigma(z_j)=\frac{e^{z_j}}{\sum_{k=1}^Ke^{z_k}}$ which returns a vector of *pseudo-probability* (sum to 1) and the retained class is chosen by keeping the position of the maximum value of the vector.

### Multi-Layer Perceptron

As described above, a single neuron can solve a linearly separable problem but fails when the data is not linearly separable. To approximate nonlinear behaviors, the idea is to add nonlinear functions in the model, they are called *activation functions* (see examples below).

<p align="center">
   <img src="/assets/images/activation_function.png" width="100%"/>
</p>

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read(x):
    return x * (x > 0)

def tanh(x):
    return np.tanh(x)
```

The model of the neural network or Perceptron Multi-Layers consists in chaining successively several linear transformations carried out by simple neurons and nonlinear transformations carried out by these functions of activations until the last operation which will return the class predicted by the model. A layer $l$ of the neural network is then composed of $m$ neurons modeled by a weight matrix $W^l \in \mathbb{R}^{(d+1)\times m}$ (for simplicity we integrates the bias $b$ in $W$) as well as an activation function $A^l$. In the end, the complete neural network can be described by a combination of **composition of functions and matrix multiplications** going from the 1st layer to the last:

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
        self.A1 = reread(self.Z1)
        # second layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        # output
        y_pred = self.A2
        return y_pred
```

**Note:** Note that we can build a network with as many hidden layers as we want and that each of these layers can also consist of an arbitrary number of neurons, regardless of the input dimension and out of the problem.
{: .notice--danger}

<p align="center">
   <img src="/assets/images/multi_layer_perceptron.png" width="70%"/>
</p>

This sequence of layers of neurons poses a problem for the training phase: the calculation of $\frac{\partial\mathcal{L}}{\partial W}$ is less trivial than for the formal neuron model since it takes take into account the weights $W^l$ of each layer $l$. The **gradient backpropagation** technique allows to train multi-layered neural networks based on the chain derivation rule. The loss gradient is calculated using the derivatives of the weights of the neurons and their activation function starting from the last layer to the first layer. It is therefore necessary to browse the network forwards (*<span style="color:green">forward pass</span>*) to obtain the value of the loss then backwards (*<span style="color :red">backward pass</span>*) to obtain the value of the derivative of the loss needed by the optimization algorithm. If we are interested in the neuron $j$ of the layer $l$ towards the neuron $i$ of the layer $l+1$, we note $a$ the value of the vector product, $o$ the output of the neuron ( after activation) and keeping the rest of the previous notations, the calculation of the gradient of $L$ as a function of $W$ is:

$$ 
\begin{align*}
    \dfrac{\partial L}{\partial w_{ij}^l} &= \underbrace{\quad \dfrac{\partial L}{\partial a_{j}^l} \ \ } \ \ \underbrace{\quad \dfrac{\partial a_j^l}{\partial w_{ij}^l} \quad} \\ 
                                      & \qquad \ \ \ \delta_j^l \quad \ \ \ \dfrac{\partial}{\partial w_{ij}^l} \sum_{n=0}^{N_{l-1}} w_{nj}^l o_n^{l-1} = o_i^{l-1}
\end{align*} 
$$

We therefore have that the derivative $\dfrac{\partial L}{\partial w_{ij}^l}$ depends on the term $\delta_j^l$ of the layer $l$ and on the output $o_i^{l- 1}$ of layer $l-1$. This makes sense since the weight $w_{ij}^l$ connects the output of neuron $i$ in layer $l-1$ to the input of neuron $j$ in layer $l$. We now expand the term $\delta_j^l$:

$$ \delta_j^l = \dfrac{\partial L}{\partial a_{j}^l} =  \sum_{n=1}^{N_{l+1}} \dfrac{\partial L}{\partial a_n^{l+1}}\dfrac{\partial  a_n^{l+1}}{\partial  a_j^l} = \sum_{n=1}^{N_{l+1}} \delta_n^{l+1} \dfrac{\partial  a_n^{l+1}}{\partial  a_j^l} $$

however, we have:

$$ 
\begin{align*}
    & a_n^{l+1} &=& \sum_{n=1}^{N_{l}} w_{jn}^{l+1}A(a_j^l) \\
    \Rightarrow & \dfrac{\partial a_n^{l+1}}{\partial  a_j^l} &=& \ w_{jn}^{l+1}A'(a_j^l)
\end{align*}
$$

and so:

$$ \dfrac{\partial L}{\partial w_{ij}^l} = \delta_j^l o_i^{l-1} = A'(a_j^l) o_i^{l-1} \sum_{n=1}^{N_{l+1}} w_{jn}^{l+1} \delta_n^{l+1} $$ 

We therefore obtain that the partial derivative of $L$ with respect to $w_{ij}$ at layer $l$ also depends on the derivative at layer $l+1$. To calculate the derivative of the entire network using the rule of chain derivation, it is therefore necessary to start with the last layer and end with the first, hence the term *error backpropagation*.

**Warning:** As the calculations of the backpropagation phase also depend on $a_j^l$ and $o_i^{l-1}$, you must first do a *pass forward* before the *pass backward* to store these values in memory.
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

**Note:** During training, only the weights $W$ of the model are optimized. The number of hidden layers as well as the number of neurons per layer are fixed and do not change. We are talking about *hyperparameters*, they must be chosen when designing the model. Optimal hyperparameter search techniques exist but are complex and computationally intensive.
{: .notice--danger}

### Go further

The popularity of neural networks in recent years is actually not due to the MLP model presented so far. Indeed, the main drawback of MLP is the large number of connections existing between each neuron which leads to high redundancy and difficulty in training when the number of neurons and the input dimension are high.

<p align="center">
   <img src="/assets/images/advanced_neural_network.png" width="100%"/>
</p>

On complex problems such as image analysis, word processing or speech processing, the effectiveness of current neural networks is, for the most part, due to more advanced operations and connections that allow modeling and effectively represent these issues. For example, for images, convolution operators are exploited, they take advantage of the local pixel structure to understand the information present, ranging from simple shapes (lines, circles, color ...) to more complex (animals, building , landscapes, etc.). For sequential data, LSTM (*long short-term memory*) connections are able to memorize important past data avoiding the problems of disappearance of gradient. On the other hand, many techniques exist to improve the learning phase and have performances that generalize the model to so-called test data never seen by the model during training (data augmentation, dropout, early stopping ...).

---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/reseau_de_neurone.ipynb) [![Generic badge](https://img.shields.io/badge/execute_le_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/reseau_de_neurone.ipynb)