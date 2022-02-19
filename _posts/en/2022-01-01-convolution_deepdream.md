---
title: "CNN : convolution, Pytorch, Deep Dream"
date: 2022-01-01T17:10:10-02:00 
lang: en
classes: wide
layout: single
categories:
  - blog
tags :
  - machine learning
  - deep learning
  - convolution
  - deep dream
header:
  teaser: /assets/images/teaser_deepdream.jpg
---

Convolutional neural networks (CNN) are the models that have made it possible to make a leap forward in image recognition problems. They are at the heart of many applications ranging from facial identification security systems to the classification of your vacation photos, including synthetic face generation and Snapchat filters. One of the founders of this model is Yann Le Cun (a Frenchman!) who, in 1989, applied gradient backpropagation to learn convolution filters and enabled a neural network to recognize handwritten digits. However, it was only in 2012 that CNNs spread widely in the computer vision scientific community with Alex Krizhevsky designing the *AlexNet* architecture and winning the *ImageNet Large Scale Visual Recognition Challenge* competition (1 million images of 1000 different classes) by implementing its algorithm on GPUs which allows the model to learn quickly from a large quantity of images. This model achieves 10% higher performance than all the others at this time and it is now one of the most influential published papers in Computer Vision (in 2021, more than 80,000 citations according to Google Scholar).

<p align="center">
   <img src="/assets/images/cnn_header.png" width="80%"/>
</p>

## Convolutions and Neural Networks

Completely connected neural network models (see [previous post](https://julienguegan.github.io/posts/2021-09-10-reseau_de_neurone/)) are not suitable for solving image processing problems . Indeed, MLPs have each neuron of a layer connected to each input unit: the number of parameters to learn quickly becomes high and a strong redundancy in the weights of the network can exist. Moreover, to use an image in such a network, all the pixels would have to be transformed into a vector and no information on the local structure of the pixels would then be taken into account.

The convolution product, denoted $\ast$, is an operator which generalizes the idea of ​​a moving average. It applies both to temporal data (in signal processing for example) and to spatial data (in image processing). For the case of images, i.e. discrete and in 2 dimensions, the convolution between an image $I$ and a kernel $w$ (or kernel) can be calculated as follows:

$$I(i,j) * \omega =\sum_{x=-a}^a{\sum_{y=-b}^b{ I(i+x,j+y)} \ \omega(x ,y)}$$

The idea is to drag the kernel spatially over the entire image and each time to make a weighted average of the pixels of the image found in the window concerned by the elements of the kernel. Depending on the value of the elements of the convolution kernel $w$, the operation can highlight particular characteristics found in the image such as contours, textures, shapes.

<p align="center">
   <img src="/assets/images/image_convolution.gif" width="40%"/>
</p>

**Note:** There are several parameters associated with the convolution operation like the size of the kernel used, the step size when dragging the window over the image, the way we handle the edges of the image, the rate of core expansion... [more info here](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215 )
{: .notice--warning}

For example, we can highlight the pixels of an image corresponding to the horizontal contours by applying a convolution with a kernel of size $3 \times 3$ with $-1$ in the 1st line, $0$ in the 2nd line and $+1$ in the 3rd row of the matrix.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
#readimage
image = np.array(Image.open("path/to/file.jpg").convert('L'))
# apply convolution
kernel = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [+1, +1, +1]])
conv_output = convolve2d(image, kernel, mode='same')
# display
plt.figure(figsize=(15.5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.axis('off')
plt.subplot(122), plt.imshow(np.abs(conv_output), cmap='gray'), plt.axis('off')
plt.tight_layout()
plt.show()
```

<p align="center">
   <img src="/assets/images/convolution_exemple.png" width="80%"/>
</p>

The idea of ​​the CNN model architecture is to keep completely connected layers for classification. However, at the inputs of these layers, the image is not directly used, but the output of several convolution operations which aim to highlight the different characteristics of an image by encoding the objects in a certain way. which are present or not. In particular, multi-channel convolutions are used which consist in applying a standard convolution to each channel of the input then summing each convolution product obtained to obtain a single 2D matrix. For example for a color image the channels are red, green and blue, we then have 3 kernels to convolve with the associated channels then the 3 products obtained are summed.

<p align="center">
   <img src="/assets/images/multichannel_convolution.png" width="100%"/>
</p>

**Note:** In 2D (1 channel only), we use the term *kernel* to talk about the kernel. In 3D (more than one channel), we use the term *filter* which is made up of as many kernels as the number of channels of the input volume.
{: .notice--info}

More precisely in CNNs, a convolutional layer is composed of a set of $N_f$ filters of size $N_W$ x $N_H$ x $N_C$ plus a bias per filter followed by a nonlinear activation function. Here, $N_W$ and $N_H$ designate the spatial sizes of the filter while $N_C$ is the number of channels (sometimes called *feature map*). Each filter performs a multi-channel convolution, we then obtain $N_f$ convolution products which are concatenated in an output volume. These $N_f$ products then become the channels of the next volume which will pass into the next convolutional layer. Note that the depth of the filters must necessarily correspond to the number of channels of the input volume of each layer but the number of filters is an architecture hyperparameter of the model. In the end, the sequence of these multichannel convolutions creates at output a volume of features (*features*) of the input image, these features are then passed to the fully connected network for classification.

<p align="center">
   <img src="/assets/images/architecture_cnn.png" width="100%"/>
</p>

**Important:** A convolutional layer is usually composed (in addition to convolution) of a nonlinear activation function and sometimes other types of operations (pooling, batch-normalization, dropout...).
{: .notice--success}

In the previous post, we defined an MLP and its training from zero. Here, the **PyTorch** library is used. It allows to easily build neural networks by taking advantage of its [automatic differentiation engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/#what-is-autograd) for the training as well as its many specialized functions (like the [convolution](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
```

```python
class My_Custom_Model(nn.Module):

    def __init__(self):
        ''' define some layers '''
        super().__init__()
        # feature learning
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # classification
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ''' create model architecture - how operations are linked '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.read(self.fc1(x))
        x = F.read(self.fc2(x))
        x = self.fc3(x)
        return x
```

As you may have understood, what is interesting with these convolution operations is that the weight of the filters can be learned during the optimization by backward propagation of the gradient since it is possible to calculate in an exact way the value of $ \frac{\ partial\mathcal{L}}{\partial W}$ by chain derivation.


```python
# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_cnn_model.parameters(), lr=0.001)
# training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data
        # Forward pass
        output = model(images)
        loss = criterion(outputs, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # optimization step
        optimizer.step()
```

**Note:** For large input data, we often use as an optimization algorithm a *stochastic gradient descent* where the loss is approximated by using a batch of some data (for example, 8, 16 or 32 pictures).
{: .notice--info}


## Deep Dream

One of the challenges of neural networks is to understand what exactly is happening at each layer. Indeed, their cascade architecture as well as their numerous interconnections make it not easy to interpret the role of each filter. The visualization of *features* is a line of research that has developed in recent years, which consists of finding methods to understand how CNNs see an image.

DeepDream is the name of one of these techniques created in 2015 by a team of engineers from Google, the idea is to use a network already trained to recognize shapes to modify an image so that a given neuron returns an output higher than the others. The algorithm looks like classic backpropagation but instead of modifying the weights of the network we adjust the pixels of the input image. Moreover, the optimization criterion is not a cross entropy but directly the norm of the output of the neuron to be visualized (it can be the entire layer or a filter) that we will seek to maximize, we then make a rise gradient (we could also minimize the opposite).

```python
# Parameters
iterations = 25 # number of gradient ascent steps per octave
at_layer = 26 # layer at which we modify image to maximize outputs
lr = 0.02 # learning rate
octave_scale = 2 # image scale between octaves
num_octaves = 4 # number of octaves
```

```python
# Load Model pretrained
network = models.vgg19(pretrained=True)
# Use only needed layers
layers = list(network.features.children())
model = nn.Sequential(*layers[: (at_layer + 1)])
# Use GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

An additional trick to obtain an interesting visualization is to operate at different spatial resolutions, here we speak of *octave*. In addition, the loss is normalized at all layers so that the contribution of large layers does not outweigh that of small layers.

```python
# loop on different resolution scale
detail = np.zeros_like(octaves[-1])
for k, octave_base in enumerate(tqdm(octaves[::-1], desc="Octaves : ")):
    # Upsample detail to new octave dimension
    if k > 0: 
      detail = nd.zoom(detail, np.array(octave_base.shape)/np.array(detail.shape), order=1) 
    # Add detail from previous octave to new base
    input_image = octave_base + detail
    # Updates the image to maximize outputs for n iterations
    input_image = Variable(torch.FloatTensor(input_image).to(device), requires_grad=True)
    for i in trange(iterations, desc="Iterations : ", leave=False):
        model.zero_grad()
        out  = model(input_image)
        loss = out.norm()
        loss.backward()
        # gradient ascent
        avg_grad = np.abs(input_image.grad.data.cpu().numpy()).mean()
        norm_lr  = lr/avg_grad
        input_image.data = input_image.data + norm_lr * input_image.grad.data
        input_image.data = clip(input_image.data)
        input_image.grad.data.zero_()
        # Extract deep dream details
        detail = input_image.cpu().data.numpy() - octave_base
```

Depending on the number of iterations, we obtain more and more abstract images with psychedelic shapes that appear as and when the name *DeepDream*. In fact, these abstract shapes are present especially for the deeper layers, the first layers usually accentuate simple *features* like edges, corners, textures...

<p align="center">
   <img src="/assets/images/deepdream_exemple.gif" width="80%"/>
</p>

With this tool, one can create very advanced artistic effects like on [DeepDreamGenerator's instagram](https://www.instagram.com/deepdreamgenerator/). But we can also accentuate the pscychedelic effect by doing a lot of iterations or by feeding the output of the algorithm as input several times. And with a little effort, one can manage to visualize what it is like to go to the supermarket in these dreams from very real images.

{% include video id="DgPaCWJL7XI" provider="youtube" %}

As presented above, Deep Dream has a drawback if you want to run it on an input white noise image to visualize what might emerge and thus have a more accurate representation of CNN *features*. Indeed, it can be seen that the image then remains dominated by high-frequency patterns.

<p align="center">
   <img src="/assets/images/deepdream_noise.png" width="80%"/>
</p>

Generally, to counter this effect, what works best is to introduce regularization in one way or another into the model. For example, transformation robustness tries to find examples that always strongly activate the optimization function when they are very weakly transformed. Concretely, this means that we shake, rotate, decrease or increase the image randomly before applying the optimization step. The [lucid](https://github.com/tensorflow/lucid) (tensorflow) and [lucent](https://github.com/greentfrapp/lucent) (pytorch) libraries are open-source packages that implement all kinds of visualization methods.

```python
# load libraries
from lucent.optvis import render
from lucent.modelzoo import vgg19
#loadmodel
model = vgg19(pretrained=True)
model = model.to(device)
model.eval()
# run optimization
image = render.render_vis(model, "features:30",thresholds=[100],show_inline=True)
```

A much more comprehensive article on feature visualization techniques is available [here](https://distill.pub/2017/feature-visualization/)

<p align="center">
   <img src="/assets/images/lucid_viz.png" width="100%"/>
</p>

---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/visualisation_CNN.ipynb) [![Generic badge](https://img.shields.io/badge/execute_le_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/visualisation_CNN.ipynb)