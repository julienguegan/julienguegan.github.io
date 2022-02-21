---
title: "Optimization: algorithm, XFOIL, airfoil"
date: 2021-08-21T22:25:30-04:00
lang: en
classes: wide
layout: single
categories:
  - blog
tags :
  - optimisation
  - gradient
  - heuristique
  - aérodynamisme
  - modélisation
header:
  teaser: /assets/images/teaser_airflow.jpg
---

In everyday life, we often seek to optimize our actions to make the least effort and well, in the world of engineering, it's the same thing. Minimization problems are ubiquitous in many systems whether to save time, money, energy, raw material, or even satisfaction. For example, we can seek to optimize a route, the shape of an object, a selling price, a chemical reaction, air traffic control, the performance of a device, the operation of an engine, etc. The complexity of the problems and their modeling makes optimization a very vast and varied branch of mathematics, in practice the quality of the results depends on the relevance of the model, the good choice of the variables that one seeks to optimize, the effectiveness of the algorithm and means for digital processing. In the field of aerodynamics, the shape of airplanes and racing cars is often designed so that the energy expended is minimal. After introducing some algorithmic aspects of the minimization problem, the following article will present how the profile of an aircraft wing can be optimized to maximize its performance.

<p align="center">
   <img src="/assets/images/optimization_problems.png" width="100%"/>
</p>

## Optimization algorithms

Faced with the resolution of an optimization problem, a 1<sup>st</sup> step is to identify to which category it belongs. Indeed, the algorithms are more or less adapted for given categories since the problem can be continuous or discrete, with or without constraints, differentiable or not, convex or not... We write an optimization problem without constraints simply:

$$ \min_{x \in X} f(x) $$

where $f$ can be called an objective function or a cost function. In theory, for unconstrained problems, one can find the minimum(s) by looking at when $ \nabla f(x) = 0 $ ([first-order condition](https://fr.wikipedia.org/wiki/Conditions_d%27optimalit%C3%A9#Conditions_du_premier_ordre_sans_constrainte)) and the positivity of the Hessian $H(x)$ ([second order condition](https://fr.wikipedia.org/wiki/Conditions_d%27optimalit%C3%A9#Conditions_du_second%C3%A8me_ordre_sans_constraint)). For a problem with constraints, the [Kuhn-Tucker conditions](https://fr.wikipedia.org/wiki/Conditions_de_Karush-Kuhn-Tucker) applied to the [Lagrangian](https://fr.wikipedia.org/wiki/Multiplicateur_de_Lagrange) make it possible to transform the problem into a new one without constraints but with additional unknowns.

**Note:** A maximization problem can easily be transposed into a minimization problem:
$$\max_{x \in X} f(x) \Leftrightarrow \min_{x \in X} - f(x)$$
{: .notice--primary}

For simple cases, we can solve the problem analytically. For example, when $f$ has a quadratic, linear and unconstrained form, canceling the gradient amounts to solving a linear system. But, in practice, the gradient may have a too complicated form or even the function $f$ may not have a known analytical form (it may be the result of a numerically solved PDE for example). There is therefore a wide variety of iterative [optimization algorithms](https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Algorithme_d%27optimisation) to try to find the minimum, some being more or less adapted to certain types of problem. On the other hand, it is common to validate these algorithms by testing them on known functions for which we know analytically the value of the true minimum, they make it possible to evaluate the characteristics of the approaches such as the speed of convergence, the robustness, the precision or general behavior. A fairly complete list of these test functions is available on [wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

For the sake of simplicity, the 3 approaches below are simple approaches that introduce the basic notions of optimization algorithms, they will be tested on the Himmelblau test function. In reality, we most often call on libraries or specialized software that implement much more sophisticated approaches.

<center>
{% include himmelblau.html %}
</center>

### Gradient descent

![image-right](/assets/images/gradient_descent.gif){: .align-right width="45%"} The gradient descent algorithm is used to minimize real differentiable functions. This iterative approach successively improves the sought point by moving in the direction opposite to the gradient so as to decrease the objective function. In its simple version, the algorithm finds a local minimum (and not a global one) and may have certain disadvantages such as the difficulty in converging if the parameter $\alpha$ (the step of the descent) is incorrectly set. There is a whole family of methods known as *with descent directions* which exploit the gradient to converge the minimum of $f$ more efficiently.

```python
def gradient_descent(f, x0, gradient, alpha=0.01, itermax=1000):
    # initialization
    x, fx = np.zeros((itermax+1, len(x0))), np.zeros((itermax+1, 1))
    x[0,:] = x0
    # iterative loop
    k = 0
    while (k < itermax):
        grad_fxk = gradient(f, x[k,:]) # use analytical expression or numerical approximation
        x[k+1,:] = x[k,:] - alpha * grad_fxk
        k = k+1
    return x
```

**Note:** If the descent step $\alpha$ is too small, the algorithm risks converging too slowly (or never). If $\alpha$ is too large, the algorithm may diverge (especially by zigzagging in narrow valleys)
{: .notice--info}

### Nelder Mead

![image-right](/assets/images/nelder_mead.gif){: .align-right width="45%"} A major problem with algorithms with descent directions is that they are mostly efficient for differentiable functions and when we know the exact expression of the gradient of $f$. We can nevertheless approximate the gradient by numerical scheme but the approximation made often makes this approach inefficient. The Nelder-Mead method is a method that exploits the concept of [simplex](https://fr.wikipedia.org/wiki/Simplexe): a figure of $N+1$ vertices for a space with $N$ dimensions . The idea consists, at each iteration, in evaluating the value of the function $f$ at each point of the simplex and, according to its values, performing geometric transformations of the simplex (reflection, expansion, contraction). For example, in a valley, the simplex will be stretched in the direction where $f$ decreases. Although simple, this algorithm makes it possible to find a minimum without calculating a gradient, however it is less efficient when the input dimension $N$ is large.

```python
def nelder_mead(f, x0, params=2, itermax=1000):
    c=params
    # initialization
    x1, x2, x3 = np.array([[x0[0]-0.5,x0[1]],[x0[0],x0[1]],[x0[0],x0[1]+0.5] ])
    x = np.array([x1, x2, x3])
    xm = np.zeros((itermax+1, len(x0)))
    # iterative loop
    k = 0
    while (k < itermax):
        # SORT SIMPLEX
        A = f(x.T)
        index = np.argsort(A)
        x_min, x_max, x_bar = x[index[0],:], x[index[2],:], (x[index[0],:] + x[index[1],:])/2
        # REFLECTION
        x_ref = x_bar + (x_bar - x_max)
        # EXPANSION
        if f(x_refl) < f(x_min):
            x_exp = x_bar + 2*(x_bar - x_max)
            if f(x_exp) < f(x_refl):
                x_max = x_exp
            else:
                x_max = x_ref
        elif (f(x_min) < f(x_refl)) and (f(x_refl) < f(x_max)):
            x_max = x_ref
        # CONTRACTION
        else:
            x_con = x_bar - (x_bar - x_max)/2
            if f(x_con) < f(x_min):
                x_max = x_con
            else:
                x[index[1],:] = x_max + (x[index[1],:] - x_min)/2
        # UPDATE DATAs
        x = np.array([x_max, x[index[1],:], x_min])
        xm[k+1,:] = x_bar
        k = k+1
    return xm[:k+1,:]
```

**Warning:** Like gradient descent, the Nelder-Mead method converges to a local minimum of the function $f$ and not a global one. It is however possible to restart the algorithm with a different initialization value $x_0$ to hope to converge towards a new smaller minimum.
{: .notice--warning}

### Evolution strategy

![image-right](/assets/images/evolution_strategy.gif){: .align-right width="45%"} The methods presented previously are able to find minima but local and not global minima. The techniques called evolution strategies are [metaheuristics](https://fr.wikipedia.org/wiki/M%C3%A9taheuristique) inspired by the theory of evolution which statistically converges towards a global minimum. The idea is to start from a population of $\mu$ *parents* which will produce $\lambda$ *children*. Of these $\lambda$ children, only the best ones are selected to be part of the next *generation*. The vocabulary used is that of evolution but, in practice, we make random draws of points and we keep those for which the function $f$ is minimal. This algorithm can find a global minimum but the main drawback is that it requires a large number of evaluations of the function $f$ which is general

```python 
def evolution_strategy(f, x0, params=[5,3,1], imax=1000):
    # parameters
    dim = len(x0)
    lambd, mu, tau = params
    # initialization
    x, xp, s = np.zeros((imax+1, dim)), np.zeros((imax+1, lambd, dim)), np.zeros((imax+1, dim))
    x[0,:] = x0
    s[0,:] = [0.1,0.1]
    # ITERATIVE LOOP
    k = 0
    while (k < imax):
        # GENERATION
        sp = s[k,:] * np.exp(tau * randn(lambd, dim))
        xp[k,:,:] = x[k,:] + sp * randn(lambd, dim)
        Zp = [f(xi) for xi in xp[k,:,:]]
        # SELECTION
        mins = np.argsort(Zp)[:mu]
        xc   = xp[k,mins,:]
        sc   = sp[mins,:]
        # UPDATE
        x[k+1,:] = np.mean(xc, 0)
        s[k+1,:] = np.mean(sc, 0)
        k = k+1
    return x[:k+1,:]
```

## Aerodynamic problem

Let's imagine that we want to create an airplane with a weight of $P=6 kg$ and which will have an average flight speed of $V=12 m/s$, the problem is to design the wings of the airplane in such a way so that the energy that will be expended is minimum. If we consider a stationary flight, there are 4 main opposing forces: thrust (produced by the engines), drag (due to air resistance, the profile of the wing, compressibility... ), weight (earth gravity), and lift (more info at [science étonnante](https://www.youtube.com/watch?v=r-ESaj_4ujc)). The goal of our wing profile optimization problem is therefore to find a wing shape that will minimize drag and maximize lift. The vertical lift $F_y$ of a wing and the horizontal drag $F_x$ are calculated using the following formulas from fluid mechanics:

$$ F_y = \frac{1}{2}\, \rho\, S\, V^2\, C_y \quad \text{et} \quad F_x = \frac{1}{2}\, \rho \, S\, V^2\, C_x $$

with
- $\rho$ the density of air ($kg/m^3$)
- $S$ the surface of the wing ($m^2$)
- $V$ the speed ($m/s$)
- $C_y$ the lift coefficient
- $C_x$ the drag coefficient

Finally, the function to be minimized is written:

$$ f(x) = F_x + \max(0, P - F_y) $$

with $$x = \left[ \text{NACA}_{M} \text{, NACA}_{P} \text{, NACA}_{XX} \text{, L, }\alpha \right]$$
{: .text-center}

```python
# constants
weight = 6
Ro = 1
V=12
# function to minimize
def cost_function(x):
    # call xfoil
    write_xfoil(x)
    os.system(r'xfoil.exe < input.dat')
    CL, CD = read_xfoil()
    # compute COST function
    L = x[3]
    c = (1/10)*L
    S = L*c
    Fx = 0.5*Ro*S*V**2*CD
    Fy = 0.5*Ro*S*V**2*CL
    y = Fx + max(0, weight-Fy)
    return y
```

The parameters to be found defining the shape of the wing are the profile geometry, the wingspan $L$ of the wing and the [angle of attack](https://fr.wikipedia.org/wiki/Incidence_(a%C3%A9rodynamics)) $\alpha$. The geometry of the profile can be defined by the code [NACA](https://fr.wikipedia.org/wiki/Profil_NACA) MPXX where M is the maximum camber, P the point of maximum camber compared to the leading edge of the chord, and XX the maximum thickness of the profile. For example, the NACA 2412 airfoil has a maximum camber of $2%$ to $40%$ from the leading edge with a maximum thickness of 12%. On the other hand, for simplicity, we will assume that the chord is 10 times smaller than the span of the wing. Then, to be able to evaluate the forces $F_x$ and $F_y$, it is necessary to know the coefficients $C_y$ and $C_x$. These coefficients depend on the shape of the wing as well as physical quantities such as the Mach number $Ma = \frac{v}{a} = \frac{12}{340}$ and the Reynolds number $Re = \frac{\rho v L}{\mu} = \frac{12L}{1.8 10^{-5}}$, estimating $C_x$ and $C_y$ is not an obvious problem. But aerodynamic solvers like [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/) implement tools to calculate these coefficients (cf [page 16](http://acversailles.free.fr/documentation/08~Documentation_Generale_M_Suire/Aerodynamics/Profiles/Programs/X%20Foil/xfoil_doc.pdf)). The idea is therefore to execute XFOIL commands and retrieve its output each time the cost function $f$ must be evaluated.

```python
def read_xfoil():
    with open("results.dat", "r") as file:
        coeffs = file.readlines()[-1]
    CL = float(coeffs.split()[1])
    CD = float(coeffs.split()[2])
    return CL, CD

def write_xfoil(x):
    NACAx, NACAy = int(x[0]), int(x[1])
    NACAep, alpha = int(x[2]), x[4]
    chord = (1/10)*x[3]
    Mach = 12/340
    reynolds = chord*12./(1.8*10e-5)
    # write command in file
    file = open("input.dat", "w")
    file.write("plop\ng\n\n")
    file.write("naca "+str(NACAx)+str(NACAy)+str(NACAep)+"\n\noper\n")
    file.write("mach "+str(mach)+"\n")
    file.write("visc "+str(reynolds)+"\n")
    file.write("pacc\nresults.dat\ny\n\n")
    file.write("alfa "+str(alpha)+"\n\nquit")
    file.close()
```

Now that we are able to calculate the function $f$ to minimize, we can apply an optimization algorithm. Since the methods presented in the previous section are basic, there is no shame in directly using libraries like [Scipy](https://docs.scipy.org/doc/scipy/reference/optimize.html) which implements the metaheuristic of [simulated annealing](https://fr.wikipedia.org/wiki/Annealing_simul%C3%A9) (inspired by metallurgical process). Scipy has several other optimization algorithms but, after experimentation, this one seemed to be the most effective.

```python
from scipy import optimize
x0 = np.array([2, 4, 12, 5, 5])
bounds = [(0,4),(2,8),(10,20),(2,6),(0,10)]
optimize.dual_annealing(cost_function, bounds, x0=x0, maxiter=10)
```

<p align="center">
   <img src="/assets/images/optimization_airfoil.gif" width="200%"/>
</p>

*todo : aerosandbox*

---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/optimisation.ipynb) [![Generic badge](https://img.shields.io/badge/execute_le_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/optimisation.ipynb)
