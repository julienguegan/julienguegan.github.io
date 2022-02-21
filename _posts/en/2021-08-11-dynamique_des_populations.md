---
title: "Population dynamics: ODE, ecology, logistics"
date: 2021-08-11T22:18:30-04:00
lang: en
classes: wide
layout: single
categories:
  - blog
tags:
  - équations différentielles ordinaires
  - écologie
  - modélisation
  - équation logistique
header:
  teaser: /assets/images/teaser_dynamique_population.jpg
---

Among the challenges of the 21<sup>th</sup> century, ecology has a major role since it is the science that studies the interactions of living beings with each other and with their environment. To model these interactions, population dynamics is the branch that is interested in the demographic fluctuations of species. Its applications are numerous since it can make it possible to respond to various problems such as the management of endangered species, the protection of crops against pests, the control of bioreactors or the prediction of epidemics.

<p align="center">
   <img src="/assets/images/ecologie.png"/>
</p>

## Verhulst model

At the end of the 18<sup>th</sup> century, the model of **Malthus** describes the variation of a population size $y$ over time $t$ by the ordinary differential equation[^1] (EDO):

[^1]: The term ordinary is used as opposed to the term partial differential equation (or partial differential equation) where the unknown function(s) may depend on more than one variable.

$$ y'(t) = (n-m) y(t) = r y(t) $$

with the constants: $n$ the birth rate, $m$ the death rate and $r$ the growth rate. This model tells us that, depending on the growth rate $r$, the size of the populations can either decrease, remain constant or increase exponentially. This model does not reflect reality since a population will never increase ad infinitum.

<p align="center">
   <img src="/assets/images/malthus_verlhust_photos.png" width="50%"/>
</p>

In 1840, **Verlhust** proposed a more suitable growth model based on the assumption that the growth rate $r$ is not a constant but is an affine function of the population size $y$:

$$ y'(t) = \big(n(y) - m(y)\big) y(t) $$

Verlhust notably starts from the hypothesis that the more the size of a population increases, the more its birth rate $n$ decreases and the more its death rate $m$ increases. Starting from this hypothesis and applying some clever algebraic manipulations, we can show that the previous differential equation can be rewritten in the form:

$$ y'(t) = r y(t) \left(1 - \frac{y(t)}{K}\right) $$

with $K$ a constant called *accommodation capacity*. We can analytically solve this equation with the initial condition $y(t=0)=y_0$, we obtain the **logistic solution**:

$$ y(t) = \frac{K}{1+\left(\frac{K}{y_0}-1\right)e^{-rt}} $$

<p align="center">
   <img src="/assets/images/verlhust_graph.png" width="70%"/>
</p>

<details>
  <summary>Detailed solution of the logistic differential equation by variable separation</summary>

  $$
  \begin{align*}
    \int_{y_0}^{y(t)} \frac{1}{y(1-y/K)}dy &= \int_0^t r \ d\tau \\
    \int_{y_0}^{y(t)} \frac{K}{y(K-y)}dy &= \int_0^t r \ d\tau \\
    \int_{y_0}^{y(t)} \frac{1}{y}dy +  \int_{y_0}^{y(t)} \frac{1}{K-1}dy &= \int_0^t r \ d\tau \\
    \ln \left| \frac{y(t)}{y_0} \right| - \ln \left| \frac{K-y(t)}{K-y_0} \right| &= r \ t \\
    \ln \left( \frac{y(t)\big(K-y_0\big)}{y_0\big(K-y(t)\big)} \right) &= r \ t \\
    \frac{y(t)}{K-y(t)} &= \frac{y_0}{K-y_0}e^{rt} \\
    y(t)\left(1+\frac{y_0}{K-y_0}e^{rt} \right) &= \frac{K y_0 e^{rt}}{K-y_0} \\
    y(t) &= \frac{Ky_0e^{rt}}{K-y_0+y_0e^{rt}} \\
    y(t) &= \frac{K y_0}{(K-y_0)e^{-rt}+y_0} \\
  \end{align*} \\
  \square
  $$
</details> {: .notice--primary}

Note that $ \lim\limits_{t\to\infty} y(t) = K $. This means that regardless of the size of the initial population $y_0$, the population will always end up tending towards $K$ the carrying capacity that is often qualified as the maximum number of individuals that the environment can accommodate (according to space, resources, etc.). This [so-called logistic function](https://fr.wikipedia.org/wiki/Fonction_logistique_(Verhulst)) introduced for the first time by Verlhulst to model the growth of populations will subsequently find many applications in various fields such as economics, chemistry, statistics and more recently artificial neural networks.

## Lotka-Volterra Model

Lotka-Volterra models are systems of simple equations that appeared at the beginning of the 20<sup>th</sup> century. They bear the name of two mathematicians who published simultaneously but independently on the subject: Volterra, in 1926, to model the populations of sardines and their predators and Lotka, in 1924, in his book _Elements of Physical Biology_. Unlike the Verlhust model, which focuses on a single population, the Lotka-Volterra models model the interactions between several species, each having an impact on the development of the other.

<p align="center">
   <img src="/assets/images/lotka_volterra_photos.png" width="50%"/>
</p>

### *Prey-predator*

The prey-predator model of Lotka-Volterra has made it possible to explain data collected from certain populations of animals such as the lynx and hare as well as the wolf and the elk in the United States. It represents the evolution of the number of prey $x$ and predators $y$ over time $t$ according to the following model:

$$
\left\{
  \begin{array}{ccc}
    x'(t) = x(t)\ \big(\alpha - \beta y(t)\big) \\
    y'(t) = y(t)\ \big( \delta x(t) - \gamma\big)
  \end{array}
\right.
$$

with the parameters $\alpha$ and $\delta$ are the reproduction rates of prey and predators respectively and $\beta$ and $\gamma$ are the mortality rates of prey and predators respectively.

**Note:** We speak of an autonomous system: the time $t$ does not appear explicitly in the equations.
{: .notice--primary}

If we develop each of the equations, we can more easily give an interpretation. For prey, we have on the one hand the term $\alpha x(t)$ which models exponential growth with an unlimited source of food and on the other hand $- \beta x(t) y(t)$ which represents predation proportional to the frequency of encounter between predators and prey. The equation for predators is very similar to that for prey, $\delta x(t)y(t)$ is the growth of predators proportional to the amount of food available (the prey) and $- \gamma y(t) $ represents the natural death of predators.

<p align="center">
   <img src="/assets/images/fox_rabbit.gif" width="70%"/>
</p>

We can calculate the equilibria of this system of differential equations and also deduce a behavior from it, but the solutions do not have a simple analytical expression. Nevertheless, it is possible to compute an approximate solution numerically (more details in the [`next section`](#numerical-method-for-odes)). 

```python
# define ODE function to resolve
r, c, m, b = 3, 4, 1, 2
def prey_predator(XY, t=0):
    dX = r*XY[0] - c*XY[0]*XY[1]
    dY = b*XY[0]*XY[1] - m*XY[1]
    return [dX, dY]
```
```python
# discretization
T0   = 0
Tmax = 12
n    = 200
T    = np.linspace(T0, Tmax, n) 
```
Here we calculate the evolution of the 2 populations as a function of time for a fixed initial condition, we see that they have a periodic behavior and a phase shift.

```python
# TEMPORAL DYNAMIC
X0 = [1,1]
solution = integrate.odeint(prey_predator, X0, T) # use scipy solver
```
<p align="center">
   <img src="/assets/images/lotka_volterra_graph2.png" width="70%"/>
</p>

Here, we calculate several solutions for different initial conditions that we display in phase space (time does not appear). You can also display the vector field generated by the system of equations with `plt.quiver()` for a grid of values.

```python
# PHASES SPACE
# some trajectories
orbits = []
for i in range(5):
    X0    = [0.2+i*0.1, 0.2+i*0.1]
    orbit = integrate.odeint(prey_predator, X0, T)
    orbits.append(orbit) 
# vector field
x, y             = np.linspace(0, 2.5, 20), np.linspace(0, 2, 20)
X_grid, Y_grid   = np.meshgrid(x, y)                      
DX_grid, DY_grid = prey_predator([X_grid, Y_grid])
N                = np.sqrt(DX_grid ** 2 + DY_grid ** 2) 
N[N==0]          = 1
DX_grid, DY_grid = DX_grid/N, DY_grid/N
```

<p align="center">
   <img src="/assets/images/lotka_volterra_graph1.png" width="70%"/>
</p>

**Warning:** The units of the simulations do not reflect reality, populations must be large enough for the modeling to be correct.
{: .notice--danger}

In the model used, predators thrive when prey is plentiful, but eventually exhaust their resources and decline. When the predator population has diminished enough, the prey taking advantage of the respite reproduce and their population increases again. This dynamic continues in a cycle of growth and decline. There are 2 equilibria: the point $(0,0)$ is an unstable saddle point which shows that the extinction of the 2 species is in fact almost impossible to obtain and the point $(\frac{\gamma}{\delta }, \frac{\alpha}{\beta})$ is a stable center, the populations oscillate around this state.

**Note:** This modeling remains quite simple, a large number of variants exist. We can add terms for the disappearance of the 2 species (due to fishing, hunting, pesticides, etc.), take into account the carrying capacity of the environment by using a logistical term.
{: .notice--info}

### *Competition*

The Lotka-Volterra competition model is a variant of the predation model where the 2 species do not have a hierarchy of prey and predators but are in competition with each other. Moreover, the basic dynamic is no longer a simple exponential growth but a logistic one (with the parameters $r_i$ and $K_i$):

$$
\left\{
  \begin{array}{ccc}
    x_1'(t) = r_1x_1(t)\left(1- \frac{x_1(t)+\alpha_{12}x_2(t)}{K_1}\right) \\
    x_2'(t) = r_2x_2(t)\left(1- \frac{x_2(t)+\alpha_{21}x_1(t)}{K_2}\right)
  \end{array}
\right.
$$

with $\alpha_{12}$ the effect of species 2 on the population of species 1 and conversely $\alpha_{21}$ the effect of species 2 on species 1. For example, for the species 1 equation, the coefficient $\alpha_{12}$ is multiplied by the population size $x_2$. When $\alpha_{12} < 1$ then the effect of species 2 on species 1 is smaller than the effect of species 1 on its own members. And conversely, when $\alpha_{12} > 1$, the effect of species 2 on species 1 is greater than the effect of species 1 on its own members.

<p align="center">
   <img src="/assets/images/competition_interspecific.jfif" width="60%"/>
</p>

To understand the predictions of the models in more detail, it is useful to plot the phase space diagrams $(x_1,x_2)$ as before. We can distinguish 4 scenarios according to the values ​​of the competition coefficients, I display below the vector fields of these scenarios with `plt.streamplot()` as well as the isoclines, the curves for which $$x_1'(t) =0$$ or $$x_2'(t)=0$$:

<p align="center">
  <img src="/assets/images/lotka_volterra_graph3.png" width="70%"/>
</p>

```python
# define ODE to resolve
r1, K1 = 3, 1
r2, K2 = 3, 1
def competition(X1X2, a1, a2):
    dX1 = r1*X1X2[0] * (1-(X1X2[0]+a1*X1X2[1])/K1)
    dX2 = r2*X1X2[1] * (1-(X1X2[1]+a2*X1X2[0])/K2)
    return [dX1, dX2]
```
```python
# compute derivatives for each scenario
N = 20
x, y = np.linspace(0, 2.5, N), np.linspace(0, 2, N)
X_grid, Y_grid = np.meshgrid(x, y)
DX_grid, DY_grid = np.zeros((4,N,N)), np.zeros((4,N,N))
coeffs = np.array([[1.5,1.5],[0.5,0.5],[1.5,0.5],[0.5,1.5]])
for k,(a1,a2) in enumerate(coeffs):
    DX_grid[k,:], DY_grid[k,:] = competition([X_grid, Y_grid], a1, a2)
```

In the end, the 4 possible behaviors depending on $\alpha_{12}$ and $\alpha_{21}$ are the following:

1. Competitive exclusion of one of the two species depending on the initial conditions.
2. Stable coexistence of the two species.
3. Competitive exclusion of species 1 by species 2.
4. Competitive exclusion of species 2 by species 1.

The stable coexistence of the 2 species is only possible if $\alpha_{12} < 1$ and $\alpha_{21} < 1$, i.e. *interspecific competition* must be weaker than *intraspecific competition*.

## Numerical method for ODEs

This section is a little bit apart from the real subject of this post since I introduce numerical methods to solve differential equations. Indeed, it is possible to deduce many properties of an ODE system based on mathematical theorems for the theory of dynamical systems ([Lyapunov method](https://fr.wikipedia.org/wiki/ Stabilit%C3%A9_de_Liapounov), [LaSalle invariance](https://en.wikipedia.org/wiki/LaSalle%27s_invariance_principle), [Poincaré-Bendixon theorem](https://fr.wikipedia.org/wiki/ Th%C3%A9or%C3%A8me_de_Poincar%C3%A9-Bendixson) ...) but only a restricted number of differential equations admit an analytical solution. In practice, we often prefer to have a method that computes an approximate solution of the problem at any time $t$. We consider the problem $$y'(t) = f\big(t,y(t)\big)$$ with $y(t_0)=y_0$. The idea of ​​numerical methods is to solve the problem on a discrete set of points $(t_n,y_n)$ with $h_n=t_{n+1}-t_n$, a fixed time step.

**Euler**

![image-right](/assets/images/euler_method.png){: .align-right width="30%"} Euler's method is the most basic numerical method for ODE, it uses the differential equation to calculate the slope of the tangent at any point on the solution curve. The solution is approached starting from the known initial point $y_0$ for which the tangent is calculated, then a time step is taken along this tangent, a new point $y_1$ is then obtained. The idea is to repeat this process, for a time step from $t_n$ to $t_{n+1}$ we can write it as $y_{n+1} = y_n + hf(t_n,y_n)$ .

This method is very simple to set up, for example in python:
```python
def Euler_method(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0,:] = y0
    for i in range(slow(t)-1):
        y[i+1] = y[i] + h*f(y[i], t[i])
    return y
```

**Runge Kutta**

![image-right](/assets/images/rungekutta_method.png){: .align-right width="45%"} Runge-Kutta methods are a family of methods, the best known is that of order 4 It is more precise than the Euler method by making a weighted average of 4 terms corresponding to different slopes of the curve in the fixed time interval. It is defined by:

$$ y_{n+1} = y_n + \frac{h}{6} (k_1+2k_2+2k_3+k_4) $$

with:
 - $k_1=f(t_n,y_n)$
 - $k_2=f(t_n+h/2,y_n+hk_1/2)$
 - $k_3=f(t_n+h/2,y_n+hk_2/2)$
 - $k_4=f(t_n,y_n+hk_3)$

```python
def RungeKutta4_method(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        k1 = f(y[i], t[i])
        k2 = f(y[i]+k1*h/2, t[i]+h/2)
        k3 = f(y[i]+k2*h/2, t[i]+h/2)
        k4 = f(y[i]+k3*h, t[i]+h)
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return y
```

**Example**

To verify the behavior of these methods, I chose to test them on a well-known model of physics: the harmonic oscillator. In its simplest form, we can calculate exactly the solution of the problem and therefore compare our approximate resolution methods. The problem can be written:

$$
\left\{
  \begin{array}{ccc}
    y''(t) + y(t) = 0 \\
    y(0) = y_0
  \end{array}
\right.
$$

and admits for solution $ y(t) = y_0 \cos(t) $.

```python
# initial condition
y0 = [2, 0]
# discretization
t = np.linspace(0.5*pi, 100)
h = t[1] - t[0]
# ODE wording
defproblem(y, t):
    return np.array([y[1], -y[0]])
# analytics solution
def exact_solution(t):
    return y0[0]*np.cos(t)
y_exact = exact_solution(t)
y_euler = Euler_method(problem, y0, t)[:, 0]
y_rk4 = RungeKutta4_method(problem, y0, t)[:, 0]
```

As expected the Runge-Kutta method is much more accurate than the Euler method (but slower). The error of the numerical methods decreases according to the size of the time step $h$ but the smaller $h$ is, the longer the calculation time. In theory, to analyze numerical methods, we base ourselves on 3 criteria:
- the convergence which guarantees that the approximate solution is close to the real solution.
- the order which quantifies the quality of the approximation for an iteration.
- the stability which judges the behavior of the error.

<p align="center">
   <img src="/assets/images/numerical_ODE.gif" width="70%"/>
</p>

**In practice**: In the [`previous problems`](#predator-predator) , I use the `integrate.odeint(f, y0, t)` function from [scipy](https://docs. scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html) which is a more advanced ODE solver that uses the method of [Adams–Bashforth](https://en.wikipedia.org /wiki/Linear_multistep_method#Adams–Bashforth_methods)
{: .notice--info}

---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/dynamique_population.ipynb) [![Generic badge](https://img.shields.io/badge/execute_le_code-binder-ff69b4.svg?style=plastic&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAMYAAADGCAMAAAC%2BRQ9vAAACOlBMVEX%2F%2F%2F9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olJXmsq%2FdJX1olLVa4pXmsrmZYH1olL1olJXmspXmsrmZYH1olJXmsr1olJXmspXmsr1olJXmsr1olJXmsrmZYH1olL1olL1olJXmspXmsrmZYH1olL1olL1olJXmsrmZYH1olL1olL1olJXmsrmZYHqdnT1olJXmsq6dZf1olJXmsrKk3rmZYH1olJXmsrCc5RXmsr0n1TtgWz1olJXmspXmsrmZYH1olJXmsqNhq%2Fzmlj1olJXmspZmshXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olL1olJXmsr1olJXmsrtgGz1olL1olJXmsr1olJXmsrmZYH1olJXmsrbaYf1olJXmsr1olJXmsr1olLIcJFXmsr1olJXmsr1olJXmsr1olJXmsr1olL1olJXmspZmshZmsldmsZemsVfl8Zgl8Zom71pk8Frm7tvm7dxkL1ykLx0m7R4m7F6jbh7jbh8nK6CnKmDirOEibOGnKaInKWNhq%2BNnKGSnZ2Vg6qegKaff6WfnZSnfKGnno6ofKGvnoeweZyxeZy3noG5dpjCcpPDcpPGn3bLb4%2FPoG%2FVa4rXoGnYoGjdaIbeaIXhoWHmZYHnaX7obXvpcHjqdHXreHLroVrtgGzuhGnuh2bxk17yl1vzm1j0nlX1olIgJPdZAAAAfnRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hYWFtgYGBkZnBwcHFxdHx8fn6AgICHiIuQkJCSnKCgoKavsLCwsLO4uMDAwMDBwcTFxsjO0NDQ09TW1tjY3Nzd4ODg4uLl5%2Bjo6uvr7O3v8PDw8%2FPz9vb39%2Fj5%2Bfv7%2FPz9%2Ff5K%2BfZ5AAAI4ElEQVR42uzWAWfDQBjG8Yc4qoihEApBIIoOOpaiFAUBBB3EjFDKRImZy0d7vtuYYWN36Zq4u5v7fYO%2FB%2B%2BLwENBEARBEAR32Zc0gpcWRXmS%2FO7SHPI5PDIvaip01TrypKGlXr2B6%2FKaV%2BirGA67v%2FBa9dKrCLWXGA5anvhXlYBjopI36DdwStrxNo2AO%2Fa8WZ%2FBEaLhGHs4YdFxnGME%2B5KeY7UCtq160v%2BOFUn%2FOxLyH3QkPafSwhrxzukcYcsrp7SFHSWnlcGGnEOaQ57i0ywrqo4DpIB5QlLruI7w07w4U%2BsZ5j1R420n8Ju46qmxhmkZ1WQBJVHq6gUM66hUCujEJ3e%2B3YIqMsWQLZVmMCmSVDgLDEskFR5h0m7kLRatC3NEckSFosPCHA%2FqitEdMxjzwbxZN7eRNGG8tcpr%2BS2vA3KFmZODoFLlDaOS4%2FXxleVj9OqYacLMzMzYR%2BHsZwtz5hnvSNOSf%2F97Vc%2F0NI%2B%2FBwM0q%2FQJMsjoynXfYFr%2BPxe9SgtVijdiLT3Jjrmxlu5UIf5wlLq%2BraqTD9dfqbSjFrhY1T5jLNkzMdbRUMVy6nsqgdpYx4TKbMViHXA2bm%2BOJqoEY7QlNpVEfayDKoD3eqzhBSqNpqo4R7dcyJdjDX%2BHuW7Ouq%2BhshqCiG9yTfPDV%2FgmUWCvpLbCmSMzqsC3%2BSvWcInvEOUyZEeL5mtzxUQEfI9%2FYw3%2F8X2mZsuOVUVxEUDGP%2FwQeZ%2BSM7pSocrL8cNciDXwowQeJaWhQjK6RfwIFzU%2Fe5UfIxpiI0M%2B4npTmduWcZmfIJ%2FU1yshIxtxiTI46tZuZAxhTipDQ659yPACLksG5712IMMLuUwZHHriMuxVYBlXGBD50pHKXgWWEbNJh72MtKgKnMX%2Fxjq8KmZxrALXVNb%2BIV9TBQyAFS4mrFqFO4oNxMDHIUGV%2Bo0sGwDdHxvoT5ChcmNcL2ITl2INF9hAlKlGLz6VjXwSgxoXE%2BI7JRZvu7GJwO8Y63jRaMJRpGcCnlNJXqkgg6aGX3ij7K9Vuig2NQwYkvcNe4GhlMkzZCrOfSKbgQxDhpjGhvH7RNQfWzKLPUMi%2BeUTVEd%2Fwgc4fggtifc0Alkjm6SmeEd%2FivWgikHmGCC3bQoSqKCBsZamtKbXwuaoL4rdqQxUATYcmusQJjNHuikW227kWEvBS7YXH22qjgOQvwX24iDS%2BI%2FHe%2FQqasBtk4KveNoCXcDB%2B6NIC2IMsEc3%2FBl4o%2B7RIFZN5eETAw0T0%2FA74YOEAVW4aDU81pKx%2Bo%2BNpvp7BQ38UPdijKgXKQpxWfdZjCiOJhpluFXp6TFkolg5FXlgooFpafAiWFiNLsaQopMSvWAzwpweG5g7je9y5sgtztw5EUoPbRF%2FUOyhCw2LbMw1PrJnx9qV6gEr1%2B48MAf%2FDfZvJ66RJ0T3GHJi21KlZ%2Fn2U%2FhK1crNQ%2FoTZEKs5dia%2BcrEos2n5GpCFO0zdrv589sWqrZZtPu83FOREKaspO5xeo1KyPz156S2yDZxSldrn16tbHhUSFNaQAZ0Dezm5zcoS%2BZvPw8zRulkEzQJuIPbP1%2FZs%2BjYg85RVIZHiXScX6FKY%2FN5tyqADDJyr847tECVysITcdxUS5WTgf18iyqHvRbeLSgj9ZYqj%2BepHcjo8Lkql5dTVZfR4RtVPp%2Bn5GXIq8A6xPMGUFF9HR5r6Gb27i%2BVK94mV6BGHPOuskY%2BXhVA1wSZp1wyjtyQt%2FTxkcotncgJOTvnSP2o2mDxxp2Hjxxn5uNHDu%2FcuFi1wXdu3Ly%2F3W5%2BijKycs9xfpTjO5YoI6%2BSC3y2qXH7mQPoD6yhd6M5tA0iF0Ro1Kch1aowH%2Fbqz8DRRpiE%2FJwSmykUSEuj4Y4PIwrxsKjxVwWZIeUcwBx1CjIv1cY0uKZZIT4mB2SSP%2ByarQC%2FD4NjVPbbNuWzAiMePB3pogA%2FdnpkcIeu59MK0JoSeXcL6kNkjG866EKe5jg6%2FSpoDi%2Fhe8E6qMK0w8xQAh3Ngg9G8snC1O%2F%2Ft%2FjICKWnn0DPoc%2FlKaWnh0kF9092FrMln4wECRL4OBC1Uf55U2mpEUgdWh2vGI4xSP7gMKV3j%2FESTYfm3XwNPkUv4MTGQGG3WfbVZ%2BFe9hoMI6UfWr3%2BBHG7RsA7NMXEFJS3Rtk8msRZdLCbigRTuH2mrXpjZMF9BBkUm2OKuxUgFgKOsG%2BeDQQ2TUurw%2BUZFvLcKvU4y3Z9xRj4RABZtk6gC9Rw8uDWdeoeq7buO8lmDA39eIFEDipEwNFbnOUE5AjSBQU9qTawdEIy0CpVj%2BAa1R6zY6BY9Qo5IhO5U%2BGTiWeVBnKF70yHT0a6CsgQ0NGfMNDH6yR1CKgAvUsXalc6oiy1ibQM8kMx7xaQgfHyXA6hRy5lCJSJVrm7%2BjJw9Y2x%2B6%2F3morIIC%2FHpTDVo2R0Een%2FNGTtPb2gi1AWHQeJ0N%2FuZkVDKDnjgYxqC4lGeWTBbJEKFwvJcxLC%2FmRFCjTjcmRyBTYT5XyypCtom0TxR4XYDrksWYEHuV1JHC878%2BjJx3vzo7te86gUUq2Vibdg7bdq3aZdd9i0blUZP90PTj%2Fl0Z5gI5VCM%2FyUPI3OJq%2F9xBY1Jf94oytjCLkGiPUO6rlnlY5XSBjzo5fmlH2ssB%2Boi98q22uVekVpSVGlaLVfouJIIV%2BJWJWlloOZwcrCxWSoUXputGuHuLKEQBSGDwaDQmAxrVFtyuDaswB2UIs4a395ueKKCcyd7g4wSX%2B%2BxJ8cWequDpMVA8nVjsiGiIEsGzReWiUrhrr0SmQOtkQMZZUtxaIvdG4xWGJbMmizmW0eo1W2aTPECjsEw3n2qDi8Cpk9ajDezr66B4NfNoqyL2CGwrf0kPRfPpRv7ZjCKe9UMEngjdRilo23UYd5hHeJmEkGVIwgwyrW74iYL%2FEi9VhBVF5RHdbgKs%2FLBqswmWdtWElQnlEc1mKEH9MN63EHPyMGS%2FKfhIjFsnzmn6hYLM2myndKNFif2yvbymbxLWyUwlfHHgy%2BjfMp5eOHpOQtHo%2FH4%2FEY7x8MZ7AAyatDDgAAAABJRU5ErkJggg%3D%3D)](https://hub.gke2.mybinder.org/user/julienguegan-notebooks_blog-z8qd9bd5/notebooks/dynamique_population.ipynb)
