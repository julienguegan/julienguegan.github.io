---
title: "Fluid Simulation in 200 Lines of Code"
lang: en
classes: wide
header:
  teaser: /assets/images/teaser_fluid_simulation.png
read_time: true
published: true
---

Fluid mechanics is a branch of physics studied for centuries by many scientists — Archimedes, Newton, Bernoulli, Lagrange and others. It helps us understand how fluids (gases and liquids) behave, and allows engineers to build turbines, predict weather, and improve aircraft aerodynamics. In everyday life it is hard to picture the trajectory of a fluid beyond watching water flow from a tap. Look a little closer and you realise those trajectories can be surprisingly chaotic, full of turbulences and vortices. In this post I simulate, using physics equations, how a fluid reacts to mouse interaction — in a small JavaScript program. I was heavily inspired by the work of Matthias Müller and his [video](https://www.youtube.com/watch?v=iKAVRgIrUOU) as well as his document [17-fluidSim.pdf](https://matthias-research.github.io/pages/tenMinutePhysics/17-fluidSim.pdf), which demystifies the magic behind Computational Fluid Dynamics (CFD).

<p align="center">
   <img src="/assets/images/fluid_simulation_header.png" width="90%"/>
</p>

## The Navier-Stokes Equations

The [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) govern fluid behaviour and describe it through nonlinear partial differential terms. Many variants exist depending on the case studied; here I use two principal equations: conservation of momentum and the incompressibility condition. The first is in fact Newton's second law $F = ma$ applied to a fluid. Denoting $u$ the velocity field, $p$ the pressure, $\rho$ the density and $\nu$ the viscosity, it can be written as:

$$
\underbrace{\frac{\partial \mathbf{u}}{\partial t}}_{\text{Acceleration}} +
\underbrace{(\mathbf{u} \cdot \nabla) \mathbf{u}}_{\text{Advection}} =
\underbrace{-\frac{1}{\rho} \nabla p}_{\text{Pressure Gradient}} +
\underbrace{\nu \nabla^2 \mathbf{u}}_{\text{Viscosity}} +
\underbrace{\mathbf{f}}_{\text{Forces}}
$$

- Acceleration $\frac{\partial u}{\partial t}$: the local acceleration. It describes how velocity at a fixed grid point changes over time. If you leave the fluid alone and it gradually comes to rest, this term captures that evolution.

- Advection $(u \cdot \nabla) u$: the term responsible for chaotic, nonlinear behaviour. It describes how the velocity field transports itself through space — this is what creates vortices.

- Pressure gradient $-\frac{1}{\rho} \nabla p$: ensures that fluid is pushed from high-pressure to low-pressure zones. It is the engine that "repairs" the velocity field to keep it incompressible.

- Viscosity $\nu \nabla^2 u$: represents internal friction, or diffusion of momentum. A highly viscous fluid resists the sliding of its internal layers (like honey).

- External forces $f$: typically gravity. In our demo, this will be the user's mouse interaction.

The second equation models the incompressibility condition. For smoke or water, we often assume the fluid cannot be compressed (its volume stays constant):

$$
\nabla \cdot \mathbf{u} = 0
$$

This zero-divergence equation means mathematically that everything entering a grid cell must leave it. It is the constraint that forces the fluid to curve and swirl rather than simply pile up against a wall.

## Modelling the Motion

When simulating fluids, two viewpoints exist, each with its own way of interpreting motion:

1. <u>Lagrangian approach</u>: the fluid is treated as a collection of particles; individual points are tracked as they collide over time. Intuitive, like simulating a pile of marbles.

    | Strengths | Weaknesses |
    | :-------- | :--------- |
    | Complex boundaries and splashes are handled naturally and efficiently. Since mass is carried by the particles, it is conserved by default. | Enforcing incompressibility requires expensive neighbour searches, as you need to know which particles are close to each other at every instant. |

   <div style="display: flex; justify-content: center; gap: 10%; margin-top: 1rem; margin-bottom: 2rem;">
   <img src="/assets/images/fluid_simulation_lagrange.png" style="width: 70%; height: auto;" alt="Lagrangian approach">
   </div>

2. <u>Eulerian approach</u>: instead of following particles, we fix our gaze on specific points in space. The domain is divided into a grid (like the pixels of an image) and we observe how properties (velocity, pressure, density) evolve in each cell.

   | Strengths | Weaknesses |
   | :-------- | :--------- |
   | Computations are more numerically stable. Calculating pressure gradients and diffusion is much simpler on a fixed structure, making it easy to simulate volume-preserving fluids. | "Numerical dissipation" (the fluid tends to become slightly too viscous or blurry over time) can occur if grid resolution is insufficient. |

   <div style="display: flex; justify-content: center; gap: 10%;margin-top: 1rem; margin-bottom: 2rem;">
   <img src="/assets/images/fluid_simulation_euler.png" style="width: 70%; height: auto;" alt="Eulerian approach">
   </div>

For this simulation I use the Eulerian approach: it is generally easier to handle for incompressibility and produces better results for smoke. It can produce the swirling vortex motion characteristic of gases while maintaining a smooth, continuous visual — where a particle approach might look "grainy".

## Numerical Solving

In an Eulerian simulation, how data is organised on the grid is crucial. A naive approach would store all variables at the centre of each cell (a collocated grid), but this makes it impossible to correctly compute gradients from one cell to the next. Instead we use a **staggered grid** (the [*Marker-and-Cell grid*](https://en.wikipedia.org/wiki/Marker-and-cell_method)): pressure $p$ sits at the cell centre $(i, j)$, horizontal velocity $u$ is shifted to the vertical faces $(i \pm \frac{1}{2}, j)$, and vertical velocity $v$ to the horizontal faces $(i, j \pm \frac{1}{2})$. This layout makes pressure computation easier and ensures numerical stability.

<p align="center">
  <img src="/assets/images/fluid_simulation_grid.png" alt="Staggered Grid Diagram" width="25%">
  <br>
  <i>The Staggered Grid Layout</i>
</p>

For the solving itself we use [**operator splitting**](https://en.wikipedia.org/wiki/Splitting_method): decompose the equation into sub-parts and solve each one sequentially. At each time step, the loop executes three steps in order:

<div style="display:flex;align-items:center;justify-content:center;gap:0.75rem;margin:2rem 0;flex-wrap:wrap;">
  <div style="border:2px solid #ff9d00;border-radius:8px;padding:0.9rem 1.2rem;text-align:center;min-width:140px;">
    <div style="color:#ff9d00;font-size:0.65rem;font-weight:bold;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">Step 1</div>
    <div style="font-weight:bold;margin-bottom:0.4rem;">Viscosity</div>
    <div style="font-size:0.85rem;opacity:0.75;">$\partial \mathbf{u}/\partial t = \nu \nabla^2 \mathbf{u}$</div>
    <div style="font-size:0.72rem;opacity:0.5;margin-top:0.35rem;">spreads velocity</div>
  </div>
  <div style="color:#ff9d00;font-size:1.5rem;font-weight:bold;">→</div>
  <div style="border:2px solid #ff9d00;border-radius:8px;padding:0.9rem 1.2rem;text-align:center;min-width:140px;">
    <div style="color:#ff9d00;font-size:0.65rem;font-weight:bold;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">Step 2</div>
    <div style="font-weight:bold;margin-bottom:0.4rem;">Projection</div>
    <div style="font-size:0.85rem;opacity:0.75;">$\nabla \cdot \mathbf{u} = 0$</div>
    <div style="font-size:0.72rem;opacity:0.5;margin-top:0.35rem;">corrects pressure</div>
  </div>
  <div style="color:#ff9d00;font-size:1.5rem;font-weight:bold;">→</div>
  <div style="border:2px solid #ff9d00;border-radius:8px;padding:0.9rem 1.2rem;text-align:center;min-width:140px;">
    <div style="color:#ff9d00;font-size:0.65rem;font-weight:bold;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">Step 3</div>
    <div style="font-weight:bold;margin-bottom:0.4rem;">Advection</div>
    <div style="font-size:0.85rem;opacity:0.75;">$\mathbf{x}' = \mathbf{x} - \mathbf{u} \cdot \Delta t$</div>
    <div style="font-size:0.72rem;opacity:0.5;margin-top:0.35rem;">transports the fluid</div>
  </div>
</div>

This technique is simpler than solving everything simultaneously, at the cost of a small approximation that is perfectly acceptable for real-time simulation.

### Viscosity

The first term is viscosity — the internal friction: a highly viscous fluid flows slowly, like honey. We solve the diffusion equation:

$$ \frac{\partial \mathbf{u}}{\partial t} = \nu \nabla^2 \mathbf{u} $$

The Laplacian $\nabla^2 \mathbf{u}$ measures how much a cell's velocity differs from its neighbours — if it is faster, viscosity slows it down toward the average. On a discrete grid, we approximate it using [finite differences](https://en.wikipedia.org/wiki/Finite_difference_method) — each cell is compared to its four neighbours.

<details style="margin:0.5rem 0 1rem;padding:0.6rem 1rem;border-left:3px solid #ff9d00;background:rgba(255,157,0,0.04);">
<summary style="cursor:pointer;font-weight:bold;color:#ff9d00;">Laplacian discretization formula</summary>

$$\nabla^2 u_{i,j} \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}$$

</details>

The next question is: how do we advance in time? The simplest reflex would be to compute the derivative at instant $n$ and use it directly to predict $n+1$. This is the **explicit scheme**, but it is unstable: if the time step or viscosity is too large, errors amplify at each iteration and the simulation blows up. We therefore use an **implicit scheme**, where the Laplacian is evaluated at instant $n+1$ (the unknown). It is like correcting a car's trajectory by anticipating the corner rather than looking at where you were before — you can never over-correct, regardless of the time step size.

The trade-off is that we can no longer compute directly: $u^{n+1}$ appears on both sides of the equation. Setting $\alpha = h^2 / (\nu \cdot dt)$, we obtain a linear system that [Gauss-Seidel](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) solves by successive iterations.

<details style="margin:0.5rem 0 1rem;padding:0.6rem 1rem;border-left:3px solid #ff9d00;background:rgba(255,157,0,0.04);">
<summary style="cursor:pointer;font-weight:bold;color:#ff9d00;">Equation to solve for each cell</summary>

$$u^{n+1}_{i,j} = \frac{u^n_{i,j} \cdot \alpha \;+\; u^{n+1}_{i+1,j} + u^{n+1}_{i-1,j} + u^{n+1}_{i,j+1} + u^{n+1}_{i,j-1}}{4 + \alpha}$$

</details>

Inverting this system exactly would be too expensive for a grid of $10\,000$ cells. Instead we use Gauss-Seidel: loop over all cells and apply this formula, immediately reusing neighbours already updated in the current iteration. It is like a rumour spreading through a crowd — after a few passes, everyone has converged. The parameter $\alpha$ controls diffusion intensity: large $\alpha$ (low viscosity) means little change; small $\alpha$ (high viscosity) means velocities strongly homogenise between neighbours.

```javascript
applyViscosity(dt, viscosity) {
    const alpha = (this.h * this.h) / (viscosity * dt);
    // initialise solver from current velocities
    this.bufferX.set(this.velocityX);
    this.bufferY.set(this.velocityY);
    // iterate until convergence (10 is enough)
    for (let iter = 0; iter < 10; iter++) {
        // for each interior cell
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

**Key takeaway —** implicit scheme: stable regardless of viscosity. Gauss-Seidel: 10 iterations are enough, without ever inverting a matrix.
{: .notice--info}

### Incompressibility

An incompressible fluid cannot be compressed: everything entering a cell must leave it ($\nabla \cdot \mathbf{u} = 0$). On the MAC grid, the divergence of a cell is simply the sum of outgoing fluxes on its four faces:

$$\text{div} = (u_x[i{+}1,j] - u_x[i,j]) + (u_y[i,j{+}1] - u_y[i,j])$$

If it is non-zero, the velocities need to be corrected.

The approach is local and direct: compute how much to adjust the four faces to cancel the divergence, distributing the correction symmetrically. Each face receives $p = -\text{div}/4$, which brings the divergence exactly to zero for that cell. But correcting one cell modifies the faces shared with its neighbours, which in turn become slightly divergent. So we cannot fix everything in a single pass — we loop and each iteration reduces the residual error until it is negligible. The `overRelaxation = 1.9` factor speeds up convergence: instead of applying exactly the computed correction, we apply 1.9× of it. This slight over-correction halves the number of iterations needed for the same precision. Beyond 2.0 the system would diverge — 1.9 is empirically the best trade-off for this type of problem.

```javascript
projectIncompressibility(iterations, overRelaxation) {
    for (let iter = 0; iter < iterations; iter++) {
        for (let i = 1; i < this.nx - 1; i++) {
            for (let j = 1; j < this.ny - 1; j++) {
                const c = this.getIdx(i,j), r = this.getIdx(i+1,j), t = this.getIdx(i,j+1);

                const divergence = this.velocityX[r] - this.velocityX[c] + this.velocityY[t] - this.velocityY[c];
                const pressure = (-divergence / 4) * overRelaxation;

                this.velocityX[c] -= pressure;   this.velocityX[r] += pressure;
                this.velocityY[c] -= pressure;   this.velocityY[t] += pressure;
            }
        }
    }
}
```

**Key takeaway —** divergence is corrected cell by cell through iteration. Over-relaxation (×1.9) halves the number of iterations needed.
{: .notice--info}

### Advection

Advection transports properties (density, velocity) along the flow. This term is the trickiest: it is nonlinear (velocity transports itself), and a direct finite-difference discretisation would be unstable — errors amplify exponentially over time steps. We instead use a **Semi-Lagrangian** scheme, which exploits a fundamental property of the advection equation: every quantity is conserved along the fluid's trajectories. Rather than pushing values forward, we trace backward in time to find where the fluid arriving at each cell came from.

$$
\mathbf{x}_{prev} = \mathbf{x} - \mathbf{u}(\mathbf{x}) \cdot \Delta t
\qquad\Rightarrow\qquad
q_{new}(\mathbf{x}) = q_{old}(\mathbf{x}_{prev})
$$

<p align="center">
  <img src="/assets/images/advection_diagram.png" alt="Semi-Lagrangian Advection Scheme" width="50%">
  <br>
  <i>Semi-Lagrangian Advection Scheme</i>
</p>

This scheme is unconditionally stable: we only ever read existing values, never extrapolate. Its only drawback is slight **numerical dissipation** — the [bilinear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation) needed to read a value between cells introduces a gentle blur, as if the fluid were slightly more viscous than it really is. That is an acceptable trade-off for stability. Note also that computations use a temporary buffer rather than writing directly into the velocity arrays: overwriting values on the fly would have cells already updated in the current step polluting reads for subsequent cells — we would no longer be reading $q^n$ but a mixture of $q^n$ and $q^{n+1}$.

```javascript
applyAdvection(dt, dissipation) {
    for (let i = 1; i < this.nx - 1; i++) {
        for (let j = 1; j < this.ny - 1; j++) {
            const idx = this.getIdx(i, j);

            // velocity at cell centre (average of adjacent MAC faces)
            const u = (this.velocityX[idx] + this.velocityX[this.getIdx(i+1,j)]) * 0.5;
            const v = (this.velocityY[idx] + this.velocityY[this.getIdx(i,j+1)]) * 0.5;

            // trace backward in time
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

**Key takeaway —** semi-Lagrangian scheme: unconditionally stable because we always read past values. The slight blur from interpolation is the price to pay.
{: .notice--info}

## Interactive Demo

These three steps are enough to produce a surprisingly realistic behaviour. Try varying viscosity or dissipation to observe their effect on the fluid:

Click and hold to add smoke:

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
            <label>Viscosity <span id="label-visc" class="valeur-bulle">0.00</span></label>
            <input type="range" id="input-visc" min="0" max="0.5" step="0.01" value="0">
        </div>
        <div class="param-group">
            <label>Radius <span id="label-radius" class="valeur-bulle">0.05</span></label>
            <input type="range" id="input-radius" min="0.01" max="0.15" step="0.01" value="0.05">
        </div>
        <div class="param-group">
            <label>Dissipation <span id="label-diss" class="valeur-bulle">0.99</span></label>
            <input type="range" id="input-diss" min="0.90" max="1.0" step="0.001" value="0.99">
        </div>
        <button class="btn-reset" onclick="location.reload()">Reset</button>
    </div>
    <canvas id="simCanvas"></canvas>
</div>

<script src="/assets/js/fluid-simulation.js"></script>
