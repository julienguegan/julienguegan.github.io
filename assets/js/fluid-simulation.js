const FIELD_TYPE = { VELOCITY_X: 0, VELOCITY_Y: 1, DENSITY: 2 };

// Palette de couleurs pré-calculée : noir → rouge → orange → blanc
const COLOR_PALETTE = new Uint8ClampedArray(256 * 3);
for (let i = 0; i < 256; i++) {
    const t = i / 255;
    COLOR_PALETTE[i*3]     = Math.min(255, t * 550);
    COLOR_PALETTE[i*3 + 1] = Math.min(255, Math.pow(t, 2.2) * 450);
    COLOR_PALETTE[i*3 + 2] = Math.pow(t, 4) * 255;
}

// ─────────────────────────────────────────────────────────────────────────────
// FluidSimulator — résout les équations de Navier-Stokes (méthode d'Euler)
//   Chaque frame exécute 3 étapes : Viscosité → Projection → Advection
// ─────────────────────────────────────────────────────────────────────────────
class FluidSimulator {
    constructor(resX, resY, h) {
        this.nx = resX + 2;  // +2 pour les cellules fantômes aux bords
        this.ny = resY + 2;
        this.h  = h;         // taille d'une cellule en unités de simulation
        this.totalCells = this.nx * this.ny;

        this.velocityX = new Float32Array(this.totalCells);
        this.velocityY = new Float32Array(this.totalCells);
        this.density   = new Float32Array(this.totalCells);

        // Buffers temporaires réutilisés à chaque étape
        this.bufferX = new Float32Array(this.totalCells);
        this.bufferY = new Float32Array(this.totalCells);
        this.bufferD = new Float32Array(this.totalCells);
    }

    getIdx(i, j) { return i * this.ny + j; }

    // ── Étape 1 : Viscosité ───────────────────────────────────────────────────
    // Diffuse la vitesse pour simuler la résistance interne du fluide.
    // Résolution par Gauss-Seidel : α = h²/(ν·dt), convergence en ~10 itérations.
    applyViscosity(dt, viscosity) {
        if (viscosity <= 0) return;
        const alpha = (this.h * this.h) / (viscosity * dt);

        // Le buffer doit partir des vitesses actuelles comme valeur initiale
        this.bufferX.set(this.velocityX);
        this.bufferY.set(this.velocityY);

        for (let iter = 0; iter < 10; iter++) {
            for (let i = 1; i < this.nx - 1; i++) {
                for (let j = 1; j < this.ny - 1; j++) {
                    const c = this.getIdx(i, j);
                    const l = this.getIdx(i-1, j), r = this.getIdx(i+1, j);
                    const b = this.getIdx(i, j-1), t = this.getIdx(i, j+1);

                    this.bufferX[c] = (this.velocityX[c] * alpha + this.bufferX[l] + this.bufferX[r] + this.bufferX[b] + this.bufferX[t]) / (4 + alpha);
                    this.bufferY[c] = (this.velocityY[c] * alpha + this.bufferY[l] + this.bufferY[r] + this.bufferY[b] + this.bufferY[t]) / (4 + alpha);
                }
            }
        }
        this.velocityX.set(this.bufferX);
        this.velocityY.set(this.bufferY);
    }

    // ── Étape 2 : Projection (incompressibilité) ──────────────────────────────
    // Corrige le champ de vitesse pour garantir ∇·u = 0 (fluide incompressible).
    // Pour chaque cellule, on annule la divergence en redistribuant la pression
    // vers les voisins, répété en boucle jusqu'à convergence.
    projectIncompressibility(iterations, overRelaxation) {
        for (let iter = 0; iter < iterations; iter++) {
            for (let i = 1; i < this.nx - 1; i++) {
                for (let j = 1; j < this.ny - 1; j++) {
                    const c = this.getIdx(i, j);
                    const r = this.getIdx(i+1, j);
                    const t = this.getIdx(i, j+1);

                    const divergence = this.velocityX[r] - this.velocityX[c]
                                     + this.velocityY[t] - this.velocityY[c];
                    const pressure = (-divergence / 4) * overRelaxation;

                    this.velocityX[c] -= pressure;
                    this.velocityX[r] += pressure;
                    this.velocityY[c] -= pressure;
                    this.velocityY[t] += pressure;
                }
            }
        }
    }

    // Interpolation bilinéaire — échantillonne un champ à la position (x, y)
    interpolate(x, y, fieldType) {
        const invH = 1.0 / this.h;
        // Décalage de demi-cellule selon le type de champ (grille MAC décalée)
        const dx = (fieldType === FIELD_TYPE.VELOCITY_Y || fieldType === FIELD_TYPE.DENSITY) ? 0.5 * this.h : 0;
        const dy = (fieldType === FIELD_TYPE.VELOCITY_X || fieldType === FIELD_TYPE.DENSITY) ? 0.5 * this.h : 0;

        x = Math.max(this.h, Math.min(x, (this.nx - 1) * this.h));
        y = Math.max(this.h, Math.min(y, (this.ny - 1) * this.h));

        const i0 = Math.floor((x - dx) * invH);
        const j0 = Math.floor((y - dy) * invH);
        const tx = ((x - dx) - i0 * this.h) * invH;
        const ty = ((y - dy) - j0 * this.h) * invH;

        const field = fieldType === FIELD_TYPE.VELOCITY_X ? this.velocityX
                    : fieldType === FIELD_TYPE.VELOCITY_Y ? this.velocityY
                    : this.density;

        return (1-tx)*(1-ty) * field[this.getIdx(i0,   j0  )]
             +    tx *(1-ty) * field[this.getIdx(i0+1, j0  )]
             +    tx *   ty  * field[this.getIdx(i0+1, j0+1)]
             + (1-tx)*   ty  * field[this.getIdx(i0,   j0+1)];
    }

    // ── Étape 3 : Advection ───────────────────────────────────────────────────
    // Transporte densité et vitesse le long du champ de vitesse (semi-Lagrangien).
    // Pour chaque cellule : on remonte en arrière dans le temps pour trouver
    // d'où venait le fluide, et on copie la valeur qui s'y trouvait.
    applyAdvection(dt, dissipation) {
        for (let i = 1; i < this.nx - 1; i++) {
            for (let j = 1; j < this.ny - 1; j++) {
                const idx = this.getIdx(i, j);

                // Vitesse au centre de la cellule (moyenne des faces adjacentes)
                const u = (this.velocityX[idx] + this.velocityX[this.getIdx(i+1, j)]) * 0.5;
                const v = (this.velocityY[idx] + this.velocityY[this.getIdx(i, j+1)]) * 0.5;

                // Position d'origine en remontant le temps
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────
const canvas = document.getElementById("simCanvas");
const ctx    = canvas.getContext("2d", { alpha: false });
// Utiliser la taille affichée réelle, pas la taille de la fenêtre
// (le canvas est dans un conteneur de 600px, pas plein écran)
const canvasRect = canvas.getBoundingClientRect();
canvas.width  = Math.round(canvasRect.width);
canvas.height = Math.round(canvasRect.height);

const SIM_RES   = 100;
const h         = 1.0 / SIM_RES;
const simNx     = Math.floor((canvas.width / canvas.height) * SIM_RES);
const simulator = new FluidSimulator(simNx, SIM_RES, h);

const config = { viscosity: 0, dissipation: 0.99, brushRadius: 0.05 };

// Sliders de contrôle
const setupSlider = (id, prop, labelId) => {
    const el = document.getElementById(id);
    el.oninput = () => {
        config[prop] = parseFloat(el.value);
        document.getElementById(labelId).innerText = el.value;
    };
};
setupSlider("input-visc",   "viscosity",   "label-visc");
setupSlider("input-radius", "brushRadius", "label-radius");
setupSlider("input-diss",   "dissipation", "label-diss");

// ─────────────────────────────────────────────────────────────────────────────
// Interaction souris
// ─────────────────────────────────────────────────────────────────────────────
let lastMouse = { x: 0, y: 0 };
canvas.addEventListener("mousemove", e => {
    // getBoundingClientRect() compense le décalage si le canvas n'est pas en (0,0)
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left)  / canvas.height;
    const y = (rect.bottom - e.clientY) / canvas.height;

    const vX = (x - lastMouse.x) * 60;
    const vY = (y - lastMouse.y) * 60;
    const r2 = config.brushRadius * config.brushRadius;

    for (let i = 1; i < simulator.nx - 1; i++) {
        for (let j = 1; j < simulator.ny - 1; j++) {
            const dx = (i + 0.5) * simulator.h - x;
            const dy = (j + 0.5) * simulator.h - y;
            if (dx*dx + dy*dy < r2) {
                const idx = simulator.getIdx(i, j);
                simulator.density[idx]   = 1.0;
                simulator.velocityX[idx] += vX;
                simulator.velocityY[idx] += vY;
            }
        }
    }
    lastMouse = { x, y };
});

// ─────────────────────────────────────────────────────────────────────────────
// Boucle de rendu
// ─────────────────────────────────────────────────────────────────────────────
function renderFrame() {
    simulator.applyViscosity(1/60, config.viscosity);
    simulator.projectIncompressibility(40, 1.9);
    simulator.applyAdvection(1/60, config.dissipation);

    const img   = ctx.createImageData(canvas.width, canvas.height);
    const scale = canvas.height;
    const size  = Math.ceil(simulator.h * scale);

    for (let i = 0; i < simulator.nx; i++) {
        for (let j = 0; j < simulator.ny; j++) {
            const d = simulator.density[simulator.getIdx(i, j)];
            if (d < 0.005) continue;

            const cIdx = Math.floor(d * 255) * 3;
            const px   = Math.floor(i * simulator.h * scale);
            const py   = Math.floor(canvas.height - (j + 1) * simulator.h * scale);

            for (let dy = py; dy < py + size; dy++) {
                if (dy < 0 || dy >= canvas.height) continue;
                for (let dx = px; dx < px + size; dx++) {
                    if (dx < 0 || dx >= canvas.width) continue;
                    const pIdx = (dy * canvas.width + dx) * 4;
                    img.data[pIdx]   = COLOR_PALETTE[cIdx];
                    img.data[pIdx+1] = COLOR_PALETTE[cIdx + 1];
                    img.data[pIdx+2] = COLOR_PALETTE[cIdx + 2];
                    img.data[pIdx+3] = 255;
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0);
    requestAnimationFrame(renderFrame);
}

renderFrame();
