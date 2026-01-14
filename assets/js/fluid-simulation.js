/** @enum {number} Types de champs pour l'interpolation */
const FIELD_TYPE = { VELOCITY_X: 0, VELOCITY_Y: 1, DENSITY: 2 };

/** @type {Uint8ClampedArray} Table de correspondance pour le rendu des couleurs */
const COLOR_PALETTE = new Uint8ClampedArray(256 * 3);
for (let i = 0; i < 256; i++) {
    let t = i / 255;
    COLOR_PALETTE[i*3]     = Math.min(255, t * 550);           // Canal Rouge
    COLOR_PALETTE[i*3 + 1] = Math.min(255, Math.pow(t, 2.2) * 450); // Canal Vert
    COLOR_PALETTE[i*3 + 2] = Math.pow(t, 4) * 255;             // Canal Bleu
}

/**
 * Classe principale gérant la résolution des équations de Navier-Stokes (Euler)
 */
class FluidSimulator {
    /**
     * @param {number} resX - Résolution horizontale.
     * @param {number} resY - Résolution verticale.
     * @param {number} h - Taille d'une cellule.
     */
    constructor(resX, resY, h) {
        this.nx = resX + 2; 
        this.ny = resY + 2; 
        this.h = h;
        this.totalCells = this.nx * this.ny;

        // Champs de données principaux
        this.velocityX = new Float32Array(this.totalCells);
        this.velocityY = new Float32Array(this.totalCells);
        this.density   = new Float32Array(this.totalCells);

        // Buffers temporaires pour les calculs d'advection
        this.bufferX   = new Float32Array(this.totalCells);
        this.bufferY   = new Float32Array(this.totalCells);
        this.bufferD   = new Float32Array(this.totalCells);
        
        // Masque d'obstacles (1.0 = fluide, 0.0 = solide)
        this.obstacleMask = new Float32Array(this.totalCells).fill(1.0);
    }

    /** Calcule l'index à plat pour une grille 2D */
    getIdx(i, j) { return i * this.ny + j; }

    /** * Résout la diffusion pour simuler la viscosité.
     * Utilise la méthode itérative de Gauss-Seidel.
     */
    applyViscosity(dt, viscosity) {
        if (viscosity <= 0) return;
        const alpha = (this.h * this.h) / (viscosity * dt);
        
        for (let iter = 0; iter < 10; iter++) {
            for (let i = 1; i < this.nx - 1; i++) {
                for (let j = 1; j < this.ny - 1; j++) {
                    const c = this.getIdx(i, j);
                    this.bufferX[c] = (this.velocityX[c] * alpha + this.bufferX[this.getIdx(i-1, j)] + this.bufferX[this.getIdx(i+1, j)] + this.bufferX[this.getIdx(i, j-1)] + this.bufferX[this.getIdx(i, j+1)]) / (4 + alpha);
                    this.bufferY[c] = (this.velocityY[c] * alpha + this.bufferY[this.getIdx(i-1, j)] + this.bufferY[this.getIdx(i+1, j)] + this.bufferY[this.getIdx(i, j-1)] + this.bufferY[this.getIdx(i, j+1)]) / (4 + alpha);
                }
            }
        }
        this.velocityX.set(this.bufferX);
        this.velocityY.set(this.bufferY);
    }

    /** * Étape de projection (Incompressibilité).
     * Garantit que la divergence du champ de vitesse est nulle.
     */
    projectIncompressibility(iterations, overRelaxation) {
        for (let iter = 0; iter < iterations; iter++) {
            for (let i = 1; i < this.nx - 1; i++) {
                for (let j = 1; j < this.ny - 1; j++) {
                    const c = this.getIdx(i, j);
                    const l = this.getIdx(i-1, j), r = this.getIdx(i+1, j), b = this.getIdx(i, j-1), t = this.getIdx(i, j+1);
                    
                    const divergence = this.velocityX[r] - this.velocityX[c] + this.velocityY[t] - this.velocityY[c];
                    const pressure = (-divergence / 4) * overRelaxation;
                    
                    this.velocityX[c] -= pressure; 
                    this.velocityX[r] += pressure;
                    this.velocityY[c] -= pressure; 
                    this.velocityY[t] += pressure;
                }
            }
        }
    }

    /** Interpolation bilinéaire pour l'échantillonnage des champs */
    interpolate(x, y, fieldType) {
        const invH = 1.0 / this.h;
        let dx = (fieldType === FIELD_TYPE.VELOCITY_Y || fieldType === FIELD_TYPE.DENSITY) ? 0.5 * this.h : 0;
        let dy = (fieldType === FIELD_TYPE.VELOCITY_X || fieldType === FIELD_TYPE.DENSITY) ? 0.5 * this.h : 0;

        x = Math.max(this.h, Math.min(x, (this.nx-1)*this.h));
        y = Math.max(this.h, Math.min(y, (this.ny-1)*this.h));

        const i0 = Math.floor((x-dx)*invH), j0 = Math.floor((y-dy)*invH);
        const tx = ((x-dx) - i0*this.h)*invH, ty = ((y-dy) - j0*this.h)*invH;
        const field = fieldType === FIELD_TYPE.VELOCITY_X ? this.velocityX : (fieldType === FIELD_TYPE.VELOCITY_Y ? this.velocityY : this.density);

        return (1-tx)*(1-ty)*field[this.getIdx(i0, j0)] + tx*(1-ty)*field[this.getIdx(i0+1, j0)] + 
               tx*ty*field[this.getIdx(i0+1, j0+1)] + (1-tx)*ty*field[this.getIdx(i0, j0+1)];
    }

    /**
     * Étape d'Advection : Transporte les propriétés (densité, vitesse) 
     * le long des lignes de courant du fluide.
     * @param {number} dt - Pas de temps (time step).
     * @param {number} dissipation - Facteur d'atténuation (0.9 à 1.0).
     */
    applyAdvection(dt, dissipation) {
        for (let i = 1; i < this.nx - 1; i++) {
            for (let j = 1; j < this.ny - 1; j++) {
                const idx = this.getIdx(i, j);
                
                // 1. On récupère la vitesse locale au centre de la cellule
                let u = (this.velocityX[idx] + this.velocityX[this.getIdx(i + 1, j)]) * 0.5;
                let v = (this.velocityY[idx] + this.velocityY[this.getIdx(i, j + 1)]) * 0.5;
                
                // 2. On recule dans le temps pour trouver la position d'origine
                let prevX = (i + 0.5) * this.h - dt * u;
                let prevY = (j + 0.5) * this.h - dt * v;

                // 3. On échantillonne les anciennes valeurs et on les applique au présent
                this.bufferD[idx] = this.interpolate(prevX, prevY, FIELD_TYPE.DENSITY) * dissipation;
                this.bufferX[idx] = this.interpolate(i * this.h - dt * this.velocityX[idx], j * this.h + 0.5 * this.h - dt * v, FIELD_TYPE.VELOCITY_X);
                this.bufferY[idx] = this.interpolate(i * this.h + 0.5 * this.h - dt * u, j * this.h - dt * this.velocityY[idx], FIELD_TYPE.VELOCITY_Y);
            }
        }
        // Mise à jour des champs réels avec les valeurs calculées dans les buffers
        this.density.set(this.bufferD);
        this.velocityX.set(this.bufferX);
        this.velocityY.set(this.bufferY);
    }
}

// --- INITIALISATION ---
const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d", {alpha: false});
canvas.width = window.innerWidth; canvas.height = window.innerHeight;

const SIM_RES = 100;
const h = 1.0 / SIM_RES;
const simulator = new FluidSimulator(Math.floor((canvas.width/canvas.height)*SIM_RES), SIM_RES, h);
const config = { viscosity: 0, dissipation: 0.99, brushRadius: 0.05 };

// --- GESTION DES SLIDERS ---
const setupSlider = (id, prop, labelId) => {
    const el = document.getElementById(id);
    el.oninput = () => {
        config[prop] = parseFloat(el.value);
        document.getElementById(labelId).innerText = el.value;
    };
};
setupSlider("input-visc", "viscosity", "label-visc");
setupSlider("input-radius", "brushRadius", "label-radius");
setupSlider("input-diss", "dissipation", "label-diss");

// --- INTERACTION ---
let lastMouse = { x: 0, y: 0 };
canvas.addEventListener("mousemove", e => {
    const x = e.clientX / canvas.height, y = (canvas.height - e.clientY) / canvas.height;
    const vX = (x - lastMouse.x) * 60, vY = (y - lastMouse.y) * 60;
    const r2 = config.brushRadius * config.brushRadius;

    for (let i = 1; i < simulator.nx - 1; i++) {
        for (let j = 1; j < simulator.ny - 1; j++) {
            const dx = (i+0.5)*simulator.h - x, dy = (j+0.5)*simulator.h - y;
            if (dx*dx + dy*dy < r2) {
                const idx = simulator.getIdx(i, j);
                simulator.density[idx] = 1.0;
                simulator.velocityX[idx] += vX; simulator.velocityY[idx] += vY;
            }
        }
    }
    lastMouse = { x, y };
});

// --- BOUCLE DE RENDU ---
function renderFrame() {
    
    simulator.applyViscosity(1/60, config.viscosity);
    simulator.projectIncompressibility(40, 1.9);
    simulator.applyAdvection(1/60, config.dissipation);

    const img = ctx.createImageData(canvas.width, canvas.height);
    const scale = canvas.height, size = Math.ceil(simulator.h * scale);

    for (let i = 0; i < simulator.nx; i++) {
        for (let j = 0; j < simulator.ny; j++) {
            const d = simulator.density[simulator.getIdx(i, j)];
            if (d < 0.005) continue;
            const cIdx = Math.floor(d * 255) * 3;
            const px = Math.floor(i * simulator.h * scale), py = Math.floor(canvas.height - (j + 1) * simulator.h * scale);
            for (let y = py; y < py + size; y++) {
                if (y < 0 || y >= canvas.height) continue;
                for (let x = px; x < px + size; x++) {
                    const pIdx = (y * canvas.width + x) * 4;
                    img.data[pIdx] = COLOR_PALETTE[cIdx]; 
                    img.data[pIdx+1] = COLOR_PALETTE[cIdx+1]; 
                    img.data[pIdx+2] = COLOR_PALETTE[cIdx+2]; 
                    img.data[pIdx+3] = 255;
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0);
    requestAnimationFrame(renderFrame);
}

renderFrame();