import { magnitude } from "../../core/vector";
import { BoundaryMode, UniverseParticle, UniverseSettings, UniverseSnapshot } from "./types";

const WIDTH = 900;
const HEIGHT = 620;
const G_2D = 1.2;
const SOFTENING = 10;
const DRAG = 0.9992;
const MAX_SPEED = 220;
const MODE_COUNT = 80;
const PM_GRID_NX = 64;
const PM_GRID_NY = 64;
const TOTAL_BOX_MASS = 1000;
const FEEDBACK_DENSE_THRESHOLD = 0.72;
const FEEDBACK_KICK_SPEED = 42.5;
const COOLING_RELAX_RATE = 2.2;
const COOLING_STRENGTH_MULTIPLIER = 2;
const COOLING_WARMUP_STEPS = 2000;

type UniverseSim = {
  step: (dt: number) => void;
  getSnapshot: () => UniverseSnapshot;
  reset: (settings: UniverseSettings) => void;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function wrapIndex(i: number, size: number): number {
  return (i % size + size) % size;
}

function minimumImageDelta(delta: number, boxSize: number): number {
  return delta - Math.round(delta / boxSize) * boxSize;
}

function hashSeed(seed: string): number {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i += 1) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return (h >>> 0) || 1;
}

function mulberry32(seedValue: number): () => number {
  let a = seedValue >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function createModes(
  spectralIndex: number,
  rng: () => number
): Array<{ kx: number; ky: number; amp: number; phase: number }> {
  const modes: Array<{ kx: number; ky: number; amp: number; phase: number }> = [];
  const kMin = 0.5;
  const kMax = 7.5;
  for (let i = 0; i < MODE_COUNT; i += 1) {
    const t = rng();
    const k = kMin * Math.pow(kMax / kMin, t);
    const theta = rng() * Math.PI * 2;
    const kx = (k * Math.cos(theta) * Math.PI * 2) / WIDTH;
    const ky = (k * Math.sin(theta) * Math.PI * 2) / HEIGHT;
    const amp = Math.pow(k, spectralIndex / 2) * gaussian(rng);
    const phase = rng() * Math.PI * 2;
    modes.push({ kx, ky, amp, phase });
  }
  return modes;
}

function densityAt(
  x: number,
  y: number,
  modes: Array<{ kx: number; ky: number; amp: number; phase: number }>
): number {
  let value = 0;
  for (const mode of modes) {
    value += mode.amp * Math.cos(mode.kx * x + mode.ky * y + mode.phase);
  }
  return value;
}

function initializeFromDensitySpectrum(
  count: number,
  particleMass: number,
  rng: () => number,
  modes: Array<{ kx: number; ky: number; amp: number; phase: number }>
): UniverseParticle[] {
  const grid = 42;
  const cellW = WIDTH / grid;
  const cellH = HEIGHT / grid;
  const cellWeights: number[] = [];
  let minDensity = Number.POSITIVE_INFINITY;
  let maxDensity = Number.NEGATIVE_INFINITY;
  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const d = densityAt((gx + 0.5) * cellW, (gy + 0.5) * cellH, modes);
      minDensity = Math.min(minDensity, d);
      maxDensity = Math.max(maxDensity, d);
      cellWeights.push(d);
    }
  }
  const spread = Math.max(maxDensity - minDensity, 1e-6);
  let total = 0;
  for (let i = 0; i < cellWeights.length; i += 1) {
    const normalized = (cellWeights[i] - minDensity) / spread;
    cellWeights[i] = Math.pow(normalized + 0.08, 1.3);
    total += cellWeights[i];
  }
  const cumulative: number[] = [];
  let running = 0;
  for (const w of cellWeights) {
    running += w / total;
    cumulative.push(running);
  }
  const particles: UniverseParticle[] = [];
  for (let i = 0; i < count; i += 1) {
    const u = rng();
    let idx = 0;
    while (idx < cumulative.length && cumulative[idx] < u) idx += 1;
    const gy = Math.floor(idx / grid);
    const gx = idx % grid;
    particles.push({
      position: { x: (gx + rng()) * cellW, y: (gy + rng()) * cellH },
      velocity: { x: gaussian(rng) * 0.45, y: gaussian(rng) * 0.45 },
      mass: particleMass
    });
  }
  return particles;
}

function initializeFromVelocitySpectrum(
  count: number,
  particleMass: number,
  rng: () => number,
  modes: Array<{ kx: number; ky: number; amp: number; phase: number }>
): UniverseParticle[] {
  const particles: UniverseParticle[] = [];
  const velNorm = 14;
  const nx = Math.ceil(Math.sqrt(count));
  const ny = Math.ceil(count / nx);
  const dx = WIDTH / nx;
  const dy = HEIGHT / ny;
  for (let iy = 0; iy < ny; iy += 1) {
    for (let ix = 0; ix < nx; ix += 1) {
      if (particles.length >= count) break;
      const x = (ix + rng()) * dx;
      const y = (iy + rng()) * dy;
      let vx = 0;
      let vy = 0;
      for (const mode of modes) {
        const phase = mode.kx * x + mode.ky * y + mode.phase;
        const s = Math.sin(phase) * mode.amp;
        vx += mode.kx * s;
        vy += mode.ky * s;
      }
      particles.push({
        position: { x, y },
        velocity: { x: vx * velNorm + gaussian(rng) * 0.22, y: vy * velNorm + gaussian(rng) * 0.22 },
        mass: particleMass
      });
    }
  }
  return particles;
}

function initializeParticles(settings: UniverseSettings): UniverseParticle[] {
  const rng = mulberry32(hashSeed(settings.seed));
  const count = clamp(Math.round(settings.particleCount), 100, 50000);
  const particleMass = TOTAL_BOX_MASS / count;
  const modes = createModes(settings.spectralIndex, rng);
  return settings.initializationMode === "velocity-spectrum"
    ? initializeFromVelocitySpectrum(count, particleMass, rng, modes)
    : initializeFromDensitySpectrum(count, particleMass, rng, modes);
}

function handleBoundary(particle: UniverseParticle, mode: BoundaryMode): "kept" | "removed" {
  if (mode === "periodic") {
    if (particle.position.x < 0) particle.position.x += WIDTH;
    if (particle.position.x >= WIDTH) particle.position.x -= WIDTH;
    if (particle.position.y < 0) particle.position.y += HEIGHT;
    if (particle.position.y >= HEIGHT) particle.position.y -= HEIGHT;
    return "kept";
  }
  if (mode === "reflective") {
    if (particle.position.x < 0 || particle.position.x > WIDTH) {
      particle.velocity.x *= -0.88;
      particle.position.x = clamp(particle.position.x, 0, WIDTH);
    }
    if (particle.position.y < 0 || particle.position.y > HEIGHT) {
      particle.velocity.y *= -0.88;
      particle.position.y = clamp(particle.position.y, 0, HEIGHT);
    }
    return "kept";
  }
  if (
    particle.position.x < 0 ||
    particle.position.x > WIDTH ||
    particle.position.y < 0 ||
    particle.position.y > HEIGHT
  ) {
    return "removed";
  }
  return "kept";
}

function applyPairAcceleration(
  p1: UniverseParticle,
  p2: UniverseParticle,
  accel1: { x: number; y: number },
  accel2: { x: number; y: number },
  gravityBoundaryMode: UniverseSettings["gravityBoundaryMode"],
  massScale = 1
): void {
  let dx = p2.position.x - p1.position.x;
  let dy = p2.position.y - p1.position.y;
  if (gravityBoundaryMode === "periodic") {
    dx = minimumImageDelta(dx, WIDTH);
    dy = minimumImageDelta(dy, HEIGHT);
  }
  const d2 = dx * dx + dy * dy + SOFTENING * SOFTENING;
  const invD = 1 / Math.sqrt(d2);
  const f = G_2D * massScale * invD * invD;
  accel1.x += dx * f;
  accel1.y += dy * f;
  accel2.x -= dx * f;
  accel2.y -= dy * f;
}

function fft1D(re: Float64Array, im: Float64Array, n: number, inverse: boolean): void {
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) { j -= m; m >>= 1; }
    j += m;
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = sign * 2 * Math.PI / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let start = 0; start < n; start += len) {
      let uRe = 1, uIm = 0;
      for (let k = 0; k < half; k++) {
        const e = start + k;
        const o = start + k + half;
        const tRe = uRe * re[o] - uIm * im[o];
        const tIm = uRe * im[o] + uIm * re[o];
        re[o] = re[e] - tRe;
        im[o] = im[e] - tIm;
        re[e] += tRe;
        im[e] += tIm;
        const nextU = uRe * wRe - uIm * wIm;
        uIm = uRe * wIm + uIm * wRe;
        uRe = nextU;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
}

function fft2D(re: Float64Array, im: Float64Array, nx: number, ny: number, inverse: boolean): void {
  const rowRe = new Float64Array(nx);
  const rowIm = new Float64Array(nx);
  for (let y = 0; y < ny; y++) {
    const off = y * nx;
    for (let x = 0; x < nx; x++) { rowRe[x] = re[off + x]; rowIm[x] = im[off + x]; }
    fft1D(rowRe, rowIm, nx, inverse);
    for (let x = 0; x < nx; x++) { re[off + x] = rowRe[x]; im[off + x] = rowIm[x]; }
  }
  const colRe = new Float64Array(ny);
  const colIm = new Float64Array(ny);
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) { colRe[y] = re[y * nx + x]; colIm[y] = im[y * nx + x]; }
    fft1D(colRe, colIm, ny, inverse);
    for (let y = 0; y < ny; y++) { re[y * nx + x] = colRe[y]; im[y * nx + x] = colIm[y]; }
  }
}

function computeDensityScores(particles: UniverseParticle[]): number[] {
  const densityScores = new Array<number>(particles.length).fill(0.5);
  if (particles.length === 0) return densityScores;

  const nx = PM_GRID_NX;
  const ny = PM_GRID_NY;
  const cellW = WIDTH / nx;
  const cellH = HEIGHT / ny;
  const massPerCell = new Float64Array(nx * ny);
  const particleCellX = new Int32Array(particles.length);
  const particleCellY = new Int32Array(particles.length);

  for (let i = 0; i < particles.length; i += 1) {
    const p = particles[i];
    const x = ((p.position.x % WIDTH) + WIDTH) % WIDTH;
    const y = ((p.position.y % HEIGHT) + HEIGHT) % HEIGHT;
    const cx = wrapIndex(Math.floor(x / cellW), nx);
    const cy = wrapIndex(Math.floor(y / cellH), ny);
    particleCellX[i] = cx;
    particleCellY[i] = cy;
    massPerCell[cy * nx + cx] += p.mass;
  }

  const totalMass = particles.reduce((sum, p) => sum + p.mass, 0);
  const cellMeanMass = totalMass / (nx * ny);
  const neighMeanMass = Math.max(1e-6, 9 * cellMeanMass);

  for (let i = 0; i < particles.length; i += 1) {
    const cx = particleCellX[i];
    const cy = particleCellY[i];
    let neighMass = 0;
    for (let oy = -1; oy <= 1; oy += 1) {
      for (let ox = -1; ox <= 1; ox += 1) {
        const nxCell = wrapIndex(cx + ox, nx);
        const nyCell = wrapIndex(cy + oy, ny);
        neighMass += massPerCell[nyCell * nx + nxCell];
      }
    }
    const contrast = (neighMass - neighMeanMass) / neighMeanMass;
    densityScores[i] = 0.5 + 0.5 * Math.tanh(contrast * 0.9);
  }

  return densityScores;
}

function createCoolingEligibility(count: number, seed: string, fraction: number): boolean[] {
  if (count <= 0) return [];
  const clipped = clamp(fraction, 0, 1);
  const target = Math.round(count * clipped);
  const eligibility = new Array<boolean>(count).fill(false);
  if (target <= 0) return eligibility;
  const indices = new Array<number>(count);
  for (let i = 0; i < count; i += 1) indices[i] = i;
  const rng = mulberry32(hashSeed(`${seed}-cooling-mask`));
  // Partial Fisher-Yates: select exactly `target` unique indices.
  for (let i = 0; i < target; i += 1) {
    const j = i + Math.floor(rng() * (count - i));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
    eligibility[indices[i]] = true;
  }
  return eligibility;
}

export function createUniverseSim(initial: UniverseSettings): UniverseSim {
  let settings = initial;
  let particles = initializeParticles(settings);
  let coolingEligibility = createCoolingEligibility(
    particles.length,
    settings.seed,
    settings.coolingFraction
  );
  let feedbackRng = mulberry32(hashSeed(`${settings.seed}-feedback`));
  let escapedCount = 0;
  let stepCount = 0;
  let penaltySink = 0;

  return {
    step(dt: number): void {
      const clampedDt = clamp(dt, 1 / 240, 1 / 30);
      const n = particles.length;
      const ax = new Float64Array(n);
      const ay = new Float64Array(n);

      if (settings.solverMode === "direct-nbody") {
        for (let i = 0; i < n; i += 1) {
          for (let j = i + 1; j < n; j += 1) {
            const ai = { x: ax[i], y: ay[i] };
            const aj = { x: ax[j], y: ay[j] };
            applyPairAcceleration(
              particles[i],
              particles[j],
              ai,
              aj,
              settings.gravityBoundaryMode,
              particles[i].mass
            );
            ax[i] = ai.x;
            ay[i] = ai.y;
            ax[j] = aj.x;
            ay[j] = aj.y;
          }
        }
        if (settings.directPenaltyFraction > 0) {
          const penalty = Math.max(0, settings.directPenaltyFraction);
          const fullPasses = Math.floor(penalty);
          const fractionalPass = penalty - fullPasses;
          for (let pass = 0; pass < fullPasses + (fractionalPass > 0 ? 1 : 0); pass += 1) {
            const passThreshold = pass < fullPasses ? 1 : fractionalPass;
            for (let i = 0; i < n; i += 1) {
              for (let j = i + 1; j < n; j += 1) {
                const selector = ((i * 131 + j * 97 + pass * 53) % 1000) / 1000;
                if (selector > passThreshold) continue;
                let dx = particles[j].position.x - particles[i].position.x;
                let dy = particles[j].position.y - particles[i].position.y;
                if (settings.gravityBoundaryMode === "periodic") {
                  dx = minimumImageDelta(dx, WIDTH);
                  dy = minimumImageDelta(dy, HEIGHT);
                }
                penaltySink += Math.sqrt(dx * dx + dy * dy + SOFTENING * SOFTENING);
              }
            }
          }
          // Keep this synthetic workload observable to avoid JIT dead-code elimination.
          if (penaltySink > 1e12 || penaltySink < -1e12) penaltySink = 0;
        }
      } else {
        const gx = PM_GRID_NX;
        const gy = PM_GRID_NY;
        const cellW = WIDTH / gx;
        const cellH = HEIGHT / gy;
        const totalCells = gx * gy;

        const rhoRe = new Float64Array(totalCells);
        const rhoIm = new Float64Array(totalCells);

        for (let i = 0; i < n; i += 1) {
          const p = particles[i];
          const px = ((p.position.x % WIDTH) + WIDTH) % WIDTH;
          const py = ((p.position.y % HEIGHT) + HEIGHT) % HEIGHT;
          const fx = px / cellW - 0.5;
          const fy = py / cellH - 0.5;
          const ix0 = Math.floor(fx);
          const iy0 = Math.floor(fy);
          const tx = fx - ix0;
          const ty = fy - iy0;
          const ix0w = ((ix0 % gx) + gx) % gx;
          const iy0w = ((iy0 % gy) + gy) % gy;
          const ix1 = (ix0w + 1) % gx;
          const iy1 = (iy0w + 1) % gy;
          rhoRe[iy0w * gx + ix0w] += (1 - tx) * (1 - ty) * p.mass;
          rhoRe[iy0w * gx + ix1]  += tx       * (1 - ty) * p.mass;
          rhoRe[iy1  * gx + ix0w] += (1 - tx) * ty       * p.mass;
          rhoRe[iy1  * gx + ix1]  += tx       * ty       * p.mass;
        }

        const cellArea = cellW * cellH;
        let rhoMean = 0;
        for (let i = 0; i < totalCells; i++) {
          rhoRe[i] /= cellArea;
          rhoMean += rhoRe[i];
        }
        rhoMean /= totalCells;
        for (let i = 0; i < totalCells; i++) rhoRe[i] -= rhoMean;

        fft2D(rhoRe, rhoIm, gx, gy, false);

        const phiRe = new Float64Array(totalCells);
        const phiIm = new Float64Array(totalCells);
        const coupling = 2 * Math.PI * G_2D;

        for (let jy = 0; jy < gy; jy++) {
          for (let jx = 0; jx < gx; jx++) {
            const idx = jy * gx + jx;
            if (jx === 0 && jy === 0) { phiRe[idx] = 0; phiIm[idx] = 0; continue; }
            const lx = 2 * (Math.cos(2 * Math.PI * jx / gx) - 1) / (cellW * cellW);
            const ly = 2 * (Math.cos(2 * Math.PI * jy / gy) - 1) / (cellH * cellH);
            const lambda = lx + ly;
            const scale = coupling / lambda;
            phiRe[idx] = rhoRe[idx] * scale;
            phiIm[idx] = rhoIm[idx] * scale;
          }
        }

        fft2D(phiRe, phiIm, gx, gy, true);

        const axGrid = new Float64Array(totalCells);
        const ayGrid = new Float64Array(totalCells);
        for (let jy = 0; jy < gy; jy++) {
          for (let jx = 0; jx < gx; jx++) {
            const idx = jy * gx + jx;
            const ixp = jy * gx + ((jx + 1) % gx);
            const ixm = jy * gx + ((jx - 1 + gx) % gx);
            const iyp = ((jy + 1) % gy) * gx + jx;
            const iym = ((jy - 1 + gy) % gy) * gx + jx;
            axGrid[idx] = -(phiRe[ixp] - phiRe[ixm]) / (2 * cellW);
            ayGrid[idx] = -(phiRe[iyp] - phiRe[iym]) / (2 * cellH);
          }
        }

        for (let i = 0; i < n; i += 1) {
          const p = particles[i];
          const px = ((p.position.x % WIDTH) + WIDTH) % WIDTH;
          const py = ((p.position.y % HEIGHT) + HEIGHT) % HEIGHT;
          const fx = px / cellW - 0.5;
          const fy = py / cellH - 0.5;
          const ix0 = Math.floor(fx);
          const iy0 = Math.floor(fy);
          const tx = fx - ix0;
          const ty = fy - iy0;
          const ix0w = ((ix0 % gx) + gx) % gx;
          const iy0w = ((iy0 % gy) + gy) % gy;
          const ix1 = (ix0w + 1) % gx;
          const iy1 = (iy0w + 1) % gy;
          const w00 = (1 - tx) * (1 - ty);
          const w10 = tx       * (1 - ty);
          const w01 = (1 - tx) * ty;
          const w11 = tx       * ty;
          const i00 = iy0w * gx + ix0w;
          const i10 = iy0w * gx + ix1;
          const i01 = iy1  * gx + ix0w;
          const i11 = iy1  * gx + ix1;
          ax[i] +=
            w00 * axGrid[i00] + w10 * axGrid[i10] + w01 * axGrid[i01] + w11 * axGrid[i11];
          ay[i] +=
            w00 * ayGrid[i00] + w10 * ayGrid[i10] + w01 * ayGrid[i01] + w11 * ayGrid[i11];
        }
      }

      let stepDensityScores: number[] | null = null;
      if (
        n > 0 &&
        ((settings.feedbackEnabled && settings.feedbackStrength > 0) ||
          (settings.coolingEnabled && settings.coolingStrength > 0))
      ) {
        stepDensityScores = computeDensityScores(particles);
      }

      if (settings.feedbackEnabled && settings.feedbackStrength > 0 && n > 0 && stepDensityScores) {
        for (let i = 0; i < n; i += 1) {
          const denseFraction = clamp(
            (stepDensityScores[i] - FEEDBACK_DENSE_THRESHOLD) / (1 - FEEDBACK_DENSE_THRESHOLD),
            0,
            1
          );
          if (denseFraction <= 0) continue;
          const angle = feedbackRng() * Math.PI * 2;
          const randomScale = 0.6 + 0.8 * feedbackRng();
          const kick =
            settings.feedbackStrength * denseFraction * FEEDBACK_KICK_SPEED * randomScale * clampedDt;
          particles[i].velocity.x += Math.cos(angle) * kick;
          particles[i].velocity.y += Math.sin(angle) * kick;
        }
      }

      if (
        settings.coolingEnabled &&
        settings.coolingStrength > 0 &&
        n > 0 &&
        stepDensityScores &&
        stepCount >= COOLING_WARMUP_STEPS
      ) {
        const nx = PM_GRID_NX;
        const ny = PM_GRID_NY;
        const cellW = WIDTH / nx;
        const cellH = HEIGHT / ny;
        const cellMass = new Float64Array(nx * ny);
        const cellMomX = new Float64Array(nx * ny);
        const cellMomY = new Float64Array(nx * ny);
        const particleCellX = new Int32Array(n);
        const particleCellY = new Int32Array(n);

        for (let i = 0; i < n; i += 1) {
          const p = particles[i];
          const x = ((p.position.x % WIDTH) + WIDTH) % WIDTH;
          const y = ((p.position.y % HEIGHT) + HEIGHT) % HEIGHT;
          const cx = wrapIndex(Math.floor(x / cellW), nx);
          const cy = wrapIndex(Math.floor(y / cellH), ny);
          const c = cy * nx + cx;
          particleCellX[i] = cx;
          particleCellY[i] = cy;
          cellMass[c] += p.mass;
          cellMomX[c] += p.mass * p.velocity.x;
          cellMomY[c] += p.mass * p.velocity.y;
        }

        const cellVx = new Float64Array(nx * ny);
        const cellVy = new Float64Array(nx * ny);
        for (let c = 0; c < nx * ny; c += 1) {
          const m = cellMass[c];
          if (m <= 0) continue;
          cellVx[c] = cellMomX[c] / m;
          cellVy[c] = cellMomY[c] / m;
        }

        for (let i = 0; i < n; i += 1) {
          if (!coolingEligibility[i]) continue;
          const densityWeight = clamp(stepDensityScores[i], 0, 1);
          if (densityWeight <= 0) continue;

          const cx = particleCellX[i];
          const cy = particleCellY[i];
          let localMass = 0;
          let localVx = 0;
          let localVy = 0;
          for (let oy = -1; oy <= 1; oy += 1) {
            for (let ox = -1; ox <= 1; ox += 1) {
              const nxCell = wrapIndex(cx + ox, nx);
              const nyCell = wrapIndex(cy + oy, ny);
              const c = nyCell * nx + nxCell;
              const m = cellMass[c];
              if (m <= 0) continue;
              localMass += m;
              localVx += m * cellVx[c];
              localVy += m * cellVy[c];
            }
          }
          if (localMass <= 0) continue;
          localVx /= localMass;
          localVy /= localMass;

          const beta =
            COOLING_RELAX_RATE *
            COOLING_STRENGTH_MULTIPLIER *
            settings.coolingStrength *
            densityWeight;
          ax[i] += -beta * (particles[i].velocity.x - localVx);
          ay[i] += -beta * (particles[i].velocity.y - localVy);
        }
      }

      const kept: UniverseParticle[] = [];
      const keptCoolingEligibility: boolean[] = [];
      const effectiveBoundaryMode: BoundaryMode =
        settings.solverMode === "fft-pm" ? "periodic" : settings.boundaryMode;
      for (let i = 0; i < n; i += 1) {
        const p = particles[i];
        p.velocity.x = (p.velocity.x + ax[i] * clampedDt) * DRAG;
        p.velocity.y = (p.velocity.y + ay[i] * clampedDt) * DRAG;
        const speed = Math.hypot(p.velocity.x, p.velocity.y);
        if (speed > MAX_SPEED) {
          const s = MAX_SPEED / speed;
          p.velocity.x *= s;
          p.velocity.y *= s;
        }
        p.position.x += p.velocity.x * clampedDt;
        p.position.y += p.velocity.y * clampedDt;
        if (handleBoundary(p, effectiveBoundaryMode) === "kept") {
          kept.push(p);
          keptCoolingEligibility.push(coolingEligibility[i]);
        } else escapedCount += 1;
      }
      particles = kept;
      coolingEligibility = keptCoolingEligibility;
      stepCount += 1;
    },
    getSnapshot(): UniverseSnapshot {
      let speedSum = 0;
      let kineticEnergy = 0;
      for (const p of particles) {
        const s = magnitude(p.velocity);
        speedSum += s;
        kineticEnergy += 0.5 * p.mass * s * s;
      }

      // Build a coarse local-density score per particle using 3x3 cell neighborhoods.
      const densityScores = computeDensityScores(particles);

      return {
        width: WIDTH,
        height: HEIGHT,
        particles,
        densityScores,
        stepCount,
        averageSpeed: particles.length > 0 ? speedSum / particles.length : 0,
        kineticEnergy,
        escapedCount,
        solverMode: settings.solverMode,
        pmGrid: { nx: PM_GRID_NX, ny: PM_GRID_NY }
      };
    },
    reset(nextSettings: UniverseSettings): void {
      settings = nextSettings;
      particles = initializeParticles(settings);
      coolingEligibility = createCoolingEligibility(
        particles.length,
        settings.seed,
        settings.coolingFraction
      );
      feedbackRng = mulberry32(hashSeed(`${settings.seed}-feedback`));
      escapedCount = 0;
      stepCount = 0;
    }
  };
}
