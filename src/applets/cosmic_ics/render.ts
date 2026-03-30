import { UniverseSnapshot } from "./types";

export function renderUniverseICs(
  ctx: CanvasRenderingContext2D,
  snapshot: UniverseSnapshot
): void {
  const { width, height, particles, densityScores, stepCount, solverMode, pmGrid } = snapshot;

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "#191b20");
  bg.addColorStop(1, "#111216");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(245, 241, 233, 0.14)";
  ctx.lineWidth = 1;
  ctx.strokeRect(1, 1, width - 2, height - 2);

  if (solverMode === "fft-pm") {
    const dx = width / pmGrid.nx;
    const dy = height / pmGrid.ny;
    ctx.strokeStyle = "rgba(230, 224, 211, 0.13)";
    ctx.lineWidth = 0.8;
    for (let ix = 1; ix < pmGrid.nx; ix += 1) {
      const x = ix * dx;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let iy = 1; iy < pmGrid.ny; iy += 1) {
      const y = iy * dy;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }

  const colourRamp = Math.min(1, stepCount / 1400);

  for (let i = 0; i < particles.length; i += 1) {
    const particle = particles[i];
    const rawScore = Math.min(1, Math.max(0, densityScores[i] ?? 0.5));
    // Keep startup near-neutral and progressively reveal density contrast.
    const score = 0.5 + (rawScore - 0.5) * colourRamp;
    // Bias the full map toward cooler tones and soften saturation.
    const coolBiased = Math.min(1, Math.max(0, score * 0.8));
    // Royal-leaning blue (underdense) -> muted warm gold (overdense)
    const r = Math.round(118 + (224 - 118) * coolBiased);
    const g = Math.round(132 + (188 - 132) * coolBiased);
    const b = Math.round(178 + (88 - 178) * coolBiased);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.90)`;
    ctx.beginPath();
    ctx.arc(particle.position.x, particle.position.y, 1.75, 0, Math.PI * 2);
    ctx.fill();
  }
}
