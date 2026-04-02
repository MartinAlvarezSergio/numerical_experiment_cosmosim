import { useEffect, useMemo, useRef, useState } from "react";
import { AppletHostAdapter } from "../../core/host";
import { ControlCard } from "../../ui/ControlCard";
import { renderUniverseICs } from "./render";
import { createUniverseSim } from "./sim";
import compare1Reference from "./compare1_reference.json";
import compare2Reference from "./compare2_reference.json";
import {
  BoundaryMode,
  GravityBoundaryMode,
  ICSourceMode,
  SolverMode,
  UniverseParticle,
  UniverseSettings
} from "./types";

const PARTICLES_MIN = 2000;
const PARTICLES_MAX = 50000;
const PARTICLES_DEFAULT = 2000;

const N3D_MIN = -3;
const N3D_MAX = 3;
const N3D_DEFAULT = 1;
const DIRECT_PENALTY_DEFAULT = 1;
const STORE_SNAPSHOTS_AT_DESIRED_STEP = true;
const STORE_RUN_ARTIFACTS_TO_DISK = false;
const COMPARE_1_REFERENCE_FINAL_STATE_IMAGE_URL = "/001_run-1_box.png";
const COMPARE_2_REFERENCE_FINAL_STATE_IMAGE_URL = "/001_run-1_box%20(1).png";
const ARBITRARY_UNITS_TO_GYR = 14 / 16.7;
const FIXED_INTERNAL_DT_SECONDS = 1 / 120;
const MAX_FRAME_DT_SECONDS = 0.1;
const MAX_SUBSTEPS_PER_FRAME = 24;

type UniverseICsCanvasProps = {
  host?: AppletHostAdapter;
};

type NearestNeighborDistribution = {
  binCenters: number[];
  pdf: number[];
  probabilities: number[];
};

type CapturedRun = {
  runNumber: number;
  label: string;
  timeSeconds: number;
  timeColor: string;
  stepCount: number;
  imageUrl: string;
  powerSpectrumUrl: string | null;
  nearestNeighborDistribution: NearestNeighborDistribution | null;
  compare1Score: number | null;
  compare2Score: number | null;
};

type PowerSpectrumArtifact = {
  imageUrl: string;
  k: number[];
  power: number[];
  logPower: number[];
};

const COMPARE_1_REFERENCE = compare1Reference;
const COMPARE_2_REFERENCE = compare2Reference;

function format(value: number): string {
  return value.toFixed(2);
}

function createPowerSpectrumImage(
  particles: UniverseParticle[],
  width: number,
  height: number
): PowerSpectrumArtifact | null {
  const n = particles.length;
  if (n < 2 || width <= 0 || height <= 0) return null;

  const kMax = 28;
  const binCount = kMax + 1;
  const powerSum = new Array<number>(binCount).fill(0);
  const modeCount = new Array<number>(binCount).fill(0);
  const twoPi = Math.PI * 2;
  const invN2 = 1 / (n * n);

  for (let kx = 0; kx <= kMax; kx += 1) {
    for (let ky = 0; ky <= kMax; ky += 1) {
      if (kx === 0 && ky === 0) continue;
      const kr = Math.hypot(kx, ky);
      const bin = Math.floor(kr);
      if (bin < 1 || bin >= binCount) continue;

      let re = 0;
      let im = 0;
      for (let i = 0; i < n; i += 1) {
        const p = particles[i].position;
        const phase = twoPi * ((kx * p.x) / width + (ky * p.y) / height);
        re += Math.cos(phase);
        im -= Math.sin(phase);
      }
      const power = (re * re + im * im) * invN2;
      powerSum[bin] += power;
      modeCount[bin] += 1;
    }
  }

  const points: Array<{ x: number; y: number }> = [];
  const kValues: number[] = [];
  const powerValues: number[] = [];
  const logPowerValues: number[] = [];
  let minLogP = Number.POSITIVE_INFINITY;
  let maxLogP = Number.NEGATIVE_INFINITY;
  for (let b = 1; b < binCount; b += 1) {
    if (modeCount[b] <= 0) continue;
    const avgPower = powerSum[b] / modeCount[b];
    const logP = Math.log10(Math.max(avgPower, 1e-14));
    points.push({ x: b, y: logP });
    kValues.push(b);
    powerValues.push(avgPower);
    logPowerValues.push(logP);
    minLogP = Math.min(minLogP, logP);
    maxLogP = Math.max(maxLogP, logP);
  }
  if (points.length < 2) return null;
  const smoothPoints = points.map((point, i) => {
    const prev = points[Math.max(0, i - 1)].y;
    const next = points[Math.min(points.length - 1, i + 1)].y;
    return { x: point.x, y: 0.2 * prev + 0.6 * point.y + 0.2 * next };
  });

  const canvas = document.createElement("canvas");
  canvas.width = 420;
  canvas.height = 280;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const w = canvas.width;
  const h = canvas.height;
  const marginLeft = 46;
  const marginBottom = 28;
  const marginRight = 12;
  const marginTop = 12;
  const plotW = w - marginLeft - marginRight;
  const plotH = h - marginTop - marginBottom;
  const ySpan = Math.max(1e-8, maxLogP - minLogP);

  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#191b20");
  bg.addColorStop(1, "#111216");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(220, 214, 204, 0.25)";
  ctx.lineWidth = 1;
  ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

  ctx.strokeStyle = "rgba(220, 214, 204, 0.15)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 3; i += 1) {
    const yy = marginTop + (i * plotH) / 4;
    ctx.beginPath();
    ctx.moveTo(marginLeft, yy);
    ctx.lineTo(marginLeft + plotW, yy);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(230, 190, 120, 0.95)";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = 0; i < smoothPoints.length; i += 1) {
    const p = smoothPoints[i];
    const px = marginLeft + ((p.x - 1) / (binCount - 2)) * plotW;
    const py = marginTop + (1 - (p.y - minLogP) / ySpan) * plotH;
    if (i === 0) {
      ctx.moveTo(px, py);
      continue;
    }
    const prev = smoothPoints[i - 1];
    const prevX = marginLeft + ((prev.x - 1) / (binCount - 2)) * plotW;
    const prevY = marginTop + (1 - (prev.y - minLogP) / ySpan) * plotH;
    const controlX = (prevX + px) / 2;
    ctx.quadraticCurveTo(controlX, prevY, px, py);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(230, 224, 212, 0.9)";
  ctx.font = "12px Inter, system-ui, sans-serif";
  const axisLabel = "scale";
  const axisLabelWidth = ctx.measureText(axisLabel).width;
  ctx.fillText(axisLabel, marginLeft + (plotW - axisLabelWidth) / 2, h - 22);
  ctx.fillText("log Power", 8, marginTop + 10);
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillStyle = "rgba(210, 204, 194, 0.88)";
  ctx.fillText("large scale", marginLeft, h - 8);
  const rightLabel = "small scale";
  const rightLabelWidth = ctx.measureText(rightLabel).width;
  ctx.fillText(rightLabel, marginLeft + plotW - rightLabelWidth, h - 8);

  return {
    imageUrl: canvas.toDataURL("image/png"),
    k: kValues,
    power: powerValues,
    logPower: logPowerValues
  };
}

function createNearestNeighborDistribution(
  particles: UniverseParticle[],
  width: number,
  height: number,
  periodic: boolean
): NearestNeighborDistribution | null {
  const n = particles.length;
  if (n < 2 || width <= 0 || height <= 0) return null;

  const distances = new Array<number>(n);
  for (let i = 0; i < n; i += 1) {
    let nearest = Number.POSITIVE_INFINITY;
    const pi = particles[i].position;
    for (let j = 0; j < n; j += 1) {
      if (i === j) continue;
      let dx = particles[j].position.x - pi.x;
      let dy = particles[j].position.y - pi.y;
      if (periodic) {
        dx -= Math.round(dx / width) * width;
        dy -= Math.round(dy / height) * height;
      }
      const d = Math.hypot(dx, dy);
      if (d < nearest) nearest = d;
    }
    distances[i] = nearest;
  }

  const dMax = Math.max(...distances);
  if (!Number.isFinite(dMax) || dMax <= 0) return null;

  const binCount = 30;
  const counts = new Array<number>(binCount).fill(0);
  for (const d of distances) {
    const t = Math.min(0.999999, d / dMax);
    const idx = Math.floor(t * binCount);
    counts[idx] += 1;
  }

  const binCenters = new Array<number>(binCount);
  const pdf = new Array<number>(binCount);
  const probabilities = new Array<number>(binCount);
  for (let i = 0; i < binCount; i += 1) {
    const left = (i / binCount) * dMax;
    const right = ((i + 1) / binCount) * dMax;
    const widthBin = right - left;
    binCenters[i] = 0.5 * (left + right);
    pdf[i] = counts[i] / (n * widthBin);
    probabilities[i] = counts[i] / n;
  }
  return { binCenters, pdf, probabilities };
}

function normalizedRmse(values: number[], reference: number[]): number {
  const length = Math.min(values.length, reference.length);
  if (length <= 0) return Number.POSITIVE_INFINITY;
  let sumSq = 0;
  let minRef = Number.POSITIVE_INFINITY;
  let maxRef = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < length; i += 1) {
    const diff = values[i] - reference[i];
    sumSq += diff * diff;
    minRef = Math.min(minRef, reference[i]);
    maxRef = Math.max(maxRef, reference[i]);
  }
  const rmse = Math.sqrt(sumSq / length);
  const span = Math.max(1e-9, maxRef - minRef);
  return rmse / span;
}

function computeCompare1Score(
  powerSpectrum: PowerSpectrumArtifact | null,
  nearestNeighbor: NearestNeighborDistribution | null
): number | null {
  return computeReferenceScore(COMPARE_1_REFERENCE, powerSpectrum, nearestNeighbor);
}

function computeCompare2Score(
  powerSpectrum: PowerSpectrumArtifact | null,
  nearestNeighbor: NearestNeighborDistribution | null
): number | null {
  return computeReferenceScore(COMPARE_2_REFERENCE, powerSpectrum, nearestNeighbor);
}

function computeReferenceScore(
  reference: { logPower: number[]; nnProbabilities: number[] },
  powerSpectrum: PowerSpectrumArtifact | null,
  nearestNeighbor: NearestNeighborDistribution | null
): number | null {
  if (!powerSpectrum || !nearestNeighbor) return null;
  const fftScore = normalizedRmse(powerSpectrum.logPower, reference.logPower);
  const nnScore = normalizedRmse(nearestNeighbor.probabilities, reference.nnProbabilities);
  return 0.65 * fftScore + 0.35 * nnScore;
}

function sanitizeName(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9-_]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "run";
}

function downloadDataUrl(dataUrl: string, filename: string): void {
  const anchor = document.createElement("a");
  anchor.href = dataUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function downloadTextFile(content: string, filename: string): void {
  const blob = new Blob([content], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function runningTimeColor(runningTimeSeconds: number, yellowSeconds: number): string {
  const yellow = Math.max(1e-6, yellowSeconds);
  const red = yellow * 2;
  const ratio = Math.min(Math.max(runningTimeSeconds / red, 0), 1);
  const hue = 120 * (1 - ratio); // 120=green, 60=yellow, 0=red
  return `hsl(${hue.toFixed(1)} 80% 62%)`;
}

export function UniverseICsCanvas({ host }: UniverseICsCanvasProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const runElapsedSecondsRef = useRef(0);
  const hasCapturedCurrentRunRef = useRef(false);
  const nextRunNumberRef = useRef(1);

  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [particleCount, setParticleCount] = useState(PARTICLES_DEFAULT);
  const [spectralIndex3D, setSpectralIndex3D] = useState(N3D_DEFAULT);
  const [initializationMode, setInitializationMode] = useState<ICSourceMode>("density-spectrum");
  const [solverMode, setSolverMode] = useState<SolverMode>("direct-nbody");
  const [gravityBoundaryMode, setGravityBoundaryMode] =
    useState<GravityBoundaryMode>("single-box");
  const [boundaryMode, setBoundaryMode] = useState<BoundaryMode>("periodic");
  const [seed, setSeed] = useState("20260330");
  const [desiredStep, setDesiredStep] = useState(2000);
  const [yellowTimeSeconds, setYellowTimeSeconds] = useState(30);
  const [feedbackEnabled, setFeedbackEnabled] = useState(false);
  const [feedbackStrength, setFeedbackStrength] = useState(0.5);
  const [coolingEnabled, setCoolingEnabled] = useState(false);
  const [coolingStrength, setCoolingStrength] = useState(2);
  const [coolingFraction, setCoolingFraction] = useState(0.75);
  const [runNameInput, setRunNameInput] = useState("");
  const [compare1Enabled, setCompare1Enabled] = useState(false);
  const [compare2Enabled, setCompare2Enabled] = useState(false);

  const [currentStep, setCurrentStep] = useState(0);
  const [runningTimeSeconds, setRunningTimeSeconds] = useState(0);
  const [timeToSolutionSeconds, setTimeToSolutionSeconds] = useState<number | null>(null);
  const [capturedRuns, setCapturedRuns] = useState<CapturedRun[]>([]);

  const settings = useMemo<UniverseSettings>(
    () => ({
      particleCount,
      // Internal 2D exponent offset to mimic 3D slope behavior.
      spectralIndex: spectralIndex3D + 1,
      initializationMode,
      solverMode,
      gravityBoundaryMode,
      boundaryMode,
      directPenaltyFraction: DIRECT_PENALTY_DEFAULT,
      feedbackEnabled,
      feedbackStrength,
      coolingEnabled,
      coolingStrength,
      coolingFraction,
      seed
    }),
    [
      particleCount,
      spectralIndex3D,
      initializationMode,
      solverMode,
      gravityBoundaryMode,
      boundaryMode,
      feedbackEnabled,
      feedbackStrength,
      coolingEnabled,
      coolingStrength,
      coolingFraction,
      seed
    ]
  );

  const sim = useMemo(() => createUniverseSim(settings), []);

  useEffect(() => {
    sim.reset(settings);
    runElapsedSecondsRef.current = 0;
    hasCapturedCurrentRunRef.current = false;
    setCurrentStep(0);
    setRunningTimeSeconds(0);
    setTimeToSolutionSeconds(null);
  }, [settings, sim]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    let last = performance.now();
    let raf = 0;
    let accumulatorSeconds = 0;

    const frame = (time: number): void => {
      const dt = (time - last) / 1000;
      last = time;

      if (running && !paused) {
        const realFrameDt = Math.max(dt, 0);
        const clampedFrameDt = Math.min(Math.max(dt, 0), MAX_FRAME_DT_SECONDS);
        runElapsedSecondsRef.current += realFrameDt;
        setRunningTimeSeconds(runElapsedSecondsRef.current);

        accumulatorSeconds += clampedFrameDt;
        let stepsThisFrame = 0;
        while (
          accumulatorSeconds >= FIXED_INTERNAL_DT_SECONDS &&
          stepsThisFrame < MAX_SUBSTEPS_PER_FRAME
        ) {
          sim.step(FIXED_INTERNAL_DT_SECONDS);
          accumulatorSeconds -= FIXED_INTERNAL_DT_SECONDS;
          stepsThisFrame += 1;
        }
        if (stepsThisFrame > 0) {
          setCurrentStep((step) => step + stepsThisFrame);
        }
      } else {
        accumulatorSeconds = 0;
      }

      const snapshot = sim.getSnapshot();
      renderUniverseICs(ctx, snapshot);
      raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [running, paused, sim]);

  useEffect(() => {
    const root = canvasRef.current?.closest(".gravity-layout");
    if (!root) return;
    const labels = root.querySelectorAll<HTMLLabelElement>("label[title]");
    for (const label of labels) {
      const hint = label.getAttribute("title");
      if (!hint) continue;
      label.setAttribute("data-hover-help", hint);
      const descendants = label.querySelectorAll<HTMLElement>("input, select, button, span, strong");
      for (const element of descendants) {
        if (!element.getAttribute("title")) {
          element.setAttribute("title", hint);
        }
        element.setAttribute("data-hover-help", hint);
      }
    }
  });

  useEffect(() => {
    const root = canvasRef.current?.closest(".gravity-layout");
    if (!root) return;
    const tooltip = document.createElement("div");
    tooltip.className = "hover-help-tooltip";
    document.body.appendChild(tooltip);

    const placeTooltip = (x: number, y: number): void => {
      const offset = 14;
      const maxX = window.innerWidth - tooltip.offsetWidth - 8;
      const maxY = window.innerHeight - tooltip.offsetHeight - 8;
      const left = Math.min(Math.max(8, x + offset), Math.max(8, maxX));
      const top = Math.min(Math.max(8, y + offset), Math.max(8, maxY));
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
    };

    const onMouseMove = (event: Event): void => {
      const mouseEvent = event as MouseEvent;
      const target = mouseEvent.target as HTMLElement | null;
      const hintTarget = target?.closest?.("[data-hover-help]") as HTMLElement | null;
      if (!hintTarget || !root.contains(hintTarget)) {
        tooltip.classList.remove("visible");
        return;
      }
      const hint = hintTarget.getAttribute("data-hover-help");
      if (!hint) {
        tooltip.classList.remove("visible");
        return;
      }
      tooltip.textContent = hint;
      tooltip.classList.add("visible");
      placeTooltip(mouseEvent.clientX, mouseEvent.clientY);
    };

    const onMouseLeave = (): void => {
      tooltip.classList.remove("visible");
    };

    root.addEventListener("mousemove", onMouseMove);
    root.addEventListener("mouseleave", onMouseLeave);
    return () => {
      root.removeEventListener("mousemove", onMouseMove);
      root.removeEventListener("mouseleave", onMouseLeave);
      tooltip.remove();
    };
  }, []);

  function onStart(): void {
    sim.reset(settings);
    runElapsedSecondsRef.current = 0;
    hasCapturedCurrentRunRef.current = false;
    setCurrentStep(0);
    setRunningTimeSeconds(0);
    setTimeToSolutionSeconds(null);
    setPaused(false);
    setRunning(true);
  }

  function onReset(): void {
    setRunning(false);
    setPaused(false);
    sim.reset(settings);
    runElapsedSecondsRef.current = 0;
    hasCapturedCurrentRunRef.current = false;
    setCurrentStep(0);
    setRunningTimeSeconds(0);
    setTimeToSolutionSeconds(null);
    host?.onResult?.({
      event: "reset",
      ...settings
    });
  }

  useEffect(() => {
    if (hasCapturedCurrentRunRef.current) {
      return;
    }
    if (currentStep >= desiredStep && desiredStep > 0) {
      hasCapturedCurrentRunRef.current = true;
      setTimeToSolutionSeconds(runningTimeSeconds);
      if (!STORE_SNAPSHOTS_AT_DESIRED_STEP) {
        return;
      }
      const canvas = canvasRef.current;
      if (canvas) {
        const snapshot = sim.getSnapshot();
        const runNumber = nextRunNumberRef.current;
        nextRunNumberRef.current += 1;
        const trimmedName = runNameInput.trim();
        const runLabel = trimmedName.length > 0 ? trimmedName : `Run ${runNumber}`;
        const imageUrl = canvas.toDataURL("image/png");
        const capturedTimeColor = runningTimeColor(runningTimeSeconds, yellowTimeSeconds);
        const powerSpectrum = createPowerSpectrumImage(
          snapshot.particles,
          snapshot.width,
          snapshot.height
        );
        const powerSpectrumUrl = powerSpectrum?.imageUrl ?? null;
        const nearestNeighborDistribution = createNearestNeighborDistribution(
          snapshot.particles,
          snapshot.width,
          snapshot.height,
          settings.solverMode === "fft-pm" || settings.boundaryMode === "periodic"
        );
        const compare1Score = compare1Enabled
          ? computeCompare1Score(powerSpectrum, nearestNeighborDistribution)
          : null;
        const compare2Score = compare2Enabled
          ? computeCompare2Score(powerSpectrum, nearestNeighborDistribution)
          : null;

        if (STORE_RUN_ARTIFACTS_TO_DISK) {
          const runSlug = sanitizeName(runLabel);
          const baseName = `${runNumber.toString().padStart(3, "0")}_${runSlug}`;
          downloadDataUrl(imageUrl, `${baseName}_box.png`);
          if (powerSpectrumUrl) {
            downloadDataUrl(powerSpectrumUrl, `${baseName}_pk.png`);
          }
          downloadTextFile(
            JSON.stringify(
              {
                runNumber,
                runLabel,
                timeSeconds: runningTimeSeconds,
                stepCount: currentStep,
                solverMode: settings.solverMode,
                boundaryMode: settings.boundaryMode,
                gravityBoundaryMode: settings.gravityBoundaryMode,
                powerSpectrum: powerSpectrum
                  ? {
                      k: powerSpectrum.k,
                      power: powerSpectrum.power,
                      logPower: powerSpectrum.logPower
                    }
                  : null,
                nearestNeighborDistribution,
                compare1Enabled,
                compare1Score,
                compare2Enabled,
                compare2Score
              },
              null,
              2
            ),
            `${baseName}_stats.json`
          );
        }
        if (trimmedName.length > 0) {
          setRunNameInput("");
        }
        setCapturedRuns((prev) => [
          {
            runNumber,
            label: runLabel,
            timeSeconds: runningTimeSeconds,
            timeColor: capturedTimeColor,
            stepCount: currentStep,
            imageUrl,
            powerSpectrumUrl,
            nearestNeighborDistribution,
            compare1Score,
            compare2Score
          },
          ...prev
        ]);
      }
    }
  }, [
    compare1Enabled,
    compare2Enabled,
    currentStep,
    desiredStep,
    runNameInput,
    runningTimeSeconds,
    sim,
    yellowTimeSeconds
  ]);

  const controlsLocked = running;
  const cosmicTimeUnits = currentStep * FIXED_INTERNAL_DT_SECONDS;
  const cosmicTimeGyr = cosmicTimeUnits * ARBITRARY_UNITS_TO_GYR;
  const runTimeColor = runningTimeColor(runningTimeSeconds, yellowTimeSeconds);
  const solutionTimeColor =
    timeToSolutionSeconds === null
      ? "inherit"
      : runningTimeColor(timeToSolutionSeconds, yellowTimeSeconds);

  function removeCapturedRun(runNumber: number): void {
    setCapturedRuns((prev) => prev.filter((run) => run.runNumber !== runNumber));
  }

  return (
    <div className="gravity-layout">
      <ControlCard
        title="Simulation Choices, Cosmic Consequences"
        subtitle={controlsLocked ? "Locked - simulation is running." : "Open - configure before you run."}
      >
        <div className="control-grid">
          <label className="field-inline control-span-2" title="Optional label used when this run snapshot is stored.">
            <span>Run name:</span>
            <input
              type="text"
              value={runNameInput}
              placeholder={`Run ${nextRunNumberRef.current}`}
              disabled={controlsLocked}
              onChange={(event) => setRunNameInput(event.target.value)}
            />
          </label>

              <div className="control-section control-span-2">
            <h4 className="section-title">Initial conditions</h4>
            <div className="control-grid">
              <label className="control-span-2" title="Total number of simulation particles in the box.">
                <span className="slider-label">
                  <span>Number of particles:</span>
                  <strong>{particleCount}</strong>
                </span>
                <input
                  type="range"
                  min={PARTICLES_MIN}
                  max={PARTICLES_MAX}
                  step={50}
                  value={particleCount}
                  disabled={controlsLocked}
                  onChange={(event) => setParticleCount(Number(event.target.value))}
                />
              </label>

              <label className="control-span-2" title="Controls how smooth or clumpy the starting Universe is. Moving it changes how much structure appears on large versus small patterns in the initial map.">
                <span className="slider-label">
                  <span>Spectral index n:</span>
                  <strong>{format(spectralIndex3D)}</strong>
                </span>
                <input
                  type="range"
                  min={N3D_MIN}
                  max={N3D_MAX}
                  step={0.1}
                  value={spectralIndex3D}
                  disabled={controlsLocked}
                  onChange={(event) => setSpectralIndex3D(Number(event.target.value))}
                />
              </label>

              <label className="field-inline control-span-2" title="Choose whether initial fluctuations are seeded in density or velocity field.">
                <span>Initialize from:</span>
                <select
                  value={initializationMode}
                  disabled={controlsLocked}
                  onChange={(event) => setInitializationMode(event.target.value as ICSourceMode)}
                >
                  <option value="density-spectrum">Density spectrum</option>
                  <option value="velocity-spectrum">Velocity spectrum</option>
                </select>
              </label>

              <label className="field-inline control-span-2" title="A starting number used by the random generator. If you keep this value the same, you can reproduce exactly the same initial particle setup; changing it gives a different Universe realization.">
                <span>Random seed:</span>
                <input
                  type="text"
                  value={seed}
                  disabled={controlsLocked}
                  onChange={(event) => setSeed(event.target.value)}
                />
              </label>
            </div>
          </div>

          <div className="control-section control-span-2">
            <h4 className="section-title">Boundary model</h4>
            <div className="control-grid">
              <label className="field-inline control-span-2" title="Chooses how gravity is computed each timestep. Direct N-body computes pair-by-pair forces, while FFT-PM computes gravity on a grid.">
                <span>Solver:</span>
                <select
                  value={solverMode}
                  disabled={controlsLocked}
                  onChange={(event) => setSolverMode(event.target.value as SolverMode)}
                >
                  <option value="direct-nbody">Direct N-body</option>
                  <option value="fft-pm">FFT-PM (64x64 mesh)</option>
                </select>
              </label>

              <label className="field-inline control-span-2" title="How particles behave at the box edges.">
                <span>Particle boundary:</span>
                <select
                  value={solverMode === "fft-pm" ? "periodic" : boundaryMode}
                  disabled={controlsLocked || solverMode === "fft-pm"}
                  onChange={(event) => setBoundaryMode(event.target.value as BoundaryMode)}
                >
                  <option value="reflective">Reflective</option>
                  <option value="outflow">Outflow</option>
                  <option value="periodic">Periodic</option>
                </select>
              </label>

              <label className="field-inline control-span-2" title="How gravity handles distances near boundaries (single box vs periodic images).">
                <span>Gravity boundary:</span>
                <select
                  value={solverMode === "fft-pm" ? "periodic" : gravityBoundaryMode}
                  disabled={controlsLocked || solverMode === "fft-pm"}
                  onChange={(event) =>
                    setGravityBoundaryMode(event.target.value as GravityBoundaryMode)
                  }
                >
                  <option value="single-box">Single box</option>
                  <option value="periodic">Periodic (3x3 images)</option>
                </select>
              </label>
            </div>
          </div>

          <div className="control-section control-span-2">
            <h4 className="section-title">Dense-region processes</h4>
            <div className="control-grid">
              <label className="checkbox control-span-2" title="Enable velocity damping process for selected particles.">
                <input
                  type="checkbox"
                  checked={coolingEnabled}
                  disabled={controlsLocked}
                  onChange={(event) => setCoolingEnabled(event.target.checked)}
                />
                <span>Cooling.</span>
              </label>

              <label className="control-span-2" title="Sets the energy-loss strength for selected particles (representing baryons). Higher values make those particles settle more quickly.">
                <span className="slider-label">
                  <span>Cooling strength:</span>
                  <strong>{format(coolingStrength)}</strong>
                </span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={coolingStrength}
                  disabled={controlsLocked || !coolingEnabled}
                  onChange={(event) => setCoolingStrength(Number(event.target.value))}
                />
              </label>

              <label className="control-span-2" title="Fraction of particles eligible for cooling each run.">
                <span className="slider-label">
                  <span>Cooling fraction:</span>
                  <strong>{(coolingFraction * 100).toFixed(0)}%</strong>
                </span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={coolingFraction}
                  disabled={controlsLocked}
                  onChange={(event) => setCoolingFraction(Number(event.target.value))}
                />
              </label>
              <p className="subtle cooling-warning control-span-2">
                Warning: the cooled-particle subset can change if initial-condition or boundary settings are changed.
              </p>

              <label className="checkbox control-span-2" title="Represents astrophysical heating processes (for example, supernova feedback). Turning this on injects energy into dense regions.">
                <input
                  type="checkbox"
                  checked={feedbackEnabled}
                  disabled={controlsLocked}
                  onChange={(event) => setFeedbackEnabled(event.target.checked)}
                />
                <span>Feedback (dense regions).</span>
              </label>

              <label className="control-span-2" title="Amplitude of random feedback kicks in dense regions.">
                <span className="slider-label">
                  <span>Feedback strength:</span>
                  <strong>{format(feedbackStrength)}</strong>
                </span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={feedbackStrength}
                  disabled={controlsLocked || !feedbackEnabled}
                  onChange={(event) => setFeedbackStrength(Number(event.target.value))}
                />
              </label>
            </div>
          </div>

          <label className="field-inline control-span-2" title="Target step at which run metrics and snapshot are captured.">
            <span>Desired step:</span>
            <input
              type="number"
              min={1}
              step={1}
              value={desiredStep}
              disabled={controlsLocked}
              onChange={(event) => setDesiredStep(Math.max(1, Number(event.target.value) || 1))}
            />
          </label>

          <label className="control-span-2" title="Runtime threshold where timing color turns yellow (red at 2x this value).">
            <span className="slider-label">
              <span>Yellow at (seconds):</span>
              <strong>{format(yellowTimeSeconds)} s</strong>
            </span>
            <input
              type="range"
              min={2}
              max={60}
              step={0.5}
              value={yellowTimeSeconds}
              disabled={controlsLocked}
              onChange={(event) => setYellowTimeSeconds(Number(event.target.value))}
            />
          </label>

          <div className="button-row control-span-2">
            <button type="button" onClick={onStart} disabled={running} title="Start a fresh run with the current configuration.">
              Start
            </button>
            <button
              type="button"
              onClick={() => setPaused((v) => !v)}
              disabled={!running}
              title="Pause or resume the current run."
            >
              {paused ? "Resume" : "Pause"}
            </button>
            <button type="button" onClick={onReset} title="Stop and reset the simulation to current settings.">
              Reset
            </button>
          </div>

          <div className="stats control-span-2">
            <div title="Total number of integration steps completed in this run.">
              Current step: <strong>{currentStep}</strong>
            </div>
            <div title="Simulation time advanced in fixed internal timestep units.">
              Cosmic time (Gyr): <strong>{format(cosmicTimeGyr)}</strong>
            </div>
            <div title="Wall-clock runtime since pressing Start.">
              Running time:{" "}
              <strong style={{ color: runTimeColor }}>{format(runningTimeSeconds)} s</strong>
            </div>
            <div title="Wall-clock time when the run first reached the desired step.">
              Time to solution:{" "}
              <strong style={{ color: solutionTimeColor }}>
                {timeToSolutionSeconds === null ? "--" : `${format(timeToSolutionSeconds)} s`}
              </strong>
            </div>
            <p className="subtle spectrum-description">
              Power spectrum (second panel for each simulation run): this shows how strongly matter
              is clustered at different sizes. In this plot, the left side corresponds to larger
              structures and the right side to smaller structures.
            </p>
          </div>

          <details className="control-section control-span-2 once-done-box">
            <summary className="section-title">Once you are done</summary>
            <div className="once-done-body">
              <div className="compare-target-grid">
                <div className="compare-target-card">
                  <img
                    className="compare-reference-image"
                    src={COMPARE_1_REFERENCE_FINAL_STATE_IMAGE_URL}
                    alt="Reference final-state simulation box for Compare 1"
                  />
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={compare1Enabled}
                      disabled={controlsLocked}
                      onChange={(event) => setCompare1Enabled(event.target.checked)}
                    />
                    <span>Compare 1</span>
                  </label>
                </div>
                <div className="compare-target-card">
                  <img
                    className="compare-reference-image"
                    src={COMPARE_2_REFERENCE_FINAL_STATE_IMAGE_URL}
                    alt="Reference final-state simulation box for Compare 2"
                  />
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={compare2Enabled}
                      disabled={controlsLocked}
                      onChange={(event) => setCompare2Enabled(event.target.checked)}
                    />
                    <span>Compare 2</span>
                  </label>
                </div>
              </div>
              <p className="subtle">Try to match one of these reference Universes.</p>
              <p className="subtle">
                Tick Compare 1 and/or Compare 2 to get similarity scores when snapshots are captured.
              </p>
              <p className="subtle">Lower scores are better.</p>
            </div>
          </details>
        </div>
      </ControlCard>

      <div className="canvas-shell card">
        <canvas ref={canvasRef} width={900} height={620} />
        {capturedRuns.length > 0 && (
          <div className="run-captures">
            {capturedRuns.map((run) => (
              <figure
                className={`run-capture-card${run.powerSpectrumUrl ? " run-capture-card-wide" : ""}`}
                key={run.runNumber}
              >
                <button
                  type="button"
                  className="run-capture-delete"
                  aria-label={`Delete ${run.label} snapshot`}
                  onClick={() => removeCapturedRun(run.runNumber)}
                >
                  x
                </button>
                <div className="run-capture-visuals">
                  <img src={run.imageUrl} alt={`${run.label} snapshot`} />
                  {run.powerSpectrumUrl ? (
                    <img src={run.powerSpectrumUrl} alt={`${run.label} power spectrum`} />
                  ) : null}
                </div>
                <figcaption>
                  {run.label} | Time:{" "}
                  <span style={{ color: run.timeColor }}>{format(run.timeSeconds)} s</span> | N:{" "}
                  {run.stepCount}
                  {run.compare1Score !== null ? ` | Compare 1 score: ${run.compare1Score.toFixed(4)}` : ""}
                  {run.compare2Score !== null ? ` | Compare 2 score: ${run.compare2Score.toFixed(4)}` : ""}
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
