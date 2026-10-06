import { useEffect, useMemo, useRef, useState } from "react";
import { setLogicalTransform } from "../../core/canvasScale";
import { AppletHostAdapter } from "../../core/host";
import { AppletStage } from "../../ui/stage/AppletStage";
import {
  StageIconButton,
  StagePills,
  StageReadout,
  StageSection,
  StageSelect,
  StageSlider,
  StageTextField,
  StageToggle
} from "../../ui/stage/StageControls";
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
const COMPARE_1_REFERENCE_FINAL_STATE_IMAGE_URL = `${import.meta.env.BASE_URL}001_run-1_box.png`;
const COMPARE_2_REFERENCE_FINAL_STATE_IMAGE_URL = `${import.meta.env.BASE_URL}001_run-1_box%20(1).png`;
const ARBITRARY_UNITS_TO_GYR = 14 / 16.7;
const FIXED_INTERNAL_DT_SECONDS = 1 / 120;
const MAX_FRAME_DT_SECONDS = 0.1;
const MAX_SUBSTEPS_PER_FRAME = 24;
/** Logical canvas size; matches the simulation box in sim.ts. */
const CANVAS_W = 900;
const CANVAS_H = 620;

const TIP = {
  play: "Start a fresh run with the current settings, or pause and resume it.",
  reset: "Stop and reset the simulation to the current settings.",
  runName: "Optional label used when this run's snapshot is stored.",
  particles: "Total number of simulation particles in the box.",
  spectralIndex:
    "Controls how smooth or clumpy the starting Universe is. Moving it changes how much structure appears on large versus small patterns in the initial map.",
  initFrom: "Choose whether initial fluctuations are seeded in the density or the velocity field.",
  seed:
    "A starting number for the random generator. Keep it the same to reproduce exactly the same initial particles; change it for a different Universe realization.",
  solver:
    "How gravity is computed each timestep. Direct N-body computes pair-by-pair forces; FFT-PM computes gravity on a grid.",
  particleBoundary: "How particles behave at the box edges.",
  gravityBoundary: "How gravity handles distances near boundaries (single box vs periodic images).",
  cooling: "Enable velocity damping for selected particles.",
  coolingStrength:
    "Energy-loss strength for selected particles (representing baryons). Higher values make those particles settle more quickly.",
  coolingFraction:
    "Fraction of particles eligible for cooling each run.\nThe cooled subset can change if initial-condition or boundary settings change.",
  feedback:
    "Represents astrophysical heating (for example, supernova feedback). Turning this on injects energy into dense regions.",
  feedbackStrength: "Amplitude of random feedback kicks in dense regions.",
  desiredStep: "Step at which the run's metrics and snapshot are captured.",
  yellowAt: "Running time at which the timing colour turns yellow (red at twice this value).",
  compare: "Get a similarity score against this reference Universe when the snapshot is captured. Lower is better.",
  currentStep: "Integration steps completed in this run.",
  cosmicTime: "Simulation time advanced in fixed internal timestep units.",
  runningTime: "Wall-clock time since pressing Start.",
  timeToSolution: "Wall-clock time when the run first reached the desired step."
} as const;

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

  // Sim is a stable handle; `settings` updates go through `sim.reset(settings)` below.
  // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional single sim instance
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
      setLogicalTransform(ctx, CANVAS_W);
      renderUniverseICs(ctx, snapshot);
      raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [running, paused, sim]);

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
    settings.boundaryMode,
    settings.gravityBoundaryMode,
    settings.solverMode,
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

  const fftLocksBoundaries = solverMode === "fft-pm";
  const playLabel = running && !paused ? "Pause" : running ? "Resume" : "Start";

  function onPlayPause(): void {
    if (!running) {
      onStart();
    } else {
      setPaused((v) => !v);
    }
  }

  const toolbar = (
    <>
      <StageIconButton icon={running && !paused ? "pause" : "play"} label={playLabel} tip={TIP.play} onClick={onPlayPause} />
      <StageIconButton icon="reset" label="Reset" tip={TIP.reset} onClick={onReset} />
    </>
  );

  const controls = (
    <>
      <StageTextField
        label="Run name"
        value={runNameInput}
        placeholder={`Run ${nextRunNumberRef.current}`}
        disabled={controlsLocked}
        tip={TIP.runName}
        onChange={setRunNameInput}
      />
      <StageSection title="Initial conditions">
        <StageSlider
          label="Number of particles"
          display={String(particleCount)}
          value={particleCount}
          min={PARTICLES_MIN}
          max={PARTICLES_MAX}
          step={50}
          disabled={controlsLocked}
          tip={TIP.particles}
          onChange={setParticleCount}
        />
        <StageSlider
          label="Spectral index n"
          display={format(spectralIndex3D)}
          value={spectralIndex3D}
          min={N3D_MIN}
          max={N3D_MAX}
          step={0.1}
          disabled={controlsLocked}
          tip={TIP.spectralIndex}
          onChange={setSpectralIndex3D}
        />
        <StageSelect
          label="Initialize from"
          value={initializationMode}
          options={[
            { value: "density-spectrum", label: "Density spectrum" },
            { value: "velocity-spectrum", label: "Velocity spectrum" }
          ]}
          disabled={controlsLocked}
          tip={TIP.initFrom}
          onChange={setInitializationMode}
        />
        <StageTextField label="Random seed" value={seed} disabled={controlsLocked} tip={TIP.seed} onChange={setSeed} />
      </StageSection>
      <StageSection title="Boundary model">
        <StageSelect
          label="Solver"
          value={solverMode}
          options={[
            { value: "direct-nbody", label: "Direct N-body" },
            { value: "fft-pm", label: "FFT-PM (64x64 mesh)" }
          ]}
          disabled={controlsLocked}
          tip={TIP.solver}
          onChange={setSolverMode}
        />
        <StageSelect
          label="Particle boundary"
          value={fftLocksBoundaries ? "periodic" : boundaryMode}
          options={[
            { value: "reflective", label: "Reflective" },
            { value: "outflow", label: "Outflow" },
            { value: "periodic", label: "Periodic" }
          ]}
          disabled={controlsLocked || fftLocksBoundaries}
          tip={TIP.particleBoundary}
          onChange={setBoundaryMode}
        />
        <StageSelect
          label="Gravity boundary"
          value={fftLocksBoundaries ? "periodic" : gravityBoundaryMode}
          options={[
            { value: "single-box", label: "Single box" },
            { value: "periodic", label: "Periodic (3x3 images)" }
          ]}
          disabled={controlsLocked || fftLocksBoundaries}
          tip={TIP.gravityBoundary}
          onChange={setGravityBoundaryMode}
        />
      </StageSection>
      <StageSection title="Dense-region processes">
        <StagePills>
          <StageToggle label="Cooling" on={coolingEnabled} disabled={controlsLocked} tip={TIP.cooling} onChange={setCoolingEnabled} />
          <StageToggle
            label="Feedback (dense regions)"
            on={feedbackEnabled}
            disabled={controlsLocked}
            tip={TIP.feedback}
            onChange={setFeedbackEnabled}
          />
        </StagePills>
        <StageSlider
          label="Cooling strength"
          display={format(coolingStrength)}
          value={coolingStrength}
          min={0}
          max={2}
          step={0.05}
          disabled={controlsLocked || !coolingEnabled}
          tip={TIP.coolingStrength}
          onChange={setCoolingStrength}
        />
        <StageSlider
          label="Cooling fraction"
          display={`${(coolingFraction * 100).toFixed(0)}%`}
          value={coolingFraction}
          min={0}
          max={1}
          step={0.01}
          disabled={controlsLocked}
          tip={TIP.coolingFraction}
          onChange={setCoolingFraction}
        />
        <StageSlider
          label="Feedback strength"
          display={format(feedbackStrength)}
          value={feedbackStrength}
          min={0}
          max={2}
          step={0.05}
          disabled={controlsLocked || !feedbackEnabled}
          tip={TIP.feedbackStrength}
          onChange={setFeedbackStrength}
        />
      </StageSection>
      <StageSection title="Run">
        <StageTextField
          label="Desired step"
          type="number"
          min={1}
          step={1}
          value={desiredStep}
          disabled={controlsLocked}
          tip={TIP.desiredStep}
          onChange={(v) => setDesiredStep(Math.max(1, Number(v) || 1))}
        />
        <StageSlider
          label="Yellow at (seconds)"
          display={`${format(yellowTimeSeconds)} s`}
          value={yellowTimeSeconds}
          min={2}
          max={60}
          step={0.5}
          disabled={controlsLocked}
          tip={TIP.yellowAt}
          onChange={setYellowTimeSeconds}
        />
      </StageSection>
      <StageSection title="Once you are done" defaultOpen={false}>
        <div className="compare-target-grid">
          <div className="compare-target-card">
            <img
              className="compare-reference-image"
              src={COMPARE_1_REFERENCE_FINAL_STATE_IMAGE_URL}
              alt="Reference final-state simulation box for Compare 1"
            />
            <StageToggle label="Compare 1" on={compare1Enabled} disabled={controlsLocked} tip={TIP.compare} onChange={setCompare1Enabled} />
          </div>
          <div className="compare-target-card">
            <img
              className="compare-reference-image"
              src={COMPARE_2_REFERENCE_FINAL_STATE_IMAGE_URL}
              alt="Reference final-state simulation box for Compare 2"
            />
            <StageToggle label="Compare 2" on={compare2Enabled} disabled={controlsLocked} tip={TIP.compare} onChange={setCompare2Enabled} />
          </div>
        </div>
      </StageSection>
    </>
  );

  const readouts = (
    <>
      <StageReadout label="Current step" value={currentStep} tip={TIP.currentStep} />
      <StageReadout label="Cosmic time (Gyr)" value={format(cosmicTimeGyr)} tip={TIP.cosmicTime} />
      <StageReadout label="Running time" value={`${format(runningTimeSeconds)} s`} valueColor={runTimeColor} tip={TIP.runningTime} />
      <StageReadout
        label="Time to solution"
        value={timeToSolutionSeconds === null ? "--" : `${format(timeToSolutionSeconds)} s`}
        valueColor={timeToSolutionSeconds === null ? undefined : solutionTimeColor}
        tip={TIP.timeToSolution}
      />
    </>
  );

  const info = (
    <>
      <h4>Running</h4>
      <ul>
        <li>Settings lock while a run is going; press reset to change them.</li>
        <li>At the desired step the run is captured below: a snapshot of the box and its power spectrum.</li>
        <li>Running-time colours: green, then yellow at the threshold you set, red at twice that.</li>
      </ul>
      <h4>Power spectrum</h4>
      <ul>
        <li>
          Shows how strongly matter is clustered at different sizes: the left side is larger structures, the right side
          smaller ones.
        </li>
      </ul>
      <h4>Once you are done</h4>
      <ul>
        <li>Try to match one of the two reference Universes. Tick Compare 1 and/or Compare 2 to get similarity scores when snapshots are captured; lower scores are better.</li>
        <li>The cooled-particle subset can change if initial-condition or boundary settings are changed.</li>
      </ul>
    </>
  );

  const below =
    capturedRuns.length > 0 ? (
      <div className="run-captures">
        {capturedRuns.map((run) => (
          <figure className={`run-capture-card${run.powerSpectrumUrl ? " run-capture-card-wide" : ""}`} key={run.runNumber}>
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
              {run.powerSpectrumUrl ? <img src={run.powerSpectrumUrl} alt={`${run.label} power spectrum`} /> : null}
            </div>
            <figcaption>
              {run.label} | Time: <span style={{ color: run.timeColor }}>{format(run.timeSeconds)} s</span> | N:{" "}
              {run.stepCount}
              {run.compare1Score !== null ? ` | Compare 1 score: ${run.compare1Score.toFixed(4)}` : ""}
              {run.compare2Score !== null ? ` | Compare 2 score: ${run.compare2Score.toFixed(4)}` : ""}
            </figcaption>
          </figure>
        ))}
      </div>
    ) : null;

  return (
    <AppletStage
      logicalWidth={CANVAS_W}
      logicalHeight={CANVAS_H}
      canvasRef={canvasRef}
      canvasLabel="Simulated matter particles in a 2D box"
      toolbar={toolbar}
      controls={controls}
      readouts={readouts}
      info={info}
      play={{ visible: !running || paused, label: playLabel, onClick: onPlayPause }}
      below={below}
    />
  );
}
