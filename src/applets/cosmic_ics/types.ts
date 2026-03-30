import { Vec2 } from "../../core/vector";

export type BoundaryMode = "reflective" | "outflow" | "periodic";
export type ICSourceMode = "density-spectrum" | "velocity-spectrum";
export type GravityBoundaryMode = "single-box" | "periodic";
export type SolverMode = "direct-nbody" | "fft-pm";

export type UniverseSettings = {
  particleCount: number;
  spectralIndex: number;
  initializationMode: ICSourceMode;
  gravityBoundaryMode: GravityBoundaryMode;
  boundaryMode: BoundaryMode;
  solverMode: SolverMode;
  directPenaltyFraction: number;
  feedbackEnabled: boolean;
  feedbackStrength: number;
  coolingEnabled: boolean;
  coolingStrength: number;
  coolingFraction: number;
  seed: string;
};

export type UniverseParticle = {
  position: Vec2;
  velocity: Vec2;
  mass: number;
};

export type UniverseSnapshot = {
  width: number;
  height: number;
  particles: UniverseParticle[];
  densityScores: number[];
  stepCount: number;
  averageSpeed: number;
  kineticEnergy: number;
  escapedCount: number;
  solverMode: SolverMode;
  pmGrid: {
    nx: number;
    ny: number;
  };
};
