# Universe-ICS Questions (with brief context)

---

## 1) Initial Conditions (Density vs Velocity)

Cosmological initial conditions (ICs) are just how you set up the particles at the beginning: where they sit and how they move before gravity has had time to pull them into webs and clumps. In this demo you can generate that starting pattern in two ways: one emphasizes more material where matter concentrates (denser and emptier patches), and the other emphasizes the initial motions (who is moving toward whom). Both approximately describe a similar early universe, but they are not the same recipe. The spectral index is a simple “texture” control for the initial pattern: more small-scale ripples or more large-scale unevenness.

- If you keep the spectral index fixed, what changes between a velocity and a density spectrum initialization?
- Which mode produces clearer overdense/underdense contrast earlier in the run? What may cause this?
- For a fixed mode and seed, what is the effect of varying the spectral index in structures?
- Are these results affected by the random seed (i.e., the simulated `universe')?

## 2) Particle and Gravity Boundaries

A cosmological box represents only a finite patch of the Universe. Particle boundary conditions dictate what happens when a particle reaches a face/edge of the box (reflect back, exit as outflow, or re-enter with periodic wrapping). Gravity boundary conditions control how the gravitational potential and forces are computed at the edges.

With a direct N-body solver:
- Compare reflective, outflowing, and periodic particle boundary conditions. Which one seems the most accurate for cosmological simulations? Why? Which artifacts at edges are most evident for each of them?
- With particle boundary fixed, compare gravity boundary single-box vs periodic. How does this affect the formation of structure?
- Which combination of particle boundary and gravity boundary do you think provides the best representation of our Universe?

## 3) Solvers: Speed vs Equivalence

A direct N-body solver sums pairwise gravitational interactions (what is the effect of each particle on every other particle); an FFT-based particle-mesh (PM) solver places the mass of the particles on a grid, solves Poisson with a Fourier transform, and interpolates forces to particles. 

- Compare the Direct N-body solver vs the FFT-PM - keeping all the other numerical parameters fixed. Are the apparent structures qualitatively similar? Which one is faster?
- Where do solver differences appear first: fine-grained small-scale clumps or broad filamentary patterns?
- If two solvers agree visually but differ in runtime, when do you think / how can we determine whether the faster one is "good enough" for a scientific question?

## 4) Extra! Added Physics: Cooling and Feedback

Note: For this part, you will have to run the models for longer periods of time (3000 steps and more). Differences will be more clear if you can use a higher number of particles: e.g., ~30000.

Cooling mimics the energy loss of normal matter as it radiates energy thermally. Our simple `feedback' intends to reproduce the energy supplied by astrophysical phenomena, such as supernova explosions. Longer runs will let you see how these processes compete with pure gravity over many crossing times.

- Turn on cooling and run the simulation for longer. What happens to the structure of the universe as its strength increases?
- Changing the proportion of particles cooling is analogous to changing the ratio of baryonic to dark matter. What is the effect on the structure of the Universe? Can we use this structure to support the existence of dark matter?
- Turn feedback while keeping cooling fixed (whether on or off). How does increasing feedback strength alter structures, and their persistence?
- Run the same setup for longer times with: i) neither, ii) cooling only, iii) feedback only, iv) both. Which case do you think is the most realistic?
- Did you find different parameter combinations that produce similar-looking outputs? What does that imply about model ambiguity (/parameter degeneration) and calibration?
