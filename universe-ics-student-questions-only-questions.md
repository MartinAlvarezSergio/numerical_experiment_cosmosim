# Universe-ICS Questions

---

## 1) Initial Conditions (Density vs Velocity)

- If you keep the spectral index fixed, what changes between a velocity and a density spectrum initialization?
- Which mode produces clearer overdense/underdense contrast earlier in the run? What may cause this?
- For a fixed mode and seed, what is the effect of varying the spectral index in structures?
- Are these results affected by the random seed (i.e., the simulated `universe')?

## 2) Particle and Gravity Boundaries

- Recommendation: start this section with a high spectral index so edge effects are easier to identify.
- With a direct N-body solver, compare reflective, outflowing, and periodic particle boundary conditions. Which one seems the most accurate for cosmological simulations? Why? Which artifacts at edges are most evident for each of them?
- With particle boundary fixed, compare gravity boundary single-box vs periodic. How does this affect the formation of structure?
- Which combination of particle boundary and gravity boundary do you think provides the best representation of our Universe?

## 3) Solvers: Speed vs Equivalence

- Compare the Direct N-body solver vs the FFT-PM - keeping all the other numerical parameters fixed. Are the apparent structures qualitatively similar? Which one is faster?
- Where do you expect solver differences to appear first: fine-grained small-scale clumps or broad filamentary patterns?
- If two solvers agree visually but differ in runtime, when do you think / how can we determine whether the faster one is "good enough" for a scientific question?

## 4) Added Physics: Cooling and Feedback
For this part, you will have to run the models for longer periods of time (3000 steps and more)

- Turn on cooling and run the simulation for longer. What happens to the structure of the universe as its strength increases?
- Changing the proportion of particles cooling is analogous to changing the ratio of baryonic to dark matter. What is the effect on the structure of the Universe? Can we use this structure to support the existence of dark matter?
- Turn feedback while keeping cooling fixed (whether on or off). How does increasing feedback strength alter structures, and their persistence?
- Run the same setup for longer times with: i) neither, ii) cooling only, iii) feedback only, iv) both. Which case do you think is the most realistic?
- Did you find different parameter combinations that produce similar-looking outputs? What does that imply about model ambiguity (/parameter degeneration) and calibration?
