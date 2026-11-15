![Euler Heston paths demo](results/heston_euler_paths.gif)

# Heston model: GPU Monte Carlo pricing in CUDA

Monte Carlo pricing of a European call under the Heston stochastic volatility model on the GPU, comparing an Euler discretisation with an exact (Poisson-Gamma) transition for the variance process and an almost-exact scheme for the log price.

Team project by **Alberto Taddei** and **Thies Weel**, *Parallelism and Accelerators for Cluster Computing*, Sorbonne Université (Dec 2025). Fork of [albtad01/HestonModel_Finance](https://github.com/albtad01/HestonModel_Finance).

## What it does

| Program | Scheme |
|---|---|
| `src/heston_1.cu` | Euler scheme for $(S_t, v_t)$ |
| `src/heston_2.cu` | Exact CIR transition for $v_t$: $v_{t+\Delta t} = \frac{\sigma^2(1-e^{-\kappa\Delta t})}{2\kappa}\,G(d+N)$, $N\sim\mathrm{Poisson}(\lambda)$, $G$ Gamma, plus almost-exact terminal price |
| `src/heston_3.cu` | Benchmark: runtime and price for Euler vs almost-exact over many $(\kappa,\theta,\sigma)$ sets and time steps $M = 30 \ldots 10^5$, under the Feller-type bound $\sigma^2 / (\kappa\theta) < 20$ |

**GPU design**
- One thread per path: 262,144 paths (1024 blocks x 256 threads).
- One `curandState` per thread, initialised once and reused across runs.
- Per-thread time loop; payoffs reduced with a block-level shared-memory sum, then a final sum on the host.
- Device buffers reused between benchmark runs; non-recursive Gamma sampling.

## Results

Representative run, $(S_0, v_0, r, \kappa, \theta, \sigma, \rho, T, K) = (1, 0.1, 0, 0.5, 0.1, 0.3, 0, 1, 1)$, $M = 1000$:

| Method | Price | Time [ms] |
|---|---|---|
| Euler | 0.123081 | 2.4 |
| Exact $v_t$ | 0.123320 | 34.0 |

Average runtime per run [ms] versus number of time steps $M$ (`results/benchmark_results.csv`, 2,400 runs):

| $M$ | 30 | $10^2$ | $10^3$ | $10^4$ | $10^5$ |
|---|---|---|---|---|---|
| Almost exact | 1.0 | 3.6 | 24.5 | 196.7 | 1425.7 |
| Euler | 0.09 | 0.25 | 2.3 | 25.3 | 266.3 |

The almost-exact scheme costs about 5-10x more per step but stays accurate at coarse time steps, where Euler shows discretisation bias (see `results/fig_error_vs_dt_ref_AE_M1e5.pdf` and `results/fig_hist_deltaP_Euler_minus_AE_ref_M1e5_M30.pdf`). Slides: `presentation.pdf`.

## Build and run

Requirements: NVIDIA GPU (tested on `sm_70`), CUDA toolkit with `curand`.

```sh
cd src
bash compile.sh
./heston_1
./heston_2
./heston_3
```

Or compile one file:

```sh
nvcc -O3 -o heston_1 heston_1.cu -lcurand -arch=sm_70
```

## Plots

Python scripts in `graphs/` (needs `numpy`, `pandas`, `matplotlib`, and `ffmpeg` for animations) turn the CSV files in `results/` into the figures and the GIF above.

## Repository layout

```
src/       CUDA programs (heston_1-3) and compile.sh
graphs/    plotting and animation scripts, path-generation kernel
results/   benchmark CSV, figures, GIF
presentation.pdf
```
