# Edgeless Lab — marimo × alphaXiv notebook competition

Three GPU-accelerated marimo notebooks that present a research paper *and* extend it —
built to be **beautiful** (Tufte-principled, bioluminescent) rather than data-dense dashboards.

Each has a JAX GPU path (attach the RTX Pro 6000 Blackwell on molab) with a NumPy CPU fallback.

| Notebook | Paper | Custom extension |
|---|---|---|
| `01_gray_scott_gpu.py` | Munafo, *Stable localized moving patterns in 2-D Gray-Scott* (arXiv:1501.01990) / Pearson 1993 | `jax.vmap` sweeps the whole (F,k) plane in one batched pass; **aesthetic autoresearch** scores every point → a novel score-landscape |
| `02_strange_attractors_gpu.py` | Sprott, *Automatic generation of strange attractors* (1993) | GPU-batched Lyapunov search over thousands of coefficient sets + a second **beauty** filter Sprott couldn't run — auto-discovers the gorgeous few |
| `03_option_pricing_gpu.py` | Boyle, *Options: A Monte Carlo Approach* (1977) | GPU Monte-Carlo with antithetic variates, 100×+ speedup, live convergence + Greek surfaces |

`_aesthetic_search.py` — the reusable aesthetic-scoring rig (coverage, entropy, edge density,
figure-ground contrast) that drives the parameter searches.

Live companions: [edgelesslab.com/lab/marimo](https://edgelesslab.com/lab/marimo) ·
[reaction-diffusion, on your phone](https://edgelesslab.com/threejs/reaction-diffusion/)

MIT License · [edgelesslab.com](https://edgelesslab.com)

## For judges — the 30-second tour

**`01_gray_scott_gpu.py`** — Pearson/Munafo's reaction-diffusion, live. Steer F/k and watch
the vocabulary change; then two extensions: the *entire* (F,k) plane computed as ONE
`jax.vmap` call (the paper explored it run-by-run), and an **aesthetic search** that scores
the beauty of every parameter combination — a score landscape over a PDE's parameter space.

**`02_strange_attractors_gpu.py`** — Sprott's 1993 automatic attractor search, but thousands
of candidates per batched GPU call instead of one-at-a-time on a 386 — plus the filter Sprott
couldn't compute: *is it beautiful?* Every discovery has a reproducible letter-code name, and
the top find breathes live (coefficients drift each tick, re-solved in real time).

**`03_option_pricing_gpu.py`** — Boyle's 1977 Monte-Carlo option pricing at 100×+ via GPU
antithetic-variate batching, with live convergence — rendered as Tufte objects (direct labels,
no chartjunk), not a spreadsheet.

All three: attach the GPU (notebook specs → RTX Pro 6000), run top-to-bottom, drag things.
NumPy fallbacks mean they degrade gracefully on CPU. Companion pieces live at
[edgelesslab.com/lab/marimo](https://edgelesslab.com/lab/marimo) and the same reaction-diffusion
runs on your phone at [edgelesslab.com/threejs/reaction-diffusion](https://edgelesslab.com/threejs/reaction-diffusion/).
