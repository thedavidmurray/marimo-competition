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
