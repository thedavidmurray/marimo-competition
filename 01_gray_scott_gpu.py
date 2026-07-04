import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium", app_title="Gray-Scott, on a GPU")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import time

    # ── GPU via JAX, with a graceful NumPy fallback ──────────────────────────
    # On molab, attach the RTX Pro 6000 Blackwell and this runs on-device.
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax

        _dev = jax.devices()[0]
        ON_GPU = _dev.platform == "gpu"
        DEVICE = f"{_dev.platform.upper()} · {getattr(_dev, 'device_kind', _dev.platform)}"
        HAS_JAX = True
    except Exception:
        jax = None
        jnp = np
        lax = None
        ON_GPU = HAS_JAX = False
        DEVICE = "CPU (NumPy) — install jax[cuda] on molab for the GPU path"
    return (
        DEVICE, HAS_JAX, LinearSegmentedColormap, ON_GPU,
        jax, jnp, lax, mo, np, plt, time,
    )


@app.cell
def _(LinearSegmentedColormap):
    # ── Edgeless "bioluminescent" palette — the aesthetic differentiator.
    # Reaction-diffusion is usually shown in magma/viridis. We render it like
    # something alive in deep water: near-black → emerald → lime.
    BIOLUM = LinearSegmentedColormap.from_list(
        "edgeless_biolum",
        ["#05070A", "#08201A", "#0C3A2A", "#2E7D46", "#7DD35F", "#C6F24E", "#F4FFE6"],
    )
    INK = "#C6F24E"      # lime accent
    PAPER = "#09090B"    # near-black ground
    FAINT = "#5B6B4A"    # muted label ink

    def tufte(ax, title=None):
        """Strip an axis to pure data-ink (Tufte): no frame, no ticks."""
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if title:
            ax.set_title(title, color="#D7E0C8", fontsize=9,
                         fontfamily="monospace", pad=4)
        return ax
    return BIOLUM, FAINT, INK, PAPER, tufte


@app.cell
def _(DEVICE, ON_GPU, mo):
    _c = "#C6F24E" if ON_GPU else "#8A94A6"
    mo.md(
        f"""
        # Gray-Scott, on a GPU
        ### Two rules, one nonlinearity, a whole vocabulary of pattern

        **Paper:** Robert P. Munafo, *Stable localized moving patterns in the 2-D
        Gray-Scott model*, [arXiv:1501.01990](https://arxiv.org/abs/1501.01990).
        Building on Pearson, *Complex Patterns in a Simple System*
        ([Science, 1993](https://www.science.org/doi/10.1126/science.261.5118.189)).

        Two chemicals diffuse and react. \\(v\\) is autocatalytic — it eats \\(u\\)
        to make more of itself. That single nonlinearity, \\(uv^2\\), is enough to
        grow spots, stripes, rings, mitosis, and — in a razor-thin band — patterns
        that *move*. Below: the paper's actual equations, integrated live.

        **Device:** <span style="color:{_c}">{DEVICE}</span>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        $$\frac{\partial u}{\partial t} = D_u\nabla^2 u - uv^2 + F(1-u)
        \qquad
        \frac{\partial v}{\partial t} = D_v\nabla^2 v + uv^2 - (F+k)v$$

        The **feed rate** \(F\) replenishes \(u\); the **kill rate** \(k\) drains
        \(v\). Everything interesting lives in that \((F,k)\) plane — so instead of
        making you hunt for it, the map below shows the whole plane at once.
        """
    )
    return


@app.cell
def _(jnp, lax, np):
    # ── The kernel. One function, two backends: jax.lax.scan when JAX is present
    # (jit-compiled, vmap-batchable), a NumPy loop otherwise.
    GRID = 160
    DX = 1.0 / 143.0
    DT = 0.5  # explicit-diffusion stability: DT·Du/DX² ≈ 0.2 < 0.25 ✓
    DU, DV = 2e-5, 1e-5  # sigma = 2, exactly as in the paper

    def _seed(grid, spots=18, seed=7):
        rng = np.random.default_rng(seed)
        u = np.ones((grid, grid), np.float32)
        v = np.zeros((grid, grid), np.float32)
        r = max(3, grid // 22)
        for _ in range(spots):
            cy = int(rng.integers(r, grid - r))
            cx = int(rng.integers(r, grid - r))
            u[cy - r:cy + r, cx - r:cx + r] = 0.5
            v[cy - r:cy + r, cx - r:cx + r] = 0.25
        u += 0.02 * rng.standard_normal((grid, grid)).astype(np.float32)
        v += 0.02 * rng.standard_normal((grid, grid)).astype(np.float32)
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    def _lap(a):
        return (
            jnp.roll(a, 1, 0) + jnp.roll(a, -1, 0)
            + jnp.roll(a, 1, 1) + jnp.roll(a, -1, 1) - 4.0 * a
        ) / (DX * DX)

    def simulate(F, k, n_steps, grid=GRID):
        """Integrate to n_steps and return the v field. GPU when JAX is live."""
        u0, v0 = _seed(grid)
        if lax is not None:
            u0j, v0j = jnp.asarray(u0), jnp.asarray(v0)

            def body(carry, _):
                u, v = carry
                uvv = u * v * v
                u = u + DT * (DU * _lap(u) - uvv + F * (1 - u))
                v = v + DT * (DV * _lap(v) + uvv - (F + k) * v)
                return (u, v), None

            (_, v), _ = lax.scan(body, (u0j, v0j), None, length=int(n_steps))
            return np.asarray(v)
        # NumPy fallback
        u, v = u0.copy(), v0.copy()
        for _ in range(int(n_steps)):
            uvv = u * v * v
            u = u + DT * (DU * _lap(u) - uvv + F * (1 - u))
            v = v + DT * (DV * _lap(v) + uvv - (F + k) * v)
        return np.asarray(v)
    return DT, GRID, simulate


@app.cell
def _(jax, jnp, lax):
    # ── The performance extension: sweep the ENTIRE (F,k) plane in one batched,
    # jit-compiled GPU call via jax.vmap. The paper explores this plane by hand,
    # one run at a time; we run the whole grid of parameters simultaneously.
    GRID_PD = 120
    STEPS_PD = 3200

    def build_sweep(seed_u, seed_v):
        if lax is None:
            return None

        def one(F, k):
            def body(carry, _):
                u, v = carry
                uvv = u * v * v
                lu = (jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0) + jnp.roll(u, 1, 1)
                      + jnp.roll(u, -1, 1) - 4.0 * u) / ((1.0 / 143.0) ** 2)
                lv = (jnp.roll(v, 1, 0) + jnp.roll(v, -1, 0) + jnp.roll(v, 1, 1)
                      + jnp.roll(v, -1, 1) - 4.0 * v) / ((1.0 / 143.0) ** 2)
                u = u + 0.5 * (2e-5 * lu - uvv + F * (1 - u))
                v = v + 0.5 * (1e-5 * lv + uvv - (F + k) * v)
                return (u, v), None
            (_, v), _ = lax.scan(body, (seed_u, seed_v), None, length=STEPS_PD)
            return v

        return jax.jit(jax.vmap(one, in_axes=(0, 0)))
    return GRID_PD, STEPS_PD, build_sweep


@app.cell
def _(mo):
    F = mo.ui.slider(0.010, 0.090, value=0.062, step=0.001,
                     label="feed  F", show_value=True)
    k = mo.ui.slider(0.045, 0.075, value=0.0609, step=0.0005,
                     label="kill  k", show_value=True)
    steps = mo.ui.slider(1000, 6000, value=3500, step=500,
                         label="steps", show_value=True)
    return F, k, steps


@app.cell
def _(BIOLUM, F, k, mo, plt, simulate, steps, tufte):
    # Controls + the field they drive, in ONE frame: move a slider, watch it re-solve live.
    _v = simulate(F.value, k.value, steps.value)
    _fig, _ax = plt.subplots(figsize=(5.6, 5.6))
    _fig.patch.set_facecolor("#09090B")
    _ax.set_facecolor("#09090B")
    _ax.imshow(_v, cmap=BIOLUM, interpolation="bilinear", vmin=0, vmax=_v.max() or 1)
    tufte(_ax)
    _ax.text(0.5, -0.04, f"v(x,y)   F={F.value:.3f}   k={k.value:.4f}",
             transform=_ax.transAxes, ha="center", va="top",
             color="#8FA378", fontsize=9, fontfamily="monospace")
    _fig.tight_layout()
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(
                        "### Steer it\n\nDrift **F** and **k** and watch the vocabulary "
                        "change — spots dissolve into worms, worms into waves. The field "
                        "re-solves live as you drag."
                    ),
                    F, k, steps,
                ],
                gap=1.1, align="stretch",
            ),
            _fig,
        ],
        widths=[1, 1.7], align="center", gap=2,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The whole plane at once — Pearson's atlas, computed in parallel

        Each tile is an *independent* Gray-Scott simulation at a different
        \((F,k)\), all launched in a **single batched GPU call** (`jax.vmap`).
        Read it like a map: the feed rate rises left→right, the kill rate
        rises top→bottom. The named regimes from the paper are marked.
        """
    )
    return


@app.cell
def _(
    BIOLUM, GRID_PD, STEPS_PD, build_sweep, jnp, mo,
    np, plt, simulate, time, tufte,
):
    # A curated diagonal walk through the (F,k) plane — every regime alive,
    # every one visibly distinct. The whole vocabulary in one glance.
    NAMED = [
        (0.018, 0.049, "waves"), (0.022, 0.051, "spots"),
        (0.026, 0.055, "chaos"), (0.030, 0.057, "maze"),
        (0.034, 0.061, "worms"), (0.038, 0.063, "loops"),
        (0.042, 0.063, "coral"), (0.046, 0.065, "fingerprint"),
        (0.050, 0.063, "cells"), (0.054, 0.063, "dense"),
        (0.058, 0.061, "mitosis"), (0.062, 0.061, "solitons"),
    ]
    F_flat = np.array([p[0] for p in NAMED], np.float32)
    k_flat = np.array([p[1] for p in NAMED], np.float32)
    _names = [p[2] for p in NAMED]

    from numpy.random import default_rng as _rng_
    _r = _rng_(7)
    _u0 = np.ones((GRID_PD, GRID_PD), np.float32)
    _v0 = np.zeros((GRID_PD, GRID_PD), np.float32)
    _rad = max(3, GRID_PD // 22)
    for _ in range(18):
        _cy = int(_r.integers(_rad, GRID_PD - _rad))
        _cx = int(_r.integers(_rad, GRID_PD - _rad))
        _u0[_cy - _rad:_cy + _rad, _cx - _rad:_cx + _rad] = 0.5
        _v0[_cy - _rad:_cy + _rad, _cx - _rad:_cx + _rad] = 0.25
    _u0 += 0.02 * _r.standard_normal((GRID_PD, GRID_PD)).astype(np.float32)
    _v0 += 0.02 * _r.standard_normal((GRID_PD, GRID_PD)).astype(np.float32)
    _u0 = np.clip(_u0, 0, 1); _v0 = np.clip(_v0, 0, 1)

    _t0 = time.time()
    _sweep = build_sweep(jnp.asarray(_u0), jnp.asarray(_v0))  # None if JAX absent
    if _sweep is not None:
        _fields = np.asarray(_sweep(jnp.asarray(F_flat), jnp.asarray(k_flat)))
    else:
        _fields = np.stack([simulate(float(a), float(b), STEPS_PD, grid=GRID_PD)
                            for a, b in zip(F_flat, k_flat)])
    _elapsed = time.time() - _t0

    _fig, _axes = plt.subplots(3, 4, figsize=(9.6, 7.4))
    _fig.patch.set_facecolor("#09090B")
    for _i, _ax in enumerate(_axes.ravel()):
        _ax.set_facecolor("#09090B")
        _f = _fields[_i]
        _ax.imshow(_f, cmap=BIOLUM, interpolation="bilinear",
                   vmin=0, vmax=max(_f.max(), 1e-3))
        tufte(_ax)
        _ax.text(0.045, 0.93, _names[_i], transform=_ax.transAxes,
                 color="#C6F24E", fontsize=8.5, fontfamily="monospace",
                 va="top", weight="bold")
        _ax.text(0.045, 0.05, f"F={F_flat[_i]:.3f} k={k_flat[_i]:.3f}",
                 transform=_ax.transAxes, color="#6E7D5A", fontsize=5.5,
                 fontfamily="monospace", va="bottom")
    _fig.suptitle("the vocabulary of Gray-Scott — one diagonal through (F,k)",
                  color="#D7E0C8", fontsize=11, fontfamily="monospace", y=0.98)
    _fig.tight_layout(rect=[0, 0, 1, 0.96])

    _n = len(NAMED)
    _mode = "GPU (single batched vmap)" if _sweep is not None else "CPU (sequential)"
    mo.vstack([
        mo.md(f"**{_n} independent simulations in {_elapsed:.2f}s** — {_mode}. "
              f"That's {_n*STEPS_PD:,} integration steps over "
              f"{_n*GRID_PD*GRID_PD:,} cells."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Where do the beautiful ones live? — an aesthetic search

        Instead of hunting for good \((F,k)\) by hand, we **score the whole plane**.
        Every point below is a full simulation, rated on how *striking* its pattern
        is — the same signal family our pen-plotter studio scores art on. On the GPU
        the entire search is one batched pass; the bright ridge is where Gray-Scott
        actually makes patterns.
        """
    )
    return


@app.cell
def _(np):
    def aesthetic_score(v):
        """Composite 0–1 on the pen-plotter autoresearch signals: coverage, visual
        entropy, edge/structure density, composition variation, and figure-ground
        contrast (the last one kills washed-out, saturated fields)."""
        if not np.isfinite(v).all():
            return 0.0
        vn = (v - v.min()) / (np.ptp(v) + 1e-9)
        cov = float((vn > 0.35).mean())
        cov_s = 1.0 - abs(cov - 0.45) / 0.45
        hist, _ = np.histogram(vn, bins=32, range=(0, 1), density=True)
        p = hist / (hist.sum() + 1e-9)
        ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(32))
        gx = np.abs(np.roll(vn, -1, 1) - vn)
        gy = np.abs(np.roll(vn, -1, 0) - vn)
        edge_s = min(1.0, float(np.hypot(gx, gy).mean()) / 0.14)
        _h = (vn.shape[0] // 8) * 8
        b = vn[:_h, :_h].reshape(8, _h // 8, 8, _h // 8).mean(axis=(1, 3))
        comp_s = min(1.0, float(b.std() / (b.mean() + 1e-9)) / 0.6)
        vmax = float(v.max())
        lo = float((v < 0.45 * vmax).mean())
        hi = float((v > 0.70 * vmax).mean())
        contrast = min(1.0, 4.0 * lo * hi)
        if vn.std() < 0.06 or contrast < 0.08:
            return 0.02
        return float(0.16 * cov_s + 0.24 * ent + 0.24 * edge_s
                     + 0.12 * comp_s + 0.24 * contrast)
    return (aesthetic_score,)


@app.cell
def _(
    GRID_PD, ON_GPU, aesthetic_score, build_sweep, jnp, mo,
    np, plt, simulate, time,
):
    # Fine grid on GPU (one batched vmap pass), coarse on CPU so it still completes.
    _NF, _NK = (26, 22) if ON_GPU else (9, 7)
    _Fs = np.linspace(0.014, 0.066, _NF)
    _ks = np.linspace(0.045, 0.069, _NK)
    _FF, _KK = np.meshgrid(_Fs, _ks)
    _Ff = _FF.ravel().astype(np.float32)
    _kf = _KK.ravel().astype(np.float32)

    from numpy.random import default_rng as _rng2
    _r2 = _rng2(7)
    _u2 = np.ones((GRID_PD, GRID_PD), np.float32)
    _v2 = np.zeros((GRID_PD, GRID_PD), np.float32)
    _rd2 = max(3, GRID_PD // 22)
    for _ in range(18):
        _cy = int(_r2.integers(_rd2, GRID_PD - _rd2))
        _cx = int(_r2.integers(_rd2, GRID_PD - _rd2))
        _u2[_cy - _rd2:_cy + _rd2, _cx - _rd2:_cx + _rd2] = 0.5
        _v2[_cy - _rd2:_cy + _rd2, _cx - _rd2:_cx + _rd2] = 0.25
    _u2 += 0.02 * _r2.standard_normal((GRID_PD, GRID_PD)).astype(np.float32)
    _v2 += 0.02 * _r2.standard_normal((GRID_PD, GRID_PD)).astype(np.float32)
    _u2 = np.clip(_u2, 0, 1); _v2 = np.clip(_v2, 0, 1)

    _t2 = time.time()
    _sw = build_sweep(jnp.asarray(_u2), jnp.asarray(_v2))
    if _sw is not None:
        _flds = np.asarray(_sw(jnp.asarray(_Ff), jnp.asarray(_kf)))
    else:
        _flds = np.stack([simulate(float(a), float(b), 3200, grid=GRID_PD)
                          for a, b in zip(_Ff, _kf)])
    _scores = np.array([aesthetic_score(f) for f in _flds])
    _land = _scores.reshape(_NK, _NF)
    _el2 = time.time() - _t2

    _order = np.argsort(-_scores)
    _picks = []
    for _i in _order:
        if _scores[_i] < 0.4:
            break
        if all(abs(_Ff[_i] - _Ff[j]) > 0.004 or abs(_kf[_i] - _kf[j]) > 0.003
               for j in _picks):
            _picks.append(int(_i))
        if len(_picks) >= 10:
            break

    _fig, _ax = plt.subplots(figsize=(7.4, 6.0))
    _fig.patch.set_facecolor("#09090B"); _ax.set_facecolor("#09090B")
    _ax.imshow(_land, origin="lower", aspect="auto", cmap="magma",
               extent=(_Fs[0], _Fs[-1], _ks[0], _ks[-1]))
    for _i in _picks:
        _ax.scatter(_Ff[_i], _kf[_i], s=26, facecolors="none",
                    edgecolors="#C6F24E", linewidths=1.3)
    _ax.set_xlabel("feed rate F", color="#8FA378", fontfamily="monospace", fontsize=9)
    _ax.set_ylabel("kill rate k", color="#8FA378", fontfamily="monospace", fontsize=9)
    _ax.tick_params(colors="#5B6B4A", labelsize=7)
    _ax.set_title("aesthetic score landscape", color="#D7E0C8",
                  fontfamily="monospace", fontsize=10)
    _fig.tight_layout()

    _mode2 = "GPU · one batched vmap" if _sw is not None else "CPU · sequential"
    mo.vstack([
        mo.md(f"**{len(_scores)} parameter sets searched in {_el2:.1f}s** ({_mode2}). "
              f"Each lime ring is a local maximum of striking-ness; the bright ridge is "
              f"the pattern-forming band, everything else decays to a uniform state. "
              f"*Searching the aesthetics of a PDE is the custom extension — and it only "
              f"stays interactive because the whole plane is one GPU call.*"),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Why this is a GPU story

        The atlas above is `jax.vmap(simulate)` — the *same* kernel the single
        run uses, mapped over an array of \((F,k)\) pairs and compiled once. On
        the Blackwell every tile integrates **in parallel**; on CPU they'd run
        one after another. The paper explored this plane by hand, a run at a
        time. Here the plane is a single call — which means you can treat "what
        does the whole parameter space *look* like" as an interactive question,
        not a weekend of compute.

        ---
        **The Edgeless loop:** real paper → real simulation → each field is also a
        level-set the pen plotter can draw. The explainer earns credibility; the
        plot is a print you can sell; both point back to *"want your paper rendered
        like this?"* — [edgelesslab.com](https://edgelesslab.com).
        """
    )
    return


if __name__ == "__main__":
    app.run()
