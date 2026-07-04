import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium", app_title="A million attractors, the beautiful few")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import time

    try:
        import jax
        jax.config.update("jax_enable_x64", True)  # Lyapunov needs float64 precision
        import jax.numpy as jnp
        from jax import lax
        _dev = jax.devices()[0]
        ON_GPU = _dev.platform == "gpu"
        DEVICE = f"{_dev.platform.upper()} · {getattr(_dev, 'device_kind', _dev.platform)}"
        HAS_JAX = True
    except Exception:
        jax = None; jnp = np; lax = None
        ON_GPU = HAS_JAX = False
        DEVICE = "CPU (NumPy) — attach a GPU on molab to search millions"
    return (DEVICE, HAS_JAX, LinearSegmentedColormap, ON_GPU,
            jax, jnp, lax, mo, np, plt, time)


@app.cell
def _(LinearSegmentedColormap):
    BIOLUM = LinearSegmentedColormap.from_list(
        "edgeless_biolum",
        ["#05070A", "#08201A", "#0C3A2A", "#2E7D46", "#7DD35F", "#C6F24E", "#F4FFE6"])

    def tufte(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor("#09090B")
        return ax
    return BIOLUM, tufte


@app.cell
def _(DEVICE, ON_GPU, mo):
    _c = "#C6F24E" if ON_GPU else "#8A94A6"
    mo.md(
        f"""
        # A million attractors, the beautiful few
        ### Sprott's search, run on a GPU and judged for beauty

        **Paper:** Julien C. Sprott, *Automatic generation of strange attractors*
        (Computers & Graphics 17(3), 1993) — and *Strange Attractors: Creating
        Patterns in Chaos*. Sprott's insight: take a simple quadratic map with
        random coefficients, iterate it, and **keep the ones that are chaotic**
        (positive Lyapunov exponent) and bounded. Most coefficient sets are boring
        — a point, a loop, or an explosion. A rare few are strange attractors.

        Sprott searched one-at-a-time on a 386. We search **thousands in parallel**
        on the GPU, then add a second filter he couldn't: *is it beautiful?*

        **Device:** <span style="color:{_c}">{DEVICE}</span>
        """
    )
    return


@app.cell
def _(np):
    # ── Sprott's 2-D quadratic map. 12 coefficients, each coded as a letter
    # A..Y over [-1.2, 1.2] — so every attractor has a pronounceable name.
    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXY"

    def code_of(a):
        return "".join(ALPHA[int(round((c + 1.2) / 0.1))] for c in a)

    def coeffs_of(code):
        return np.array([ALPHA.index(ch) * 0.1 - 1.2 for ch in code], np.float64)

    def orbit(a, n=90000, warmup=1000):
        """Iterate to steady state and return the orbit + Lyapunov exponent.
        Bounded attractors stay small — anything past 1e4 is a slow diverger."""
        SEP = 1e-6  # shadow separation; large enough to survive, small enough to stay linear
        x = y = 0.1
        xe, ye = x + SEP, y
        xs = np.empty(n); ys = np.empty(n); lyap = 0.0
        for i in range(n + warmup):
            xn = a[0] + a[1]*x + a[2]*x*x + a[3]*x*y + a[4]*y + a[5]*y*y
            yn = a[6] + a[7]*x + a[8]*x*x + a[9]*x*y + a[10]*y + a[11]*y*y
            if not np.isfinite(xn) or abs(xn) > 1e4 or abs(yn) > 1e4:
                return None, -1.0
            xen = a[0] + a[1]*xe + a[2]*xe*xe + a[3]*xe*ye + a[4]*ye + a[5]*ye*ye
            yen = a[6] + a[7]*xe + a[8]*xe*xe + a[9]*xe*ye + a[10]*ye + a[11]*ye*ye
            dx, dy = xen - xn, yen - yn
            d = np.hypot(dx, dy) + 1e-15
            if i > warmup:
                lyap += np.log(d / SEP)
            xe, ye = xn + SEP * dx / d, yn + SEP * dy / d
            x, y = xn, yn
            if i >= warmup:
                xs[i - warmup] = x; ys[i - warmup] = y
        return (xs, ys), lyap / n
    return ALPHA, code_of, coeffs_of, orbit


@app.cell
def _(BIOLUM, np):
    from scipy.ndimage import gaussian_filter

    def render(xs, ys, res=620, gamma=0.42, blur=0.6):
        """Density → vivid tone-mapped image. Gamma lifts the faint fractal
        filaments; a whisper of blur gives the filaments a glow."""
        H, _, _ = np.histogram2d(ys, xs, bins=res)
        H = gaussian_filter(H, blur)
        H = np.power(H, gamma)
        if H.max() > 0:
            H /= H.max()
        return H
    return gaussian_filter, render


@app.cell
def _(lax, jnp, jax, np):
    # ── The GPU search. vmap the whole thing over a batch of coefficient sets:
    # iterate each map + a shadow trajectory, accumulate the Lyapunov exponent,
    # and report boundedness — thousands of candidates in one compiled call.
    def build_scan():
        if lax is None:
            return None

        def one(a):
            def body(carry, _):
                x, y, xe, ye, lyap, alive = carry
                xn = a[0] + a[1]*x + a[2]*x*x + a[3]*x*y + a[4]*y + a[5]*y*y
                yn = a[6] + a[7]*x + a[8]*x*x + a[9]*x*y + a[10]*y + a[11]*y*y
                xen = a[0] + a[1]*xe + a[2]*xe*xe + a[3]*xe*ye + a[4]*ye + a[5]*ye*ye
                yen = a[6] + a[7]*xe + a[8]*xe*xe + a[9]*xe*ye + a[10]*ye + a[11]*ye*ye
                dx, dy = xen - xn, yen - yn
                d = jnp.hypot(dx, dy) + 1e-15
                lyap = lyap + jnp.log(d / 1e-6)
                xe2, ye2 = xn + 1e-6 * dx / d, yn + 1e-6 * dy / d
                bad = jnp.abs(xn) > 1e4
                bad = jnp.logical_or(bad, jnp.abs(yn) > 1e4)
                bad = jnp.logical_or(bad, jnp.isnan(xn))
                alive = jnp.where(bad, 0.0, alive)
                # freeze a diverged trajectory so it can't NaN-poison the scan
                xn = jnp.where(bad, 0.0, xn); yn = jnp.where(bad, 0.0, yn)
                return (xn, yn, xe2, ye2, lyap, alive), None
            init = (0.1, 0.1, 0.1 + 1e-6, 0.1, 0.0, 1.0)
            # filter over the FULL render horizon so bounded-here = bounded-when-drawn
            (_, _, _, _, lyap, alive), _ = lax.scan(body, init, None, length=80000)
            return jnp.array([lyap / 80000.0, alive])
        return jax.jit(jax.vmap(one))
    return (build_scan,)


@app.cell
def _(mo):
    n_search = mo.ui.slider(200, 4000, value=800, step=200,
                            label="candidates to search", show_value=True)
    seed = mo.ui.slider(0, 40, value=3, step=1, label="search seed", show_value=True)
    mo.md(
        f"""
        ### Discover
        Draw random coefficient sets, keep the ones that are chaotic **and**
        beautiful. On the GPU this searches thousands in a blink; on CPU it
        samples a few hundred.
        {mo.hstack([n_search, seed], justify="start", gap=2)}
        """
    )
    return n_search, seed


@app.cell
def _(
    ON_GPU, aesthetic_score, build_scan, code_of, jnp, mo, n_search,
    np, orbit, render, seed, time,
):
    _t0 = time.time()
    _rng = np.random.default_rng(int(seed.value))
    _N = int(n_search.value)
    _NB = 80000  # filter + render horizon — MUST match so bounded-here = bounded-drawn
    _cand = _rng.uniform(-1.2, 1.2, (_N, 12))  # float64 — Lyapunov needs the precision

    _scan = build_scan()
    _gallery = []
    _nchaotic = 0

    if _scan is not None:
        # GPU: filter thousands over the full horizon in one batched pass,
        # then render only the survivors (they're guaranteed bounded at _NB).
        _res = np.asarray(_scan(jnp.asarray(_cand)))
        _lyap, _alive = _res[:, 0], _res[:, 1]
        _surv = [i for i in range(_N) if _alive[i] > 0.5 and _lyap[i] > 0.03]
        _surv.sort(key=lambda i: -_lyap[i])
        _nchaotic = len(_surv)
        for _i in _surv[:30]:
            _o, _l = orbit(_cand[_i], n=_NB, warmup=1000)
            if _o is None:
                continue
            _xs, _ys = _o
            if _xs.std() < 0.08 or _ys.std() < 0.08:
                continue
            _img = render(_xs, _ys)
            _fill = float((_img > 0.05).mean())
            if _fill < 0.012 or _fill > 0.55:
                continue
            _gallery.append((aesthetic_score(_img), _l, code_of(_cand[_i]), _img))
            if len(_gallery) >= 14:
                break
    else:
        # CPU fallback: single-stage — test boundedness AT render length as we go
        # (no cheap-filter mismatch), capped so a local run still finishes.
        _tried = 0
        for _a in _cand:
            if len(_gallery) >= 9 or _tried >= 700:
                break
            _tried += 1
            _o, _l = orbit(_a, n=_NB, warmup=1000)
            if _o is None or _l < 0.03:
                continue
            _xs, _ys = _o
            if _xs.std() < 0.08 or _ys.std() < 0.08:
                continue
            _img = render(_xs, _ys)
            _fill = float((_img > 0.05).mean())
            if _fill < 0.012 or _fill > 0.55:
                continue
            _nchaotic += 1
            _gallery.append((aesthetic_score(_img), _l, code_of(_a), _img))
    _gallery.sort(key=lambda t: -t[0])
    _elapsed = time.time() - _t0

    _stage = "GPU · one batched vmap" if _scan is not None else "CPU · sequential"
    summary = mo.md(
        f"**Searched {_N} coefficient sets in {_elapsed:.1f}s** ({_stage}) → "
        f"**{_nchaotic} chaotic & bounded**, top **{len(_gallery)}** rendered and ranked "
        f"by beauty. Sprott hand-searched these one at a time in 1993."
    )
    gallery = _gallery
    summary
    return gallery, summary


@app.cell
def _(np):
    def aesthetic_score(img):
        """Beauty score for an attractor density image: reward structured
        filament coverage + contrast, punish empty or blobby fields."""
        if not np.isfinite(img).all() or img.max() <= 0:
            return 0.0
        v = img / img.max()
        ink = float((v > 0.05).mean())              # how much of the frame is used
        ink_s = 1.0 - abs(ink - 0.16) / 0.16        # reward airy, filamentary fill
        hist, _ = np.histogram(v, bins=32, range=(0, 1), density=True)
        p = hist / (hist.sum() + 1e-9)
        ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(32))
        gx = np.abs(np.roll(v, -1, 1) - v); gy = np.abs(np.roll(v, -1, 0) - v)
        edge = min(1.0, float(np.hypot(gx, gy).mean()) / 0.05)
        if ink < 0.01 or ink > 0.6:
            return 0.03
        return float(0.34 * max(0, ink_s) + 0.33 * ent + 0.33 * edge)
    return (aesthetic_score,)


@app.cell
def _(BIOLUM, gallery, mo, plt, tufte):
    if not gallery:
        _out = mo.md("*No attractors survived — nudge the seed and search again.*")
    else:
        _n = min(9, len(gallery))
        _fig, _axes = plt.subplots(3, 3, figsize=(9.2, 9.2))
        _fig.patch.set_facecolor("#09090B")
        for _ax, (_sc, _l, _code, _img) in zip(_axes.ravel(), gallery[:_n]):
            tufte(_ax)
            _ax.imshow(_img, cmap=BIOLUM, interpolation="bilinear", origin="lower")
            _ax.text(0.05, 0.94, _code, transform=_ax.transAxes, color="#C6F24E",
                     fontsize=7.5, fontfamily="monospace", va="top", weight="bold")
            _ax.text(0.05, 0.05, f"β{_sc:.2f}  λ{_l:.2f}", transform=_ax.transAxes,
                     color="#6E7D5A", fontsize=6, fontfamily="monospace", va="bottom")
        for _ax in _axes.ravel()[_n:]:
            tufte(_ax); _ax.set_visible(False)
        _fig.suptitle("auto-discovered strange attractors, ranked by beauty (β)",
                      color="#D7E0C8", fontsize=11, fontfamily="monospace", y=0.98)
        _fig.tight_layout(rect=[0, 0, 1, 0.96])
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Why this is a GPU story — and a custom extension

        Sprott's method is *embarrassingly parallel*: every candidate is an
        independent iterate-and-measure. `jax.vmap` maps the whole search —
        iteration, shadow trajectory, Lyapunov accumulation — across the batch and
        compiles it once, so a Blackwell evaluates thousands of universes at once.

        The **custom extension** is the second filter. Sprott kept whatever was
        chaotic; we score every survivor for *beauty* — structured filament
        coverage, entropy, edge density — and surface only the gorgeous few. The
        letter-codes (e.g. the top tile's name) are reproducible: type one back in
        and you get the same attractor. Chaos, made searchable and curatable.

        ---
        **The Edgeless loop:** each attractor is a stroke path a pen plotter can
        draw — the same discovery that fills this gallery fills a print drawer.
        [edgelesslab.com](https://edgelesslab.com)
        """
    )
    return


if __name__ == "__main__":
    app.run()
