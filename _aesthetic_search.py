#!/usr/bin/env python3
"""Aesthetic autoresearch over the Gray-Scott (F,k) plane.

Same signal family as pen-plotter-art/autoresearch/scoring.py (coverage,
visual entropy, edge/structure density, composition variation) — computed
in-memory on the v-field so we can score hundreds of candidates fast.

Searches the parameter plane, scores every candidate on how *striking* its
pattern is, and selects a top set that is both high-scoring AND diverse (so
the atlas shows range, not twelve near-identical mazes).

Output: marimo-competition/_search_results.json  +  a score-landscape PNG.
"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BIOLUM = LinearSegmentedColormap.from_list(
    "b", ["#05070A", "#08201A", "#0C3A2A", "#2E7D46", "#7DD35F", "#C6F24E", "#F4FFE6"])

G = 96
DX = 1 / 143.0
DT = 0.5
DU, DV = 2e-5, 1e-5

# fixed seed so every candidate starts identically → differences are pure (F,k)
_r = np.random.default_rng(7)
_U0 = np.ones((G, G), np.float32)
_V0 = np.zeros((G, G), np.float32)
_rad = max(3, G // 22)
for _ in range(16):
    cy = int(_r.integers(_rad, G - _rad)); cx = int(_r.integers(_rad, G - _rad))
    _U0[cy - _rad:cy + _rad, cx - _rad:cx + _rad] = 0.5
    _V0[cy - _rad:cy + _rad, cx - _rad:cx + _rad] = 0.25
_U0 += 0.02 * _r.standard_normal((G, G)).astype(np.float32)
_V0 += 0.02 * _r.standard_normal((G, G)).astype(np.float32)
_U0 = np.clip(_U0, 0, 1); _V0 = np.clip(_V0, 0, 1)


def _lap(a):
    return (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1)
            + np.roll(a, -1, 1) - 4 * a) / (DX * DX)


def simulate(F, k, n=2600):
    u, v = _U0.copy(), _V0.copy()
    for _ in range(n):
        uvv = u * v * v
        u = u + DT * (DU * _lap(u) - uvv + F * (1 - u))
        v = v + DT * (DV * _lap(v) + uvv - (F + k) * v)
    return v


def score(v):
    """Aesthetic composite 0-1. Mirrors the pen-plotter scorer's signals."""
    if not np.isfinite(v).all():
        return 0.0, {}
    vn = (v - v.min()) / (np.ptp(v) + 1e-9)
    # 1. coverage — fraction 'active'; reward the mid-band, punish empty/saturated
    cov = float((vn > 0.35).mean())
    cov_score = 1.0 - abs(cov - 0.45) / 0.45
    # 2. visual entropy — histogram Shannon entropy, normalized
    hist, _ = np.histogram(vn, bins=32, range=(0, 1), density=True)
    p = hist / (hist.sum() + 1e-9)
    ent = -(p * np.log(p + 1e-12)).sum() / np.log(32)
    # 3. edge/structure density — mean gradient magnitude (Sobel-ish)
    gx = np.abs(np.roll(vn, -1, 1) - vn); gy = np.abs(np.roll(vn, -1, 0) - vn)
    edge = float(np.hypot(gx, gy).mean())
    edge_score = min(1.0, edge / 0.14)
    # 4. composition variation — CoV of local means over an 8x8 grid (asymmetry)
    b = vn.reshape(8, G // 8, 8, G // 8).mean(axis=(1, 3))
    comp = float(b.std() / (b.mean() + 1e-9))
    comp_score = min(1.0, comp / 0.6)
    # 5. figure-ground contrast — striking patterns are BIMODAL (clear bright
    # structures on a dark ground). Washed-out/saturated fields are unimodal;
    # this term kills them even when their entropy/edges look busy.
    vmax = float(v.max())
    lo_frac = float((v < 0.45 * vmax).mean())   # background
    hi_frac = float((v > 0.70 * vmax).mean())   # structures
    contrast_score = min(1.0, 4.0 * lo_frac * hi_frac)  # peaks when both ~0.5
    # 6. degeneracy guard — near-uniform fields score ~0
    if vn.std() < 0.06 or contrast_score < 0.08:
        return 0.02, {"cov": cov, "ent": ent, "edge": edge, "comp": comp}
    composite = (0.16 * cov_score + 0.24 * ent + 0.24 * edge_score
                 + 0.12 * comp_score + 0.24 * contrast_score)
    return float(composite), {"cov": round(cov, 3), "ent": round(float(ent), 3),
                              "edge": round(edge, 3), "comp": round(comp, 3),
                              "contrast": round(contrast_score, 3),
                              "std": round(float(vn.std()), 3)}


def main():
    t0 = time.time()
    # fine sweep of the plane
    Fs = np.linspace(0.014, 0.066, 27)
    ks = np.linspace(0.045, 0.069, 25)
    results = []
    land = np.zeros((len(ks), len(Fs)))
    for i, k in enumerate(ks):
        for j, F in enumerate(Fs):
            v = simulate(float(F), float(k))
            s, sig = score(v)
            land[i, j] = s
            results.append({"F": round(float(F), 4), "k": round(float(k), 4),
                            "score": round(s, 4), **sig})
    results.sort(key=lambda r: -r["score"])

    # diverse top set: greedily pick high scorers that are far apart in (F,k)
    chosen = []
    for r in results:
        if r["score"] < 0.4:
            break
        if all((abs(r["F"] - c["F"]) > 0.004 or abs(r["k"] - c["k"]) > 0.003)
               for c in chosen):
            chosen.append(r)
        if len(chosen) >= 12:
            break

    json.dump({"searched": len(results), "elapsed_s": round(time.time() - t0, 1),
               "top12_diverse": chosen, "top20_raw": results[:20]},
              open("marimo-competition/_search_results.json", "w"), indent=2)

    # score landscape
    fig, ax = plt.subplots(figsize=(7.5, 6.4)); fig.patch.set_facecolor("#09090B")
    ax.set_facecolor("#09090B")
    im = ax.imshow(land, origin="lower", aspect="auto", cmap="magma",
                   extent=[Fs[0], Fs[-1], ks[0], ks[-1]])
    for c in chosen:
        ax.scatter(c["F"], c["k"], s=28, facecolors="none",
                   edgecolors="#C6F24E", linewidths=1.4)
    ax.set_xlabel("feed rate F", color="#8FA378", fontfamily="monospace", fontsize=9)
    ax.set_ylabel("kill rate k", color="#8FA378", fontfamily="monospace", fontsize=9)
    ax.tick_params(colors="#5B6B4A", labelsize=7)
    ax.set_title(f"aesthetic score landscape · {len(results)} candidates searched",
                 color="#D7E0C8", fontfamily="monospace", fontsize=10)
    fig.tight_layout(); fig.savefig("marimo-competition/_score_landscape.png",
                                    dpi=115, facecolor="#09090B")
    print(f"searched {len(results)} in {time.time()-t0:.1f}s | "
          f"top score {results[0]['score']:.3f} @ F={results[0]['F']} k={results[0]['k']} | "
          f"{len(chosen)} diverse picks")
    for c in chosen:
        print(f"  {c['score']:.3f}  F={c['F']:.4f} k={c['k']:.4f}  "
              f"ent={c['ent']} edge={c['edge']} comp={c['comp']}")


if __name__ == "__main__":
    main()
