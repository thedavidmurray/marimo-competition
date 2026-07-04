# TouchDesigner Reskin Plan — Edgeless Lime Gray-Scott + Sprott Attractors

**Goal:** a TouchDesigner (TD) piece that renders the *same* two generative systems as the marimo competition
entries (`01_gray_scott_gpu.py`, `02_strange_attractors_gpu.py`), skinned in Edgeless-lime instead of the
existing Hermes navy/indigo hero, for a future Marimo ↔ TouchDesigner ↔ Three.js field note (same math, three
renderers).

**Source of truth for the math** (copied verbatim from the marimo entries so TD matches parameter-for-parameter):

- Gray-Scott: `GRID=160`, `DX=1/143`, `DT=0.5`, `DU=2e-5`, `DV=1e-5` (σ=2). Update:
  `u += DT·(DU·∇²u − u·v² + F·(1−u))`, `v += DT·(DV·∇²v + u·v² − (F+k)·v)`. Sliders: `F∈[.min,.max]` (feed),
  `k` (kill), `steps∈[1000,6000]`.
- Sprott 2-D quadratic map: 12 coefficients `a[0..11]` each in `[-1.2,1.2]` (step 0.1, letter-coded A–Y):
  `x' = a0 + a1·x + a2·x² + a3·xy + a4·y + a5·y²`
  `y' = a6 + a7·x + a8·x² + a9·xy + a10·y + a11·y²`
  Rendered as a density histogram (`res=620`), Gaussian-blurred (`blur=0.6`), gamma-lifted (`gamma=0.42`),
  normalized to [0,1].
- Palette (**BIOLUM**, both pieces): `["#05070A","#08201A","#0C3A2A","#2E7D46","#7DD35F","#C6F24E","#F4FFE6"]`.
  Ink/accent: `#C6F24E`. Muted mono annotation greens: `#8FA378` / `#6E7D5A`.

**Existing TD asset to re-skin:** `/Users/djm/Desktop/NewProject.1.toe`, network at `/project1/hero_base/`.
Per `touchdesigner-hero-project.md`, it's currently a screen-blend compositing chain (noise → color-grade →
torus wireframe → text overlays → bloom → scanlines/vignette → chromatic aberration → `hero_out` nullTOP) in
Hermes navy/indigo. That chain is a *good compositing shell* (bloom/scanline/vignette/CA stages are
palette-agnostic) but its *generator* stages (`bg_noise`, `noise_var_blend`, `swarm_render` torus) are the
wrong systems — they need to be swapped for Gray-Scott and Sprott generators, not just recolored. Plan: fork
a new .toe from the compositing shell rather than mutate the hero in place, since the hero is still the
Hermes-branded deliverable.

---

## 1. GPU Reaction-Diffusion in TD (feedback-loop chain)

TD's native way to do Gray-Scott is a **Feedback TOP** ping-pong loop around a **GLSL TOP**, not
CHOP math — this runs the whole simulation on GPU every frame at real-time rates.

**Operator chain (`/project1/gray_scott/`):**

```
gs_seed        (Noise TOP, GPU type=Sparse→ NO, use type=Random via rendered circleTOP splats instead;
                or a Ramp/Circle TOP composite: paint ~18 filled circles at random positions,
                R channel=u≈0.5, G channel=v≈0.25, matches _seed() in the marimo file)
     ↓ (first-frame only, via a Switch TOP gated by absTime.frame<2 or a Logic CHOP "reset" pulse)
gs_feedback    (Feedback TOP) — holds (u,v) as RG channels of a 160×160 (or upres to 512×512 for TD's
                sharper output — GRID is a knob, not fixed) two-channel image
     ↓
gs_step        (GLSL TOP) — one Gray-Scott Euler step per cook, reading gs_feedback as sTD2DInputs[0]:
                samples 4-neighbor Laplacian (vec4 via textureOffset), computes:
                  float uvv = u*v*v;
                  u += DT*(DU*lap_u - uvv + F*(1.0-u));
                  v += DT*(DV*lap_v + uvv - (F+k)*v);
                DU/DV/DT as uniforms (2e-5, 1e-5, 0.5) fed from Custom Parameters on a container COMP
                (so F/k/steps-per-frame are exposed exactly like the marimo sliders).
     ↓ (feeds back into gs_feedback input 1 — this IS the ping-pong: Feedback TOP's own output
        becomes next frame's input, no separate double-buffer op needed)
gs_field_out   (Null TOP) — the raw v-channel field, analogous to `simulate()`'s return value
```

**Key TD specifics:**
- Feedback TOP is the idiomatic ping-pong primitive — one operator IS the two buffers (it caches its own
  last output). Do NOT build a manual A/B swap with two TOPs + a Switch; Feedback TOP already does this
  and cooks once per frame regardless of downstream reads.
- Run the GLSL step **multiple times per displayed frame** if you want to match marimo's 1000–6000 step
  counts without waiting minutes of real-time playback: either (a) accept TD's continuous real-time
  evolution as its own aesthetic (the diffusion visibly grows — arguably *more* alive than marimo's static
  endpoint), or (b) use a Render Pass / feedback loop count via a `for`-loop in a Script TOP wrapping
  multiple GLSL TOP cooks per frame (CHOP Execute DAT `onFrameStart` calling `op('gs_step').cook(force=True)`
  N times) if an exact single-shot match to a marimo `steps` value is required for a side-by-side export.
- Reset/reseed: wire a Button/Pulse CHOP → Logic CHOP → drives a Switch TOP between `gs_seed` and
  `gs_feedback`'s current state, so re-randomizing spots doesn't require restarting TD.
- Resolution: keep GLSL sim resolution at 160×160–256×256 (matches marimo GRID; cheap), upscale only in a
  final Resize/TOP *after* colorization for crisp export resolution (1920×1080+).

## 2. Sprott Strange Attractor in TD (GPU compute → instanced points)

Two viable TD paths; recommend (b) for GPU parity with the marimo JAX path.

**(a) CHOP/SOP iterative (simpler, CPU-bound, fine for ≤50k points):**
```
sprott_coeffs  (Constant CHOP, 12 channels a0..a11 as Custom Parameters on a container — the "DNA" knobs,
                letter-codeable exactly like marimo's ALPHA scheme)
     ↓
sprott_iter    (Script CHOP) — runs the quadratic map in Python for n=20k-90k iterations at cook time
                (mirrors `orbit()` in 02_strange_attractors_gpu.py, minus the Lyapunov/shadow-trajectory
                bookkeeping — that's a search-time diagnostic, not needed for a fixed known-good attractor
                code), outputs two channels (x,y) as a timeslice=False static array (per mcp-patterns.md
                §"noiseCHOP Timeslice": False = static array, ideal for instancing)
     ↓
sprott_sop     (CHOP to SOP) — one point per orbit sample (x,y,0), `par.chop` reference per known gotcha
     ↓
sprott_instance (Geometry COMP with instancing, instanceop=sprott_sop) — tiny point/sprite geometry
                (per mcp-patterns.md §"3D Particle Instancing": COMP scale=1, keep the SOP geometry itself
                tiny, use P(0)/P(1)/P(2) attrs from choptoSOP, NOT channel names)
     ↓
sprott_render  (Render TOP, orthographic camera framed to the orbit's bounding box)
```

**(b) GPU compute (matches marimo's JAX vmap parity, scales to millions of points):**
```
sprott_state   (GLSL TOP, RG32F, e.g. 1024×1024 = ~1M parallel orbit instances) — each texel holds one
                (x,y) state; a compute-style GLSL TOP applies the quadratic map per-texel per frame
                (same a0..a11 uniforms), Feedback-TOP-looped exactly like the Gray-Scott step —
                this is "Gray-Scott's ping-pong pattern, reused for point-map iteration instead of PDE diffusion"
     ↓
sprott_splat   (Point Sprite / GLSL TOP with additive/Screen accumulation) — histogram-style density
                accumulation: read sprott_state, scatter each texel's (x,y) into a density-accumulation
                buffer via a second Feedback TOP that ADDS a soft-splat kernel at each point's screen
                position (approximates marimo's `np.histogram2d` + gaussian_filter density image)
     ↓
sprott_gamma   (Level TOP / GLSL TOP) — gamma=0.42 lift + normalize-to-max, matching `render()` in the
                marimo file
```
Path (a) is the pragmatic MVP (ships fastest, CPU cost is trivial for one fixed attractor at ~50-90k
points); path (b) is the "true GPU parity" version worth building for the field note's comparison narrative
(same order-of-magnitude particle counts as the marimo GPU search). Start with (a), note (b) as a stretch
goal in Risks below.

## 3. Re-skinning to Edgeless Lime

Re-skin is a **palette swap of the compositing shell**, done by editing exactly the operators that touch
color — the structural chain (bloom/vignette/CA/scanlines) stays:

| Hero chain operator | Current (Hermes navy/indigo) | Reskin action |
|---|---|---|
| `color_grade` (Level TOP, triadic blue/red/green expressions on `absTime.seconds`) | 3-way triadic phase-offset hue cycle | **Replace** with a Ramp TOP driven by the BIOLUM 7-stop gradient (`#05070A→#08201A→#0C3A2A→#2E7D46→#7DD35F→#C6F24E→#F4FFE6`) used as a **Lookup TOP** keyed off the Gray-Scott `v` field intensity (0→1 maps along the ramp) — this directly reproduces marimo's `LinearSegmentedColormap` behavior in TD. Build the ramp once as a 256×1 Ramp TOP with 7 keyframes at the BIOLUM hex stops, feed it into a Lookup TOP with `gs_field_out` as the index input. |
| `hermes_scr` text overlay ("E D G E L E S S") | flat white | keep flat, but text color → `#FAFAFA` (near-white, per Edgeless surface convention, not pure `#FFFFFF`) |
| `edgeless_text` ("L A B") | flat white | same — `#FAFAFA` |
| `hermes_sub` ("powered by Hermes", muted blue 16px) | rename/recolor: text → "gray-scott · sprott — edgeless field notes", color `#8FA378` (muted mono-green annotation, Tufte direct-label convention) |
| `timestamp_text` ("SWARM PROTOCOL v0.6") | muted blue | color → `#6E7D5A`, content → run params as direct labels (`F=0.0367 k=0.0649` or attractor code, not a legend) |
| `bloom_pulse` (oscillating brightness on bloom_comp) | tuned for navy | re-tune bloom threshold/gain so bloom fires on the `#C6F24E` lime highs specifically (lime sits at the 5th BIOLUM stop — high luminance channel — bloom threshold ~0.75 on the ramp-mapped output catches only the lime/white filament tips, giving the "glowing lime filaments on near-black" look) |
| `swarm_render` (wireframe torus) | **Delete/bypass** — replaced structurally by `gs_field_out` (RD piece) or `sprott_render` (attractor piece); this op is Hermes-specific geometry, not part of either target system |
| `crt_vignette` (circleTOP) | keep as-is (fillcolor stays near-black `#09090B`, vignette is palette-neutral) |
| `chroma_red/green/blue` (channel-split CA) | keep structurally; tune split magnitude down slightly since lime-on-black CA reads busier than white-on-navy CA — test at half the current ±0.003 offset first |
| Background/canvas base | wherever a flat background color constant exists (e.g. a Constant TOP under the composite stack) | set to Edgeless `bg` `#09090B` exactly (not the old navy) |

**New Tufte-compliance pass (not in the original hero, required for this piece):** remove any residual
frame/grid decoration on the render camera; no legend box for the colormap — instead burn the F/k values
(or attractor code) directly as monospace text next to the field, per the annotation convention above. Font:
set every `textTOP.par.fonttype` to `"Menlo"` or `"Monaco"` (mono), consistent with the marimo pieces'
`fontfamily="monospace"`.

## 4. Headless drive + export via the skill

Per `~/.hermes/skills/touchdesigner-generative-art/SKILL.md`, the render path is `hermes touchdesigner`,
which shells to `TOUCHDESIGNER_PATH` (`/Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner`) with a
`.toe` + frame range, non-interactively (`-q`/quiet GUI flag under the hood). Exact commands:

```bash
# one-time: confirm the CLI sees the local install
hermes touchdesigner check

# Gray-Scott loop, matching marimo's default slider values as a documented baseline
hermes touchdesigner render \
  --project /Users/djm/Desktop/edgeless-gray-scott.toe \
  --output  /Users/djm/Desktop/edgeless-gray-scott.mp4 \
  --frames 300 --fps 30 \
  --params "F=0.0367,k=0.0649,steps_per_frame=8" \
  --resolution 1920x1080

# Sprott attractor loop — pick a known-good bounded attractor code from
# marimo-competition/_search_results.json rather than re-searching in TD
hermes touchdesigner render \
  --project /Users/djm/Desktop/edgeless-sprott.toe \
  --output  /Users/djm/Desktop/edgeless-sprott.mp4 \
  --frames 300 --fps 30 \
  --params "attractor_code=<12-letter code from _search_results.json>" \
  --resolution 1920x1080
```

GIF conversion (matches the existing hero's `.gif` deliverable pattern at
`/Users/djm/Desktop/edgeless-lab-hero.gif`):
```bash
ffmpeg -i /Users/djm/Desktop/edgeless-gray-scott.mp4 \
  -vf "fps=24,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  -loop 0 /Users/djm/Desktop/edgeless-gray-scott.gif
```

**Parameter injection mechanism** (per skill's "Project Structure" section): the .toe's container COMP
exposes `F`, `k`, `steps_per_frame` / `attractor_code` as Custom Parameters; a **File In DAT** inside the
project watches a JSON param file the CLI writes before launch — do not build a bespoke IPC path, reuse this
existing convention so both new .toe files stay consistent with any other TD template in the skill's library.

**MCP-driven build alternative** (per `touchdesigner-mcp-patterns.md`): if building the operator network
programmatically rather than by hand in the TD GUI, drive it via the TwoZero MCP JSON-RPC bridge
(`http://localhost:40404/mcp`), using the `td()` curl helper documented there, with `td_create_operator` /
`td_execute_python` calls. Respect the hard-won gotchas in that memory file verbatim: GPU-only noise types,
Screen-blend instead of alpha for renderTOP compositing, `par.period` not `periodx/periody`, `par.chop`
reference (not inputConnectors) for CHOP-to-SOP, COMP scale=1 with tiny SOP geometry for instancing.

## 5. Next steps + risks

**Next steps (in order):**
1. Duplicate `/Users/djm/Desktop/NewProject.1.toe` → `edgeless-gray-scott.toe`, strip to the compositing
   shell only (delete `swarm_render` torus + old noise generators), keep bloom/vignette/CA/text stages.
2. Build the Feedback-TOP + GLSL-TOP Gray-Scott loop (Section 1), wire its `v`-field output into the kept
   Ramp/Lookup TOP palette stage (Section 3), verify visually against a marimo render at matching F/k before
   touching text/branding.
3. Repeat for a second file `edgeless-sprott.toe` using CHOP/SOP path 1(a) first (fastest to a working
   render); revisit GPU compute path 1(b) only if the field note explicitly wants a particle-count callout.
4. Re-skin text/annotation layers (Section 3 table), confirm Tufte pass (no gridlines/legend/frame).
5. Wire Custom Parameters + File In DAT for headless param injection; smoke-test `hermes touchdesigner
   render` end-to-end on both files at low frame count (30 frames) before committing to full 300-frame runs.
6. Export MP4→GIF pair for each system; hand both + the two marimo notebooks to the field-note draft
   (Marimo ↔ TouchDesigner ↔ Three.js comparison).

**Risks:**
- **Feedback TOP + multi-step-per-frame** is the main unknown: TD's real-time cook model wants one GLSL
  eval per frame, but marimo runs 1000–6000 explicit steps per rendered image. Sub-looping the GLSL TOP N
  times per frame via a Script/Execute DAT is the workaround (noted in Section 1) but is unverified
  perf-wise at N≈8-20; test early, not at the end.
- **GPU compute path for Sprott (1b)** is genuinely novel TD work (no prior pattern in the mcp-patterns
  memory) — treat as R&D, not a scheduled deliverable; ship path (a) first.
- **renderTOP alpha is broken** (documented gotcha) — every new render pass must use Screen blend, same as
  the existing hero; do not attempt Over/alpha compositing on the new field-render outputs.
- **Bloom retuning for lime-on-black** is untested; lime at `#C6F24E` is less luminance-separated from
  near-black than white was from navy in the old piece — may need threshold tuning to avoid either no-bloom
  or blown-out oversaturation.
- **No headless verification in this session** — TouchDesigner.app is installed but this plan was produced
  without opening/running it (per task instructions); Section 4's exact CLI flags should be smoke-tested
  against the actual `hermes touchdesigner --help` output before the first real render, in case the skill's
  documented flags have drifted from the installed CLI version.
