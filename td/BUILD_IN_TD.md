# Edgeless Gray-Scott in TouchDesigner — build guide

The lime reaction-diffusion from the phone demo / Entry 1, running as a realtime TD network.
~10 min in the TD UI. Two shaders live next to this file: `gray_scott_TOP.glsl`, `display_lime_TOP.glsl`.

## The network (GPU ping-pong)

```
[Constant/Noise TOP: seed] ─┐
                            ▼
                    [Feedback TOP] ──► [GLSL TOP: gray_scott] ──► (loop back into Feedback TOP)
                                                │
                                                ▼
                                       [GLSL TOP: display_lime] ──► [Bloom/Level] ──► Out
```

## Steps

1. **Two GLSL TOPs.** Create a `GLSL TOP`, set its **Pixel Shader** page → paste `gray_scott_TOP.glsl`
   (or point it at a Text DAT holding the file). Name it `gray_scott`. Make a second GLSL TOP
   `display_lime` with `display_lime_TOP.glsl`.

2. **32-bit float.** On `gray_scott` AND the Feedback TOP: Common page → **Pixel Format = 32-bit float (RGBA)**.
   Resolution ~1024×1024 (or 720² for lighter GPUs). RD needs float precision — 8-bit will smear.

3. **Feedback loop.** Create a `Feedback TOP` named `rd_feedback`. Set its **Target TOP = gray_scott**.
   Wire: `rd_feedback` → input 0 of `gray_scott`. That closes the ping-pong (each cook = one RD step).

4. **Seed it.** Make a `Constant TOP` (or Noise/Circle) that's u=1,v=0 with a few v-blobs, wire into the
   Feedback TOP's **first frame**, or just left-click-drag a Circle TOP into it to paint. On `gray_scott`
   set custom uniforms (Vectors 1 page): `uF=0.037`, `uK=0.060`, `uSeedR=0.03`, `uSeed=(0.5,0.5,0)`.
   For touch/mouse seeding, drive `uSeed` from a **Mouse In CHOP** (xy → uSeed.xy, button → uSeed.z).

5. **Sub-step for speed.** RD needs ~10 steps/frame to move at a nice rate. Either raise the Feedback/GLSL
   cook rate, or put the `gray_scott` step in a small **loop** (a Replicator or a `for` in a Script TOP)
   — 8–12 iterations per frame.

6. **Display + bloom.** Wire `gray_scott` → `display_lime` → a **Bloom TOP** (or Blur+Composite Add) →
   a subtle **Vignette** (Level/Ramp mask). That's the glow that makes it luminesce, matching the phone demo.

## Re-skinning the existing hero (`~/Desktop/NewProject.1.toe`)

Per `../TD_RESKIN_PLAN.md`: keep the bloom/vignette/CA/scanline compositing shell; **replace** the navy
triadic color-grade + Hermes torus/text with this RD network + lime ramp. The one palette constant to
change everywhere: Hermes navy `#818CF8`-ish → Edgeless lime **#C6F24E** (0.776, 0.949, 0.306).

## Headless render (from the skill)

`~/.hermes/skills/touchdesigner-generative-art` runs `.toe` files headlessly + exports. Once the network
looks right, use its render command to bake a GIF/MP4 loop for the field note.

---
Test in TD, paste me any GLSL/console errors, and I'll fix the shaders. The GLSL is adapted from the
verified WebGL demo (`edgelesslab.com/threejs/reaction-diffusion/`), so the math is proven — only TD's
GLSL-TOP conventions (`sTD2DInputs`, `uTD2DInfos`, `TDOutputSwizzle`) are new surface area.
