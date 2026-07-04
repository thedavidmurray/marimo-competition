// Gray-Scott reaction-diffusion — TouchDesigner GLSL TOP (pixel shader)
// Wire as: [Feedback TOP] -> [this GLSL TOP] -> back into the Feedback TOP.
// State is stored in R,G = (u, v). Set the GLSL TOP + Feedback TOP to 32-bit float RGBA.
//
// Custom uniforms (GLSL TOP -> Vectors 1 page):
//   uF  (float)  feed rate   ~0.037
//   uK  (float)  kill  rate   ~0.060
//   uSeed (vec3) x,y = seed position (0..1), z = 1.0 while seeding (e.g. from a Mouse CHOP)
//   uSeedR (float) seed radius ~0.03
// Karl-Sims parameterization (Du=0.16, Dv=0.08, dt=1) — stable in realtime.

out vec4 fragColor;

uniform float uF;
uniform float uK;
uniform vec3  uSeed;
uniform float uSeedR;

void main()
{
    vec2 res   = uTD2DInfos[0].res.zw;   // width, height
    vec2 texel = 1.0 / res;
    vec2 uv    = vUV.st;

    vec2 c = texture(sTD2DInputs[0], uv).rg;

    // 9-point Laplacian (orthogonal 1.0, diagonal 0.5, center -6), normalized.
    vec2 l =
          texture(sTD2DInputs[0], uv + vec2(-texel.x, 0.0)).rg
        + texture(sTD2DInputs[0], uv + vec2( texel.x, 0.0)).rg
        + texture(sTD2DInputs[0], uv + vec2(0.0, -texel.y)).rg
        + texture(sTD2DInputs[0], uv + vec2(0.0,  texel.y)).rg
        + 0.5 * ( texture(sTD2DInputs[0], uv + vec2(-texel.x, -texel.y)).rg
                + texture(sTD2DInputs[0], uv + vec2( texel.x, -texel.y)).rg
                + texture(sTD2DInputs[0], uv + vec2(-texel.x,  texel.y)).rg
                + texture(sTD2DInputs[0], uv + vec2( texel.x,  texel.y)).rg )
        - 6.0 * c;
    l /= 6.0;

    float u = c.x, v = c.y, uvv = u * v * v;
    vec2 n = c + vec2( 0.16 * l.x - uvv + uF * (1.0 - u),
                       0.08 * l.y + uvv - (uF + uK) * v );

    // inject v where the seed input is active
    if (uSeed.z > 0.5 && distance(uv, uSeed.xy) < uSeedR) {
        n.y = 0.5;
        n.x = 0.2;
    }

    n = clamp(n, 0.0, 1.0);
    fragColor = TDOutputSwizzle(vec4(n, 0.0, 1.0));
}
