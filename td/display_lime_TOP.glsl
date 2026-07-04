// Edgeless bioluminescent display — TouchDesigner GLSL TOP
// Wire the reaction-diffusion state (or any single-channel field) into input 0.
// Maps the v channel through the near-black -> emerald -> lime ramp.

out vec4 fragColor;

vec3 ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.020, 0.027, 0.039);
    vec3 c1 = vec3(0.031, 0.125, 0.102);
    vec3 c2 = vec3(0.047, 0.227, 0.165);
    vec3 c3 = vec3(0.180, 0.490, 0.275);
    vec3 c4 = vec3(0.490, 0.827, 0.373);
    vec3 c5 = vec3(0.776, 0.949, 0.306);
    vec3 c6 = vec3(0.957, 1.000, 0.902);
    if (t < 0.16) return mix(c0, c1, t / 0.16);
    if (t < 0.34) return mix(c1, c2, (t - 0.16) / 0.18);
    if (t < 0.52) return mix(c2, c3, (t - 0.34) / 0.18);
    if (t < 0.72) return mix(c3, c4, (t - 0.52) / 0.20);
    if (t < 0.88) return mix(c4, c5, (t - 0.72) / 0.16);
    return mix(c5, c6, (t - 0.88) / 0.12);
}

void main() {
    float v = texture(sTD2DInputs[0], vUV.st).g;   // v = green channel of the RD state
    vec3 col = ramp(pow(v * 3.4, 0.9));
    fragColor = TDOutputSwizzle(vec4(col, 1.0));
}
