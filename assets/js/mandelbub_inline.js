let theShader;
let startTime;

// Inline GLSL source as template strings:
const vertSrc = `
attribute vec3 aPosition;
void main() {
  gl_Position = vec4(aPosition, 1.0);
}
`;

const fragSrc = `#ifdef GL_ES
precision highp float;
#endif

uniform float uTime;
uniform vec2  uResolution;
uniform float uPixelDensity;
uniform float uRotAngle;
uniform float uDistance;

#define M_PI  3.1415926535897932384626433832795
#define M_INF 1e10

#define MARCHING_ITERATIONS       64
#define MANDELBULB_MAXDIST       1.5 
#define MANDELBULB_MAXITERATIONS   4
#define PHONG_SHADING              0
#define ADVANCED_COLOR             0

float sdSphere(vec3 p, float s) {
  return length(p) - s;
}

vec3 mandelbulbColor(float dr) {
#if !ADVANCED_COLOR
  return vec3(dr / 400000.0);
#else
  float c = 1.5 * pow(dr, 1.0 / 20.0);
  vec3 color = sin(2.0 * c + vec3(0.0, 0.0, 1.0));
  return 0.5 + 0.5 * color;
#endif
}

vec3 mandelbulbStep(vec3 zeta, float n) {
  float r = length(zeta);
  float theta = acos(zeta.z / r);
  float phi = atan(zeta.y, zeta.x);
  r = pow(r, n);
  theta = theta * n;
  phi = phi * n;
  return vec3(
    r * sin(theta) * cos(phi),
    r * sin(theta) * sin(phi),
    r * cos(theta)
  );
}

float mandelbulbDerivative(vec3 zeta, float n, float dr) {
  return n * pow(length(zeta), n - 1.0) * dr + 1.0;
}

vec4 sdMandelbulb(vec3 pos, float n) {
  vec3 zeta = pos;
  float dr = 1.0;
  float r = 0.0;

  for (int i = 0; i < MANDELBULB_MAXITERATIONS; i++) {
    r = length(zeta);
    if (r > MANDELBULB_MAXDIST) break;
    dr = mandelbulbDerivative(zeta, n, dr);
    zeta = mandelbulbStep(zeta, n) + pos;
  }

  vec3  c = mandelbulbColor(dr);
  float t = 0.5 * log(r) * r / dr;
  return vec4(c, t);
}

vec4 select(vec4 d1, vec4 d2) {
  return (d1.w < d2.w) ? d1 : d2;
}

vec4 mapScene(vec3 p) {
  vec4 sample0 = vec4(vec3(0.0), M_INF);
  vec4 sample1 = sdMandelbulb(p, 8.0);
  return select(sample0, sample1);
}

vec3 renderShaded(vec3 pos, vec3 dir, vec3 baseColor) {
  vec2 e = vec2(1.0, -1.0) * 0.015;
  vec3 normal = normalize(
    e.xyy * mapScene(pos + e.xyy).w +
    e.yyx * mapScene(pos + e.yyx).w +
    e.yxy * mapScene(pos + e.yxy).w +
    e.xxx * mapScene(pos + e.xxx).w
  );
  vec3 lightDir = normalize(vec3(-0.5, 0.4, -0.6));
  float Ka = 0.05; vec3 La = baseColor;
  float Kd = 1.0; vec3 Ld = baseColor;
  float Ks = 1.0; vec3 Ls = vec3(1.0);
  float shininess = 32.0;
  vec3 N = normalize(normal);
  vec3 L = normalize(lightDir);
  vec3 R = reflect(-L, N);
  vec3 V = normalize(-dir);
  float fd = max(dot(N, L), 0.0);
  float fs = pow(max(dot(R, V), 0.0), shininess);
  return Ka * La + Kd * fd * Ld + Ks * fs * Ls;
}

vec4 trace(vec3 org, vec3 dir) {
  float t = 0.0;
  vec3  c = vec3(0.0);
  for (int i = 0; i < MARCHING_ITERATIONS; i++) {
    vec3 pos = org + dir * t;
    vec4 res = mapScene(pos);
    c = res.xyz;
    t += res.w;
  }
  return vec4(c, t);
}

mat3 calculateCamera() {
  return mat3(
     cos(uRotAngle),  0.0,  sin(uRotAngle),
                0.0,  1.0,             0.0,
    -sin(uRotAngle),  0.0,  cos(uRotAngle)
  );
}

vec4 sampleScene(vec2 uv) {
  mat3 rotMatrix = calculateCamera();
  vec3 org = rotMatrix * uDistance * vec3(0.0, 0.0, -1.0);
  vec3 dir = rotMatrix * normalize(vec3(uv, 1.0));
  vec4 res = trace(org, dir);
  vec3 c = res.xyz;
  float t = res.w;
  if (t >= 100.0) return vec4(vec3(0.0), t);
  #if PHONG_SHADING
    c = renderShaded(org+t*dir, dir, c);
  #endif
  return vec4(c, t);
}

void main() {
  vec2 fragCoord = gl_FragCoord.xy / uPixelDensity;
  vec2 uv = (2.0 * fragCoord - uResolution) / uResolution.y;
  vec4 result = sampleScene(uv);
  vec3 color = pow(result.xyz, vec3(0.5));
  gl_FragColor = vec4(color, 1.0);
}
`;

function setup() {
  createCanvas(windowWidth, windowHeight, WEBGL);
  noStroke();
  startTime = millis();
  theShader = createShader(vertSrc, fragSrc);
}

function draw() {
  shader(theShader);
  theShader.setUniform('uTime', (millis() - startTime) / 1000.0);
  theShader.setUniform('uResolution', [width, height]);
  theShader.setUniform('uPixelDensity', pixelDensity());
  theShader.setUniform('uRotAngle', millis() / 10000.0);
  theShader.setUniform('uDistance', 3.0);
  rect(-width / 2, -height / 2, width, height);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
