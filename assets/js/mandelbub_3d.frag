#ifdef GL_ES
precision highp float;
#endif

// port of
//   https://www.shadertoy.com/view/wdjGWR
// base raymarching rendering thing
//   https://www.youtube.com/watch?v=yxNnRSefK94
// fractal by
//   http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
// http://www.imajeenyus.com/mathematics/20121112_distance_estimates/point_to_isoline.pdf

uniform float uTime;
uniform vec2  uResolution;
uniform float uPixelDensity;
uniform float uRotAngleY;
uniform float uRotAngleX;
uniform float uDistance;
varying vec2 vTexCoord;

#define M_PI  3.1415926535897932384626433832795
#define M_INF 1e10

#define MARCHING_ITERATIONS       64
#define MANDELBULB_MAXDIST       1.5 
#define MANDELBULB_MAXITERATIONS   4
#define PHONG_SHADING              0 // turns Phong shading on/off
#define ADVANCED_COLOR             0 // switches grayscale/colorful


// calculates the distance to a sphere
float sdSphere(vec3 p, float s) {
  return length(p) - s;
}



// maps the run length of a sample to a color
vec3 mandelbulbColor(float dr) {
#if !ADVANCED_COLOR
  // simple grayscale coloring
  return vec3(dr / 400000.0);
#else
  // advanced coloring
  float c = 1.5 * pow(dr, 1.0 / 20.0);
  vec3 color = sin(2.0 * c + vec3(0.0, 0.0, 1.0));
  return 0.5 + 0.5 * color;
#endif
}

// calculates one iteration of the mandelbulb
vec3 mandelbulbStep(vec3 zeta, float n) {
  // convert to polar coordinates
  float r     = length(zeta);
  float theta = acos(zeta.z / r);
  float phi   = atan(zeta.y, zeta.x);
  // scale and rotate the point
  r     = pow(r, n);
  theta = theta * n;
  phi   = phi * n;
  // convert back to cartesian coordinates
  return vec3(
    r * sin(theta) * cos(phi),
    r * sin(theta) * sin(phi),
    r * cos(theta)
  );
}

// calculates one iteration of the run length of the mandelbulb
//  z_new  =     z^(n)        + c
//  z_new' = n * z^(n-1) * z' + 1
float mandelbulbDerivative(vec3 zeta, float n, float dr) {
  return n * pow(length(zeta), n - 1.0) * dr + 1.0;
}

// calculates the distance to the mandelbulb set
//  xyz : the color
//    w : the distance to the set
vec4 sdMandelbulb(vec3 pos, float n) {
  vec3  zeta = pos;
  float dr   = 1.0;
  float r    = 0.0;

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



// select the smallest of 2 samples
vec4 select(vec4 d1, vec4 d2) {
	return (d1.w < d2.w) ? d1 : d2;
}

// mapping function
//  xyz : the color
//    w : the distance of the nearest object in the direction p
vec4 mapScene(vec3 p) {
  // start with an "infinitly" distant sample with back color
  vec4 sample0 = vec4(vec3(0.0), M_INF);

  vec4 sample1 = sdMandelbulb(p, 8.0);

  // vec4 sample2 = vec4(
  //   vec3(1.0,1.0,0.0), 
  //   sdSphere(p-vec3( 1.75, 0.0, 0.0), 0.5)
  // );

  // vec4 sample3 = vec4(
  //   vec3(1.0,0.0,0.0), 
  //   sdSphere(p-vec3(-1.75, 0.0, 0.0), 0.5)
  // );
  
  return select(sample0, sample1);

  // return select(sample0, select(sample1, select(sample2, sample3)));

  // res = select(res, );
  
  // res = select(res, vec4(
  //   vec3(1.0,0.0,0.0), 
  //   sdSphere(p-vec3(-1.75, 0.0, 0.0), 0.5)
  // ));
  
  // res = select(res, vec4(
  //   materialColor, 
  //   sdf(p - objOrigin, ...objParameter)
  // ));

  // return res;
}



// calculates the shaded color at a given position
vec3 renderShaded(vec3 pos, vec3 dir, vec3 baseColor) {
  // calculate normal by sampling in 4 directions
  vec2 e = vec2(1.0, -1.0) * 0.015;
  vec3 normal = normalize(
    e.xyy * mapScene(pos + e.xyy).w +
    e.yyx * mapScene(pos + e.yyx).w +
    e.yxy * mapScene(pos + e.yxy).w +
    e.xxx * mapScene(pos + e.xxx).w
  );
  
  // define the direction of the light
  vec3 lightDir  = normalize(vec3(-0.5,  0.4, -0.6));
  
  // shading constants
  float Ka = 0.05;        // ambient intensity
  vec3  La = baseColor;   // ambient color
  float Kd = 1.0;         // diffuse intensity
  vec3  Ld = baseColor;   // diffuse color
  float Ks = 1.0;         // specular intesity
  vec3  Ls = vec3(1.0);   // specular color
  float shininess = 32.0; // specular size

  // define 
  vec3 N = normalize(normal);   // face normal
  vec3 L = normalize(lightDir); // light direction
  vec3 R = reflect(-L, N);      // reflected light vector
  vec3 V = normalize(-dir);     // vector to viewer

  // calculate factors
  float fd = max(dot(N, L), 0.0);
  float fs = pow(max(dot(R, V), 0.0), shininess);
  
  return 
    Ka * La +
    Kd * fd * Ld +
    Ks * fs * Ls;
}



// measuring the distance to the nearest object
vec4 trace(vec3 org, vec3 dir) {
  float t = 0.0;
  vec3  c = vec3(0.0);

  for (int i = 0; i < MARCHING_ITERATIONS; i++) {
    vec3 pos  = org + dir * t;
    vec4 res = mapScene(pos);
    c  = res.xyz;
    t += res.w;
  }

  return vec4(c, t);
}

// returns the current camera to world matrix
mat3 calculateCamera() {
  // Rotation autour de Y (horizontale)
  mat3 rotY = mat3(
     cos(uRotAngleY),  0.0,  sin(uRotAngleY),
                 0.0,  1.0,              0.0,
    -sin(uRotAngleY),  0.0,  cos(uRotAngleY)
  );
  
  // Rotation autour de X (verticale)
  mat3 rotX = mat3(
    1.0,               0.0,                0.0,
    0.0,  cos(uRotAngleX), -sin(uRotAngleX),
    0.0,  sin(uRotAngleX),  cos(uRotAngleX)
  );
  
  // Combiner les deux rotations
  return rotY * rotX;
}

// samples the scene at the given coordinate
vec4 sample(vec2 uv) {
  // rotation matrix for rotating the camera
  mat3 rotMatrix = calculateCamera();

  // ray origin & direction
  vec3 org = rotMatrix * uDistance * vec3(0.0, 0.0, -1.0);
  vec3 dir = rotMatrix * normalize(vec3(uv,  1.0));

  // tracing the ray
  vec4 res = trace(org, dir);
  // splitting the result into 
  // its components color and distance
  vec3  c = res.xyz;
  float t = res.w;

  // cut-off distance to remove artefacts
  if (t >= 100.0) {
    return vec4(vec3(0.0), t);
  }
  
  #if PHONG_SHADING
    c = renderShaded(org+t*dir, dir, c);
  #endif

  return vec4(c, t);
}


void main() {
  vec2 uv = (vTexCoord * 2.0 - 1.0) * vec2(uResolution.x / uResolution.y, 1.0);
  
  vec4 result = sample(uv);
  vec3  color = result.xyz;
  float depth = result.w;

  color = pow(color, vec3(0.5));
  gl_FragColor = vec4(color, 1.0);
}
