const FUSED_DISPATCH_WIDTH = 256u;

struct Frame {
  view: mat4x4<f32>,
  camera: vec4<f32>, // world position, distance_scale
  intrinsics: vec4<f32>, // width, height, fx, fy
  anchorCount: u32, // remaining uniform bytes are alignment padding
}

struct Anchor {
  position: vec4<f32>, // xyz, exp(log_position_scaling)
  scale: vec4<f32>, // exp(log_gaussian_scaling), padding
  features: array<vec4<f32>, 8>,
}

struct ActiveOffset {
  position: vec3<f32>,
  slot: u32, // Original anchor * 10 + offset index. Keep integer bits integer.
}

struct Control {
  vertices: u32,
  count: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
  diagnostics: array<u32>, // populated only by the validation pipeline
}

struct Splat {
  centerConic: vec4<f32>, // center.xy, inverse covariance xx, xy
  colorOpacity: vec4<f32>,
  shape: vec4<f32>, // inverse covariance yy
  bounds: vec4<f32>, // tile-aligned pixel bounds
}

@group(0) @binding(8) var<uniform> frame: Frame;

fn cameraPoint(world: vec3<f32>) -> vec3<f32> {
  return (frame.view * vec4<f32>(world, 1.0)).xyz;
}

fn imagePoint(p: vec3<f32>) -> vec2<f32> {
  // COSA/CUDA ndc2Pix: pixel centers have integer coordinates.
  return frame.intrinsics.zw * p.xy / (p.z + 1e-7) + (frame.intrinsics.xy - 1.0) * 0.5;
}

fn covariance2d(p: vec3<f32>, covariance: mat3x3<f32>) -> vec3<f32> {
  let focal = frame.intrinsics.zw;
  let limit = 1.3 * frame.intrinsics.xy / (2.0 * focal);
  let slope = clamp(p.xy / p.z, -limit, limit);
  let viewRotation = mat3x3<f32>(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz);
  let jx = transpose(viewRotation) * vec3<f32>(focal.x / p.z, 0.0, -focal.x * slope.x / p.z);
  let jy = transpose(viewRotation) * vec3<f32>(0.0, focal.y / p.z, -focal.y * slope.y / p.z);
  return vec3<f32>(
    dot(jx, covariance * jx) + 0.3,
    dot(jx, covariance * jy),
    dot(jy, covariance * jy) + 0.3,
  );
}

fn tileBounds(center: vec2<f32>, cov: vec3<f32>) -> vec4<f32> {
  let det = cov.x * cov.z - cov.y * cov.y;
  let mid = 0.5 * (cov.x + cov.z);
  let radius = ceil(3.0 * sqrt(mid + sqrt(max(0.1, mid * mid - det))));
  let grid = vec2<i32>(ceil(frame.intrinsics.xy / 16.0));
  // Integer conversion intentionally truncates toward zero, matching getRect.
  let lo = clamp(vec2<i32>((center - radius) / 16.0), vec2<i32>(0), grid);
  let hi = clamp(vec2<i32>((center + radius + 15.0) / 16.0), vec2<i32>(0), grid);
  return vec4<f32>(vec2<f32>(lo) * 16.0, vec2<f32>(hi) * 16.0);
}

fn nonempty(bounds: vec4<f32>) -> bool {
  return bounds.z > bounds.x && bounds.w > bounds.y;
}
