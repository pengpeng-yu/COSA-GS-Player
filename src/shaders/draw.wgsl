@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> keys: array<vec2<u32>>;
struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) centerConic: vec4<f32>,
  @location(1) @interpolate(flat) colorOpacity: vec4<f32>,
  @location(2) @interpolate(flat) conicY: f32,
}
@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOut {
  let splat = splats[keys[instance].y];
  let corners = array<vec2<f32>, 6>(vec2<f32>(0,0), vec2<f32>(1,0), vec2<f32>(0,1),
    vec2<f32>(0,1), vec2<f32>(1,0), vec2<f32>(1,1));
  let pixel = mix(splat.bounds.xy, splat.bounds.zw, corners[vertex]);
  var result: VertexOut;
  result.position = vec4<f32>(pixel.x / frame.intrinsics.x * 2.0 - 1.0,
                              1.0 - pixel.y / frame.intrinsics.y * 2.0, 0.0, 1.0);
  result.centerConic = splat.centerConic;
  result.colorOpacity = splat.colorOpacity;
  result.conicY = splat.shape.x;
  return result;
}
@fragment
fn fs(input: VertexOut) -> @location(0) vec4<f32> {
  let d = input.position.xy - vec2<f32>(0.5) - input.centerConic.xy;
  let power = -0.5 * (input.centerConic.z * d.x * d.x + input.conicY * d.y * d.y)
              - input.centerConic.w * d.x * d.y;
  let alpha = min(0.99, input.colorOpacity.w * exp(power));
  if (power > 0.0 || alpha < 1.0 / 255.0) { discard; }
  return vec4<f32>(input.colorOpacity.xyz * alpha, alpha);
}
