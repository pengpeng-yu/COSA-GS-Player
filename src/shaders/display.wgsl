override EXPLICIT_RGB8: bool = true;
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
  let points = array<vec2<f32>, 3 > (vec2<f32>( - 1, -1), vec2<f32>(3, -1), vec2<f32>( - 1, 3));
  return vec4<f32>(points[vertex], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  // COSA RGB values are displayed directly: no tone mapping or gamma transform.
  var rgb = clamp(textureLoad(source, vec2<i32>(position.xy), 0).rgb, vec3<f32>(0), vec3<f32>(1));
  if (EXPLICIT_RGB8) {
    rgb = floor(rgb * 255.0 + 0.5) / 255.0;
  }
  return vec4<f32>(rgb, 1.0);
}
