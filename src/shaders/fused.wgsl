// Fixed COSA model layout. Mirrored by layout.ts and export-web-scene.py.
const FEATURE_COUNT = 32u;
const OFFSET_COUNT = 10u;
const HEAD_COUNT = 3u;
const INPUT_COUNT = FEATURE_COUNT + 4u;
const OPACITY_CHANNELS = OFFSET_COUNT;
const COLOR_CHANNELS = OFFSET_COUNT * 3u;
const COV_CHANNELS = OFFSET_COUNT * 7u;
const COLOR_START = OPACITY_CHANNELS;
const COV_START = COLOR_START + COLOR_CHANNELS;
const OUTPUT_CHANNELS = COV_START + COV_CHANNELS;
const SAMPLE_STRIDE = 1u + OUTPUT_CHANNELS;
const HIDDEN_CHANNELS = HEAD_COUNT * FEATURE_COUNT;
const ANCHORS_PER_WORKGROUP = 4u;
const FIRST_BIAS_START = INPUT_COUNT * FEATURE_COUNT;
const SECOND_WEIGHT_START = FIRST_BIAS_START + FEATURE_COUNT;
const SECOND_BIAS_START = SECOND_WEIGHT_START + COV_CHANNELS * FEATURE_COUNT;
const WEIGHTS_PER_HEAD = SECOND_BIAS_START + COV_CHANNELS + 2u;

// The production pipeline specializes this to false. Validation uses the same
// arithmetic with sampling enabled and a temporary control buffer.
override RECORD_MLP: bool = false;

struct AnchorMetadata {
  offsetStart: u32,
  slotMask: u32,
}

@group(0) @binding(0) var<storage, read> anchors: array<Anchor>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> metadata: array<AnchorMetadata>;
@group(0) @binding(3) var<storage, read> worklist: array<u32>;
@group(0) @binding(4) var<storage, read> offsets: array<ActiveOffset>;
@group(0) @binding(5) var<storage, read_write> control: Control;
@group(0) @binding(6) var<storage, read_write> splats: array<Splat>;
@group(0) @binding(7) var<storage, read_write> keys: array<vec2<u32>>;

var<workgroup> hidden: array<f32, ANCHORS_PER_WORKGROUP * HIDDEN_CHANNELS>;
var<workgroup> outputs: array<f32, ANCHORS_PER_WORKGROUP * OUTPUT_CHANNELS>;

fn firstLayer(
  anchor: u32,
  neuron: u32,
  head: u32,
  direction: vec3<f32>,
  distance: f32,
) -> f32 {
  let base = head * WEIGHTS_PER_HEAD;
  var sum = 0.0;
  for (var channel = 0u; channel < FEATURE_COUNT; channel++) {
    sum += anchors[anchor].features[channel / 4u][channel % 4u]
         * weights[base + channel * FEATURE_COUNT + neuron];
  }
  for (var axis = 0u; axis < 3u; axis++) {
    sum += direction[axis] * weights[base + (FEATURE_COUNT + axis) * FEATURE_COUNT + neuron];
  }
  if (head != 1u) {
    sum += distance * weights[base + (INPUT_COUNT - 1u) * FEATURE_COUNT + neuron];
  }
  return max(sum + weights[base + FIRST_BIAS_START + neuron], 0.0);
}

fn headOutput(localAnchor: u32, head: u32, row: u32) -> f32 {
  let base = head * WEIGHTS_PER_HEAD;
  var sum = 0.0;
  for (var channel = 0u; channel < FEATURE_COUNT; channel++) {
    sum += hidden[localAnchor * HIDDEN_CHANNELS + head * FEATURE_COUNT + channel]
         * weights[base + SECOND_WEIGHT_START + row * FEATURE_COUNT + channel];
  }
  sum += weights[base + SECOND_BIAS_START + row];
  if (head == 0u) {
    return tanh(sum);
  }
  if (head == 1u) {
    return 1.0 / (1.0 + exp(-sum));
  }
  return sum;
}

fn sigmoid(value: vec3<f32>) -> vec3<f32> {
  return 1.0 / (1.0 + exp(-value));
}

fn projectSplat(
  i: u32,
  xyz: vec3<f32>,
  anchorScale: vec3<f32>,
  opacity: f32,
  color: vec3<f32>,
  covariance: array<f32, 7>,
) {
  let p = cameraPoint(xyz);
  if (p.z <= 0.2) {
    return;
  }

  let scale = anchorScale * sigmoid(vec3<f32>(covariance[0], covariance[1], covariance[2]));
  var q = vec4<f32>(covariance[3], covariance[4], covariance[5], covariance[6]);
  q /= max(length(q), 1e-12); // F.normalize. Quaternion order is w, x, y, z.
  let w = q.x;
  let x = q.y;
  let y = q.z;
  let z = q.w;
  let rotation = mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
  );
  let transform = mat3x3<f32>(rotation[0] * scale.x, rotation[1] * scale.y, rotation[2] * scale.z);
  let cov = covariance2d(p, transform * transpose(transform));
  let det = cov.x * cov.z - cov.y * cov.y;
  if (det <= 0.0) {
    return;
  }

  // alpha >= 1/255 implies d^T inv(cov) d <= 2 log(255 opacity).
  // One pixel of conservative padding covers pixel-center and rounding error.
  let center = imagePoint(p);
  let tile = tileBounds(center, cov);
  let cutoff = 2.0 * log(max(255.0 * opacity, 1.0));
  let extent = sqrt(cutoff * vec2<f32>(cov.x, cov.z)) + vec2<f32>(1.0);
  let bounds = vec4<f32>(
    max(tile.xy, floor(center - extent)),
    min(tile.zw, ceil(center + extent) + vec2<f32>(1.0)),
  );
  if (!nonempty(bounds)) {
    return;
  }

  splats[i].centerConic = vec4<f32>(center, cov.z / det, -cov.y / det);
  splats[i].colorOpacity = vec4<f32>(color, opacity);
  splats[i].shape = vec4<f32>(cov.x / det, 0.0, 0.0, 0.0);
  splats[i].bounds = bounds;
  keys[i] = vec2<u32>(bitcast<u32>(p.z), i);
  atomicAdd(&control.count, 1u);
}

@compute @workgroup_size(ANCHORS_PER_WORKGROUP * FEATURE_COUNT)
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let localAnchor = lid / FEATURE_COUNT;
  let neuron = lid % FEATURE_COUNT;
  let index = (wid.x + wid.y * FUSED_DISPATCH_WIDTH) * ANCHORS_PER_WORKGROUP + localAnchor;
  let count = worklist[0];
  var anchor = 0u;

  if (index < count) {
    anchor = worklist[4u + index];
    let delta = anchors[anchor].position.xyz - frame.camera.xyz;
    let distance = length(delta);
    let direction = delta / max(distance, 1e-12);
    for (var head = 0u; head < HEAD_COUNT; head++) {
      hidden[localAnchor * HIDDEN_CHANNELS + head * FEATURE_COUNT + neuron] =
        firstLayer(anchor, neuron, head, direction, distance / frame.camera.w);
    }
  }
  workgroupBarrier();

  if (index < count) {
    for (var channel = neuron; channel < OUTPUT_CHANNELS; channel += FEATURE_COUNT) {
      var head = 0u;
      var row = channel;
      var slot = channel;
      if (channel >= COV_START) {
        head = 2u;
        row = channel - COV_START;
        slot = row / 7u;
      } else if (channel >= COLOR_START) {
        head = 1u;
        row = channel - COLOR_START;
        slot = row / 3u;
      }
      if ((metadata[anchor].slotMask & (1u << slot)) != 0u) {
        let value = headOutput(localAnchor, head, row);
        outputs[localAnchor * OUTPUT_CHANNELS + channel] = value;
        if (RECORD_MLP) {
          // Validation buffer: anchor-to-sample map, then sampled head outputs.
          let sample = control.diagnostics[anchor];
          if (sample > 0u) {
            let start = frame.anchorCount + (sample - 1u) * SAMPLE_STRIDE;
            control.diagnostics[start + 1u + channel] = bitcast<u32>(value);
          }
        }
      }
    }
  }
  workgroupBarrier();

  if (RECORD_MLP) {
    if (index < count && neuron == 0u) {
      let sample = control.diagnostics[anchor];
      if (sample > 0u) {
        control.diagnostics[frame.anchorCount + (sample - 1u) * SAMPLE_STRIDE] = 1u;
      }
    }
  }

  if (index >= count || neuron >= OFFSET_COUNT) {
    return;
  }

  let mask = metadata[anchor].slotMask;
  if ((mask & (1u << neuron)) == 0u) {
    return;
  }

  let base = localAnchor * OUTPUT_CHANNELS;
  let opacity = outputs[base + neuron];
  if (opacity < (1.0 / 255.0)) {
    return;
  }

  let offsetIndex = metadata[anchor].offsetStart + countOneBits(mask & ((1u << neuron) - 1u));
  let offset = offsets[offsetIndex];
  let color = vec3<f32>(
    outputs[base + COLOR_START + neuron * 3u],
    outputs[base + COLOR_START + neuron * 3u + 1u],
    outputs[base + COLOR_START + neuron * 3u + 2u],
  );
  var covariance: array<f32, 7>;
  for (var channel = 0u; channel < 7u; channel++) {
    covariance[channel] = outputs[base + COV_START + neuron * 7u + channel];
  }
  let position = anchors[anchor].position;
  projectSplat(
    offsetIndex,
    position.xyz + offset.position * position.w,
    anchors[anchor].scale.xyz,
    opacity,
    color,
    covariance,
  );
}
