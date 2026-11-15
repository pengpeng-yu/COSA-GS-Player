override LAYERS: u32 = 2u;

struct Dispatch {
  rows: u32,
  groupsX: u32,
  unused0: u32,
  unused1: u32,
}

@group(0) @binding(0) var<storage, read> parameters: array<u32>;
@group(0) @binding(1) var<storage, read> inputs: array<i32>;
@group(0) @binding(2) var<storage, read> geluTable: array<i32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<i32>;
@group(0) @binding(4) var<uniform> dispatch: Dispatch;

var<workgroup> activations: array<i32, 64>;
var<workgroup> packed: array<u32, 16>;

fn packedActivations(base: u32) -> u32 {
  return (bitcast<u32>(activations[base]) & 255u)
    | ((bitcast<u32>(activations[base + 1u]) & 255u) << 8u)
    | ((bitcast<u32>(activations[base + 2u]) & 255u) << 16u)
    | ((bitcast<u32>(activations[base + 3u]) & 255u) << 24u);
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_index) channel: u32,
) {
  let row = group.x + group.y * dispatch.groupsX;
  if (row >= dispatch.rows) {
    return;
  }

  let inputWidth = parameters[6u];
  var activation = 0;
  if (channel < inputWidth) {
    let value = wideRoundShift(
      wideMulSigned(inputs[row * inputWidth + channel], parameters[1u]), parameters[2u],
    );
    activation = wideClampI8(wideAdd(value, wideSigned(bitcast<i32>(parameters[3u]))));
  }
  activations[channel] = activation;
  workgroupBarrier();

  // One workgroup keeps both layers' INT8 activations in shared memory.
  // Layer count is specialized to keep every barrier in uniform control flow.
  for (var layer = 0u; layer < LAYERS; layer++) {
    let base = parameters[4u + layer];
    let width = parameters[base];
    let outputWidth = parameters[base + 1u];
    let wordsPerRow = (width + 3u) / 4u;

    // Pack each group of four inputs once, then share it across output neurons.
    if (channel < wordsPerRow) {
      packed[channel] = packedActivations(channel * 4u);
    }
    workgroupBarrier();

    let weights = base + 8u;
    let biases = weights + outputWidth * wordsPerRow;
    let multipliers = biases + outputWidth;
    var next = 0;
    if (channel < outputWidth) {
      var sum = 0;
      for (var k = 0u; k < wordsPerRow; k++) {
        sum += dot4I8Packed(packed[k], parameters[weights + channel * wordsPerRow + k]);
      }
      let biased = sum + bitcast<i32>(parameters[biases + channel]);
      var value = wideRoundShift(
        wideMulSigned(biased, parameters[multipliers + channel]), parameters[base + 3u],
      );
      let hasGelu = parameters[base + 2u] != 0u;
      if (hasGelu) {
        value = wideSigned(fixedGelu(wideLowI32(value)));
      }

      if (layer + 1u < LAYERS) {
        value = wideRoundShift(wideMulSmall(value, parameters[base + 4u]), parameters[base + 5u]);
        next = wideClampI8(wideAdd(value, wideSigned(bitcast<i32>(parameters[base + 6u]))));
      } else {
        if (!hasGelu) {
          value = wideAdd(value, wideSigned(bitcast<i32>(parameters[base + 6u])));
        }
        outputs[row * outputWidth + channel] = wideLowI32(value);
      }
    }

    if (layer + 1u < LAYERS) {
      // This layer only reads packed. Next-layer activations use a separate array.
      // Finish all neurons before the next iteration repacks their outputs.
      activations[channel] = next;
      workgroupBarrier();
    }
  }
}
