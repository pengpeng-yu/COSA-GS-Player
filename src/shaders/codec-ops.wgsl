// Utility kernels for staged decoding. All indexing is in 32-bit words.
struct Params {
  elementCount: u32,
  channelCount: u32,
  attributeKind: u32,
  secondChannelCount: u32,
  multiplierLow: u32,
  multiplierHigh: u32,
  shift: u32,
  floatStep: f32,
}

const ATTRIBUTE_FEATURE = 0u;
const ATTRIBUTE_GAUSSIAN_SCALING = 1u;
const ATTRIBUTE_OFFSET = 2u;

@group(0) @binding(0) var<storage, read> sourceValues: array<i32>;
@group(0) @binding(1) var<storage, read> secondInput: array<u32>;
@group(0) @binding(2) var<storage, read> activeSlots: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> geluTable: array<i32>;

fn linearIndex(id: vec3<u32>) -> u32 {
  return id.x + id.y * 65535u * 256u;
}

fn decodedSymbol(index: u32) -> i32 {
  let packed = secondInput[index / 2u];
  let bits = (packed >> ((index % 2u) * 16u)) & 65535u;
  return bitcast<i32>(bits << 16u) >> 16u;
}

fn distributionMeanIndex(index: u32) -> u32 {
  if (params.attributeKind == ATTRIBUTE_OFFSET) {
    let slot = activeSlots[index / 3u];
    return (slot / 10u) * 60u + (slot % 10u) * 3u + index % 3u;
  }
  return (index / params.channelCount) * params.channelCount * 2u + index % params.channelCount;
}

@compute @workgroup_size(256)
fn packCoordinates(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = linearIndex(id);
  if (i < params.elementCount) {
    output[(i / 3u) * 40u + i % 3u] = bitcast<u32>(sourceValues[i]);
  }
}

@compute @workgroup_size(256)
fn concatenate(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = linearIndex(id);
  if (i >= params.elementCount) {
    return;
  }
  let width = params.channelCount + params.secondChannelCount;
  let row = i / width;
  let col = i % width;
  if (col < params.channelCount) {
    output[i] = bitcast<u32>(sourceValues[row * params.channelCount + col]);
  } else {
    output[i] = secondInput[row * params.secondChannelCount + col - params.channelCount];
  }
}

@compute @workgroup_size(256)
fn addContext(@builtin(global_invocation_id) id: vec3<u32>) {
  // Reuse the prior network's output. Every invocation owns one element.
  let i = linearIndex(id);
  if (i < params.elementCount) {
    output[i] += secondInput[i];
  }
}

@compute @workgroup_size(256)
fn packCdfIndices(@builtin(global_invocation_id) id: vec3<u32>) {
  let word = linearIndex(id);
  if (word * 4u >= params.elementCount) {
    return;
  }

  var packed = 0u;
  for (var lane = 0u; lane < 4u; lane++) {
    let i = word * 4u + lane;
    if (i < params.elementCount) {
      let value = sourceValues[distributionMeanIndex(i) + params.channelCount];
      let rounded = wideLowI32(wideRoundShift(wideSigned(value), 20u));
      packed |= u32(clamp(rounded, 0, 127)) << (lane * 8u);
    }
  }
  output[word] = packed;
}

@compute @workgroup_size(256)
fn reconstructFixed(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = linearIndex(id);
  if (i >= params.elementCount) {
    return;
  }
  let symbol = decodedSymbol(i);
  let magnitude = select(u32(symbol), 0u - u32(symbol), symbol < 0);
  var product = wideMulSmall(vec2<u32>(params.multiplierLow, params.multiplierHigh), magnitude);
  if (symbol < 0) {
    product = wideNeg(product);
  }
  let residual = wideRoundShift(product, params.shift);
  output[i] = wideAdd(wideSigned(sourceValues[i * 2u]), residual).x;
}

@compute @workgroup_size(256)
fn packPositionScaling(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = linearIndex(id);
  if (i >= params.elementCount) {
    return;
  }
  let value = f32(sourceValues[i]) / 1048576.0;
  output[i * 40u + 3u] = bitcast<u32>(exp(value));
}

@compute @workgroup_size(256)
fn reconstructFloat(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = linearIndex(id);
  if (i >= params.elementCount) {
    return;
  }
  let mean = f32(sourceValues[distributionMeanIndex(i)]) / 1048576.0;
  let residual = f32(decodedSymbol(i)) * params.floatStep;
  let value = mean + residual;

  if (params.attributeKind == ATTRIBUTE_FEATURE) {
    output[(i / 32u) * 40u + 8u + i % 32u] = bitcast<u32>(value);
  } else if (params.attributeKind == ATTRIBUTE_GAUSSIAN_SCALING) {
    output[(i / 3u) * 40u + 4u + i % 3u] = bitcast<u32>(exp(value));
  } else {
    output[(i / 3u) * 4u + i % 3u] = bitcast<u32>(value);
    if (i % 3u == 0u) {
      output[(i / 3u) * 4u + 3u] = activeSlots[i / 3u];
    }
  }
}
