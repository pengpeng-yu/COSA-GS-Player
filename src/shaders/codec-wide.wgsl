// Two's-complement 64-bit values, represented as (low word, high word).
// Only the operations required by the integer codec are implemented.
fn wideAdd(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let low = a.x + b.x;
  return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}

fn wideNeg(value: vec2<u32>) -> vec2<u32> {
  return wideAdd(~value, vec2<u32>(1u, 0u));
}

fn wideSigned(value: i32) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(value), select(0u, 0xffffffffu, value < 0));
}

fn wideLowI32(value: vec2<u32>) -> i32 {
  return bitcast<i32>(value.x);
}

// Exact unsigned 32 x 32 -> 64, using 16-bit partial products.
fn wideMul(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 65535u;
  let a1 = a >> 16u;
  let b0 = b & 65535u;
  let b1 = b >> 16u;
  let lowProduct = a0 * b0;
  let middle = a1 * b0 + (lowProduct >> 16u);
  let lowMiddle = (middle & 65535u) + a0 * b1;
  let high = a1 * b1 + (middle >> 16u) + (lowMiddle >> 16u);
  return vec2<u32>((lowMiddle << 16u) | (lowProduct & 65535u), high);
}

// The caller ensures its signed mathematical product fits int64.
// Modulo arithmetic also correctly handles a negative two's-complement input.
fn wideMulSmall(a: vec2<u32>, b: u32) -> vec2<u32> {
  let low = wideMul(a.x, b);
  return vec2<u32>(low.x, low.y + a.y * b);
}

fn wideMulSigned(value: i32, multiplier: u32) -> vec2<u32> {
  let negative = value < 0;
  let bits = bitcast<u32>(value);
  let magnitude = select(bits, 0u - bits, negative);
  let product = wideMul(magnitude, multiplier);
  if (negative) { return wideNeg(product); }
  return product;
}

fn wideUnsignedShift(value: vec2<u32>, shift: u32) -> vec2<u32> {
  if (shift == 0u) { return value; }
  if (shift < 32u) {
    return vec2<u32>((value.x >> shift) | (value.y << (32u - shift)), value.y >> shift);
  }
  if (shift == 32u) { return vec2<u32>(value.y, 0u); }
  return vec2<u32>(value.y >> (shift - 32u), 0u);
}

// Round to nearest, with exact half-way values rounded away from zero.
// The supported shift range is 0..62, matching the exported codec parameters.
fn wideRoundShift(value: vec2<u32>, shift: u32) -> vec2<u32> {
  if (shift == 0u) { return value; }
  let negative = (value.y & 0x80000000u) != 0u;
  var magnitude = value;
  if (negative) { magnitude = wideNeg(value); }
  var rounded = wideUnsignedShift(magnitude, shift);
  let bit = shift - 1u;
  var half = 0u;
  if (bit < 32u) { half = (magnitude.x >> bit) & 1u; }
  else { half = (magnitude.y >> (bit - 32u)) & 1u; }
  rounded = wideAdd(rounded, vec2<u32>(half, 0u));
  if (negative) { return wideNeg(rounded); }
  return rounded;
}

fn wideClampI8(value: vec2<u32>) -> i32 {
  let negative = (value.y & 0x80000000u) != 0u;
  var magnitude = value;
  if (negative) { magnitude = wideNeg(value); }
  var clamped = 127u;
  if (magnitude.y == 0u) { clamped = min(magnitude.x, 127u); }
  return select(i32(clamped), -i32(clamped), negative);
}

// The shader that includes this file supplies geluTable: array<i32>.
fn fixedGelu(input: i32) -> i32 {
  if (input >= 6291456) { return input; }
  if (input <= -6291456) { return 0; }
  let magnitude = u32(abs(input));
  let index = magnitude >> 11u;
  let fraction = magnitude & 2047u;
  let value = geluTable[index];
  let delta = geluTable[index + 1u] - value;
  let numerator = wideAdd(wideMul(u32(value), 2048u), wideMulSigned(delta, fraction));
  return max(input, 0) - wideLowI32(wideRoundShift(numerator, 15u));
}
