import wideSource from './shaders/codec-wide.wgsl?raw';
import { check } from './types';

const source = `
@group(0) @binding(0) var<storage, read> geluTable: array<i32>;
@group(0) @binding(1) var<storage, read> cases: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&cases) / 8u) { return; }
  let base = id.x * 8u;
  let product = wideMulSmall(vec2<u32>(cases[base], cases[base + 1u]), cases[base + 2u]);
  let rounded = wideRoundShift(product, cases[base + 3u]);
  let output = id.x * 4u;
  results[output] = rounded.x;
  results[output + 1u] = rounded.y;
  results[output + 2u] = bitcast<u32>(fixedGelu(bitcast<i32>(cases[base + 4u])));
  results[output + 3u] = bitcast<u32>(wideClampI8(
    wideAdd(rounded, wideSigned(bitcast<i32>(cases[base + 5u])))));
}
`;

interface ArithmeticCase {
  value: bigint;
  multiplier: number;
  shift: number;
  gelu: number;
  zero: number;
}

function roundedShift(value: bigint, shift: number) {
  if (!shift) return value;

  const magnitude = value < 0n ? -value : value;
  const half = 1n << BigInt(shift - 1);
  const rounded = (magnitude + half) >> BigInt(shift);
  return value < 0n ? -rounded : rounded;
}

function gelu(input: number, table: Int32Array) {
  if (input >= 6 * 1048576) return input;
  if (input <= -6 * 1048576) return 0;

  const magnitude = Math.abs(input);
  const index = magnitude >>> 11;
  const fraction = magnitude & 2047;
  const numerator =
    BigInt(table[index]) * 2048n + BigInt(table[index + 1] - table[index]) * BigInt(fraction);
  return Math.max(input, 0) - Number(roundedShift(numerator, 15));
}

export async function checkCodecMath(device: GPUDevice, table: Int32Array) {
  check(table.length === 3073, 'Expected the 3073-entry integer GELU table');

  const cases: ArithmeticCase[] = [];
  const add = (value: bigint, multiplier: number, shift: number, input = 0, zero = 0) => {
    const product = value * BigInt(multiplier);
    if (product < -(1n << 63n) || product >= (1n << 63n)) return;
    cases.push({ value, multiplier, shift, gelu: input, zero });
  };
  const values = [
    0n,
    1n,
    -1n,
    127n,
    -127n,
    2147483647n,
    -2147483648n,
    (1n << 47n) - 1n,
    -((1n << 47n) - 1n),
    (1n << 63n) - 1n,
    -(1n << 63n),
  ];
  for (const value of values) {
    for (const multiplier of [1, 2, 127, 32768, 2147483647]) {
      for (const shift of [0, 1, 15, 20, 31, 32, 33, 47, 62]) {
        add(value, multiplier, shift, Number(BigInt.asIntN(32, value)), -17);
      }
    }
  }

  for (let shift = 1; shift <= 62; shift++) {
    const half = 1n << BigInt(shift - 1);
    for (const delta of [-1n, 0n, 1n]) {
      for (const sign of [-1n, 1n]) {
        add(sign * (half + delta), 1, shift, 0, 1);
      }
    }
  }

  // Check interpolation within every LUT interval and both signs.
  for (let index = 0; index < 3072; index++) {
    for (const fraction of [0, 1, 1023, 1024, 2047]) {
      for (const sign of [-1, 1]) {
        add(0n, 1, 0, sign * (index * 2048 + fraction));
      }
    }
  }

  for (const input of [-2147483648, -6291457, -6291456, 6291456, 6291457, 2147483647]) {
    add(0n, 1, 0, input);
  }

  let random = 0x13579bdf;
  const next = () => {
    random ^= random << 13;
    random ^= random >>> 17;
    random ^= random << 5;
    return random >>> 0;
  };
  for (let i = 0; i < 4096; i++) {
    add(BigInt(next() | 0), (next() & 0x7fffffff) || 1, next() % 63, next() | 0, next() | 0);
  }

  const words = new Uint32Array(cases.length * 8);
  cases.forEach((entry, i) => {
    const bits = BigInt.asUintN(64, entry.value);
    words.set(
      [
        Number(bits & 0xffffffffn),
        Number(bits >> 32n),
        entry.multiplier,
        entry.shift,
        entry.gelu >>> 0,
        entry.zero >>> 0,
        0,
        0,
      ],
      i * 8,
    );
  });

  const upload = (label: string, values: Int32Array | Uint32Array) => {
    const buffer = device.createBuffer({
      label,
      size: values.byteLength,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
    );
    buffer.unmap();
    return buffer;
  };

  const lut = upload('Integer math check GELU', table);
  const input = upload('Integer math check cases', words);
  const output = device.createBuffer({
    label: 'Integer math check output',
    size: cases.length * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: 'Integer math check readback',
    size: output.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  try {
    const module = device.createShaderModule({
      label: 'Integer codec arithmetic checks',
      code: wideSource + source,
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    check(!errors.length, errors.map(message => message.lineNum + ': ' + message.message).join('\n'));

    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [lut, input, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    const commands = device.createCommandEncoder();
    const pass = commands.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(cases.length / 64));
    pass.end();

    commands.copyBufferToBuffer(output, 0, readback, 0, output.size);
    device.queue.submit([commands.finish()]);
    await readback.mapAsync(GPUMapMode.READ);

    const actual = new Uint32Array(readback.getMappedRange());
    for (let index = 0; index < cases.length; index++) {
      const entry = cases[index];
      const rounded = roundedShift(entry.value * BigInt(entry.multiplier), entry.shift);
      const bits = BigInt.asUintN(64, rounded);
      const biased = BigInt.asIntN(64, rounded + BigInt(entry.zero));
      const clamped = biased < -127n ? -127 : biased > 127n ? 127 : Number(biased);
      const expected = [
        Number(bits & 0xffffffffn),
        Number(bits >> 32n),
        gelu(entry.gelu, table) >>> 0,
        clamped >>> 0,
      ];
      for (let field = 0; field < 4; field++) {
        check(
          actual[index * 4 + field] === expected[field],
          'Integer math case ' + index + ' field ' + field + ': actual=' +
            actual[index * 4 + field] + ', expected=' + expected[field],
        );
      }
    }

    readback.unmap();
    return { passed: true, cases: cases.length, checkedValues: cases.length * 4 };
  } finally {
    lut.destroy();
    input.destroy();
    output.destroy();
    readback.destroy();
  }
}
