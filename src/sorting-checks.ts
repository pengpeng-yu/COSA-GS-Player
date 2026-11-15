import { GpuSorter } from './sorting';
import { GpuTimer } from './timing';
import { check, type AllocateBuffer } from './types';

// Includes capacities that are not powers of two, partial groups and depth ties.
export async function checkSorting(device: GPUDevice) {
  const timer = new GpuTimer(device);
  const capacities = [256, 768, 1280, 2048];
  const tested: { capacity: number; counts: number[] }[] = [];
  device.pushErrorScope('validation');

  try {
    for (const capacity of capacities) {
      const owned: GPUBuffer[] = [];
      const allocate: AllocateBuffer = (label, size, data, usage = GPUBufferUsage.STORAGE) => {
        const buffer = device.createBuffer({
          label,
          size: Math.ceil(size / 16) * 16,
          usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        owned.push(buffer);
        if (data) device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };

      try {
        const keys = allocate('sort test keys', capacity * 8);
        const sorter = new GpuSorter(device, capacity, keys, allocate);
        await sorter.initialize(async (label, code) => {
          const module = device.createShaderModule({ label, code });
          const info = await module.getCompilationInfo();
          check(!info.messages.some(message => message.type === 'error'), 'Sorting test shader failed');
          return module;
        });

        const counts = [...new Set([0, 1, 255, 256, 257, 1023, 1024, capacity - 1, capacity])].filter(
          count => count <= capacity,
        );
        for (const count of counts) {
          const data = new Uint32Array(capacity * 2);
          const expected: number[][] = [];
          for (let i = 0; i < capacity; i++) data[i * 2 + 1] = 0xffffffff;

          for (let n = 0; n < count; n++) {
            const index = n * 997 % capacity;
            const depth = 0x3f800000 + (n % 13) * 8192;
            data[index * 2] = depth;
            data[index * 2 + 1] = index;
            expected.push([depth, index]);
          }
          expected.sort((a, b) => b[0] - a[0] || b[1] - a[1]);

          device.queue.writeBuffer(keys, 0, data);
          const encoder = device.createCommandEncoder();
          sorter.compact(encoder, timer);
          const pass = encoder.beginComputePass();
          sorter.sort(pass);
          pass.end();

          const readback = device.createBuffer({
            size: capacity * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          });
          try {
            encoder.copyBufferToBuffer(sorter.scratch, 0, readback, 0, capacity * 8);
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(GPUMapMode.READ);

            const result = new Uint32Array(readback.getMappedRange());
            for (let i = 0; i < capacity; i++) {
              const pair = i < count ? expected[i] : [0, 0xffffffff];
              check(
                result[i * 2] === pair[0] && result[i * 2 + 1] === pair[1],
                'compact-radix capacity=' + capacity + ', count=' + count + ' mismatch at ' + i,
              );
            }
          } finally {
            readback.destroy();
          }
        }
        tested.push({ capacity, counts });
      } finally {
        owned.forEach(buffer => buffer.destroy());
      }
    }

    return {
      passed: true,
      cases: tested.reduce((sum, test) => sum + test.counts.length, 0),
      capacities: tested,
      equalDepthOrder: 'descending original index',
    };
  } finally {
    timer.destroy();
    const error = await device.popErrorScope();
    check(!error, error?.message ?? 'GPU sorting error');
  }
}
