import { check } from './types';

// Pass intervals may overlap on mobile GPUs. Use totalMs for frame comparisons.
export class GpuTimer {
  private query?: GPUQuerySet;
  private resolve?: GPUBuffer;
  private readback?: GPUBuffer;
  readonly names = ['cull', 'fused', 'compact', 'sort', 'draw', 'display'];
  readonly supported: boolean;
  enabled = false;

  constructor(private readonly device: GPUDevice) {
    this.supported = device.features.has('timestamp-query');
  }

  writes(name: string): GPUComputePassTimestampWrites | undefined {
    if (!this.supported || !this.enabled) return;

    if (!this.query) {
      const size = this.names.length * 16;
      this.query = this.device.createQuerySet({ type: 'timestamp', count: this.names.length * 2 });
      this.resolve = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.readback = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    const index = this.names.indexOf(name);
    check(index >= 0, 'Unknown GPU timing interval');
    return {
      querySet: this.query,
      beginningOfPassWriteIndex: index * 2,
      endOfPassWriteIndex: index * 2 + 1,
    };
  }

  finish(encoder: GPUCommandEncoder) {
    if (!this.query || !this.enabled) return;

    encoder.resolveQuerySet(this.query, 0, this.names.length * 2, this.resolve!, 0);
    encoder.copyBufferToBuffer(this.resolve!, 0, this.readback!, 0, this.names.length * 16);
  }

  async read() {
    if (!this.readback || !this.enabled) return null;

    await this.readback.mapAsync(GPUMapMode.READ);
    try {
      const values = new BigUint64Array(this.readback.getMappedRange());
      const intervalsMs: Record<string, number> = {};
      this.names.forEach((name, index) => {
        intervalsMs[name] = Number(values[index * 2 + 1] - values[index * 2]) / 1e6;
      });
      return {
        intervalsMs,
        note: 'Raw GPU pass intervals may overlap. Use totalMs for full-frame comparisons',
      };
    } finally {
      this.readback.unmap();
    }
  }

  destroy() {
    this.query?.destroy();
    this.resolve?.destroy();
    this.readback?.destroy();
  }
}
