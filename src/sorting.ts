import compactSource from './shaders/compact.wgsl?raw';
import radixSource from './shaders/radix.wgsl?raw';
import {
  DYNAMIC_UNIFORM_STRIDE,
  MAX_ACTIVE_OFFSETS,
  RADIX_BITS,
  RADIX_BUCKETS,
  RADIX_PASSES,
  SORT_DISPATCH_OFFSET,
  SORT_INFO_BYTES,
  SORT_WORKGROUP_SIZE,
} from './layout';
import type { GpuTimer } from './timing';
import { check, type AllocateBuffer, type CompileShader } from './types';

type CompactStage = 'local' | 'prefix' | 'scatter';
type RadixStage = 'histogramPass' | 'prefixPass' | 'scatterPass';

export class GpuSorter {
  readonly scratch: GPUBuffer;
  private readonly info: GPUBuffer;
  private readonly ranks: GPUBuffer;
  private readonly blocks: GPUBuffer;
  private readonly histogram: GPUBuffer;
  private readonly totals: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly shifts: GPUBuffer;
  private readonly groups: number;
  private compactPipelines = {} as Record<CompactStage, GPUComputePipeline>;
  private radixPipelines = {} as Record<RadixStage, GPUComputePipeline>;
  private compactGroup!: GPUBindGroup;
  private radixGroups: GPUBindGroup[] = [];

  constructor(
    private readonly device: GPUDevice,
    capacity: number,
    private readonly keys: GPUBuffer,
    allocate: AllocateBuffer,
  ) {
    check(capacity > 0 && capacity <= MAX_ACTIVE_OFFSETS, 'Invalid compaction capacity');

    this.groups = Math.ceil(capacity / SORT_WORKGROUP_SIZE);
    this.scratch = allocate('compact/sort scratch', capacity * 8);
    this.ranks = allocate('stable ranks', capacity * 4);
    this.blocks = allocate('compact block prefixes', this.groups * 4);
    this.histogram = allocate('radix histogram', RADIX_BUCKETS * this.groups * 4);
    this.totals = allocate('radix totals', RADIX_BUCKETS * 4);
    this.info = allocate(
      'sort sizes and indirect dispatch',
      SORT_INFO_BYTES,
      undefined,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    );
    this.params = allocate(
      'compact parameters',
      16,
      new Uint32Array([capacity, this.groups, 0, 0]).buffer,
      GPUBufferUsage.UNIFORM,
    );

    const shifts = new Uint32Array(RADIX_PASSES * DYNAMIC_UNIFORM_STRIDE / 4);
    for (let pass = 0; pass < RADIX_PASSES; pass++) {
      shifts[pass * DYNAMIC_UNIFORM_STRIDE / 4] = pass * RADIX_BITS;
    }
    this.shifts = allocate('radix shifts', shifts.byteLength, shifts.buffer, GPUBufferUsage.UNIFORM);
  }

  async initialize(compile: CompileShader) {
    const [compact, radix] = await Promise.all([
      compile('stable compaction', compactSource),
      compile('stable radix', radixSource),
    ]);

    const layout = (types: GPUBufferBindingType[], dynamicBinding = -1) =>
      this.device.createBindGroupLayout({
        entries: types.map((type, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type,
            ...(binding === dynamicBinding ? { hasDynamicOffset: true, minBindingSize: 16 } : {}),
          },
        })),
      });
    const compactLayout = layout([
      'read-only-storage',
      'storage',
      'storage',
      'storage',
      'storage',
      'uniform',
    ]);
    const radixLayout = layout(
      ['read-only-storage', 'storage', 'storage', 'storage', 'storage', 'read-only-storage', 'uniform'],
      6,
    );

    const pipeline = (module: GPUShaderModule, bindGroupLayout: GPUBindGroupLayout, entryPoint: string) =>
      this.device.createComputePipelineAsync({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module, entryPoint },
      });
    const compactStages: CompactStage[] = ['local', 'prefix', 'scatter'];
    const radixStages: RadixStage[] = ['histogramPass', 'prefixPass', 'scatterPass'];
    await Promise.all([
      ...compactStages.map(async stage => {
        this.compactPipelines[stage] = await pipeline(compact, compactLayout, stage);
      }),
      ...radixStages.map(async stage => {
        this.radixPipelines[stage] = await pipeline(radix, radixLayout, stage);
      }),
    ]);

    const group = (bindGroupLayout: GPUBindGroupLayout, buffers: GPUBuffer[], dynamicBinding = -1) =>
      this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: buffers.map((buffer, binding) => ({
          binding,
          resource: { buffer, ...(binding === dynamicBinding ? { size: 16 } : {}) },
        })),
      });
    this.compactGroup = group(compactLayout, [
      this.keys,
      this.scratch,
      this.ranks,
      this.blocks,
      this.info,
      this.params,
    ]);
    this.radixGroups = [
      group(
        radixLayout,
        [this.scratch, this.keys, this.ranks, this.histogram, this.totals, this.info, this.shifts],
        6,
      ),
      group(
        radixLayout,
        [this.keys, this.scratch, this.ranks, this.histogram, this.totals, this.info, this.shifts],
        6,
      ),
    ];
  }

  compact(encoder: GPUCommandEncoder, timer: GpuTimer) {
    const pass = encoder.beginComputePass({ timestampWrites: timer.writes('compact') });
    pass.setBindGroup(0, this.compactGroup);

    for (const [stage, groups] of [
      ['local', this.groups],
      ['prefix', 1],
      ['scatter', this.groups],
    ] as const) {
      pass.setPipeline(this.compactPipelines[stage]);
      pass.dispatchWorkgroups(groups);
    }

    pass.end();
  }

  sort(pass: GPUComputePassEncoder) {
    for (let step = 0; step < RADIX_PASSES; step++) {
      pass.setBindGroup(0, this.radixGroups[step % 2], [step * DYNAMIC_UNIFORM_STRIDE]);
      pass.setPipeline(this.radixPipelines.histogramPass);
      pass.dispatchWorkgroupsIndirect(this.info, SORT_DISPATCH_OFFSET);

      pass.setPipeline(this.radixPipelines.prefixPass);
      pass.dispatchWorkgroups(RADIX_BUCKETS);

      pass.setPipeline(this.radixPipelines.scatterPass);
      pass.dispatchWorkgroupsIndirect(this.info, SORT_DISPATCH_OFFSET);
    }
  }
}
