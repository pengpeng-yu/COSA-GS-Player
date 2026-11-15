import common from './shaders/common.wgsl?raw';
import visibilitySource from './shaders/visibility.wgsl?raw';
import fusedSource from './shaders/fused.wgsl?raw';
import { MLP_HEAD_COUNT, OFFSETS_PER_ANCHOR, WEIGHTS_PER_HEAD } from './layout';
import type { GpuTimer } from './timing';
import { check, fetchChecked, type AllocateBuffer, type CompileShader } from './types';

interface MlpBuffers {
  anchors: GPUBuffer;
  offsets: GPUBuffer;
  uniform: GPUBuffer;
  control: GPUBuffer;
  projected: GPUBuffer;
  keys: GPUBuffer;
}

interface MlpResources {
  device: GPUDevice;
  anchorCount: number;
  buffers: MlpBuffers;
  allocate: AllocateBuffer;
}

export interface MlpTarget {
  pipeline: GPUComputePipeline;
  group: GPUBindGroup;
}

export class Mlp {
  readonly worklist: GPUBuffer;
  private readonly metadataValues: Uint32Array<ArrayBuffer>;
  private readonly metadata: GPUBuffer;
  private weights!: GPUBuffer;
  private fusedModule!: GPUShaderModule;
  private fused!: MlpTarget;
  private diagnosticPipeline?: GPUComputePipeline;
  private prefilter!: GPUComputePipeline;
  private finalize!: GPUComputePipeline;
  private prefilterGroup!: GPUBindGroup;
  private finalizeGroup!: GPUBindGroup;

  constructor(
    private readonly resources: MlpResources,
    offsetData: ArrayBuffer | Uint32Array,
  ) {
    const { anchorCount, allocate } = resources;
    const packed = offsetData instanceof Uint32Array;
    const slots = packed ? offsetData : new Uint32Array(offsetData);
    const stride = packed ? 1 : 4;
    this.metadataValues = new Uint32Array(anchorCount * 2);
    let previousSlot = -1;

    for (let offset = 0; offset < slots.length / stride; offset++) {
      const globalSlot = slots[offset * stride + stride - 1];
      const anchor = Math.floor(globalSlot / OFFSETS_PER_ANCHOR);
      const localSlot = globalSlot % OFFSETS_PER_ANCHOR;
      check(
        globalSlot > previousSlot && anchor < anchorCount,
        'Offsets must be unique and ordered by anchor/slot',
      );
      previousSlot = globalSlot;

      const base = anchor * 2;
      if (this.metadataValues[base + 1] === 0) {
        this.metadataValues[base] = offset;
      }
      this.metadataValues[base + 1] |= 1 << localSlot;
    }

    this.metadata = allocate(
      'anchor offset starts and masks',
      this.metadataValues.byteLength,
      this.metadataValues.buffer,
    );
    this.worklist = allocate(
      'visible anchors and dispatch',
      WORKLIST_HEADER_BYTES + anchorCount * 4,
      undefined,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    );
  }

  offsetMask(anchor: number) {
    return this.metadataValues[anchor * 2 + 1];
  }

  private group(pipeline: GPUComputePipeline, entries: [number, GPUBuffer][]) {
    return this.resources.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
    });
  }

  private fusedGroup(pipeline: GPUComputePipeline, control: GPUBuffer) {
    const { anchors, offsets, uniform, projected, keys } = this.resources.buffers;
    return this.group(pipeline, [
      [0, anchors],
      [1, this.weights],
      [2, this.metadata],
      [3, this.worklist],
      [4, offsets],
      [5, control],
      [6, projected],
      [7, keys],
      [8, uniform],
    ]);
  }

  async initialize(compile: CompileShader, weights: string | GPUBuffer) {
    const { device, allocate, buffers } = this.resources;
    const data =
      typeof weights === 'string'
        ? await fetchChecked(weights).then(response => response.arrayBuffer())
        : weights;
    const size = data instanceof GPUBuffer ? data.size : data.byteLength;
    check(size === MLP_HEAD_COUNT * WEIGHTS_PER_HEAD * 4, 'Invalid FP32 MLP weight layout');
    this.weights = data instanceof GPUBuffer ? data : allocate('FP32 MLP weights', size, data);

    const [visibility, fused] = await Promise.all([
      compile('anchor visibility', common + visibilitySource),
      compile('fused MLP and projection', common + fusedSource),
    ]);
    this.fusedModule = fused;

    const pipeline = (module: GPUShaderModule, entryPoint: string) =>
      device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint } });
    const [prefilter, finalize, production] = await Promise.all([
      pipeline(visibility, 'prefilter'),
      pipeline(visibility, 'finalize'),
      device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: fused, entryPoint: 'main', constants: { RECORD_MLP: 0 } },
      }),
    ]);

    this.prefilter = prefilter;
    this.finalize = finalize;
    this.prefilterGroup = this.group(prefilter, [
      [0, buffers.anchors],
      [4, this.worklist],
      [8, buffers.uniform],
    ]);
    this.finalizeGroup = this.group(finalize, [[4, this.worklist]]);
    this.fused = { pipeline: production, group: this.fusedGroup(production, buffers.control) };
  }

  // Only called by the lazily loaded validation module.
  async diagnosticTarget(control: GPUBuffer): Promise<MlpTarget> {
    this.diagnosticPipeline ??= await this.resources.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: this.fusedModule, entryPoint: 'main', constants: { RECORD_MLP: 1 } },
    });
    return {
      pipeline: this.diagnosticPipeline,
      group: this.fusedGroup(this.diagnosticPipeline, control),
    };
  }

  encode(encoder: GPUCommandEncoder, timer?: GpuTimer, target = this.fused) {
    encoder.clearBuffer(this.worklist, 0, WORKLIST_HEADER_BYTES);
    encoder.clearBuffer(this.resources.buffers.keys);

    const visibility = encoder.beginComputePass({ timestampWrites: timer?.writes('cull') });
    visibility.setPipeline(this.prefilter);
    visibility.setBindGroup(0, this.prefilterGroup);
    visibility.dispatchWorkgroups(Math.ceil(this.resources.anchorCount / 128));
    visibility.setPipeline(this.finalize);
    visibility.setBindGroup(0, this.finalizeGroup);
    visibility.dispatchWorkgroups(1);
    visibility.end();

    const fused = encoder.beginComputePass({ timestampWrites: timer?.writes('fused') });
    fused.setPipeline(target.pipeline);
    fused.setBindGroup(0, target.group);
    fused.dispatchWorkgroupsIndirect(this.worklist, 4);
    fused.end();
  }
}

const WORKLIST_HEADER_BYTES = 16;
