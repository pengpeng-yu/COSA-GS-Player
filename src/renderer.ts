import common from './shaders/common.wgsl?raw';
import drawSource from './shaders/draw.wgsl?raw';
import displaySource from './shaders/display.wgsl?raw';
import { frameUniform } from './camera';
import { GpuSorter } from './sorting';
import { GpuTimer } from './timing';
import { Mlp } from './mlp';
import type { MlpDiagnostics } from './diagnostics';
import {
  ACTIVE_OFFSET_BYTES,
  ANCHOR_STRIDE_FLOATS,
  CONTROL_BUFFER_BYTES,
  FEATURE_COUNT,
  FRAME_UNIFORM_BYTES,
  MAX_ACTIVE_OFFSETS,
  OFFSETS_PER_ANCHOR,
  PROJECTED_SPLAT_BYTES,
  SORT_WORKGROUP_SIZE,
} from './layout';
import { DEFAULT_SETTINGS, validateSettings, type RenderSettings } from './settings';
import { assetUrl, check, fetchChecked, type Camera, type Scene, type GpuSceneBuffers } from './types';

export interface FrameTiming {
  totalMs: number;
  gpu: Awaited<ReturnType<GpuTimer['read']>>;
}

export class Renderer {
  device!: GPUDevice;
  adapterInfo!: Pick<GPUAdapterInfo, 'vendor' | 'architecture' | 'device' | 'description'>;
  accumulationFormat!: GPUTextureFormat;
  readonly sortCapacity: number;
  readonly errors: string[] = [];

  private context!: GPUCanvasContext;
  private uniform!: GPUBuffer;
  private control!: GPUBuffer;
  private drawPipeline!: GPURenderPipeline;
  private displayPipeline!: GPURenderPipeline;
  private drawLayout!: GPUBindGroupLayout;
  private displayLayout!: GPUBindGroupLayout;
  private drawGroup!: GPUBindGroup;
  private displayGroup!: GPUBindGroup;
  private drawModule!: GPUShaderModule;
  private displayModule!: GPUShaderModule;
  private drawPipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
  private displayPipelines = new Map<boolean, GPURenderPipeline>();

  private accumulationTexture?: GPUTexture;
  private outputTexture!: GPUTexture;
  private accumulationView!: GPUTextureView;
  private outputView!: GPUTextureView;
  private canvasFormat!: GPUTextureFormat;
  private buffers: GPUBuffer[] = [];
  private sorter!: GpuSorter;
  private timer!: GpuTimer;
  private mlp!: Mlp;

  private rendering = false;
  private disposed = false;
  private currentSettings = { ...DEFAULT_SETTINGS };
  private validation?: Promise<MlpDiagnostics>;
  private float32Blendable = false;

  constructor(
    readonly canvas: HTMLCanvasElement,
    readonly scene: Scene,
    private base = assetUrl('scene/'),
  ) {
    this.sortCapacity = Math.ceil(scene.activeOffsetCount / SORT_WORKGROUP_SIZE) * SORT_WORKGROUP_SIZE;
  }

  get settings() {
    return { ...this.currentSettings };
  }

  get capabilities() {
    return {
      float32Blendable: this.float32Blendable,
      timestamps: this.timer.supported,
      enabledFeatures: [...this.device.features],
    };
  }

  get memory() {
    const pixels = this.canvas.width * this.canvas.height;
    return {
      trackedBuffersMiB: this.buffers.reduce((sum, buffer) => sum + buffer.size, 0) / 1048576,
      renderTexturesMiB: (pixels * ((this.accumulationFormat === 'rgba32float' ? 16 : 8) + 4)) / 1048576,
      note:
        'Persistent renderer buffers and textures. ' +
        'Excludes canvas, validation, readback, driver and browser allocations',
    };
  }

  private buffer(label: string, size: number, data?: ArrayBuffer, usage = GPUBufferUsage.STORAGE) {
    const buffer = this.device.createBuffer({
      label,
      size: Math.ceil(size / 16) * 16,
      usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.buffers.push(buffer);
    if (data) this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  private async module(label: string, code: string) {
    const module = this.device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    check(
      !errors.length,
      label + ': ' + errors.map(message => message.lineNum + ': ' + message.message).join('\n'),
    );
    return module;
  }

  async initializeDevice(progress: (message: string) => void) {
    if (this.device) return;

    check(
      this.scene.featureCount === FEATURE_COUNT &&
        this.scene.offsetsPerAnchor === OFFSETS_PER_ANCHOR &&
        this.scene.anchorStrideFloats === ANCHOR_STRIDE_FLOATS,
      'Unsupported scene layout',
    );
    check(
      this.scene.anchorCount > 0 &&
        this.scene.activeOffsetCount > 0 &&
        this.sortCapacity <= MAX_ACTIVE_OFFSETS,
      'Invalid scene size (requires anchors and active offsets, with at most 2^20 active offsets)',
    );
    check(
      navigator.gpu,
      'WebGPU is unavailable in this browser. Use a version of Chrome with WebGPU support',
    );

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    check(adapter, 'No WebGPU adapter found');

    const largest =
      Math.ceil(
        Math.max(
          this.scene.anchorCount * ANCHOR_STRIDE_FLOATS * 4,
          this.scene.activeOffsetCount * PROJECTED_SPLAT_BYTES,
          this.sortCapacity * 8,
        ) / 16,
      ) * 16;
    check(
      adapter.limits.maxStorageBufferBindingSize >= largest && adapter.limits.maxBufferSize >= largest,
      'Device buffer limits are too small for this scene. Chunking is required',
    );

    const requiredFeatures: GPUFeatureName[] = [];
    for (const feature of ['float32-blendable', 'timestamp-query'] as GPUFeatureName[]) {
      if (adapter.features.has(feature)) requiredFeatures.push(feature);
    }

    this.device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: Math.max(134217728, largest),
        maxBufferSize: Math.max(268435456, largest),
      },
    });
    this.float32Blendable = this.device.features.has('float32-blendable');

    const info = adapter.info;
    this.adapterInfo = {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    };

    this.device.addEventListener('uncapturederror', event => {
      this.errors.push(event.error.message);
      console.error('WebGPU:', event.error.message);
    });
    this.device.lost.then(info => {
      if (!this.disposed) {
        this.errors.push('GPU device lost: ' + info.message);
        progress('GPU device lost. Reload the page: ' + info.message);
      }
    });
  }

  async initialize(
    progress: (message: string) => void,
    settings: RenderSettings,
    data: GpuSceneBuffers | string,
  ) {
    await this.initializeDevice(progress);
    this.timer = new GpuTimer(this.device);
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    const camera = this.scene.cameras[0];
    check(camera, 'Scene has no cameras');
    this.canvas.width = camera.width;
    this.canvas.height = camera.height;

    this.context = this.canvas.getContext('webgpu')!;
    check(this.context, 'Failed to initialize the WebGPU canvas');
    this.context.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });

    this.device.pushErrorScope('validation');
    try {
      const gpu = typeof data === 'string' ? undefined : data;
      const decodedBase = typeof data === 'string' ? data : '';
      progress(gpu ? 'Preparing decoded scene for rendering…' : 'Loading optional predecoded scene…');

      let anchors: GPUBuffer;
      let offsets: GPUBuffer;
      let offsetData: ArrayBuffer | Uint32Array;
      if (gpu) {
        anchors = gpu.anchors;
        offsets = gpu.offsets;
        offsetData = gpu.activeSlots;
        this.buffers.push(anchors, offsets, gpu.weights);
      } else {
        const [anchorData, loadedOffsets] = await Promise.all([
          fetchChecked(decodedBase + 'anchors.bin').then(response => response.arrayBuffer()),
          fetchChecked(decodedBase + 'offsets.bin').then(response => response.arrayBuffer()),
        ]);
        check(
          anchorData.byteLength === this.scene.anchorCount * ANCHOR_STRIDE_FLOATS * 4,
          'Anchor file size mismatch',
        );
        check(
          loadedOffsets.byteLength === this.scene.activeOffsetCount * ACTIVE_OFFSET_BYTES,
          'Offset file size mismatch',
        );
        anchors = this.buffer('anchors', anchorData.byteLength, anchorData);
        offsets = this.buffer('active offsets', loadedOffsets.byteLength, loadedOffsets);
        offsetData = loadedOffsets;
      }

      this.uniform = this.buffer('camera', FRAME_UNIFORM_BYTES, undefined, GPUBufferUsage.UNIFORM);
      this.control = this.buffer(
        'indirect draw arguments',
        CONTROL_BUFFER_BYTES,
        new Uint32Array([6, 0, 0, 0]).buffer,
        GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
      );
      const projected = this.buffer('projected splats', this.scene.activeOffsetCount * PROJECTED_SPLAT_BYTES);
      const keys = this.buffer('depth/index pairs', this.sortCapacity * 8);
      this.mlp = new Mlp(
        {
          device: this.device,
          anchorCount: this.scene.anchorCount,
          buffers: { anchors, offsets, projected, keys, uniform: this.uniform, control: this.control },
          allocate: this.buffer.bind(this),
        },
        offsetData,
      );
      this.sorter = new GpuSorter(this.device, this.sortCapacity, keys, this.buffer.bind(this));

      progress('Compiling fused MLP, sorting, and drawing shaders…');
      const [, , drawModule, displayModule] = await Promise.all([
        this.mlp.initialize(this.module.bind(this), gpu?.weights ?? decodedBase + 'weights.bin'),
        this.sorter.initialize(this.module.bind(this)),
        this.module('splat draw', common + drawSource),
        this.module('display', displaySource),
      ]);
      this.drawModule = drawModule;
      this.displayModule = displayModule;

      this.drawLayout = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 8, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        ],
      });

      // textureLoad needs no filtering feature, including for rgba32float.
      this.displayLayout = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        ],
      });
      this.drawGroup = this.device.createBindGroup({
        layout: this.drawLayout,
        entries: [
          { binding: 0, resource: { buffer: projected } },
          { binding: 1, resource: { buffer: this.sorter.scratch } },
          { binding: 8, resource: { buffer: this.uniform } },
        ],
      });

      this.outputTexture = this.device.createTexture({
        label: 'persistent display image',
        size: [camera.width, camera.height],
        format: this.canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      this.outputView = this.outputTexture.createView();
      await this.setSettings(settings);
    } finally {
      const error = await this.device.popErrorScope();
      check(!error, 'WebGPU initialization: ' + error?.message);
    }

    check(!this.errors.length, this.errors.join('\n'));
  }

  async setSettings(settings: RenderSettings) {
    check(!this.rendering && !this.disposed, 'Wait for the current frame before changing settings');
    const next = validateSettings(settings);
    check(
      next.accumulation !== 'rgba32float' || this.float32Blendable,
      'rgba32float blending is not supported on this device. Select rgba16float',
    );
    const format = next.accumulation;

    if (!this.drawPipelines.has(format)) {
      this.drawPipelines.set(
        format,
        await this.device.createRenderPipelineAsync({
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.drawLayout] }),
          vertex: { module: this.drawModule, entryPoint: 'vs' },
          fragment: {
            module: this.drawModule,
            entryPoint: 'fs',
            targets: [
              {
                format,
                blend: {
                  color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                },
              },
            ],
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
        }),
      );
    }

    if (!this.displayPipelines.has(next.explicitRgb8)) {
      this.displayPipelines.set(
        next.explicitRgb8,
        await this.device.createRenderPipelineAsync({
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.displayLayout] }),
          vertex: { module: this.displayModule, entryPoint: 'vs' },
          fragment: {
            module: this.displayModule,
            entryPoint: 'fs',
            constants: { EXPLICIT_RGB8: Number(next.explicitRgb8) },
            targets: [{ format: this.canvasFormat }],
          },
          primitive: { topology: 'triangle-list' },
        }),
      );
    }

    if (format !== this.accumulationFormat) {
      const color = this.device.createTexture({
        label: 'float accumulation',
        size: [this.canvas.width, this.canvas.height],
        format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.accumulationView = color.createView();
      this.displayGroup = this.device.createBindGroup({
        layout: this.displayLayout,
        entries: [{ binding: 0, resource: this.accumulationView }],
      });
      this.accumulationTexture?.destroy();
      this.accumulationTexture = color;
    }

    this.drawPipeline = this.drawPipelines.get(format)!;
    this.displayPipeline = this.displayPipelines.get(next.explicitRgb8)!;
    this.accumulationFormat = format;
    this.currentSettings = next;
  }

  async render(camera: Camera, profileGpu = false): Promise<FrameTiming> {
    check(!this.rendering && !this.disposed && this.accumulationTexture, 'Renderer is unavailable or busy');
    check(!this.errors.length, this.errors.join('\n'));
    check(camera.width === this.canvas.width && camera.height === this.canvas.height, 'Resolution mismatch');

    this.rendering = true;
    const start = performance.now();

    try {
      this.timer.enabled = profileGpu;
      this.device.queue.writeBuffer(this.uniform, 0, frameUniform(camera, this.scene));
      const encoder = this.device.createCommandEncoder();
      encoder.clearBuffer(this.control, 4, 4); // Only the indirect instance count changes each frame.
      this.mlp.encode(encoder, this.timer);
      this.sorter.compact(encoder, this.timer);

      const compute = encoder.beginComputePass({ timestampWrites: this.timer.writes('sort') });
      this.sorter.sort(compute);
      compute.end();

      const background = this.scene.background;
      const draw = encoder.beginRenderPass({
        timestampWrites: this.timer.writes('draw'),
        colorAttachments: [
          {
            view: this.accumulationView,
            clearValue: { r: background[0], g: background[1], b: background[2], a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      draw.setPipeline(this.drawPipeline);
      draw.setBindGroup(0, this.drawGroup);
      draw.drawIndirect(this.control, 0);
      draw.end();

      const display = encoder.beginRenderPass({
        timestampWrites: this.timer.writes('display'),
        colorAttachments: [
          { view: this.outputView, clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' },
        ],
      });
      display.setPipeline(this.displayPipeline);
      display.setBindGroup(0, this.displayGroup);
      display.draw(3);
      display.end();

      encoder.copyTextureToTexture(
        { texture: this.outputTexture },
        { texture: this.context.getCurrentTexture() },
        [camera.width, camera.height],
      );
      this.timer.finish(encoder);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const totalMs = performance.now() - start;

      check(!this.errors.length, this.errors.join('\n'));
      return { totalMs, gpu: await this.timer.read() };
    } finally {
      this.rendering = false;
    }
  }

  async pixels() {
    check(!this.rendering, 'Wait for rendering before readback');

    const { width, height } = this.canvas;
    const stride = Math.ceil((width * 4) / 256) * 256;
    const readback = this.device.createBuffer({
      size: stride * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture: this.outputTexture },
        { buffer: readback, bytesPerRow: stride },
        [width, height],
      );
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);

      const bytes = new Uint8Array(readback.getMappedRange());
      const rgba = new Uint8ClampedArray(width * height * 4);
      const bgra = this.canvasFormat.startsWith('bgra');
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const source = y * stride + x * 4;
          const destination = (y * width + x) * 4;
          rgba[destination] = bytes[source + (bgra ? 2 : 0)];
          rgba[destination + 1] = bytes[source + 1];
          rgba[destination + 2] = bytes[source + (bgra ? 0 : 2)];
          rgba[destination + 3] = 255;
        }
      }

      return rgba;
    } finally {
      readback.destroy();
    }
  }

  async diagnostics(camera: Camera, view: number) {
    check(!this.rendering && !this.disposed, 'Wait for rendering before diagnostics');
    this.validation ??= import('./diagnostics').then(({ MlpDiagnostics }) =>
      MlpDiagnostics.load({
        device: this.device,
        scene: this.scene,
        mlp: this.mlp,
        uniform: this.uniform,
        control: this.control,
        referenceUrl: this.base + this.scene.files.renderMlpReference,
      }),
    );
    return (await this.validation).check(camera, view);
  }

  async dispose() {
    this.disposed = true;
    this.timer?.destroy();
    this.buffers.forEach(buffer => buffer.destroy());
    this.accumulationTexture?.destroy();
    this.outputTexture?.destroy();
    this.context?.unconfigure();
    this.device?.destroy();
  }
}
