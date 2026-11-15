import wideSource from './shaders/codec-wide.wgsl?raw';
import opsSource from './shaders/codec-ops.wgsl?raw';
import { CodecNetwork, type CodecNetworkParameters } from './codec-network';
import { createHybridCodec, type CodecStep, type DecodeMetrics, type HybridCodec } from './codec-client';
import { combineCodecHashes } from './codec-hashes';
import {
  ATTRIBUTE_KIND,
  CODEC_DISPATCH_WIDTH,
  CODEC_WORKGROUP_SIZE,
  CONTEXT_WIDTH,
  LATENT_NETWORKS,
  NETWORK_ID,
  NETWORK_NAMES,
  STEP_ID,
  STREAM_ID,
  type AttributeKind,
  type NetworkName,
  type StepName,
} from './codec-layout';
import { ACTIVE_OFFSET_BYTES, ANCHOR_STRIDE_FLOATS, FEATURE_COUNT, OFFSETS_PER_ANCHOR } from './layout';
import { check, type Scene, type GpuSceneBuffers } from './types';

const align4 = (size: number) => Math.ceil(size / 4) * 4;
const OPERATION_NAMES = [
  'packCoordinates',
  'concatenate',
  'addContext',
  'packCdfIndices',
  'reconstructFixed',
  'packPositionScaling',
  'reconstructFloat',
] as const;
type OperationName = (typeof OPERATION_NAMES)[number];

interface Operation {
  source?: GPUBuffer;
  secondInput?: GPUBuffer;
  output: GPUBuffer;
  elementCount: number;
  width?: number;
  secondWidth?: number;
  attributeKind?: AttributeKind;
  step?: CodecStep;
}

interface Attribute {
  network: NetworkName;
  priorNetwork?: NetworkName;
  stream: number;
  width: number;
  kind: AttributeKind;
  step: StepName;
  symbolsName: string;
  stageName: string;
}

interface DecodedScene {
  data: GpuSceneBuffers;
  distanceScale: number;
  metrics: DecodeMetrics;
}

async function prepareOperations(device: GPUDevice) {
  const layout = device.createBindGroupLayout({
    entries: [
      ...[0, 1, 2, 5].map(binding => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' as const },
      })),
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const module = device.createShaderModule({
    label: 'Codec reconstruction',
    code: wideSource + '\n' + opsSource,
  });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(message => message.type === 'error');
  check(!errors.length, errors.map(message => message.lineNum + ': ' + message.message).join('\n'));

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const entries = await Promise.all(
    OPERATION_NAMES.map(
      async name =>
        [
          name,
          await device.createComputePipelineAsync({
            layout: pipelineLayout,
            compute: { module, entryPoint: name },
          }),
        ] as const,
    ),
  );
  return { layout, pipelines: new Map(entries) };
}

export async function decodeGpuScene(
  device: GPUDevice,
  scene: Scene,
  progress: (message: string) => void,
  { diagnostic = false, referenceUrl = '' }: { diagnostic?: boolean; referenceUrl?: string } = {},
): Promise<DecodedScene> {
  const start = performance.now();
  const checkReferences = diagnostic && Boolean(referenceUrl);
  let codec: HybridCodec | undefined;
  let engine: CodecNetwork | undefined;
  let scopeClosed = false;
  let encoder: GPUCommandEncoder | undefined;
  let readbackBuffer: GPUBuffer | undefined;
  const allocations = new Set<GPUBuffer>();
  const buffersPendingRelease: GPUBuffer[] = [];

  let currentBytes = 0;
  let peakBytes = 0;
  let readbackMs = 0;
  let readbackBytes = 0;
  let uploadBytes = 0;
  let submissions = 0;
  const stages: DecodeMetrics['stages'] = [];
  const checks: DecodeMetrics['checks'] = [];
  const networkChecks: DecodeMetrics['networkChecks'] = [];
  let mathChecks: DecodeMetrics['mathChecks'];

  const track = (buffer: GPUBuffer) => {
    allocations.add(buffer);
    currentBytes += buffer.size;
    peakBytes = Math.max(peakBytes, currentBytes);
    return buffer;
  };

  // Encoded commands must be submitted before their temporary resources are destroyed.
  const releaseAfterSubmit = (buffer: GPUBuffer) => {
    if (allocations.delete(buffer)) buffersPendingRelease.push(buffer);
  };

  const commands = () => (encoder ??= device.createCommandEncoder({ label: 'Codec stage' }));

  const submit = () => {
    if (encoder) {
      device.queue.submit([encoder.finish()]);
      encoder = undefined;
      submissions++;
    }

    for (const buffer of buffersPendingRelease) {
      currentBytes -= buffer.size;
      buffer.destroy();
    }
    buffersPendingRelease.length = 0;
  };

  const upload = (buffer: GPUBuffer, data: ArrayBufferView<ArrayBuffer>) => {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const alignedBytes = bytes.length - (bytes.length % 4);
    if (alignedBytes) device.queue.writeBuffer(buffer, 0, bytes, 0, alignedBytes);
    if (alignedBytes < bytes.length) {
      const tail = new Uint8Array(4);
      tail.set(bytes.subarray(alignedBytes));
      device.queue.writeBuffer(buffer, alignedBytes, tail);
    }
    uploadBytes += bytes.length;
  };

  const allocate = (label: string, size: number, data?: ArrayBufferView<ArrayBuffer>) => {
    const buffer = track(
      device.createBuffer({
        label,
        size: Math.max(16, Math.ceil(size / 16) * 16),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    );
    if (data) upload(buffer, data);
    return buffer;
  };

  // All readbacks are awaited serially. Only this buffer is mapped.
  const readBuffer = async (buffer: GPUBuffer, size: number) => {
    const began = performance.now();
    const bytes = align4(size);
    if (!readbackBuffer || readbackBuffer.size < bytes) {
      readbackBuffer?.destroy();
      readbackBuffer = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    try {
      commands().copyBufferToBuffer(buffer, 0, readbackBuffer, 0, bytes);
      submit();
      await readbackBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
      readbackBytes += size;
      return readbackBuffer.getMappedRange(0, bytes).slice(0, size);
    } finally {
      readbackBuffer.unmap();
      readbackMs += performance.now() - began;
    }
  };

  const checkArray = async (name: string, actual: Int32Array | Int16Array) => {
    if (!checkReferences) return;

    const { checkCodecArray } = await import('./codec-checks');
    checks.push(await checkCodecArray(referenceUrl, name, actual));
  };

  device.pushErrorScope('validation');
  try {
    const geluTable = allocate('Fixed GELU table', 3073 * 4);
    engine = new CodecNetwork(device, geluTable, scene.anchorCount);

    // Wait for all initializers so a failed shader compilation can still close the Worker.
    const prepared = await Promise.allSettled([
      createHybridCodec(progress, checkReferences),
      engine.initialize(),
      prepareOperations(device),
    ]);
    if (prepared[0].status === 'fulfilled') codec = prepared[0].value;
    for (const task of prepared) {
      if (task.status === 'rejected') throw task.reason;
    }
    check(codec && prepared[2].status === 'fulfilled', 'Codec initialization failed');

    const { layout, pipelines } = prepared[2].value;
    const { initial } = codec;
    const { anchorCount, activeOffsetCount } = initial;
    check(
      anchorCount === scene.anchorCount && activeOffsetCount === scene.activeOffsetCount,
      'Compressed scene metadata mismatch',
    );

    const activeSlots = new Uint32Array(initial.arrays.activeSlots);
    const anchors = allocate('Decoded anchors', anchorCount * ANCHOR_STRIDE_FLOATS * 4);
    const positions = allocate(
      'Decoded positions',
      anchorCount * 3 * 4,
      new Uint8Array(initial.arrays.positions),
    );
    const offsets = allocate('Decoded active offsets', activeOffsetCount * ACTIVE_OFFSET_BYTES);
    const weights = allocate(
      'Decoded render weights',
      initial.arrays.weights.byteLength,
      new Uint8Array(initial.arrays.weights),
    );
    const normalized = allocate(
      'Normalized integer coordinates',
      anchorCount * 3 * 4,
      new Uint8Array(initial.arrays.normalized),
    );
    const slots = allocate('Active anchor slots', activeSlots.byteLength, activeSlots);
    const lutValues = new Int32Array(initial.arrays.gelu);
    upload(geluTable, lutValues);

    const unusedInput = allocate('Unused decoder input', 16);
    const parameters = Object.fromEntries(
      NETWORK_NAMES.map(name => {
        const network = engine!.uploadParameters(new Int32Array(initial.networks[NETWORK_ID[name]]), name);
        track(network.buffer);
        return [name, network];
      }),
    ) as Record<NetworkName, CodecNetworkParameters>;
    stages.push({ name: 'prepare', ms: performance.now() - start });

    if (diagnostic) {
      const { checkCodecMath } = await import('./codec-math-check');
      mathChecks = await checkCodecMath(device, lutValues);
    }
    if (checkReferences) {
      await checkArray('anchor_coords', new Int32Array(initial.arrays.anchor_coords));
      await checkArray('offset_mask_symbols', new Int16Array(initial.arrays.offset_mask_symbols));
    }

    for (const name of Object.keys(initial.arrays)) initial.arrays[name] = new ArrayBuffer(0);
    initial.networks.length = 0;

    const encodeOperation = (
      name: OperationName,
      {
        source = unusedInput,
        secondInput = unusedInput,
        output,
        elementCount,
        width = 0,
        secondWidth = 0,
        attributeKind = ATTRIBUTE_KIND.feature,
        step,
      }: Operation,
    ) => {
      const words = new Uint32Array(8);
      words.set([
        elementCount,
        width,
        attributeKind,
        secondWidth,
        step?.multiplierLow ?? 0,
        step?.multiplierHigh ?? 0,
        step?.shift ?? 0,
        0,
      ]);
      new Float32Array(words.buffer)[7] = step?.floatStep ?? 0;

      const uniform = track(
        device.createBuffer({
          size: words.byteLength,
          usage: GPUBufferUsage.UNIFORM,
          mappedAtCreation: true,
        }),
      );
      new Uint32Array(uniform.getMappedRange()).set(words);
      uniform.unmap();

      const resources = [source, secondInput, slots, output, uniform, geluTable];
      const group = device.createBindGroup({
        layout,
        entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });

      const pass = commands().beginComputePass({ label: name });
      pass.setPipeline(pipelines.get(name)!);
      pass.setBindGroup(0, group);
      const invocations = name === 'packCdfIndices' ? Math.ceil(elementCount / 4) : elementCount;
      const groups = Math.ceil(invocations / CODEC_WORKGROUP_SIZE);
      pass.dispatchWorkgroups(
        Math.min(groups, CODEC_DISPATCH_WIDTH),
        Math.ceil(groups / CODEC_DISPATCH_WIDTH),
      );
      pass.end();
      releaseAfterSubmit(uniform);
    };

    const concatenate = (first: GPUBuffer, second: GPUBuffer, firstWidth: number, secondWidth: number) => {
      const elementCount = anchorCount * (firstWidth + secondWidth);
      const output = allocate('Concatenated context', elementCount * 4);
      encodeOperation('concatenate', {
        source: first,
        secondInput: second,
        output,
        elementCount,
        width: firstWidth,
        secondWidth,
      });
      return output;
    };

    const runNetwork = async (name: NetworkName, input: GPUBuffer) => {
      const network = parameters[name];
      const began = diagnostic ? performance.now() : 0;
      const output = track(engine!.encode(network, input, commands(), name));

      if (diagnostic) {
        const inputValues = new Int32Array(await readBuffer(input, anchorCount * network.inputWidth * 4));
        const expected = await codec!.referenceNetwork(NETWORK_ID[name], inputValues, anchorCount);
        const actual = new Int32Array(await readBuffer(output, anchorCount * network.outputWidth * 4));
        check(actual.length === expected.length, 'Network reference size mismatch');

        for (let index = 0; index < actual.length; index++) {
          if (actual[index] !== expected[index])
            throw new Error(
              name + ' integer mismatch at ' + index + ': ' + actual[index] + ' / ' + expected[index],
            );
        }
        networkChecks.push({ name, count: actual.length, unequal: 0, ms: performance.now() - began });
      }
      return output;
    };

    let cdfIndexBuffer: GPUBuffer | undefined;
    let symbolBuffer: GPUBuffer | undefined;

    const readCdfIndices = async (
      distribution: GPUBuffer,
      elementCount: number,
      width: number,
      attributeKind: AttributeKind,
    ) => {
      if (!cdfIndexBuffer || cdfIndexBuffer.size < align4(elementCount)) {
        if (cdfIndexBuffer) releaseAfterSubmit(cdfIndexBuffer);
        cdfIndexBuffer = allocate('Packed CDF indices', align4(elementCount));
      }

      encodeOperation('packCdfIndices', {
        source: distribution,
        output: cdfIndexBuffer,
        elementCount,
        width,
        attributeKind,
      });
      return new Uint8Array(await readBuffer(cdfIndexBuffer, elementCount));
    };

    const uploadSymbols = (symbols: Int16Array<ArrayBuffer>) => {
      // Submit any use of the previous symbols before overwriting their buffer.
      submit();
      if (!symbolBuffer || symbolBuffer.size < align4(symbols.byteLength)) {
        if (symbolBuffer) releaseAfterSubmit(symbolBuffer);
        symbolBuffer = allocate('Packed decoded symbols', symbols.byteLength);
      }
      upload(symbolBuffer, symbols);
      return symbolBuffer;
    };

    const decodeFixedAttribute = async (
      distribution: GPUBuffer,
      stream: number,
      step: StepName,
      symbolsName: string,
    ) => {
      const cdfIndices = await readCdfIndices(distribution, anchorCount, 1, ATTRIBUTE_KIND.feature);
      const symbols = await codec!.entropy(stream, cdfIndices, symbolsName, [anchorCount, 1]);
      await checkArray(symbolsName, symbols);

      const output = allocate('Reconstructed fixed attribute', anchorCount * 4);
      encodeOperation('reconstructFixed', {
        source: distribution,
        secondInput: uploadSymbols(symbols),
        output,
        elementCount: anchorCount,
        width: 1,
        step: codec!.steps[STEP_ID[step]],
      });
      return output;
    };

    const beginStage = (name: string) => {
      progress('Decoding ' + name + ' on WebGPU…');
      return performance.now();
    };

    const endStage = (name: string, began: number) => stages.push({ name, ms: performance.now() - began });

    encodeOperation('packCoordinates', { source: positions, output: anchors, elementCount: anchorCount * 3 });
    releaseAfterSubmit(positions);
    let context = await runNetwork('coordinate', normalized);
    releaseAfterSubmit(normalized);

    let latent: GPUBuffer | undefined;
    for (const [stage, name] of LATENT_NETWORKS.entries()) {
      const began = beginStage('latent ' + stage);
      const input = stage ? concatenate(context, latent!, CONTEXT_WIDTH, stage) : context;
      const distribution = await runNetwork(name, input);
      if (stage) releaseAfterSubmit(input);

      const channel = await decodeFixedAttribute(
        distribution,
        STREAM_ID.firstLatent + stage,
        'latent',
        'anchor_latent_symbols_' + stage,
      );
      releaseAfterSubmit(distribution);

      if (latent) {
        const previous = latent;
        latent = concatenate(previous, channel, stage, 1);
        releaseAfterSubmit(previous);
        releaseAfterSubmit(channel);
      } else {
        latent = channel;
      }
      endStage('latent_' + stage, began);
    }

    let began = beginStage('position scaling');
    const latentContext = await runNetwork('latentContext', latent!);
    releaseAfterSubmit(latent!);

    const combined = concatenate(context, latentContext, CONTEXT_WIDTH, CONTEXT_WIDTH);
    releaseAfterSubmit(context);
    releaseAfterSubmit(latentContext);
    submit(); // Release the two large inputs before allocating the fused context.
    context = await runNetwork('latentFusion', combined);
    releaseAfterSubmit(combined);

    const positionScalingDistribution = await runNetwork('positionScaling', context);
    const logPositionScalingQ20 = await decodeFixedAttribute(
      positionScalingDistribution,
      STREAM_ID.positionScaling,
      'positionScaling',
      'position_scaling_symbols',
    );
    releaseAfterSubmit(positionScalingDistribution);

    encodeOperation('packPositionScaling', {
      source: logPositionScalingQ20,
      output: anchors,
      elementCount: anchorCount,
    });

    const positionScalingContext = await runNetwork('positionScalingContext', logPositionScalingQ20);
    releaseAfterSubmit(logPositionScalingQ20);
    endStage('position_scaling', began);

    const attributes: Attribute[] = [
      {
        network: 'offset',
        priorNetwork: 'offsetPrior',
        stream: STREAM_ID.offset,
        width: OFFSETS_PER_ANCHOR * 3,
        kind: ATTRIBUTE_KIND.offset,
        step: 'offset',
        symbolsName: 'offset_symbols',
        stageName: 'offset',
      },
      {
        network: 'gaussianScaling',
        priorNetwork: 'gaussianScalingPrior',
        stream: STREAM_ID.gaussianScaling,
        width: 3,
        kind: ATTRIBUTE_KIND.gaussianScaling,
        step: 'gaussianScaling',
        symbolsName: 'gaussian_scaling_symbols',
        stageName: 'gaussian_scaling',
      },
      {
        network: 'feature',
        stream: STREAM_ID.feature,
        width: FEATURE_COUNT,
        kind: ATTRIBUTE_KIND.feature,
        step: 'feature',
        symbolsName: 'feat_symbols',
        stageName: 'anchor_feat',
      },
    ];

    const predictAttribute = async (attribute: Attribute) => {
      let input = context;
      if (attribute.priorNetwork) {
        input = await runNetwork(attribute.priorNetwork, positionScalingContext);
        encodeOperation('addContext', {
          secondInput: context,
          output: input,
          elementCount: anchorCount * CONTEXT_WIDTH,
        });
      }

      const distribution = await runNetwork(attribute.network, input);
      if (input !== context) releaseAfterSubmit(input);
      // Gaussian scaling is the last consumer of the position-scaling context.
      if (attribute.kind === ATTRIBUTE_KIND.gaussianScaling) releaseAfterSubmit(positionScalingContext);
      return distribution;
    };

    let distribution = await predictAttribute(attributes[0]);
    for (let index = 0; index < attributes.length; index++) {
      const attribute = attributes[index];
      const { width, kind, step } = attribute;
      const isOffset = kind === ATTRIBUTE_KIND.offset;
      began = beginStage(attribute.stageName);
      const elementCount = isOffset ? activeOffsetCount * 3 : anchorCount * width;
      const cdfIndices = await readCdfIndices(distribution, elementCount, width, kind);

      // Keep at most one prediction ahead. The Worker remains serial while the
      // GPU computes the next independent attribute. Means stay alive until used.
      const [symbols, nextDistribution] = await Promise.all([
        codec.entropy(
          attribute.stream,
          cdfIndices,
          attribute.symbolsName,
          isOffset ? [elementCount] : [anchorCount, width],
        ),
        index + 1 < attributes.length
          ? predictAttribute(attributes[index + 1]).then(buffer => {
              submit();
              return buffer;
            })
          : Promise.resolve(undefined),
      ]);
      await checkArray(attribute.symbolsName, symbols);

      encodeOperation('reconstructFloat', {
        source: distribution,
        secondInput: uploadSymbols(symbols),
        output: isOffset ? offsets : anchors,
        elementCount,
        width,
        attributeKind: kind,
        step: codec.steps[STEP_ID[step]],
      });
      releaseAfterSubmit(distribution);

      if (nextDistribution) distribution = nextDistribution;
      endStage(attribute.stageName, began);
    }

    releaseAfterSubmit(context);
    submit();
    await device.queue.onSubmittedWorkDone();
    const hashStart = performance.now();
    const hashes = await combineCodecHashes(codec.arrayHashes);
    codec.metrics.hashMs += performance.now() - hashStart;
    await codec.close();

    const error = await device.popErrorScope();
    scopeClosed = true;
    check(!error, 'GPU decoder: ' + error?.message);
    const totalMs = performance.now() - start;

    for (const buffer of [anchors, offsets, weights]) {
      allocations.delete(buffer);
      currentBytes -= buffer.size;
    }

    return {
      data: { anchors, offsets, weights, activeSlots },
      distanceScale: initial.distanceScale,
      metrics: {
        source: 'webgpu+wasm',
        totalMs,
        ...codec.metrics,
        stages,
        readbackMs,
        readbackBytes,
        uploadBytes,
        peakTrackedGpuBytes: peakBytes,
        readbackBufferBytes: readbackBuffer?.size ?? 0,
        submissions,
        diagnostic,
        hashes,
        referenceCheck: diagnostic
          ? checkReferences
            ? { status: 'PASS' }
            : { status: 'SKIPPED', reason: 'NPY reference files were not provided' }
          : undefined,
        checks,
        networkChecks,
        mathChecks,
      },
    };
  } finally {
    engine?.dispose();
    readbackBuffer?.destroy();
    for (const buffer of allocations) buffer.destroy();
    for (const buffer of buffersPendingRelease) buffer.destroy();
    await codec?.close();
    if (!scopeClosed) {
      const error = await device.popErrorScope();
      check(!error, 'GPU decoder: ' + error?.message);
    }
  }
}
