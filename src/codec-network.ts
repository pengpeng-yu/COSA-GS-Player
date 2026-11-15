import integerSource from './shaders/codec-integer.wgsl?raw';
import wideSource from './shaders/codec-wide.wgsl?raw';
import { check } from './types';

export interface CodecNetworkParameters {
  buffer: GPUBuffer;
  inputWidth: number;
  outputWidth: number;
  layers: number;
}

export class CodecNetwork {
  private pipelines = new Map<number, GPUComputePipeline>();
  private readonly dispatchUniform: GPUBuffer;
  private readonly groupsX: number;

  constructor(
    private readonly device: GPUDevice,
    private readonly gelu: GPUBuffer,
    private readonly rowCount: number,
  ) {
    check(Number.isSafeInteger(rowCount) && rowCount > 0, 'Invalid integer network row count');
    this.groupsX = Math.min(rowCount, device.limits.maxComputeWorkgroupsPerDimension);
    check(
      Math.ceil(rowCount / this.groupsX) <= device.limits.maxComputeWorkgroupsPerDimension,
      'Integer network dispatch is too large',
    );

    this.dispatchUniform = device.createBuffer({
      label: 'Integer network row count',
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    new Uint32Array(this.dispatchUniform.getMappedRange()).set([rowCount, this.groupsX, 0, 0]);
    this.dispatchUniform.unmap();
  }

  async initialize() {
    const module = this.device.createShaderModule({
      label: 'Exact integer codec networks',
      code: wideSource + '\n' + integerSource,
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    check(!errors.length, errors.map(message => message.lineNum + ': ' + message.message).join('\n'));

    await Promise.all(
      [1, 2].map(async layers => {
        const pipeline = await this.device.createComputePipelineAsync({
          label: 'Integer network with ' + layers + ' layers',
          layout: 'auto',
          compute: { module, entryPoint: 'main', constants: { LAYERS: layers } },
        });
        this.pipelines.set(layers, pipeline);
      }),
    );
  }

  uploadParameters(words: Int32Array | Uint32Array, label: string): CodecNetworkParameters {
    check(words.length >= 8, 'Invalid packed integer network');

    const layers = words[0];
    const inputWidth = words[6];
    const outputWidth = words[7];
    check(
      (layers === 1 || layers === 2) &&
        inputWidth > 0 &&
        inputWidth <= 64 &&
        outputWidth > 0 &&
        outputWidth <= 64,
      'Unsupported integer network dimensions',
    );

    for (let layer = 0; layer < layers; layer++) {
      const base = words[4 + layer];
      check(base >= 8 && base + 8 <= words.length, 'Invalid packed network layer');

      const inputs = words[base];
      const outputs = words[base + 1];
      check(
        inputs > 0 &&
          inputs <= 64 &&
          outputs > 0 &&
          outputs <= 64 &&
          base + 8 + outputs * (Math.ceil(inputs / 4) + 2) <= words.length,
        'Invalid packed network parameters',
      );
    }

    const buffer = this.device.createBuffer({
      label,
      size: words.byteLength,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(words.buffer, words.byteOffset, words.byteLength),
    );
    buffer.unmap();

    return { buffer, inputWidth, outputWidth, layers };
  }

  encode(
    network: CodecNetworkParameters,
    input: GPUBuffer,
    encoder: GPUCommandEncoder,
    label: string,
  ): GPUBuffer {
    check(input.size >= this.rowCount * network.inputWidth * 4, 'Integer network input buffer is too small');
    const pipeline = this.pipelines.get(network.layers);
    check(pipeline, 'Integer network engine has not been initialized');

    const output = this.device.createBuffer({
      label,
      size: this.rowCount * network.outputWidth * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const bindings = [network.buffer, input, this.gelu, output, this.dispatchUniform];
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(this.groupsX, Math.ceil(this.rowCount / this.groupsX));
    pass.end();
    return output;
  }

  dispose() {
    this.dispatchUniform.destroy();
    this.pipelines.clear();
  }
}
