import { frameUniform } from './camera';
import { DRAW_ARGUMENT_BYTES, MLP_SAMPLE_STRIDE, OFFSETS_PER_ANCHOR } from './layout';
import type { Mlp } from './mlp';
import { check, fetchChecked, type Camera, type RenderMlpReference, type Scene } from './types';

interface DiagnosticResources {
  device: GPUDevice;
  scene: Scene;
  mlp: Mlp;
  uniform: GPUBuffer;
  control: GPUBuffer;
  referenceUrl: string;
}

const HEADS = [
  { name: 'opacity', channels: OFFSETS_PER_ANCHOR, start: 0, channelsPerSlot: 1 },
  { name: 'color', channels: OFFSETS_PER_ANCHOR * 3, start: 10, channelsPerSlot: 3 },
  { name: 'cov', channels: OFFSETS_PER_ANCHOR * 7, start: 40, channelsPerSlot: 7 },
] as const;

// Loaded only by an explicit comparison. GPU sampling buffers live for one check.
export class MlpDiagnostics {
  private constructor(
    private readonly resources: DiagnosticResources,
    private readonly reference: RenderMlpReference,
  ) {}

  static async load(resources: DiagnosticResources) {
    const reference: RenderMlpReference = await fetchChecked(resources.referenceUrl).then(
      response => response.json(),
    );

    const seen = new Set<number>();
    check(reference.rows.length > 0, 'MLP reference contains no sample anchors');
    for (const row of reference.rows) {
      check(
        Number.isInteger(row) && row >= 0 && row < resources.scene.anchorCount && !seen.has(row),
        'Invalid or duplicate MLP reference row',
      );
      seen.add(row);
    }

    return new MlpDiagnostics(resources, reference);
  }

  private async read(control: GPUBuffer, sampleOffset = 0, sampleBytes = 0) {
    const { device, mlp } = this.resources;
    const headerBytes = DRAW_ARGUMENT_BYTES * 2;
    const readback = device.createBuffer({
      size: headerBytes + sampleBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    try {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(control, 0, readback, 0, DRAW_ARGUMENT_BYTES);
      encoder.copyBufferToBuffer(mlp.worklist, 0, readback, DRAW_ARGUMENT_BYTES, DRAW_ARGUMENT_BYTES);
      if (sampleBytes) {
        encoder.copyBufferToBuffer(control, sampleOffset, readback, headerBytes, sampleBytes);
      }
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return readback.getMappedRange().slice(0);
    } finally {
      readback.destroy();
    }
  }

  async check(camera: Camera, view: number) {
    const { device, scene, mlp, uniform, control } = this.resources;
    const viewReference = this.reference.views.find(item => item.view === view);
    let memory: ArrayBuffer;

    if (viewReference) {
      // The runtime array starts with an anchor-to-sample map, followed by outputs.
      const sampleOffset = DRAW_ARGUMENT_BYTES + scene.anchorCount * 4;
      const sampleBytes = this.reference.rows.length * MLP_SAMPLE_STRIDE * 4;
      const data = new Uint32Array(Math.ceil((sampleOffset + sampleBytes) / 16) * 4);
      data[0] = 6;
      this.reference.rows.forEach((anchor, sample) => {
        data[DRAW_ARGUMENT_BYTES / 4 + anchor] = sample + 1;
      });

      const sampledControl = device.createBuffer({
        label: 'temporary MLP validation samples',
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

      try {
        const target = await mlp.diagnosticTarget(sampledControl);
        device.queue.writeBuffer(sampledControl, 0, data);
        device.queue.writeBuffer(uniform, 0, frameUniform(camera, scene));
        const encoder = device.createCommandEncoder();
        mlp.encode(encoder, undefined, target);
        device.queue.submit([encoder.finish()]);
        memory = await this.read(sampledControl, sampleOffset, sampleBytes);
      } finally {
        sampledControl.destroy();
      }
    } else {
      memory = await this.read(control);
    }

    const words = new Uint32Array(memory);
    const values = new Float32Array(memory);
    const results: Record<string, { maxAbsError: number; failures: number; checkedValues: number }> = {};
    if (viewReference) {
      for (const head of HEADS) {
        let maxAbsError = 0;
        let failures = 0;
        let checkedValues = 0;
        for (let sample = 0; sample < this.reference.rows.length; sample++) {
          const base = DRAW_ARGUMENT_BYTES / 4 * 2 + sample * MLP_SAMPLE_STRIDE;
          if (!words[base]) continue;

          const mask = mlp.offsetMask(this.reference.rows[sample]);
          for (let channel = 0; channel < head.channels; channel++) {
            const slot = Math.floor(channel / head.channelsPerSlot);
            if (!(mask & (1 << slot))) continue;

            const actual = values[base + 1 + head.start + channel];
            const expected = viewReference.outputs[head.name][sample][channel];
            const difference = Math.abs(actual - expected);
            checkedValues++;
            maxAbsError = Math.max(maxAbsError, difference);
            if (!Number.isFinite(actual) || difference > 5e-5 + 1e-4 * Math.abs(expected)) failures++;
          }
        }
        results[head.name] = { maxAbsError, failures, checkedValues };
      }
    }

    return {
      visibleGaussians: words[1],
      visibleAnchors: words[DRAW_ARGUMENT_BYTES / 4],
      mlp: results,
      mlpDiagnosticPath: viewReference ? 'fused kernel with compile-time sampling enabled' : 'not sampled',
      mlpSampleAnchors: viewReference ? this.reference.rows.length : 0,
    };
  }
}
