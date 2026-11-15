import type { CodecArrayCheck } from './codec-checks';
import type { checkCodecMath } from './codec-math-check';
import type { CodecArrayHash, CodecHashes } from './codec-hashes';
import { assetUrl } from './types';

export interface CodecStep {
  multiplierLow: number;
  multiplierHigh: number;
  shift: number;
  floatStep: number;
}

export interface CodecInitial {
  anchorCount: number;
  activeOffsetCount: number;
  distanceScale: number;
  arrays: Record<string, ArrayBuffer> & {
    positions: ArrayBuffer;
    normalized: ArrayBuffer;
    activeSlots: ArrayBuffer;
    weights: ArrayBuffer;
    gelu: ArrayBuffer;
  };
  networks: ArrayBuffer[];
}

export interface CodecWorkerMetrics {
  inputBytes: number;
  tableBytes: number;
  workerInitMs: number;
  wasmPrepareMs: number;
  entropyMs: number;
  /** Accumulated digest time, including asynchronous waits. Included in decoder totalMs. */
  hashMs: number;
  entropyCalls: number;
  indexBytes: number;
  symbolBytes: number;
  linearMemoryBytes: number;
}

export interface NetworkCheck {
  name: string;
  count: number;
  unequal: number;
  ms: number;
}

export interface DecodeMetrics extends CodecWorkerMetrics {
  source: 'webgpu+wasm';
  /** Wall time from starting the decoder to having render buffers ready. */
  totalMs: number;
  stages: { name: string; ms: number }[];
  readbackMs: number;
  readbackBytes: number;
  uploadBytes: number;
  peakTrackedGpuBytes: number;
  readbackBufferBytes: number;
  submissions: number;
  diagnostic: boolean;
  hashes: CodecHashes;
  referenceCheck?: { status: 'PASS' | 'SKIPPED'; reason?: string };
  checks: CodecArrayCheck[];
  networkChecks: NetworkCheck[];
  mathChecks?: Awaited<ReturnType<typeof checkCodecMath>>;
}

export type CodecRequest =
  | { id: number; type: 'open'; manifestUrl: string; includeReferenceArrays: boolean }
  | {
      id: number;
      type: 'entropy';
      stream: number;
      indices: Uint8Array<ArrayBuffer>;
      name: string;
      shape: number[];
    }
  | { id: number; type: 'reference'; network: number; input: Int32Array<ArrayBuffer>; rows: number }
  | { id: number; type: 'close' };

export type CodecResponse =
  | { type: 'progress'; message: string }
  | { id: number; type: 'error'; message: string }
  | {
      id: number;
      type: 'open';
      initial: CodecInitial;
      steps: CodecStep[];
      metrics: CodecWorkerMetrics;
      arrayHashes: CodecArrayHash[];
    }
  | {
      id: number;
      type: 'entropy';
      symbols: Int16Array<ArrayBuffer>;
      ms: number;
      linearMemoryBytes: number;
      arrayHash: CodecArrayHash;
      hashMs: number;
    }
  | { id: number; type: 'reference'; output: Int32Array<ArrayBuffer> }
  | { id: number; type: 'close' };

type Reply = Exclude<CodecResponse, { type: 'progress' } | { type: 'error' }>;

export interface HybridCodec {
  initial: CodecInitial;
  steps: CodecStep[];
  metrics: CodecWorkerMetrics;
  arrayHashes: CodecArrayHash[];

  /** Transfers ownership of the index buffer to the worker. */
  entropy(
    stream: number,
    indices: Uint8Array<ArrayBuffer>,
    name: string,
    shape: number[],
  ): Promise<Int16Array<ArrayBuffer>>;

  /** Explicit diagnostic only. Transfers ownership of the input buffer. */
  referenceNetwork(
    network: number,
    input: Int32Array<ArrayBuffer>,
    rows: number,
  ): Promise<Int32Array<ArrayBuffer>>;

  close(): Promise<void>;
}

export async function createHybridCodec(
  progress: (message: string) => void,
  includeReferenceArrays = false,
): Promise<HybridCodec> {
  const worker = new Worker(new URL('./codec-worker.ts', import.meta.url), { type: 'module' });
  const pending = new Map<number, { resolve(reply: Reply): void; reject(error: Error): void }>();
  let nextId = 1;
  let closed = false;

  const stop = (error: Error) => {
    closed = true;
    worker.terminate();
    for (const task of pending.values()) task.reject(error);
    pending.clear();
  };

  const request = (message: CodecRequest, transfers: Transferable[] = []) =>
    new Promise<Reply>((resolve, reject) => {
      if (closed) {
        reject(new Error('Codec worker is closed'));
        return;
      }

      pending.set(message.id, { resolve, reject });
      try {
        worker.postMessage(message, transfers);
      } catch (error) {
        stop(error instanceof Error ? error : new Error(String(error)));
      }
    });

  worker.onmessage = ({ data }: MessageEvent<CodecResponse>) => {
    if (data.type === 'progress') {
      progress(data.message);
      return;
    }

    if (data.type === 'error') {
      stop(new Error(data.message));
      return;
    }

    const task = pending.get(data.id);
    pending.delete(data.id);
    task?.resolve(data);
  };
  worker.onerror = event => stop(new Error(event.message || 'Codec worker failed'));
  worker.onmessageerror = () => stop(new Error('Could not receive codec worker data'));

  try {
    const reply = await request({
      id: nextId++,
      type: 'open',
      manifestUrl: assetUrl('codec/manifest.json'),
      includeReferenceArrays,
    });
    if (reply.type !== 'open') throw new Error('Unexpected codec initialization reply');

    const { initial, steps, metrics, arrayHashes } = reply;
    return {
      initial,
      steps,
      metrics,
      arrayHashes,

      async entropy(stream, indices, name, shape) {
        const bytes = indices.byteLength;
        const response = await request({ id: nextId++, type: 'entropy', stream, indices, name, shape }, [
          indices.buffer,
        ]);
        if (response.type !== 'entropy') throw new Error('Unexpected entropy reply');

        metrics.entropyMs += response.ms;
        metrics.hashMs += response.hashMs;
        arrayHashes.push(response.arrayHash);
        metrics.entropyCalls++;
        metrics.indexBytes += bytes;
        metrics.symbolBytes += response.symbols.byteLength;
        metrics.linearMemoryBytes = response.linearMemoryBytes;
        return response.symbols;
      },

      async referenceNetwork(network, input, rows) {
        const response = await request({ id: nextId++, type: 'reference', network, input, rows }, [
          input.buffer,
        ]);
        if (response.type !== 'reference') throw new Error('Unexpected network reference reply');
        return response.output;
      },

      async close() {
        if (closed) return;

        try {
          await request({ id: nextId++, type: 'close' });
        } finally {
          stop(new Error('Codec worker is closed'));
        }
      },
    };
  } catch (error) {
    stop(error instanceof Error ? error : new Error(String(error)));
    throw error;
  }
}
