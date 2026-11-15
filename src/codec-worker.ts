import { check, fetchChecked } from './types';
import type {
  CodecInitial,
  CodecWorkerMetrics,
  CodecRequest,
  CodecResponse,
  CodecStep,
} from './codec-client';
import { NETWORK_NAMES, STEP_ID } from './codec-layout';
import { hashCodecArray, type CodecArrayHash } from './codec-hashes';

interface Manifest {
  scene: string;
  tables: string;
  config: {
    anchor_feat_channels: number;
    n_offsets: number;
    latent_context_channels: number;
    anchor_context_channels: number;
    expand_offset_entropy_hidden: boolean;
    anchor_latent_channels: number;
    grid_size_in_stream: boolean;
    add_opacity_dist: boolean;
    add_cov_dist: boolean;
    add_color_dist: boolean;
  };
  anchorGridSize: number;
}

interface CodecModule {
  HEAPU8: Uint8Array<ArrayBuffer>;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  UTF8ToString(pointer: number): string;
  _hybrid_open(
    bytes: number,
    byteCount: number,
    tables: number,
    tableBytes: number,
    config: number,
    configCount: number,
    anchorGridSize: number,
  ): number;
  _hybrid_clear_arrays(): void;
  _hybrid_entropy(stream: number, indices: number, count: number): number;
  _hybrid_symbols_data(): number;
  _hybrid_symbols_bytes(): number;
  _hybrid_symbols_count(): number;
  _hybrid_network_count(): number;
  _hybrid_network_data(index: number): number;
  _hybrid_network_bytes(index: number): number;
  _hybrid_network_reference(index: number, input: number, rows: number): number;
  _hybrid_reference_bytes(): number;
  _hybrid_step_mul_lo(index: number): number;
  _hybrid_step_mul_hi(index: number): number;
  _hybrid_step_shift(index: number): number;
  _hybrid_step_float(index: number): number;
  _codec_error(): number;
  _codec_array_count(): number;
  _codec_array_name(index: number): number;
  _codec_array_data(index: number): number;
  _codec_array_bytes(index: number): number;
  _codec_anchor_count(): number;
  _codec_active_count(): number;
  _codec_distance_scale(): number;
  _codec_release(): void;
}

const worker = self as unknown as {
  onmessage: ((event: MessageEvent<CodecRequest>) => void) | null;
  postMessage(message: CodecResponse, transfers?: Transferable[]): void;
};
let module: CodecModule | undefined;
const inputs: number[] = [];
let entropyInput = 0;
let entropyCapacity = 0;
const progress = (message: string) => worker.postMessage({ type: 'progress', message });

function upload(bytes: Uint8Array): number {
  check(module, 'Codec is not initialized');
  // malloc(0) is implementation-defined. Empty streams still use a valid pointer.
  const pointer = module._malloc(Math.max(1, bytes.byteLength));
  check(pointer, 'Could not allocate codec input memory');
  module.HEAPU8.set(bytes, pointer);
  return pointer;
}

function copy(pointer: number, bytes: number): ArrayBuffer {
  check(module, 'Codec is not initialized');
  return module.HEAPU8.slice(pointer, pointer + bytes).buffer;
}

function release() {
  if (!module) return;

  module._codec_release();
  module._free(entropyInput);
  entropyInput = entropyCapacity = 0;
  inputs.splice(0).forEach(pointer => module!._free(pointer));
  module = undefined;
}

async function open(id: number, manifestUrl: string, includeReferenceArrays: boolean) {
  check(!module, 'Codec is already initialized');
  const start = performance.now();
  progress('Loading compressed scene and staged WASM decoder…');

  const manifest: Manifest = await fetchChecked(manifestUrl).then(response => response.json());
  const config = manifest.config;

  // Parameter order matches the C++ decoder's hybrid_open configuration.
  const configValues = [
    config.anchor_feat_channels,
    config.n_offsets,
    config.latent_context_channels,
    config.anchor_context_channels,
    Number(config.expand_offset_entropy_hidden),
    config.anchor_latent_channels,
    Number(config.grid_size_in_stream),
    Number(config.add_opacity_dist),
    Number(config.add_cov_dist),
    Number(config.add_color_dist),
  ];
  check(configValues.every(Number.isInteger), 'Invalid codec configuration');

  const moduleUrl = new URL('decoder.mjs', manifestUrl).href;
  const [loadedModule, scene, tables] = await Promise.all([
    import(/* @vite-ignore */ moduleUrl).then(({ default: createModule }) =>
      createModule(),
    ) as Promise<CodecModule>,
    fetchChecked(new URL(manifest.scene, manifestUrl).href).then(response => response.arrayBuffer()),
    fetchChecked(new URL(manifest.tables, manifestUrl).href).then(response => response.arrayBuffer()),
  ]);
  module = loadedModule;

  const persistent = (bytes: Uint8Array) => {
    const pointer = upload(bytes);
    inputs.push(pointer);
    return pointer;
  };
  const scenePointer = persistent(new Uint8Array(scene));
  const tablePointer = persistent(new Uint8Array(tables));
  const configPointer = persistent(new Uint8Array(new Int32Array(configValues).buffer));

  progress('Decoding geometry and offset masks…');
  const openStart = performance.now();
  const status = module._hybrid_open(
    scenePointer,
    scene.byteLength,
    tablePointer,
    tables.byteLength,
    configPointer,
    configValues.length,
    manifest.anchorGridSize,
  );
  check(status === 0, 'Codec: ' + module.UTF8ToString(module._codec_error()));
  const wasmPrepareMs = performance.now() - openStart;

  const arrays: Record<string, ArrayBuffer> = {};
  const arrayHashes: CodecArrayHash[] = [];
  let hashMs = 0;
  for (let index = 0; index < module._codec_array_count(); index++) {
    const name = module.UTF8ToString(module._codec_array_name(index));
    const pointer = module._codec_array_data(index);
    const byteCount = module._codec_array_bytes(index);
    if (name === 'anchor_coords' || name === 'offset_mask_symbols') {
      const coordinates = name === 'anchor_coords';
      const shape = coordinates ? [module._codec_anchor_count(), 3] : [module._codec_anchor_count()];
      const hashStart = performance.now();
      arrayHashes.push(
        await hashCodecArray(
          name,
          coordinates ? '<i4' : '<i2',
          shape,
          module.HEAPU8.subarray(pointer, pointer + byteCount),
        ),
      );
      hashMs += performance.now() - hashStart;
    }

    if (
      name === 'positions' ||
      name === 'normalized' ||
      name === 'activeSlots' ||
      name === 'weights' ||
      name === 'gelu' ||
      (includeReferenceArrays && (name === 'anchor_coords' || name === 'offset_mask_symbols'))
    ) {
      arrays[name] = copy(pointer, byteCount);
    }
  }

  check(
    arrays.positions && arrays.normalized && arrays.activeSlots && arrays.weights && arrays.gelu,
    'Codec did not produce GPU initialization arrays',
  );
  if (includeReferenceArrays)
    check(arrays.anchor_coords && arrays.offset_mask_symbols, 'Codec did not produce diagnostic arrays');

  const networks = Array.from({ length: module._hybrid_network_count() }, (_, index) =>
    copy(module!._hybrid_network_data(index), module!._hybrid_network_bytes(index)),
  );
  check(networks.length === NETWORK_NAMES.length, 'Unexpected integer network count');

  const steps: CodecStep[] = Array.from({ length: Object.keys(STEP_ID).length }, (_, index) => ({
    multiplierLow: module!._hybrid_step_mul_lo(index) >>> 0,
    multiplierHigh: module!._hybrid_step_mul_hi(index) >>> 0,
    shift: module!._hybrid_step_shift(index),
    floatStep: module!._hybrid_step_float(index),
  }));

  const initial: CodecInitial = {
    anchorCount: module._codec_anchor_count(),
    activeOffsetCount: module._codec_active_count(),
    distanceScale: module._codec_distance_scale(),
    arrays: {
      ...arrays,
      positions: arrays.positions,
      normalized: arrays.normalized,
      activeSlots: arrays.activeSlots,
      weights: arrays.weights,
      gelu: arrays.gelu,
    },
    networks,
  };

  const metrics: CodecWorkerMetrics = {
    inputBytes: scene.byteLength,
    tableBytes: tables.byteLength,
    workerInitMs: performance.now() - start,
    wasmPrepareMs,
    entropyMs: 0,
    hashMs,
    entropyCalls: 0,
    indexBytes: 0,
    symbolBytes: 0,
    linearMemoryBytes: module.HEAPU8.buffer.byteLength,
  };

  module._hybrid_clear_arrays();
  worker.postMessage({ id, type: 'open', initial, steps, metrics, arrayHashes }, [
    ...Object.values(initial.arrays),
    ...networks,
  ]);
}

async function handle(request: CodecRequest) {
  if (request.type === 'open') {
    await open(request.id, request.manifestUrl, request.includeReferenceArrays);
    return;
  }

  if (request.type === 'close') {
    release();
    worker.postMessage({ id: request.id, type: 'close' });
    return;
  }

  check(module, 'Codec is not initialized');

  if (request.type === 'entropy') {
    if (request.indices.byteLength > entropyCapacity) {
      module._free(entropyInput);
      entropyInput = 0;
      entropyCapacity = 0;
      entropyInput = module._malloc(request.indices.byteLength);
      check(entropyInput, 'Could not allocate entropy input memory');
      entropyCapacity = request.indices.byteLength;
    }

    module.HEAPU8.set(request.indices, entropyInput);
    const start = performance.now();
    const status = module._hybrid_entropy(request.stream, entropyInput, request.indices.length);
    const ms = performance.now() - start;

    check(status === 0, 'Entropy decoder: ' + module.UTF8ToString(module._codec_error()));
    check(module._hybrid_symbols_count() === request.indices.length, 'Entropy symbol count mismatch');

    const symbols = new Int16Array(copy(module._hybrid_symbols_data(), module._hybrid_symbols_bytes()));
    const hashStart = performance.now();
    const arrayHash = await hashCodecArray(request.name, '<i2', request.shape, new Uint8Array(symbols.buffer));
    const hashMs = performance.now() - hashStart;
    worker.postMessage(
      {
        id: request.id,
        type: 'entropy',
        symbols,
        ms,
        linearMemoryBytes: module.HEAPU8.buffer.byteLength,
        arrayHash,
        hashMs,
      },
      [symbols.buffer],
    );
  } else {
    const pointer = upload(
      new Uint8Array(request.input.buffer, request.input.byteOffset, request.input.byteLength),
    );
    try {
      const outputPointer = module._hybrid_network_reference(request.network, pointer, request.rows);
      check(outputPointer, 'Network reference: ' + module.UTF8ToString(module._codec_error()));
      const output = new Int32Array(copy(outputPointer, module._hybrid_reference_bytes()));
      worker.postMessage({ id: request.id, type: 'reference', output }, [output.buffer]);
    } finally {
      module._free(pointer);
    }
  }
}

// Keep WASM state mutations serialized even if callers queue independent work.
let queue = Promise.resolve();
worker.onmessage = ({ data }) => {
  queue = queue
    .then(() => handle(data))
    .catch(error => {
      release();
      worker.postMessage({
        id: data.id,
        type: 'error',
        message: error instanceof Error ? error.message : String(error),
      });
    });
};
