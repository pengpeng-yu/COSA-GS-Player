import './style.css';
import { Renderer, type FrameTiming } from './renderer';
import { validateSettings, type RenderSettings } from './settings';
import { orbit, dolly } from './camera';
import {
  assetUrl,
  check,
  fetchChecked,
  type Camera,
  type Scene,
  type GpuSceneBuffers,
  type CudaPsnrReference,
} from './types';
import type { DecodeMetrics } from './codec-client';
import { measurePsnr } from './quality';
import { showCodecHashes } from './hash-panel';

const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = element<HTMLCanvasElement>('render');
const status = element('status');
const select = element<HTMLSelectElement>('view');
const precision = element<HTMLSelectElement>('precision');
const query = new URLSearchParams(location.search);
const requestedSource = query.get('source') ?? 'compressed';
let source = requestedSource;
const skipped: string[] = [];
const profileGpu = query.has('profile');
const target = element<HTMLImageElement>('target');
const psnr = element('psnr');
const cudaPsnr = element('cuda-psnr');
const metrics = element('metrics');
const differenceCanvas = element<HTMLCanvasElement>('difference');
const differencePanel = element<HTMLDetailsElement>('difference-panel');

let scene: Scene;
let cudaPsnrReference: CudaPsnrReference;
let renderer: Renderer;
let currentCamera: Camera;
let selectedViewIndex = 0;
let isInteractiveCamera = false;
let lastTiming: FrameTiming | undefined;
let decoderMetrics: DecodeMetrics | undefined;
let quality: (ReturnType<typeof measurePsnr> & { comparisonMs: number }) | undefined;
let targetPixels: { url: string; pixels: Uint8ClampedArray<ArrayBuffer> } | undefined;

let operationQueue = Promise.resolve();
let ready = false;
let interactiveDirty = false;
let interactionLoopRunning = false;
let pendingOperationCount = 0;

function message(text: string) {
  status.textContent = text;
}

function fail(error: unknown) {
  const text = error instanceof Error ? (error.stack ?? error.message) : String(error);
  status.textContent = text;
  status.classList.add('error');
  console.error(error);
  api.error = text;
}

function controls() {
  for (const button of document.querySelectorAll<HTMLButtonElement>('nav button'))
    button.disabled = !ready || pendingOperationCount > 0;
  select.disabled = !ready || pendingOperationCount > 0;
  precision.disabled = !ready || pendingOperationCount > 0;
  element<HTMLButtonElement>('compare').disabled = !ready || pendingOperationCount > 0 || isInteractiveCamera;
}

function enqueue<T>(operation: () => Promise<T>): Promise<T> {
  pendingOperationCount++;
  controls();

  const next = operationQueue.then(operation);
  operationQueue = next
    .then(
      () => undefined,
      () => undefined,
    )
    .finally(() => {
      pendingOperationCount--;
      controls();
    });

  return next;
}

function clearComparison() {
  quality = undefined;
  psnr.textContent = 'N/A';
  differencePanel.hidden = true;
  if (differenceCanvas.width > 0) {
    differenceCanvas.width = 0;
    differenceCanvas.height = 0;
  }
}

async function readTarget(camera: Camera) {
  const url = assetUrl('scene/' + camera.target);
  if (targetPixels?.url === url) return targetPixels.pixels;

  await target.decode();
  check(
    target.naturalWidth === camera.width && target.naturalHeight === camera.height,
    'Ground-truth dimensions do not match the camera',
  );

  const targetCanvas = document.createElement('canvas');
  targetCanvas.width = camera.width;
  targetCanvas.height = camera.height;
  const context = targetCanvas.getContext('2d', { willReadFrequently: true })!;
  context.drawImage(target, 0, 0);

  const pixels = context.getImageData(0, 0, camera.width, camera.height).data;
  targetPixels = { url, pixels };
  return pixels;
}

async function renderCurrent() {
  clearComparison();
  const camera = structuredClone(currentCamera);
  lastTiming = await renderer.render(camera, profileGpu);

  if (!isInteractiveCamera) {
    message('Computing PSNR for ' + camera.name + '…');
    const start = performance.now();
    const [actual, expected] = await Promise.all([renderer.pixels(), readTarget(camera)]);

    // Pointer events may move the camera while image decoding or GPU readback is pending.
    if (!isInteractiveCamera) {
      quality = { ...measurePsnr(actual, expected), comparisonMs: performance.now() - start };
      psnr.textContent = quality.psnr === null ? '∞ dB' : quality.psnr.toFixed(4) + ' dB';
    }
  }

  metrics.textContent = JSON.stringify(
    {
      source,
      skipped: skipped.length ? skipped : undefined,
      decodeTotalMs: decoderMetrics?.totalMs,
      hashMs: decoderMetrics?.hashMs,
      view: camera.name,
      accumulation: renderer.accumulationFormat,
      customCamera: isInteractiveCamera,
      timing: lastTiming,
      quality,
    },
    null,
    2,
  );
  message('Rendered');
}

async function renderView(view: number) {
  return enqueue(async () => {
    check(Number.isInteger(view) && view >= 0 && view < scene.cameras.length, 'Invalid view index');

    selectedViewIndex = view;
    isInteractiveCamera = false;
    currentCamera = structuredClone(scene.cameras[view]);
    select.value = String(view);

    const baseline = cudaPsnrReference.views.find(
      item =>
        item.name === currentCamera.name &&
        item.width === currentCamera.width &&
        item.height === currentCamera.height,
    );
    check(baseline && Number.isFinite(baseline.psnr), 'Missing CUDA PSNR for ' + currentCamera.name);
    cudaPsnr.textContent = baseline.psnr.toFixed(4) + ' dB';

    const url = assetUrl('scene/' + currentCamera.target);
    if (target.src !== url) target.src = url;

    message('Rendering ' + currentCamera.name + '…');
    await renderCurrent();
    element('render-resolution').textContent = canvas.width + ' × ' + canvas.height + ' px';

    return { view: selectedViewIndex, name: currentCamera.name, timing: lastTiming, quality };
  });
}

function imageCanvas(pixels: Uint8ClampedArray<ArrayBuffer>, width: number, height: number) {
  const result = document.createElement('canvas');
  result.width = width;
  result.height = height;
  result.getContext('2d')!.putImageData(new ImageData(pixels, width, height), 0, 0);
  return result;
}

async function compare() {
  return enqueue(async () => {
    check(!isInteractiveCamera, 'Reset to the reference camera first');
    check(quality, 'Wait for the view quality measurement first');

    const measuredQuality = quality;
    const camera = scene.cameras[selectedViewIndex];
    const [actual, expected] = await Promise.all([renderer.pixels(), readTarget(camera)]);

    const difference = new Uint8ClampedArray(actual.length);
    let absolute = 0;
    let max = 0;
    for (let i = 0; i < actual.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const d = Math.abs(actual[i + c] - expected[i + c]);
        absolute += d;
        max = Math.max(max, d);
        difference[i + c] = Math.min(255, d * 4);
      }
      difference[i + 3] = 255;
    }

    const count = camera.width * camera.height * 3;
    differenceCanvas.width = camera.width;
    differenceCanvas.height = camera.height;
    differenceCanvas
      .getContext('2d')!
      .putImageData(new ImageData(difference, camera.width, camera.height), 0, 0);

    const diagnostics = await renderer.diagnostics(camera, selectedViewIndex);
    const result = {
      source,
      skipped,
      decoder: decoderMetrics,
      view: selectedViewIndex,
      name: camera.name,
      width: camera.width,
      height: camera.height,
      ...measuredQuality,
      mae: absolute / count,
      maxAbsByteError: max,
      ...diagnostics,
      timing: lastTiming,
      settings: renderer.settings,
      memory: renderer.memory,
      adapter: renderer.adapterInfo,
      accumulationFormat: renderer.accumulationFormat,
      capabilities: renderer.capabilities,
      gpuErrors: [...renderer.errors],
    };

    if (!isInteractiveCamera) {
      metrics.textContent = JSON.stringify(result, null, 2);
      differencePanel.hidden = false;
    }
    return result;
  });
}

async function png() {
  return enqueue(async () =>
    imageCanvas(await renderer.pixels(), canvas.width, canvas.height).toDataURL('image/png'),
  );
}

async function interactive() {
  clearComparison();
  interactiveDirty = true;
  if (interactionLoopRunning) return;

  interactionLoopRunning = true;
  try {
    while (interactiveDirty) {
      interactiveDirty = false;
      await enqueue(renderCurrent);
    }
  } catch (error) {
    fail(error);
  } finally {
    interactionLoopRunning = false;
    controls();
  }
}

const api = {
  state: 'loading',
  error: null as string | null,
  ready: Promise.resolve(),
  renderView,
  compare,
  png,
  setSettings: (settings: Partial<RenderSettings>) =>
    enqueue(async () => {
      await renderer.setSettings({ ...renderer.settings, ...settings });
      precision.value = renderer.settings.accumulation;
      await renderCurrent();
    }),
  get source() {
    return source;
  },
  get skipped() {
    return skipped;
  },
  get decoder() {
    return decoderMetrics;
  },
  get quality() {
    return quality;
  },
  get settings() {
    return renderer?.settings;
  },
  get capabilities() {
    return renderer?.capabilities;
  },
  checkSorting: () =>
    enqueue(async () => {
      const { checkSorting } = await import('./sorting-checks');
      return checkSorting(renderer.device);
    }),
  fingerprint: () =>
    enqueue(async () => {
      let hash = 2166136261;
      for (const byte of await renderer.pixels()) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
      return hash.toString(16);
    }),
  get busy() {
    return pendingOperationCount > 0 || interactionLoopRunning;
  },
  get customCamera() {
    return isInteractiveCamera;
  },
  get selectedView() {
    return selectedViewIndex;
  },
  get cameraCount() {
    return scene?.cameras.length ?? 0;
  },
  get errors() {
    return renderer?.errors ?? [];
  },
};

(window as unknown as { cosa: typeof api }).cosa = api;

async function initialize() {
  check(source === 'decoded' || source === 'compressed', 'Invalid scene source');
  const decodedBase = query.get('decodedBase');
  if (source === 'decoded' && !decodedBase) {
    skipped.push('SKIPPED predecoded input: optional files were not provided. Using the compressed scene');
    source = 'compressed';
  }

  [scene, cudaPsnrReference] = await Promise.all([
    fetchChecked(assetUrl('scene/scene.json')).then(response => response.json()),
    fetchChecked(assetUrl('scene/cuda-psnr.json')).then(response => response.json()),
  ]);
  element('scene-source').textContent = scene.source;

  let data: GpuSceneBuffers | string = decodedBase ?? '';
  renderer = new Renderer(canvas, scene);
  await renderer.initializeDevice(message);

  if (source === 'compressed') {
    const { decodeGpuScene } = await import('./decoder-gpu');
    const decoded = await decodeGpuScene(renderer.device, scene, message, {
      diagnostic: query.get('codecCheck') === '1',
      referenceUrl: query.get('codecReference') ?? '',
    });
    scene.distanceScale = decoded.distanceScale;
    data = decoded.data;
    decoderMetrics = decoded.metrics;
    if (decoderMetrics.referenceCheck?.status === 'SKIPPED')
      skipped.push('SKIPPED NPY comparison: ' + decoderMetrics.referenceCheck.reason);
  }

  showCodecHashes(decoderMetrics?.hashes);

  const defaultAccumulation = renderer.device.features.has('float32-blendable')
    ? 'rgba32float'
    : 'rgba16float';
  const initial = validateSettings({
    accumulation: (query.get('accumulation') ?? defaultAccumulation) as RenderSettings['accumulation'],
    explicitRgb8: query.get('rgb8') !== 'auto',
  });
  await renderer.initialize(message, initial, data);

  const formatMiB = (bytes: number) => (bytes / 1048576).toFixed(2) + ' MiB';
  if (decoderMetrics) {
    const { inputBytes, tableBytes } = decoderMetrics;
    element('compressed-size').textContent = formatMiB(inputBytes + tableBytes);
  } else {
    element('compressed-size').textContent = 'Not loaded';
  }

  element<HTMLOptionElement>('precision-float32').disabled = !renderer.capabilities.float32Blendable;
  precision.value = initial.accumulation;
  precision.addEventListener(
    'change',
    () =>
      void api
        .setSettings({
          accumulation: precision.value as RenderSettings['accumulation'],
        })
        .catch(fail),
  );

  scene.cameras.forEach((camera, i) => {
    const option = document.createElement('option');
    option.value = String(i);
    option.textContent = camera.name;
    select.append(option);
  });

  await renderView(0);
  ready = true;
  api.state = 'ready';
  controls();

  select.addEventListener('change', () => void renderView(Number(select.value)).catch(fail));
  element('previous').addEventListener(
    'click',
    () => void renderView((selectedViewIndex + scene.cameras.length - 1) % scene.cameras.length).catch(fail),
  );
  element('next').addEventListener(
    'click',
    () => void renderView((selectedViewIndex + 1) % scene.cameras.length).catch(fail),
  );
  element('reset').addEventListener('click', () => void renderView(selectedViewIndex).catch(fail));
  element('compare').addEventListener('click', () => void compare().catch(fail));
  element('download').addEventListener(
    'click',
    () =>
      void png()
        .then(url => {
          const link = document.createElement('a');
          link.href = url;
          link.download = currentCamera.name + '-webgpu.png';
          link.click();
        })
        .catch(fail),
  );

  const pointers = new Map<number, { x: number; y: number }>();
  const pinchDistance = () => {
    const [a, b] = pointers.values();
    return Math.hypot(b.x - a.x, b.y - a.y);
  };

  canvas.addEventListener('pointerdown', event => {
    if (!ready) return;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener('pointermove', event => {
    const pointer = pointers.get(event.pointerId);
    if (!pointer) return;

    const dx = event.clientX - pointer.x;
    const dy = event.clientY - pointer.y;
    if (dx === 0 && dy === 0) return;

    const previousDistance = pointers.size === 2 ? pinchDistance() : 0;
    pointer.x = event.clientX;
    pointer.y = event.clientY;

    if (pointers.size === 1) {
      currentCamera = orbit(currentCamera, scene.orbitTarget, dx, dy);
    } else if (pointers.size === 2) {
      const distance = pinchDistance();
      if (previousDistance === 0 || distance === 0) return;

      // Reuse wheel dolly: spreading the fingers reduces the camera-to-target distance.
      currentCamera = dolly(currentCamera, scene.orbitTarget, Math.log(previousDistance / distance) * 1000);
    } else {
      return;
    }

    isInteractiveCamera = true;
    void interactive();
  });

  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture'] as const)
    canvas.addEventListener(type, event => pointers.delete(event.pointerId));

  canvas.addEventListener(
    'wheel',
    event => {
      event.preventDefault();
      currentCamera = dolly(currentCamera, scene.orbitTarget, event.deltaY);
      isInteractiveCamera = true;
      void interactive();
    },
    { passive: false },
  );

  window.addEventListener('pagehide', () => void renderer.dispose());
}

api.ready = initialize().catch(async error => {
  api.state = 'error';
  fail(error);
  await renderer?.dispose().catch(() => {});
});
