export type AllocateBuffer = (label: string, size: number, data?: ArrayBuffer, usage?: number) => GPUBuffer;
export type CompileShader = (label: string, source: string) => Promise<GPUShaderModule>;

export interface Camera {
  name: string;
  target: string;
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
  fx: number;
  fy: number;
  position: number[];
  rotation: number[][];
}

export interface Scene {
  source: string;
  anchorCount: number;
  activeOffsetCount: number;
  featureCount: number;
  offsetsPerAnchor: number;
  anchorStrideFloats: number;
  distanceScale: number;
  background: number[];
  orbitTarget: number[];
  cameras: Camera[];
  files: { renderMlpReference: string };
}

export interface GpuSceneBuffers {
  anchors: GPUBuffer;
  offsets: GPUBuffer;
  weights: GPUBuffer;
  activeSlots: Uint32Array<ArrayBuffer>;
}

export interface RenderMlpReference {
  rows: number[];
  views: { view: number; outputs: Record<string, number[][]> }[];
}

export interface CudaPsnrReference {
  views: { name: string; width: number; height: number; psnr: number }[];
}

export function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

export async function fetchChecked(url: string) {
  const response = await fetch(url);
  check(response.ok, `HTTP ${response.status}: ${url}`);
  return response;
}

export function assetUrl(path: string) {
  return new URL(path, new URL(import.meta.env.BASE_URL, location.href)).href;
}
