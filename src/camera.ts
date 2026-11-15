import { FRAME_UNIFORM_BYTES } from './layout';
import type { Camera, Scene } from './types';

export function frameUniform(camera: Camera, scene: Scene) {
  const result = new ArrayBuffer(FRAME_UNIFORM_BYTES);
  const f = new Float32Array(result);
  const u = new Uint32Array(result);

  // JSON rotation is C2W. Build column-major W2C explicitly, with camera +Z forward.
  for (let col = 0; col < 3; col++) {
    for (let row = 0; row < 3; row++) {
      f[col * 4 + row] = camera.rotation[col][row];
    }
  }

  for (let row = 0; row < 3; row++) {
    f[12 + row] = -camera.position.reduce((sum, p, col) => sum + camera.rotation[col][row] * p, 0);
  }
  f[15] = 1;

  f.set([...camera.position, scene.distanceScale], 16);
  f.set([camera.width, camera.height, camera.fx, camera.fy], 20);
  u[24] = scene.anchorCount;
  return result;
}

function rotate(v: number[], axis: number[], angle: number) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const dot = v.reduce((a, x, i) => a + x * axis[i], 0);
  const cross = [
    axis[1] * v[2] - axis[2] * v[1],
    axis[2] * v[0] - axis[0] * v[2],
    axis[0] * v[1] - axis[1] * v[0],
  ];

  return v.map((x, i) => x * c + cross[i] * s + axis[i] * dot * (1 - c));
}

export function orbit(camera: Camera, target: number[], dx: number, dy: number) {
  const result = structuredClone(camera);
  const axes = [0, 1, 2].map(col => result.rotation.map(row => row[col]));
  let position = result.position.map((x, i) => x - target[i]);

  for (const [axis, angle] of [
    [axes[1], dx * 0.005],
    [axes[0], -dy * 0.005],
  ] as [number[], number][]) {
    position = rotate(position, axis, angle);
    for (let i = 0; i < 3; i++) {
      axes[i] = rotate(axes[i], axis, angle);
    }
  }

  result.position = position.map((x, i) => x + target[i]);
  result.rotation = [0, 1, 2].map(row => axes.map(axis => axis[row]));
  return result;
}

export function dolly(camera: Camera, target: number[], delta: number) {
  const result = structuredClone(camera);
  const factor = Math.exp(Math.max(-0.3, Math.min(0.3, delta * 0.001)));
  result.position = camera.position.map((x, i) => target[i] + (x - target[i]) * factor);
  return result;
}
