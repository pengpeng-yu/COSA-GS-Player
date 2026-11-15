import { spawnSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const output = join(root, 'public/codec');
await mkdir(output, { recursive: true });
const compiler = process.env.EMCC ?? (process.platform === 'win32' ? 'emcc.exe' : 'emcc');
const args = [
  'decoder/decoder.cpp',
  'decoder/network.cpp',
  'decoder/entropy.cpp',
  '-sDEFAULT_TO_CXX=1',
  '-std=c++17',
  '-O3',
  '-msimd128',
  '-ffp-contract=off',
  '-fexceptions',
  '-sDISABLE_EXCEPTION_CATCHING=0',
  '-sMODULARIZE=1',
  '-sEXPORT_ES6=1',
  '-sENVIRONMENT=worker,node',
  '-sALLOW_MEMORY_GROWTH=1',
  '-sINITIAL_MEMORY=33554432',
  '-sMAXIMUM_MEMORY=2147483648',
  '-sFILESYSTEM=0',
  '-sEXPORTED_FUNCTIONS=["_malloc","_free"]',
  '-sEXPORTED_RUNTIME_METHODS=["UTF8ToString","HEAPU8"]',
  '-o',
  join(output, 'decoder.mjs'),
];
const result = spawnSync(compiler, args, { cwd: root, stdio: 'inherit', windowsHide: true });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
console.log('Built single-worker SIMD WASM decoder: ' + output);
