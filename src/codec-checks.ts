import { check, fetchChecked } from './types';
import { sha256 } from './codec-hashes';

type IntegerArray = Int16Array | Int32Array;

export interface CodecArrayCheck {
  name: string;
  dtype: string;
  shape: number[];
  count: number;
  sha256: string;
  mismatches: number;
  firstMismatch: { index: number; actual: number; expected: number } | null;
}

function readNpy(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer);
  check(
    bytes.length >= 10 && bytes[0] === 0x93 && String.fromCharCode(...bytes.subarray(1, 6)) === 'NUMPY',
    'Invalid NPY magic',
  );

  const version = bytes[6];
  check(version === 1 || version === 2, 'Unsupported NPY version');

  const view = new DataView(buffer);
  const headerStart = version === 1 ? 10 : 12;
  check(bytes.length >= headerStart, 'Truncated NPY header');
  const headerLength = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const dataStart = headerStart + headerLength;
  check(dataStart <= bytes.length, 'Truncated NPY header');

  const header = new TextDecoder().decode(bytes.subarray(headerStart, dataStart));
  const dtype = header.match(/'descr':\s*'([^']+)'/)?.[1];
  check(dtype, 'Missing NPY dtype');
  check(/'fortran_order':\s*False/.test(header), 'Fortran-order NPY is not supported');

  const shapeText = header.match(/'shape':\s*\(([^)]*)\)/)?.[1];
  check(shapeText, 'Missing NPY shape');
  const shape = shapeText
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
    .map(Number);
  check(
    shape.length > 0 && shape.every(value => Number.isSafeInteger(value) && value > 0),
    'Invalid NPY shape',
  );
  const count = shape.reduce((a, b) => a * b, 1);
  check(Number.isSafeInteger(count), 'NPY element count exceeds integer range');

  const readers: Record<string, { width: number; value(offset: number): number }> = {
    '<i2': { width: 2, value: offset => view.getInt16(offset, true) },
    '<i4': { width: 4, value: offset => view.getInt32(offset, true) },
  };
  const reader = readers[dtype];
  check(reader, 'Unsupported NPY dtype: ' + dtype);
  check(count * reader.width === bytes.length - dataStart, 'NPY payload size mismatch');
  return {
    dtype,
    shape,
    count,
    data: bytes.subarray(dataStart),
    value: (index: number) => reader.value(dataStart + index * reader.width),
  };
}

/** Loaded only by explicit codec diagnostics. Reference arrays are never production inputs. */
export async function checkCodecArray(
  baseUrl: string,
  name: string,
  actual: IntegerArray,
): Promise<CodecArrayCheck> {
  const url = baseUrl.replace(/\/?$/, '/') + encodeURIComponent(name) + '.npy';
  const expected = readNpy(await fetchChecked(url).then(response => response.arrayBuffer()));
  check(actual.length === expected.count, name + ': reference element count differs');
  check(expected.dtype === (actual instanceof Int16Array ? '<i2' : '<i4'), name + ': reference dtype differs');

  const result: CodecArrayCheck = {
    name,
    dtype: expected.dtype,
    shape: expected.shape,
    count: actual.length,
    sha256: await sha256(expected.data),
    mismatches: 0,
    firstMismatch: null,
  };

  for (let index = 0; index < actual.length; index++) {
    const value = actual[index];
    const reference = expected.value(index);
    if (value !== reference) {
      result.mismatches++;
      result.firstMismatch ??= { index, actual: value, expected: reference };
    }
  }

  check(
    result.mismatches === 0,
    name + ': ' + result.mismatches + ' reference mismatches. First mismatch: ' +
      JSON.stringify(result.firstMismatch),
  );
  return result;
}
