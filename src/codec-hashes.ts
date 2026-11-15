export interface CodecArrayHash {
  name: string;
  dtype: '<i2' | '<i4';
  shape: number[];
  sha256: string;
}

export interface CodecHashes {
  algorithm: 'SHA-256';
  combinedSha256: string;
  arrays: CodecArrayHash[];
}

export async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
}

/** Hash row-major little-endian integers without NPY headers or buffer padding. */
export async function hashCodecArray(
  name: string,
  dtype: CodecArrayHash['dtype'],
  shape: number[],
  bytes: Uint8Array<ArrayBuffer>,
): Promise<CodecArrayHash> {
  return { name, dtype, shape, sha256: await sha256(bytes) };
}

export async function combineCodecHashes(entries: CodecArrayHash[]): Promise<CodecHashes> {
  const arrays = [...entries].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  // Sort ASCII names, then hash UTF-8 JSON tuples with no whitespace or trailing newline.
  // Each tuple is [name, dtype, shape, lowercase SHA-256 hex].
  const tuples = arrays.map(({ name, dtype, shape, sha256 }) => [name, dtype, shape, sha256]);
  const combinedSha256 = await sha256(new TextEncoder().encode(JSON.stringify(tuples)));
  return { algorithm: 'SHA-256', combinedSha256, arrays };
}
