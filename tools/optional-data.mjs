import { readFile, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { join } from 'node:path';

const PREDECODED_FILES = ['anchors.bin', 'offsets.bin', 'weights.bin'];
export const CODEC_REFERENCE_NAMES = [
  'anchor_coords',
  'anchor_latent_symbols_0',
  'anchor_latent_symbols_1',
  'anchor_latent_symbols_2',
  'anchor_latent_symbols_3',
  'position_scaling_symbols',
  'offset_mask_symbols',
  'offset_symbols',
  'gaussian_scaling_symbols',
  'feat_symbols',
];
const REFERENCE_FILES = CODEC_REFERENCE_NAMES.map(name => name + '.npy');

// Optional data stay outside public/. Only explicitly requested groups are inspected.
export async function prepareOptionalData(
  root,
  {
    predecoded = false,
    codecCheck = false,
    decodedDirectory = join(root, 'local_dev/predecoded'),
    referenceDirectory = join(root, 'local_dev/codec_reference'),
  } = {},
) {
  const checks = {};
  const files = new Map();
  const inspect = async (group, directory, names) => {
    const missing = (
      await Promise.all(
        names.map(async name => {
          try {
            return (await stat(join(directory, name))).isFile() ? null : name;
          } catch (error) {
            if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return name;
            throw error;
          }
        }),
      )
    ).filter(Boolean);
    if (missing.length) {
      checks[group] = { status: 'SKIPPED', reason: 'Missing optional files', missing };
    } else {
      checks[group] = { status: 'AVAILABLE', files: names.length };
      for (const name of names) files.set('/' + group + '/' + name, join(directory, name));
    }
  };
  await Promise.all([
    ...(predecoded ? [inspect('predecoded', decodedDirectory, PREDECODED_FILES)] : []),
    ...(codecCheck ? [inspect('codec_reference', referenceDirectory, REFERENCE_FILES)] : []),
  ]);
  for (const [group, result] of Object.entries(checks)) {
    if (result.status === 'SKIPPED') console.log('SKIPPED ' + group + ': ' + result.reason);
  }
  if (!files.size) return { checks, server: undefined };

  const server = createServer(async (request, response) => {
    const path = files.get(request.url?.split('?')[0]);
    if (!path) {
      response.writeHead(404);
      response.end();
      return;
    }
    try {
      const bytes = await readFile(path);
      response.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-store',
      });
      response.end(bytes);
    } catch (error) {
      // A present file that cannot be read is an error, not a skipped check.
      response.writeHead(500);
      response.end(String(error));
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return { checks, server };
}

export function applyOptionalData(url, checks, baseUrl) {
  if (checks.predecoded?.status === 'AVAILABLE') url.searchParams.set('decodedBase', baseUrl + 'predecoded/');
  if (checks.codec_reference?.status === 'AVAILABLE')
    url.searchParams.set('codecReference', baseUrl + 'codec_reference/');
}
