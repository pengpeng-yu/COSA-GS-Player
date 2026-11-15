import type { CodecHashes } from './codec-hashes';

export function showCodecHashes(hashes: CodecHashes | undefined) {
  const status = document.getElementById('hash-status')!;
  if (!hashes) {
    status.textContent = 'Not available for predecoded input.';
    return;
  }

  status.hidden = true;
  document.getElementById('hash-results')!.hidden = false;
  document.getElementById('combined-hash')!.textContent = hashes.combinedSha256;
  document.getElementById('hash-count')!.textContent = hashes.arrays.length + ' matrices';

  const rows = hashes.arrays.map(entry => {
    const row = document.createElement('tr');
    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = entry.name;
    row.append(name);

    for (const text of [entry.dtype === '<i4' ? 'INT32' : 'INT16', JSON.stringify(entry.shape)]) {
      const cell = document.createElement('td');
      cell.textContent = text;
      row.append(cell);
    }

    const cell = document.createElement('td');
    const code = document.createElement('code');
    code.textContent = entry.sha256;
    cell.append(code);
    row.append(cell);
    return row;
  });
  document.getElementById('hash-rows')!.replaceChildren(...rows);

  const copyButton = document.getElementById('copy-hashes') as HTMLButtonElement;
  const copyStatus = document.getElementById('hash-copy-status')!;
  copyButton.disabled = false;
  copyButton.addEventListener('click', async () => {
    copyButton.disabled = true;
    try {
      await navigator.clipboard.writeText(JSON.stringify(hashes, null, 2));
      copyStatus.textContent = 'Copied';
    } catch {
      copyStatus.textContent = 'Copy failed. Select the hash text to copy it.';
    } finally {
      copyButton.disabled = false;
    }
  });
}
