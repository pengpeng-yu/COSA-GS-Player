import { access } from 'node:fs/promises';
import { delimiter, join } from 'node:path';

// Browser validation needs an installed Chrome/Chromium, not a machine-specific path.
export async function findChrome(explicit = process.env.CHROME_PATH) {
  if (explicit) return explicit;
  const names =
    process.platform === 'win32'
      ? ['chrome.exe', 'chromium.exe']
      : ['google-chrome', 'chromium', 'chromium-browser'];
  const candidates = (process.env.PATH ?? '')
    .split(delimiter)
    .filter(Boolean)
    .flatMap(directory => names.map(name => join(directory, name)));
  if (process.platform === 'win32') {
    for (const directory of [
      process.env.PROGRAMFILES,
      process.env['PROGRAMFILES(X86)'],
      process.env.LOCALAPPDATA,
    ])
      if (directory) candidates.push(join(directory, 'Google/Chrome/Application/chrome.exe'));
  } else if (process.platform === 'darwin') {
    candidates.push('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
  }
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch (error) {
      if (error.code !== 'ENOENT' && error.code !== 'ENOTDIR') throw error;
    }
  }
  throw new Error('Chrome was not found. Set CHROME_PATH or pass --chrome=<executable>.');
}
