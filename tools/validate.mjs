import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { preview, createServer } from 'vite';
import { connectCdp, browserErrors } from './cdp.mjs';
import { option, validateScene, assertCodecChecks } from './validation.mjs';
import { prepareOptionalData, applyOptionalData } from './optional-data.mjs';
import { findChrome } from './chrome.mjs';

const root = resolve(import.meta.dirname, '..'),
  dev = process.argv.includes('--dev');
const run = option('run', dev ? 'desktop-dev' : 'desktop');
assert.match(run, /^[\w-]+$/);
const artifacts = join(root, 'local_dev/artifacts', run),
  profile = join(tmpdir(), 'cosa-validate-' + randomUUID());
await mkdir(artifacts, { recursive: true });
let server,
  chrome,
  cdp,
  sessionId,
  optionalServer,
  stderr = '';
const report = {
  status: 'FAIL',
  started: new Date().toISOString(),
  build: dev ? 'development' : 'production',
  headless: process.argv.includes('--headless'),
};
try {
  if (dev) {
    server = await createServer({ root, server: { host: '127.0.0.1', port: 4173, strictPort: false } });
    await server.listen();
  } else server = await preview({ root, preview: { host: '127.0.0.1', port: 4173, strictPort: false } });
  const url = new URL(server.resolvedUrls.local[0]);
  const accumulation = option('accumulation', '');
  if (accumulation) url.searchParams.set('accumulation', accumulation);
  url.searchParams.set('rgb8', option('rgb8', 'explicit'));
  const source = option('source', 'compressed');
  const codecCheck = process.argv.includes('--codec-check');
  url.searchParams.set('source', source);
  if (codecCheck) {
    assert.equal(url.searchParams.get('source'), 'compressed', '--codec-check requires --source=compressed');
    url.searchParams.set('codecCheck', '1');
  }
  const optional = await prepareOptionalData(root, {
    predecoded: source === 'decoded',
    codecCheck,
    decodedDirectory: option('decoded-dir', undefined),
    referenceDirectory: option('references', undefined),
  });
  optionalServer = optional.server;
  report.optionalData = optional.checks;
  report.requestedSource = source;
  if (optionalServer)
    applyOptionalData(url, optional.checks, 'http://127.0.0.1:' + optionalServer.address().port + '/');
  const chromePath = await findChrome(option('chrome', undefined));
  const endpoint = await new Promise((resolveEndpoint, reject) => {
    const timer = setTimeout(
      () => reject(new Error('Chrome startup timed out: ' + stderr.slice(-1000))),
      30000,
    );
    chrome = spawn(
      chromePath,
      [
        '--remote-debugging-address=127.0.0.1',
        '--remote-debugging-port=0',
        '--user-data-dir=' + profile,
        '--no-first-run',
        '--no-default-browser-check',
        '--disable-background-networking',
        '--force-high-performance-gpu',
        '--window-size=1500,1000',
        ...(process.argv.includes('--headless') ? ['--headless=new'] : []),
        'about:blank',
      ],
      { stdio: ['ignore', 'ignore', 'pipe'] },
    );
    chrome.once('error', e => {
      clearTimeout(timer);
      reject(e);
    });
    chrome.once('exit', code => {
      clearTimeout(timer);
      reject(new Error('Chrome exited: ' + code));
    });
    chrome.stderr.setEncoding('utf8');
    chrome.stderr.on('data', chunk => {
      stderr = (stderr + chunk).slice(-20000);
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) {
        clearTimeout(timer);
        resolveEndpoint(match[1]);
      }
    });
  });
  cdp = await connectCdp(endpoint);
  report.browser = await cdp.send('Browser.getVersion');
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
  ({ sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true }));
  await cdp.send('Page.enable', {}, sessionId);
  await cdp.send('Runtime.enable', {}, sessionId);
  if (process.argv.includes('--disable-float32-blendable')) {
    // Test capability fallback without changing browser flags or the application.
    await cdp.send(
      'Page.addScriptToEvaluateOnNewDocument',
      {
        source:
          'const original=GPUAdapter.prototype.requestDevice;GPUAdapter.prototype.requestDevice=function(desc={}){' +
          'return original.call(this,{...desc,requiredFeatures:[...(desc.requiredFeatures??[])].filter(f=>f!=="float32-blendable")});};',
      },
      sessionId,
    );
    report.simulatedMissingFloat32Blendable = true;
  }
  await cdp.send('Page.navigate', { url: String(url) }, sessionId);
  await cdp.send('Page.bringToFront', {}, sessionId);
  await validateScene(cdp, sessionId, {
    root,
    artifacts,
    report,
    all: process.argv.includes('--all'),
    views: option('views', '') ? option('views', '').split(',').map(Number) : undefined,
    saveAll: process.argv.includes('--save-all'),
    soakSeconds: Number(option('soak', '0')),
  });
  if (process.argv.includes('--codec-check')) {
    assertCodecChecks(report);
  }
  report.status = 'PASS';
} catch (error) {
  report.error = error.stack ?? String(error);
  report.chromeStderr = stderr.slice(-2000);
  process.exitCode = 1;
} finally {
  if (cdp && sessionId) {
    report.browserErrors = browserErrors(cdp, sessionId);
    if (report.browserErrors.length) {
      report.status = 'FAIL';
      process.exitCode = 1;
    }
    try {
      const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' }, sessionId);
      await writeFile(join(artifacts, 'browser.png'), Buffer.from(data, 'base64'));
    } catch {}
  }
  report.completed = new Date().toISOString();
  await writeFile(join(artifacts, 'validation.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(
    JSON.stringify(
      {
        status: report.status,
        views: report.views?.length,
        error: report.error,
        benchmark: report.benchmark,
        artifacts,
      },
      null,
      2,
    ),
  );
  if (cdp) {
    await Promise.race([cdp.send('Browser.close').catch(() => {}), delay(3000)]);
    cdp.close();
  }
  if (chrome && chrome.exitCode === null && !chrome.killed) chrome.kill();
  if (optionalServer) {
    optionalServer.closeAllConnections();
    await new Promise(done => optionalServer.close(done));
  }
  if (server) {
    if (dev) await server.close();
    else await new Promise(resolveClose => server.httpServer.close(resolveClose));
  }
  await rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 300 }).catch(() => {});
}
