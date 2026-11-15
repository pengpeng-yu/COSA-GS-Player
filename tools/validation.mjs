import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { evaluator, browserErrors } from './cdp.mjs';
import { CODEC_REFERENCE_NAMES } from './optional-data.mjs';

const MAX_PSNR_DROP_DB = 0.01;

function assertCodecHashes(hashes) {
  assert.equal(hashes.algorithm, 'SHA-256');
  assert.deepEqual(hashes.arrays.map(entry => entry.name), [...CODEC_REFERENCE_NAMES].sort());
  for (const entry of hashes.arrays) {
    assert.equal(entry.dtype, entry.name === 'anchor_coords' ? '<i4' : '<i2');
    assert.match(entry.sha256, /^[0-9a-f]{64}$/);
  }

  const tuples = hashes.arrays.map(({ name, dtype, shape, sha256 }) => [name, dtype, shape, sha256]);
  const expected = createHash('sha256').update(JSON.stringify(tuples), 'utf8').digest('hex');
  assert.equal(hashes.combinedSha256, expected, 'Combined matrix hash mismatch');
}

export const option = (name, fallback) =>
  process.argv.find(x => x.startsWith('--' + name + '='))?.slice(name.length + 3) ?? fallback;
export async function waitReady(evaluate) {
  for (let i = 0; i < 300; i++) {
    const state = await evaluate('({state:window.cosa?.state,error:window.cosa?.error})');
    if (state?.state === 'ready') return;
    if (state?.state === 'error') throw new Error(state.error);
    if (i % 10 === 0) console.log('Waiting for renderer…');
    await delay(1000);
  }
  throw new Error('Renderer initialization timed out');
}
function qualityCheck(q, references) {
  assert.deepEqual(q.gpuErrors, []);
  assert.ok(q.visibleGaussians > 0 && q.visibleAnchors > 0, 'Empty scene');
  for (const [name, result] of Object.entries(q.mlp)) {
    assert.ok(result.checkedValues > 0, name + ' MLP reference did not check any values');
    assert.equal(result.failures, 0, name + ' MLP mismatch');
  }
  if (q.mlpSampleAnchors) assert.equal(Object.keys(q.mlp).length, 3, 'Missing MLP heads');
  const baseline = references.views.find(
    v => v.name === q.name && v.width === q.width && v.height === q.height,
  );
  assert.ok(baseline && Number.isFinite(baseline.psnr), 'Missing COSA-GS baseline for ' + q.name);
  const floor = baseline.psnr - MAX_PSNR_DROP_DB;
  assert.ok(
    q.exactMatch || (Number.isFinite(q.psnr) && q.psnr >= floor),
    'PSNR ' + q.psnr + ' is below ' + floor + ' for ' + q.name,
  );
  return {
    referencePsnr: baseline.psnr,
    psnrDelta: q.psnr === null ? null : q.psnr - baseline.psnr,
    qualityFloor: floor,
  };
}
export async function validateScene(
  cdp,
  sessionId,
  {
    root,
    artifacts,
    report,
    all = false,
    views,
    saveAll = false,
    soakSeconds = 0,
    touch = false,
  },
) {
  const evaluate = evaluator(cdp, sessionId);
  await waitReady(evaluate);
  const references = JSON.parse(await readFile(join(root, 'public/scene/cuda-psnr.json'), 'utf8'));
  const automaticQuality = async () => {
    const state = await evaluate(
      '({quality:window.cosa.quality,text:document.getElementById("psnr").textContent,' +
        'viewName:document.getElementById("view").selectedOptions[0].textContent,' +
        'cudaPsnrText:document.getElementById("cuda-psnr").textContent,' +
        'differenceHidden:document.getElementById("difference-panel").hidden,' +
        'differenceWidth:document.getElementById("difference").width})',
    );
    assert.ok(state.quality && Number.isFinite(state.quality.psnr), 'Automatic PSNR is missing');
    assert.equal(state.text, state.quality.psnr.toFixed(4) + ' dB');
    const baseline = references.views.find(view => view.name === state.viewName);
    assert.ok(baseline, 'Missing CUDA PSNR for ' + state.viewName);
    assert.equal(state.cudaPsnrText, baseline.psnr.toFixed(4) + ' dB');
    assert.ok(state.quality.comparisonMs >= 0);
    assert.equal(state.differenceHidden, true, 'Automatic PSNR must not generate a difference image');
    assert.equal(state.differenceWidth, 0);
    return state.quality;
  };
  const mlpReferenceRequests = await evaluate(
    'performance.getEntriesByType("resource").filter(entry=>new URL(entry.name).pathname.endsWith("/render-mlp-reference.json")).length',
  );
  assert.equal(mlpReferenceRequests, 0, 'Normal startup must not load the render MLP reference');
  const startup = await evaluate(
    '({source:window.cosa.source,decoder:window.cosa.decoder,requests:performance.getEntriesByType("resource").map(entry=>new URL(entry.name).pathname)})',
  );
  report.normalStartup = {
    mlpReferenceRequests,
    source: startup.source,
    decoder: startup.decoder,
    quality: await automaticQuality(),
  };
  report.skipped = await evaluate('window.cosa.skipped');
  if (report.requestedSource === 'decoded') {
    const available = report.optionalData.predecoded.status === 'AVAILABLE';
    assert.equal(startup.source, available ? 'decoded' : 'compressed');
    if (!available) assert.ok(report.skipped.some(message => message.includes('predecoded')));
  }
  if (startup.source === 'compressed') {
    assert.equal(startup.decoder?.source, 'webgpu+wasm', 'Unexpected decoder backend');
    assert.ok(startup.decoder.totalMs > 0, 'Missing decoder wall time');
    assert.ok(startup.decoder.hashMs >= 0, 'Missing hash timing');
    assertCodecHashes(startup.decoder.hashes);
    assert.equal(
      startup.requests.filter(path => /\/(anchors|offsets|weights)\.bin$/.test(path)).length,
      0,
      'Compressed startup must not fetch predecoded buffers',
    );
  }
  const hashPanel = await evaluate(`(() => {
    const panel = document.getElementById('codec-hashes');
    return {
      hidden: document.getElementById('hash-results').hidden,
      status: document.getElementById('hash-status').textContent,
      combined: document.getElementById('combined-hash').textContent,
      copyDisabled: document.getElementById('copy-hashes').disabled,
      rows: [...document.querySelectorAll('#hash-rows tr')].map(row =>
        [...row.cells].map(cell => cell.textContent)),
      belowImages: panel.getBoundingClientRect().top >= document.querySelector('main').getBoundingClientRect().bottom,
      aboveValidation: panel.getBoundingClientRect().bottom <= document.getElementById('validation-panel').getBoundingClientRect().top,
    };
  })()`);
  assert.equal(hashPanel.belowImages && hashPanel.aboveValidation, true, 'Unexpected hash panel position');
  if (startup.source === 'compressed') {
    assert.equal(hashPanel.hidden, false);
    assert.equal(hashPanel.copyDisabled, false);
    assert.equal(hashPanel.combined, startup.decoder.hashes.combinedSha256);
    assert.deepEqual(hashPanel.rows, startup.decoder.hashes.arrays.map(entry => [
      entry.name,
      entry.dtype === '<i4' ? 'INT32' : 'INT16',
      JSON.stringify(entry.shape),
      entry.sha256,
    ]));
  } else {
    assert.equal(hashPanel.hidden, true);
    assert.equal(hashPanel.copyDisabled, true);
    assert.equal(hashPanel.status, 'Not available for predecoded input.');
  }

  const targetRequests = startup.requests.filter(path => path.includes('/targets/'));
  assert.equal(targetRequests.length, 1, 'Startup must load only the current ground-truth image');
  assert.equal(startup.requests.some(path => path.includes('/references/')), false);
  report.normalStartup.targetRequests = targetRequests.length;
  const cudaPsnrRequests = startup.requests.filter(path => path.endsWith('/scene/cuda-psnr.json')).length;
  assert.equal(cudaPsnrRequests, 1, 'Startup must load the CUDA PSNR reference once');
  report.normalStartup.cudaPsnrRequests = cudaPsnrRequests;
  const count = await evaluate('window.cosa.cameraCount');
  const indices = all
    ? Array.from({ length: count }, (_, i) => i)
    : (views ?? [0, Math.floor(count / 2), count - 1]);
  const initial = await evaluate('window.cosa.settings');
  const capabilities = await evaluate('window.cosa.capabilities');
  const requestedAccumulation = await evaluate('new URLSearchParams(location.search).get("accumulation")');
  assert.equal(initial.accumulation, requestedAccumulation ?? (capabilities.float32Blendable ? 'rgba32float' : 'rgba16float'));
  assert.deepEqual(
    await evaluate('[...document.getElementById("precision").options].map(option => option.value)'),
    ['rgba32float', 'rgba16float'],
  );
  report.capabilities = capabilities;
  report.views = [];
  for (const view of indices) {
    assert.ok(Number.isInteger(view) && view >= 0 && view < count, 'Invalid view');
    const rendered = await evaluate('window.cosa.renderView(' + view + ')');
    assert.deepEqual(rendered.quality, await automaticQuality());
    const q = await evaluate('window.cosa.compare()');
    assert.equal(q.psnr, rendered.quality.psnr);
    report.views.push(q);
    Object.assign(q, qualityCheck(q, references));
    assert.equal(await evaluate('document.getElementById("precision").value'), q.settings.accumulation);
    assert.equal(q.accumulationFormat, initial.accumulation);
    let png;
    if (view === indices[0] || saveAll) png = await evaluate('window.cosa.png()');
    if (view === indices[0] || saveAll) {
      await writeFile(join(artifacts, 'view-' + view + '.png'), Buffer.from(png.split(',')[1], 'base64'));
      const diff = await evaluate('document.getElementById("difference").toDataURL("image/png")');
      await writeFile(
        join(artifacts, 'difference-' + view + '.png'),
        Buffer.from(diff.split(',')[1], 'base64'),
      );
    }
    console.log(
      JSON.stringify({
        view,
        psnr: q.psnr,
        psnrDelta: q.psnrDelta,
        comparisonMs: q.comparisonMs,
        format: q.accumulationFormat,
        ms: q.timing.totalMs,
        memoryMiB: q.memory.trackedBuffersMiB,
      }),
    );
  }
  report.sorting = await evaluate('window.cosa.checkSorting()');
  assert.equal(report.sorting.passed, true);

  // Keep the two independent fallbacks covered without restoring legacy MLP modes.
  const sampleViews = [...new Set([0, Math.floor(count / 2), count - 1])];
  report.alternatives = [];
  for (const settings of [
    { accumulation: 'rgba16float', explicitRgb8: true },
    { accumulation: 'rgba16float', explicitRgb8: false },
    ...(capabilities.float32Blendable ? [{ accumulation: 'rgba32float', explicitRgb8: false }] : []),
  ]) {
    await evaluate('window.cosa.setSettings(' + JSON.stringify(settings) + ')');
    for (const view of sampleViews) {
      await evaluate('window.cosa.renderView(' + view + ')');
      await automaticQuality();
      const q = await evaluate('window.cosa.compare()');
      Object.assign(q, qualityCheck(q, references));
      report.alternatives.push(q);
    }
  }
  await evaluate('window.cosa.setSettings(' + JSON.stringify(initial) + ')');
  await evaluate('window.cosa.renderView(0)');
  const fingerprint = await evaluate('window.cosa.fingerprint()');
  const cudaPsnrText = await evaluate('document.getElementById("cuda-psnr").textContent');
  await evaluate('window.cosa.setSettings({accumulation:"rgba16float",explicitRgb8:false})');
  await evaluate('window.cosa.setSettings(' + JSON.stringify(initial) + ')');
  assert.equal(
    await evaluate('window.cosa.fingerprint()'),
    fingerprint,
    'Settings restoration changed pixels',
  );

  const idle = async () => {
    for (let i = 0; i < 300; i++) {
      if (await evaluate('!window.cosa.busy')) return;
      await delay(100);
    }
    throw new Error('Interaction did not finish');
  };
  await evaluate('document.getElementById("render").scrollIntoView({block:"center"})');
  const point = await evaluate(
    '(()=>{const r=document.getElementById("render").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()',
  );
  if (touch) {
    await cdp.send(
      'Input.dispatchTouchEvent',
      { type: 'touchStart', touchPoints: [{ ...point }] },
      sessionId,
    );
    await cdp.send(
      'Input.dispatchTouchEvent',
      { type: 'touchMove', touchPoints: [{ x: point.x + 25, y: point.y + 10 }] },
      sessionId,
    );
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] }, sessionId);
  } else {
    await cdp.send(
      'Input.dispatchMouseEvent',
      { type: 'mousePressed', ...point, button: 'left', buttons: 1, clickCount: 1 },
      sessionId,
    );
    await cdp.send(
      'Input.dispatchMouseEvent',
      { type: 'mouseMoved', x: point.x + 25, y: point.y + 10, button: 'left', buttons: 1 },
      sessionId,
    );
    await cdp.send(
      'Input.dispatchMouseEvent',
      { type: 'mouseReleased', x: point.x + 25, y: point.y + 10, button: 'left', buttons: 0, clickCount: 1 },
      sessionId,
    );
  }
  await idle();
  assert.equal(await evaluate('window.cosa.customCamera'), true);
  assert.equal(await evaluate('document.getElementById("compare").disabled'), true);
  assert.equal(await evaluate('window.cosa.quality === undefined'), true);
  assert.equal(await evaluate('document.getElementById("psnr").textContent'), 'N/A');
  assert.equal(await evaluate('document.getElementById("cuda-psnr").textContent'), cudaPsnrText);
  const dragged = await evaluate('window.cosa.fingerprint()');
  assert.notEqual(dragged, fingerprint, 'Drag did not change pixels');
  if (!touch) {
    await cdp.send(
      'Input.dispatchMouseEvent',
      { type: 'mouseWheel', ...point, deltaY: 100, deltaX: 0 },
      sessionId,
    );
    await delay(200);
    await idle();
    assert.notEqual(await evaluate('window.cosa.fingerprint()'), dragged, 'Wheel did not change pixels');
    assert.equal(await evaluate('window.cosa.quality === undefined'), true);
  }
  // Exercise multi-touch in desktop Chrome as well as on a connected phone.
  if (!touch)
    await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 2 }, sessionId);
  await evaluate('window.cosa.renderView(0)');
  const sendTouch = async (type, touchPoints) => {
    await cdp.send('Input.dispatchTouchEvent', { type, touchPoints }, sessionId);
    // Chrome may acknowledge input before delivering the coalesced pointer events.
    await delay(50);
    await idle();
  };
  const left = { id: 1, x: point.x - 20, y: point.y };
  const right = { id: 2, x: point.x + 20, y: point.y };
  const spreadLeft = { ...left, x: left.x - 5 };
  const spreadRight = { ...right, x: right.x + 5 };
  await sendTouch('touchStart', [left]);
  await sendTouch('touchStart', [left, right]);
  assert.equal(await evaluate('window.cosa.customCamera'), false, 'Adding a finger moved the camera');
  await sendTouch('touchMove', [spreadLeft, spreadRight]);
  assert.notEqual(await evaluate('window.cosa.fingerprint()'), fingerprint, 'Pinch did not change pixels');
  assert.equal(await evaluate('window.cosa.customCamera'), true);
  assert.equal(await evaluate('window.cosa.quality === undefined'), true);
  assert.equal(await evaluate('document.getElementById("psnr").textContent'), 'N/A');
  assert.equal(await evaluate('document.getElementById("cuda-psnr").textContent'), cudaPsnrText);
  assert.equal(await evaluate('document.getElementById("compare").disabled'), true);
  await sendTouch('touchMove', [left, right]);
  assert.equal(await evaluate('window.cosa.fingerprint()'), fingerprint, 'Reverse pinch did not restore the view');
  await sendTouch('touchEnd', [right]);
  assert.equal(await evaluate('window.cosa.fingerprint()'), fingerprint, 'Removing a finger moved the camera');
  await sendTouch('touchMove', [{ ...left, x: left.x + 10 }]);
  assert.notEqual(await evaluate('window.cosa.fingerprint()'), fingerprint, 'One-finger orbit did not resume');
  await sendTouch('touchCancel', []);
  await evaluate('window.cosa.renderView(0)');
  await sendTouch('touchStart', [left]);
  await sendTouch('touchMove', [{ ...left, x: left.x + 10 }]);
  await sendTouch('touchEnd', []);
  assert.notEqual(await evaluate('window.cosa.fingerprint()'), fingerprint, 'Touch did not recover after cancellation');
  if (!touch) await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false }, sessionId);
  await evaluate('document.getElementById("reset").click()');
  await idle();
  await automaticQuality();
  assert.equal(
    await evaluate('window.cosa.fingerprint()'),
    fingerprint,
    'Reset did not reproduce the camera',
  );
  await evaluate('document.getElementById("next").click()');
  await idle();
  assert.equal(await evaluate('window.cosa.selectedView'), 1);
  await automaticQuality();
  await evaluate('document.getElementById("previous").click()');
  await idle();
  assert.equal(await evaluate('window.cosa.selectedView'), 0);
  await automaticQuality();
  await evaluate(
    '(()=>{const el=document.getElementById("view");el.value="' +
      (count - 1) +
      '";el.dispatchEvent(new Event("change"));})()',
  );
  await idle();
  assert.equal(await evaluate('window.cosa.selectedView'), count - 1);
  await automaticQuality();
  await evaluate('window.cosa.renderView(0)');
  // Exercise the actual precision selector, then restore the initial settings.
  await evaluate(
    '(()=>{const el=document.getElementById("precision");el.value="rgba16float";el.dispatchEvent(new Event("change"));})()',
  );
  await idle();
  assert.equal(await evaluate('window.cosa.settings.accumulation'), 'rgba16float');
  await automaticQuality();
  assert.equal(
    await evaluate('document.getElementById("precision-float32").disabled'),
    !capabilities.float32Blendable,
  );
  if (capabilities.float32Blendable) {
    await evaluate(
      '(()=>{const el=document.getElementById("precision");el.value="rgba32float";el.dispatchEvent(new Event("change"));})()',
    );
    await idle();
    assert.equal(await evaluate('window.cosa.settings.accumulation'), 'rgba32float');
    assert.equal(await evaluate('document.getElementById("precision").value'), 'rgba32float');
    await automaticQuality();
  }
  await evaluate('window.cosa.setSettings(' + JSON.stringify(initial) + ')');
  assert.equal(await evaluate('window.cosa.fingerprint()'), fingerprint);
  report.interaction = {
    drag: true,
    input: touch ? 'touch' : 'mouse',
    wheel: !touch,
    pinch: true,
    touchTransitions: true,
    touchCancellation: true,
    reset: true,
    previousNext: true,
    viewSelector: true,
    precisionSwitch: true,
    automaticPsnr: true,
    clearPsnrOnCameraMove: true,
  };

  report.benchmark = [];
  for (const view of sampleViews) {
    for (let i = 0; i < 3; i++) await evaluate('window.cosa.renderView(' + view + ')');
    const frames = [];
    for (let i = 0; i < 7; i++)
      frames.push((await evaluate('window.cosa.renderView(' + view + ')')).timing.totalMs);
    const sorted = [...frames].sort((a, b) => a - b);
    report.benchmark.push({ view, frames, medianMs: sorted[3], minMs: sorted[0], maxMs: sorted.at(-1) });
  }
  if (soakSeconds > 0) {
    assert.ok(Number.isFinite(soakSeconds) && soakSeconds <= 300, 'Soak duration must be at most 300 s');
    console.log('Continuous view-0 rendering: ' + soakSeconds + ' s');
    report.soak = await evaluate(
      '(async()=>{const duration=' +
        soakSeconds * 1000 +
        ';const start=performance.now(),frames=[];' +
        'while(performance.now()-start<duration){const r=await window.cosa.renderView(0);frames.push({elapsedMs:performance.now()-start,ms:r.timing.totalMs});}' +
        'return {durationMs:performance.now()-start,frames,errors:window.cosa.errors};})()',
    );
    assert.deepEqual(report.soak.errors, []);
    report.afterSoak = await evaluate('window.cosa.compare()');
    Object.assign(report.afterSoak, qualityCheck(report.afterSoak, references));
  }
  await evaluate('window.cosa.renderView(0)');
  assert.deepEqual(
    await evaluate('window.cosa.decoder?.hashes ?? null'),
    startup.decoder?.hashes ?? null,
    'Matrix hashes changed during rendering',
  );
  assert.deepEqual(await evaluate('window.cosa.errors'), []);
  assert.deepEqual(browserErrors(cdp, sessionId), []);
  return report;
}

export function assertCodecChecks(report) {
  const decoder = report.normalStartup.decoder;
  assert.equal(decoder.diagnostic, true, 'Codec diagnostics did not run');
  assert.equal(decoder.networkChecks.length, 14, 'Missing integer network checks');
  assert.equal(decoder.mathChecks.passed, true, 'Wide-integer checks did not pass');
  const referencesAvailable = report.optionalData.codec_reference.status === 'AVAILABLE';
  assert.equal(decoder.referenceCheck.status, referencesAvailable ? 'PASS' : 'SKIPPED');
  assert.deepEqual(
    decoder.checks.map(check => check.name).sort(),
    referencesAvailable ? [...CODEC_REFERENCE_NAMES].sort() : [],
    'Unexpected NPY reference checks',
  );
  assert.ok(decoder.checks.every(check => check.mismatches === 0), 'Integer reference mismatch');
  for (const reference of decoder.checks) {
    const actual = decoder.hashes.arrays.find(entry => entry.name === reference.name);
    assert.equal(actual.sha256, reference.sha256, reference.name + ': NPY payload hash mismatch');
    assert.equal(actual.dtype, reference.dtype);
    assert.deepEqual(actual.shape, reference.shape);
  }
}
