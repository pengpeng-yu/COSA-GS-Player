export function measurePsnr(actual: Uint8ClampedArray, expected: Uint8ClampedArray) {
  const squared = [0, 0, 0];
  for (let i = 0; i < actual.length; i += 4) {
    for (let channel = 0; channel < 3; channel++) {
      const error = actual[i + channel] - expected[i + channel];
      squared[channel] += error * error;
    }
  }

  const pixels = actual.length / 4;
  // COSA evaluate_views uses psnr(CHW_image).mean(): average the three channel PSNRs.
  const psnr = squared.reduce((sum, error) => sum + 10 * Math.log10((255 * 255 * pixels) / error), 0) / 3;
  const mse = (squared[0] + squared[1] + squared[2]) / (pixels * 3);

  return {
    psnr: Number.isFinite(psnr) ? psnr : null,
    exactMatch: mse === 0,
    rmse: Math.sqrt(mse),
  };
}
