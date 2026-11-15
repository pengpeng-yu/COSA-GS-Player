export type Accumulation = 'rgba16float' | 'rgba32float';

export interface RenderSettings {
  accumulation: Accumulation;
  explicitRgb8: boolean;
}

export const DEFAULT_SETTINGS: RenderSettings = { accumulation: 'rgba32float', explicitRgb8: true };

export function validateSettings(settings: RenderSettings) {
  if (
    !['rgba16float', 'rgba32float'].includes(settings.accumulation) ||
    typeof settings.explicitRgb8 !== 'boolean'
  ) {
    throw new Error('Invalid rendering settings');
  }
  return { ...settings };
}
