// Serialized network, stream and reconstruction-step order in COSA's integer codec.
export const NETWORK_ID = {
  latent0: 0,
  latent1: 1,
  latent2: 2,
  latent3: 3,
  latentContext: 4,
  latentFusion: 5,
  coordinate: 6,
  positionScalingContext: 7,
  offsetPrior: 8,
  gaussianScalingPrior: 9,
  offset: 10,
  positionScaling: 11,
  gaussianScaling: 12,
  feature: 13,
} as const;
export type NetworkName = keyof typeof NETWORK_ID;
export const NETWORK_NAMES = Object.keys(NETWORK_ID) as NetworkName[];
export const LATENT_NETWORKS = ['latent0', 'latent1', 'latent2', 'latent3'] as const;
export const CONTEXT_WIDTH = 24;

export const STREAM_ID = {
  firstLatent: 1,
  positionScaling: 6,
  offset: 7,
  gaussianScaling: 8,
  feature: 9,
} as const;
export const STEP_ID = { latent: 0, positionScaling: 1, offset: 2, gaussianScaling: 3, feature: 4 } as const;
export type StepName = keyof typeof STEP_ID;

// Keep these values aligned with codec-ops.wgsl.
export const ATTRIBUTE_KIND = { feature: 0, gaussianScaling: 1, offset: 2 } as const;
export type AttributeKind = (typeof ATTRIBUTE_KIND)[keyof typeof ATTRIBUTE_KIND];
export const CODEC_WORKGROUP_SIZE = 256;
export const CODEC_DISPATCH_WIDTH = 65535;
