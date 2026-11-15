// Host-side model and buffer layout. Keep in sync with common.wgsl and fused.wgsl.
export const FEATURE_COUNT = 32;
export const OFFSETS_PER_ANCHOR = 10;
export const ANCHOR_STRIDE_FLOATS = 40;
export const ACTIVE_OFFSET_BYTES = 16;
export const PROJECTED_SPLAT_BYTES = 64;
export const WEIGHTS_PER_HEAD = 3496;
export const MLP_HEAD_COUNT = 3;

export const FRAME_UNIFORM_BYTES = 112;
export const DRAW_ARGUMENT_BYTES = 16;
// Control has a trailing runtime array. Its minimum binding size is 20 bytes.
export const CONTROL_BUFFER_BYTES = 32;
export const MLP_OUTPUT_COUNT = 110;
export const MLP_SAMPLE_STRIDE = 1 + MLP_OUTPUT_COUNT;

export const SORT_WORKGROUP_SIZE = 256;
export const MAX_ACTIVE_OFFSETS = 1 << 20;
export const SORT_INFO_BYTES = 32;
export const SORT_DISPATCH_OFFSET = 16;
export const RADIX_PASSES = 8;
export const RADIX_BITS = 4;
export const RADIX_BUCKETS = 1 << RADIX_BITS;
export const DYNAMIC_UNIFORM_STRIDE = 256;
