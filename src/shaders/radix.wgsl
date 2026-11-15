// Matches the first three words of the compaction Info buffer.
struct Info {
  count: u32,
  groups: u32,
  maxGroups: u32
}

struct Parameters {
  shift: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32
}

@group(0) @binding(0) var<storage, read> source: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> destinationKeys: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> ranks: array<u32>;
@group(0) @binding(3) var<storage, read_write> histogram: array<u32>;
@group(0) @binding(4) var<storage, read_write> totals: array<u32>;
@group(0) @binding(5) var<storage, read> info: Info;
@group(0) @binding(6) var<uniform> params: Parameters;
var < workgroup > words: array<atomic<u32>, 128 > ;
var < workgroup > scan: array<u32, 256>;
@compute @workgroup_size(256)
fn histogramPass(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  if (lid < 128u) {
    atomicStore(&words[lid], 0u);
  }
  workgroupBarrier();
  var digit = 0u;
  if (gid.x < info.count) {
    digit = ((~source[gid.x].x) >> params.shift) & 15u;
    atomicOr(&words[digit * 8u + lid / 32u], 1u << (lid % 32u));
  }
  workgroupBarrier();
  if (gid.x < info.count) {
    var rank = 0u;
    for (var w = 0u; w < lid / 32u; w++) {
      rank += countOneBits(atomicLoad(&words[digit * 8u + w]));
    }
    rank += countOneBits(atomicLoad(&words[digit * 8u + lid / 32u]) & ((1u << (lid % 32u)) - 1u));
    ranks[gid.x] = rank;
  }
  if (lid < 16u) {
    var n = 0u;
    for (var w = 0u; w < 8u; w++) {
      n += countOneBits(atomicLoad(&words[lid * 8u + w]));
    }
    histogram[lid * info.maxGroups + wid.x] = n;
  }
}

@compute @workgroup_size(256)
fn prefixPass(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let digit = wid.x;
  let perThread = (info.groups + 255u) / 256u;
  let start = lid * perThread;
  var sum = 0u;
  for (var i = 0u; i < perThread; i++) {
    if (start + i < info.groups) {
      sum += histogram[digit * info.maxGroups + start + i];
    }
  }
  scan[lid] = sum;
  workgroupBarrier();
  for (var stride = 1u; stride < 256u; stride *= 2u) {
    var add = 0u;
    if (lid >= stride) {
      add = scan[lid - stride];
    }
    workgroupBarrier();
    scan[lid] += add;
    workgroupBarrier();
  }
  var base = scan[lid] - sum;
  for (var i = 0u; i < perThread; i++) {
    if (start + i < info.groups) {
      let offset = digit * info.maxGroups + start + i;
      let n = histogram[offset];
      histogram[offset] = base;
      base += n;
    }
  }
  if (lid == 255u) {
    totals[digit] = scan[255];
  }
}

@compute @workgroup_size(256)
fn scatterPass(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = gid.x;
  if (i >= info.count) {
    return;
  }
  let pair = source[i];
  let digit = ((~pair.x) >> params.shift) & 15u;
  var base = 0u;
  for (var d = 0u; d < digit; d++) {
    base += totals[d];
  }
  let destination = base + histogram[digit * info.maxGroups + wid.x] + ranks[i];
  destinationKeys[destination] = pair;
}
