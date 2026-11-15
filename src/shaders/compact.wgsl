struct Info {
  count: u32,
  groups: u32,
  maxGroups: u32,
  radixDispatch: vec3<u32>, // aligned to byte 16 for dispatchWorkgroupsIndirect
}

@group(0) @binding(0) var<storage, read> source: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> destinationKeys: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> ranks: array<u32>;
@group(0) @binding(3) var<storage, read_write> blocks: array<u32>;
@group(0) @binding(4) var<storage, read_write> info: Info;
struct Parameters {
  capacity: u32,
  maxGroups: u32,
  pad0: u32,
  pad1: u32
}

@group(0) @binding(5) var<uniform> params: Parameters;
var < workgroup > scan: array<u32, 256>;
@compute @workgroup_size(256)
fn local(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  var valid = 0u;
  if (gid.x < params.capacity) {
    valid = select(0u, 1u, source[gid.x].x > 0u);
  }
  scan[lid] = valid;
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
  if (gid.x < params.capacity) {
    ranks[gid.x] = scan[lid] - valid;
  }
  if (lid == 255u) {
    blocks[wid.x] = scan[255];
  }
}

@compute @workgroup_size(256)
fn prefix(@builtin(local_invocation_index) lid: u32) {
  let perThread = (params.maxGroups + 255u) / 256u;
  let start = lid * perThread;
  var total = 0u;
  for (var i = 0u; i < perThread; i++) {
    if (start + i < params.maxGroups) {
      total += blocks[start + i];
    }
  }
  scan[lid] = total;
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
  var base = scan[lid] - total;
  for (var i = 0u; i < perThread; i++) {
    if (start + i < params.maxGroups) {
      let n = blocks[start + i];
      blocks[start + i] = base;
      base += n;
    }
  }
  let count = scan[255];
  if (lid == 0u) {
    info.count = count;
    info.groups = (count + 255u) / 256u;
    info.maxGroups = params.maxGroups;
    info.radixDispatch = vec3<u32>(info.groups, 1u, 1u);
  }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.capacity) {
    return;
  }
  if (i >= info.count) {
    destinationKeys[i] = vec2<u32>(0u, 0xffffffffu);
  }
  let key = source[i];
  if (key.x == 0u) {
    return;
  }
  // Reverse original indices before stable radix sorting, so equal depths
  // retain descending original index order.
  let rank = blocks[i / 256u] + ranks[i];
  destinationKeys[info.count - 1u - rank] = key;
}
