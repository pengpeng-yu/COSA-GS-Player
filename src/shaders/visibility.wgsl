@group(0) @binding(0) var<storage, read> anchors: array<Anchor>;
struct Worklist {
  count: atomic<u32>,
  x: u32,
  y: u32,
  z: u32,
  indices: array<u32>
}

@group(0) @binding(4) var<storage, read_write> worklist: Worklist;
@compute @workgroup_size(128)
fn prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= frame.anchorCount) {
    return;
  }
  let position = anchors[i].position;
  let p = cameraPoint(position.xyz);
  if (p.z <= 0.2) {
    return;
  }
  let s2 = position.w * position.w;
  let cov = covariance2d(p, mat3x3<f32>(
    vec3<f32>(s2, 0.0, 0.0),
    vec3<f32>(0.0, s2, 0.0),
    vec3<f32>(0.0, 0.0, s2)
  ));
  if (nonempty(tileBounds(imagePoint(p), cov))) {
    let index = atomicAdd(&worklist.count, 1u);
    worklist.indices[index] = i;
  }
}

@compute @workgroup_size(1)
fn finalize() {
  let anchorsPerRow = FUSED_DISPATCH_WIDTH * 4u;
  worklist.x = FUSED_DISPATCH_WIDTH;
  worklist.y = (atomicLoad(&worklist.count) + anchorsPerRow - 1u) / anchorsPerRow;
  worklist.z = 1u;
}
