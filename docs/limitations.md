# Limitations

**What you get:** the known gaps in avbd2d v1.0.0 (ABI 3.0), with a workaround for each. Check these before you commit to the library.

Capability values match `avbd2d_get_capabilities()`. Items marked "ignored" are accepted by the API and have no effect.

| Gap | Effect | Workaround |
| :--- | :--- | :--- |
| **Restitution** has no effect (`restitution` is stored, `capabilities.restitution = 0`) | No bouncing balls, no bouncy pickups | Use zero restitution. For a bounce, read the contact begin event and set the post-impact velocity yourself with `avbd2d_body_set_linear_velocity` |
| **Joint `dampingRatio` ignored** | Springs, suspensions and soft welds oscillate more than in Box2D | Lower `hertz`, or add damping with body `linearDamping` / `angularDamping` |
| **Per-body `enableSleep` ignored** (sleeping is velocity-gated, not island-based) | A body cannot be kept awake on its own; a resting body can stay asleep | Set world `sleepFrames = 0` to turn sleep off for the whole world, or call `avbd2d_body_set_awake` / `avbd2d_wake_all` when your game needs bodies awake |
| **Explosion `maskBits` ignored** | Explosions cannot be filtered by category | Query with `avbd2d_world_overlap_aabb` and a `categoryBits`/`maskBits` filter, then apply `avbd2d_body_apply_linear_impulse_to_center` to the bodies you want |
| **No determinism** (a stated non-goal, `capabilities.determinism = 0`) | Results differ across GPUs and drivers; no bit-exact replay | Do not use lockstep or rollback netcode. Sync state snapshots from an authoritative host instead |
| **Tested on one machine only** | Other GPUs, drivers and operating systems are unverified; cross-vendor float behaviour is unknown | Test on your target hardware, and fall back to a CPU engine when `avbd2d_create` returns `AVBD2D_ERR_DEVICE` |
| **Vulkan 1.3 and required features**: `shaderInt64`, `bufferDeviceAddress`, 64-bit atomics (buffer and shared), `scalarBlockLayout`, `shaderInt8`, `timelineSemaphore`, `hostQueryReset`, `synchronization2`, compute subgroup basic/ballot/arithmetic; subgroup width 32 or 64 (32 is pinned when the driver allows it) | Older drivers and GPUs fail to create a world with `AVBD2D_ERR_DEVICE` | Update the driver. When you pass your own device, enable the full feature list. There is no device-support query yet, so catch `AVBD2D_ERR_DEVICE` and fall back |
| **Pipeline cache not persisted** | About 50 compute pipelines are recompiled on every launch | Create the world once at startup, during a loading screen. Do not recreate it per level |
| **Fixed cost of about 0.5 ms per step** (measured on RTX 4070 SUPER, 2026-10-06) | Small worlds cost more on the GPU than on the CPU; the GPU also shares time with your renderer | Use it for large worlds (about 10k bodies and up). Overlap the step with the frame using `avbd2d_step_begin` / `avbd2d_step_end` |
