# avbd2d API cheatsheet

All calls return `Avbd2dResult` unless noted. Header: `include/avbd2d/avbd2d.h`. Units and queueing rules are in [concepts.md](concepts.md).

## World

| Function | Purpose |
|---|---|
| `avbd2d_default_world_def` | Fill a world def with defaults (gravity (0,-10), dt 1/60). |
| `avbd2d_create` | Create a world on its own Vulkan device, default def. |
| `avbd2d_create_world` | Create a world from a def. |
| `avbd2d_create_with_vulkan` | Create a world on a host-owned `Avbd2dVulkanDevice`. |
| `avbd2d_create_world_with_vulkan` | Same, with a def. |
| `avbd2d_destroy` | Destroy the world. Required even after `DEVICE_LOST`. |
| `avbd2d_set_world_def` | Apply solver fields of a def to a live world. |
| `avbd2d_get_world_def` | Read the world's current def. |
| `avbd2d_world_set_gravity` | Queue a new gravity. |
| `avbd2d_world_get_gravity` | Read gravity. |
| `avbd2d_set_memory_budget` | Cap device memory (0 = unlimited). |
| `avbd2d_wake_all` | Wake every sleeping body at the next step. |
| `avbd2d_get_stats` | Counts and timings of the last step. |
| `avbd2d_get_capabilities` | Which features this build supports (1/0). |
| `avbd2d_abi_version` | `(major << 16) \| minor`. |
| `avbd2d_result_string` | Readable name of a result code. |
| `avbd2d_make_rot` / `avbd2d_rot_get_angle` | Build a rotation from radians / read its angle. |

```c
Avbd2dWorldDef wd = avbd2d_default_world_def();
wd.gravity = (Avbd2dVec2){0.0f, -9.8f};
Avbd2dWorld *world = NULL;
if (avbd2d_create_world(&wd, &world) != AVBD2D_OK) { /* handle */ }
/* ... */
avbd2d_destroy(world);
```

## Stepping

| Function | Purpose |
|---|---|
| `avbd2d_step` | One full step (begin + end). |
| `avbd2d_step_begin` | Submit the step to the GPU and return. |
| `avbd2d_step_end` | Wait for the step in flight. No-op if none. |

```c
avbd2d_step_begin(world);
/* read the previous state, queue next-step work */
avbd2d_step_end(world);
```

## Bodies

| Function | Purpose |
|---|---|
| `avbd2d_default_body_def` | Fill a body def. |
| `avbd2d_create_body` | Create a body (valid at once, enters sim at next step). |
| `avbd2d_destroy_body` | Destroy the body, its shapes and joints. |
| `avbd2d_body_is_valid` | Check an id. |
| `avbd2d_body_get_position` / `_get_rotation` / `_get_angle` / `_get_transform` | Origin, rotation, angle, or both. |
| `avbd2d_body_get_world_center` / `_get_local_center` | Centre of mass, world or body frame. |
| `avbd2d_body_get_linear_velocity` / `_get_angular_velocity` | Velocity. |
| `avbd2d_body_get_mass` / `_get_rotational_inertia` | Mass and inertia about the centre. |
| `avbd2d_body_get_type` / `_is_awake` / `_is_enabled` | Type (`Avbd2dBodyType`), sleep state, enabled flag. |
| `avbd2d_body_get_user_data` / `_set_user_data` | Host pointer. |
| `avbd2d_body_set_transform` | Queue a new origin and rotation. |
| `avbd2d_body_set_linear_velocity` / `_set_angular_velocity` | Queue a velocity. |
| `avbd2d_body_apply_force` / `_apply_force_to_center` | Force for the next step, at a point or the centre. |
| `avbd2d_body_apply_torque` | Torque for the next step. |
| `avbd2d_body_apply_linear_impulse` / `_..._to_center` | Impulse now (before next step). |
| `avbd2d_body_apply_angular_impulse` | Angular impulse. |
| `avbd2d_body_set_awake` | Queue sleep or wake. |
| `avbd2d_body_enable` / `avbd2d_body_disable` | Queue enable or disable. |

The `wake` flag on `apply_*` wakes the body.

```c
Avbd2dBodyDef bd = avbd2d_default_body_def();
bd.type = AVBD2D_DYNAMIC_BODY;
bd.position = (Avbd2dVec2){0.0f, 4.0f};
Avbd2dBodyId body;
avbd2d_create_body(world, &bd, &body);
avbd2d_body_apply_linear_impulse_to_center(body, (Avbd2dVec2){2.0f, 0.0f}, 1);
```

## Shapes

| Function | Purpose |
|---|---|
| `avbd2d_default_shape_def` | Fill a shape def (friction, density, filter, sensor flags). |
| `avbd2d_make_box` / `avbd2d_make_offset_box` / `avbd2d_make_rounded_box` | Box polygons. |
| `avbd2d_make_polygon` | Convex hull of up to 8 points, with rounding radius. |
| `avbd2d_create_polygon_shape` | Add a polygon to a body. |
| `avbd2d_create_circle_shape` | Add a circle. |
| `avbd2d_create_capsule_shape` | Add a capsule. |
| `avbd2d_create_segment_shape` | Add a massless two-sided segment. |
| `avbd2d_destroy_shape` | Remove one shape (body's shape set is replaced at next step). |
| `avbd2d_shape_is_valid` | Check an id. |
| `avbd2d_shape_get_body` / `avbd2d_shape_get_user_data` | Owning body; user pointer. |

```c
Avbd2dShapeDef sd = avbd2d_default_shape_def();
sd.density = 1.0f;
sd.friction = 0.6f;
Avbd2dPolygon box = avbd2d_make_box(0.5f, 0.5f);
Avbd2dShapeId shape;
avbd2d_create_polygon_shape(body, &sd, &box, &shape);
```

## Joints

Every def takes two body ids and body-local anchors. Use a static body for ground. `breakForce` defaults to unbreakable (>= 1e38).

| Function | Purpose |
|---|---|
| `avbd2d_default_<type>_joint_def` | Fill a def. `<type>` is revolute, weld, prismatic, wheel, distance, motor, filter. |
| `avbd2d_create_revolute_joint` | Pin joint, with optional limit, motor, spring. |
| `avbd2d_create_weld_joint` | Rigid (or soft, with hertz) link. |
| `avbd2d_create_prismatic_joint` | Slide along a local axis. |
| `avbd2d_create_wheel_joint` | Slide along an axis with spring and motor. |
| `avbd2d_create_distance_joint` | Keep a length, or a min/max range. |
| `avbd2d_create_motor_joint` | Drive B toward an offset of A. |
| `avbd2d_create_filter_joint` | Only disables collision between the pair. |
| `avbd2d_destroy_joint` | Stops acting at the next step. |
| `avbd2d_joint_is_valid` | Check an id. |
| `avbd2d_joint_get_user_data` | Host pointer. |
| `avbd2d_joint_get_constraint_force` / `_get_constraint_torque` | Last step's load on B. |
| `avbd2d_joint_wake_bodies` | Wake both bodies. |

Per-type setters (queued):

| Joint | Functions |
|---|---|
| Revolute | `_enable_limit`, `_set_limits`, `_enable_motor`, `_set_motor_speed`, `_set_max_motor_torque` |
| Prismatic | `_enable_limit`, `_set_limits`, `_enable_motor`, `_set_motor_speed`, `_set_max_motor_force` |
| Wheel | `_enable_limit`, `_set_limits`, `_enable_motor`, `_set_motor_speed`, `_set_max_motor_torque` |
| Distance | `_enable_limit`, `_set_length_range`, `_enable_motor`, `_set_motor_speed`, `_set_max_motor_force` |

All are prefixed `avbd2d_<type>_joint_`.

```c
Avbd2dRevoluteJointDef jd = avbd2d_default_revolute_joint_def();
jd.bodyIdA = ground;
jd.bodyIdB = body;
jd.localAnchorA = (Avbd2dVec2){0.0f, 0.0f};
jd.localAnchorB = (Avbd2dVec2){-0.5f, 0.0f};
Avbd2dJointId joint;
avbd2d_create_revolute_joint(world, &jd, &joint);
avbd2d_revolute_joint_enable_motor(joint, 1);
avbd2d_revolute_joint_set_motor_speed(joint, 3.0f);
```

## Events

Lists are valid until the next `avbd2d_step_end`. Read them after `step_end`.

| Function | Purpose |
|---|---|
| `avbd2d_world_get_body_events` | Bodies awake at start or end of the step (moves, sleep transitions). |
| `avbd2d_world_get_contact_events` | Contact begin, end, and hit events. |
| `avbd2d_world_get_sensor_events` | Sensor begin and end. |
| `avbd2d_world_get_joint_events` | Joints cut by their break force. |
| `avbd2d_world_get_transforms` | Every live body's transform from the host mirror (no GPU access). |

```c
avbd2d_step(world);
Avbd2dBodyEvents be;
avbd2d_world_get_body_events(world, &be);
for (int32_t i = 0; i < be.moveCount; ++i) {
    const Avbd2dBodyMoveEvent *m = &be.moveEvents[i];
    /* update the sprite for m->bodyId from m->transform */
}
```

## Queries

Queries answer from the last finished pose, so they are legal while a step is in flight. Single queries settle the step in flight first.

| Function | Purpose |
|---|---|
| `avbd2d_default_query_filter` | Filter with category and mask bits. |
| `avbd2d_world_cast_ray_closest` | Closest ray hit. |
| `avbd2d_world_cast_ray` | Ray with a callback (return -1 ignore, 0 stop, fraction clip, 1 continue). |
| `avbd2d_world_cast_rays_batch` | Many closest-hit rays in one GPU submit. |
| `avbd2d_world_overlap_aabb` | Shapes overlapping an AABB, via callback. |
| `avbd2d_world_overlap_aabbs_batch` | Many AABB overlaps into caller arrays. |
| `avbd2d_world_overlap_shape` | Shapes overlapping a proxy, via callback. |
| `avbd2d_world_cast_shape` | Sweep a proxy along a translation, via callback. |
| `avbd2d_make_proxy` | Build a shape proxy from up to 8 world points plus radius. |

```c
Avbd2dRayResult hit;
avbd2d_world_cast_ray_closest(world, (Avbd2dVec2){0,10}, (Avbd2dVec2){0,-20},
                              avbd2d_default_query_filter(), &hit);
if (hit.hit) { /* hit.point, hit.normal, hit.fraction */ }
```

Batch outputs: `avbd2d_world_overlap_aabbs_batch` writes `count * maxPerQuery` ids. `outCounts[i]` may exceed `maxPerQuery`; extras are dropped.

## Grab and explosions

| Function | Purpose |
|---|---|
| `avbd2d_grab_begin` | Spring from a world point to the dynamic body under it. Returns the body. |
| `avbd2d_grab_move` | Move the grab point. |
| `avbd2d_grab_end` | Release. |
| `avbd2d_default_explosion_def` | Fill an explosion def. |
| `avbd2d_world_explode` | Radial impulse, applied at the next step. |

```c
Avbd2dBodyId held;
if (avbd2d_grab_begin(world, mouseWorld, &held) == AVBD2D_OK) {
    avbd2d_grab_move(world, mouseWorld);
}
/* on release */
avbd2d_grab_end(world);
```

## Chains

| Function | Purpose |
|---|---|
| `avbd2d_default_chain_def` | Fill a chain def. |
| `avbd2d_create_chain` | One-sided terrain segments on a body, with ghost vertices. |
| `avbd2d_destroy_chain` | Remove the chain. |
| `avbd2d_chain_is_valid` | Check an id. |

```c
Avbd2dVec2 pts[4] = {{-10,0},{-3,-1},{3,-1},{10,0}};
Avbd2dChainDef cd = avbd2d_default_chain_def();
cd.points = pts;
cd.count = 4;      /* open: >= 2 points, loop: >= 3 */
Avbd2dChainId chain;
avbd2d_create_chain(ground, &cd, &chain);
```

## Misc and render interop

| Function | Purpose |
|---|---|
| `avbd2d_register_render_target` | Always `AVBD2D_ERR_UNSUPPORTED` in this build. |

Math helpers `avbd2d_make_rot`, `avbd2d_rot_get_angle` and the Vulkan device struct `Avbd2dVulkanDevice` are listed under World.
