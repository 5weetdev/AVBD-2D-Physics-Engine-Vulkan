# Porting from Box2D v3

**What you get:** the call mapping, what behaves differently, and what is missing.

The API is shaped after Box2D v3: `*Def` structs with defaults, ids instead of pointers, one world handle.
Most ports are a rename.

## Name mapping

| Box2D v3 | avbd2d |
| :--- | :--- |
| `b2DefaultWorldDef` / `b2CreateWorld` / `b2DestroyWorld` | `avbd2d_default_world_def` / `avbd2d_create_world` / `avbd2d_destroy` |
| `b2World_Step` | `avbd2d_step` (or `avbd2d_step_begin` + `avbd2d_step_end`) |
| `b2World_SetGravity` | `avbd2d_world_set_gravity` |
| `b2DefaultBodyDef` / `b2CreateBody` / `b2DestroyBody` | `avbd2d_default_body_def` / `avbd2d_create_body` / `avbd2d_destroy_body` |
| `b2Body_GetPosition` / `GetRotation` / `GetTransform` | `avbd2d_body_get_position` / `_get_rotation` / `_get_transform` |
| `b2Body_SetLinearVelocity` / `ApplyForce` / `ApplyLinearImpulse` | `avbd2d_body_set_linear_velocity` / `_apply_force` / `_apply_linear_impulse` |
| `b2DefaultShapeDef` | `avbd2d_default_shape_def` |
| `b2CreatePolygonShape` / `CreateCircleShape` / `CreateCapsuleShape` / `CreateSegmentShape` | `avbd2d_create_polygon_shape` / `_circle_shape` / `_capsule_shape` / `_segment_shape` |
| `b2MakeBox` / `b2MakeOffsetBox` / `b2MakeRoundedBox` / `b2ComputeHull` + `b2MakePolygon` | `avbd2d_make_box` / `_make_offset_box` / `_make_rounded_box` / `_make_polygon` |
| `b2CreateChain` | `avbd2d_create_chain` |
| `b2CreateRevoluteJoint`, `Weld`, `Prismatic`, `Wheel`, `Distance`, `Motor` | `avbd2d_create_revolute_joint`, `_weld_`, `_prismatic_`, `_wheel_`, `_distance_`, `_motor_joint` |
| `b2World_GetBodyEvents` / `ContactEvents` / `SensorEvents` / `JointEvents` | `avbd2d_world_get_body_events` / `_contact_events` / `_sensor_events` / `_joint_events` |
| `b2World_CastRayClosest` / `OverlapAABB` / `CastShape` | `avbd2d_world_cast_ray_closest` / `_overlap_aabb` / `_cast_shape` |
| `b2World_Explode` | `avbd2d_world_explode` |

## Behaves differently

| Topic | Box2D | avbd2d |
| :--- | :--- | :--- |
| Writes | Immediate | Queued; applied at the next step. See [concepts](concepts.md) |
| Errors | Asserts | Every call returns `Avbd2dResult` |
| Callbacks | Custom filter and friction callbacks | Data only: filters, categories and masks on shapes |
| Pose reads | CPU memory | GPU-resident; moved bodies are read back once per step |
| Stepping | Synchronous | `avbd2d_step_begin` returns after submit; `avbd2d_step_end` waits |

## Not supported

| Feature | State |
| :--- | :--- |
| Restitution | Stored, no effect |
| Joint `dampingRatio` | Accepted, ignored |
| Per-body `enableSleep` | Accepted, ignored |
| Explosion `maskBits` | Accepted, not applied |
| Determinism | Not provided |

Details and workarounds: [limitations](limitations.md). Whether the GPU pays off for your body count:
[when to use it](when-to-use.md).
