// The C ABI of the 2D solver (avbd2d.h, ABI 3) on VkWorld2D + Scene2D.
//
// Shape of the layer: the host keeps slot tables (with free lists and generations) of bodies,
// shapes and joints. A created object is a record in its table and is "pending" until the next
// step_begin, which replays every pending body (with its shapes) and joint into the live world's
// authored Scene2D in one go and commits them (VkWorld2D::commitAuthored). Setters on a live
// object go to VkWorld2D's mutation queue; on a pending object they edit its record. Getters read
// VkWorld2D's host mirrors (the last completed step plus queued pose/velocity sets).
//
// Every entry point validates its handle or id first, so a null, never-created or destroyed world
// and a stale id are error codes and not crashes.


// World lifecycle, defaults, ids and capability queries of the 2D C ABI.

#include "avbd2d_world.h"

extern "C"
{

    AVBD2D_API uint32_t avbd2d_abi_version(void)
    {
        return ((uint32_t)AVBD2D_ABI_VERSION_MAJOR << 16) | (uint32_t)AVBD2D_ABI_VERSION_MINOR;
    }

    AVBD2D_API const char *avbd2d_result_string(Avbd2dResult r)
    {
        switch (r)
        {
        case AVBD2D_OK:
            return "ok";
        case AVBD2D_ERR_NULL_HANDLE:
            return "null, unknown or destroyed world handle";
        case AVBD2D_ERR_INVALID_ARG:
            return "invalid argument (or a stale id)";
        case AVBD2D_ERR_UNSUPPORTED:
            return "unsupported";
        case AVBD2D_ERR_NOT_COMMITTED:
            return "unused";
        case AVBD2D_ERR_DEVICE:
            return "no usable Vulkan device";
        case AVBD2D_ERR_CAPACITY:
            return "a hard capacity ceiling or the memory budget clamped the step";
        case AVBD2D_ERR_DEVICE_LOST:
            return "the Vulkan device failed or was lost; destroy the world and create a new one";
        case AVBD2D_ERR_OUT_OF_MEMORY:
            return "out of memory (host, device or the memory budget)";
        }
        return "unknown result";
    }

    AVBD2D_API Avbd2dRot avbd2d_make_rot(float radians)
    {
        return rotOf(radians);
    }
    AVBD2D_API float avbd2d_rot_get_angle(Avbd2dRot q)
    {
        return angleOf(q);
    }

    AVBD2D_API int32_t avbd2d_body_is_valid(Avbd2dBodyId id)
    {
        Avbd2dWorld *w = worldFromId(id.world0);
        return w && w->bodies.find(id.index1, id.generation) ? 1 : 0;
    }
    AVBD2D_API int32_t avbd2d_shape_is_valid(Avbd2dShapeId id)
    {
        Avbd2dWorld *w = worldFromId(id.world0);
        return w && w->shapes.find(id.index1, id.generation) ? 1 : 0;
    }
    AVBD2D_API int32_t avbd2d_joint_is_valid(Avbd2dJointId id)
    {
        Avbd2dWorld *w = worldFromId(id.world0);
        return w && w->joints.find(id.index1, id.generation) ? 1 : 0;
    }
    AVBD2D_API int32_t avbd2d_chain_is_valid(Avbd2dChainId id)
    {
        Avbd2dWorld *w = worldFromId(id.world0);
        if (!w || id.index1 < 1 || id.index1 > (int32_t)w->shapes.items.size())
            return 0;
        // A chain id names its first segment shape.
        const ShapeRec &s = w->shapes.items[(size_t)id.index1 - 1];
        return s.alive && s.gen == id.generation && s.kind == SK_CHAIN ? 1 : 0;
    }

    // ----- Defaults --------------------------------------------------------------------------------
    AVBD2D_API Avbd2dWorldDef avbd2d_default_world_def(void)
    {
        Avbd2dWorldDef d{};
        fromParams(SolverParams2D(), d);
        d.hitEventThreshold = 1.0f;
        return d;
    }

    AVBD2D_API Avbd2dBodyDef avbd2d_default_body_def(void)
    {
        Avbd2dBodyDef d{};
        d.type = AVBD2D_STATIC_BODY;
        d.rotation = Avbd2dRot{1.0f, 0.0f};
        d.gravityScale = 1.0f;
        d.enableSleep = 1;
        d.isAwake = 1;
        d.isEnabled = 1;
        return d;
    }

    AVBD2D_API Avbd2dShapeDef avbd2d_default_shape_def(void)
    {
        Avbd2dShapeDef d{};
        d.friction = 0.6f;
        d.density = 1.0f;
        d.filter.categoryBits = 1;
        d.filter.maskBits = ~uint64_t(0);
        d.enableSensorEvents = 1;
        d.enableContactEvents = 1;
        return d;
    }

    AVBD2D_API Avbd2dChainDef avbd2d_default_chain_def(void)
    {
        Avbd2dChainDef d{};
        d.friction = 0.6f;
        d.filter.categoryBits = 1;
        d.filter.maskBits = ~uint64_t(0);
        return d;
    }

    AVBD2D_API Avbd2dRevoluteJointDef avbd2d_default_revolute_joint_def(void)
    {
        Avbd2dRevoluteJointDef d{};
        d.breakForce = 3.4e38f;
        return d;
    }
    AVBD2D_API Avbd2dWeldJointDef avbd2d_default_weld_joint_def(void)
    {
        Avbd2dWeldJointDef d{};
        d.breakForce = 3.4e38f;
        return d;
    }
    AVBD2D_API Avbd2dPrismaticJointDef avbd2d_default_prismatic_joint_def(void)
    {
        Avbd2dPrismaticJointDef d{};
        d.localAxisA = Avbd2dVec2{1.0f, 0.0f};
        d.breakForce = 3.4e38f;
        return d;
    }
    AVBD2D_API Avbd2dWheelJointDef avbd2d_default_wheel_joint_def(void)
    {
        Avbd2dWheelJointDef d{};
        d.localAxisA = Avbd2dVec2{1.0f, 0.0f};
        d.enableSpring = 1;
        d.hertz = 1.0f;
        d.breakForce = 3.4e38f;
        return d;
    }
    AVBD2D_API Avbd2dDistanceJointDef avbd2d_default_distance_joint_def(void)
    {
        Avbd2dDistanceJointDef d{};
        d.length = 1.0f;
        d.maxLength = 1.0e5f;
        d.breakForce = 3.4e38f;
        return d;
    }
    AVBD2D_API Avbd2dMotorJointDef avbd2d_default_motor_joint_def(void)
    {
        Avbd2dMotorJointDef d{};
        d.maxForce = 1.0f;
        d.maxTorque = 1.0f;
        d.breakForce = 3.4e38f;
        return d;
    }
    AVBD2D_API Avbd2dFilterJointDef avbd2d_default_filter_joint_def(void)
    {
        Avbd2dFilterJointDef d{};
        return d;
    }
    AVBD2D_API Avbd2dExplosionDef avbd2d_default_explosion_def(void)
    {
        Avbd2dExplosionDef d{};
        d.maskBits = ~uint64_t(0);
        return d;
    }
    AVBD2D_API Avbd2dQueryFilter avbd2d_default_query_filter(void)
    {
        Avbd2dQueryFilter f{};
        f.categoryBits = 1;
        f.maskBits = ~uint64_t(0);
        return f;
    }

    // ----- World lifetime --------------------------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_create(Avbd2dWorld **out_world)
    {
        NEED(out_world);
        return createImpl(nullptr, nullptr, out_world);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_world(const Avbd2dWorldDef *def, Avbd2dWorld **out_world)
    {
        NEED(out_world);
        *out_world = nullptr;
        NEED(def);
        return createImpl(def, nullptr, out_world);
    }

    static Avbd2dResult validateDevice(const Avbd2dVulkanDevice *device)
    {
        if (!device || device->struct_size < sizeof(Avbd2dVulkanDevice) || !device->instance ||
            !device->physical_device || !device->device)
            return AVBD2D_ERR_INVALID_ARG;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_create_with_vulkan(const Avbd2dVulkanDevice *device, Avbd2dWorld **out_world)
    {
        NEED(out_world);
        *out_world = nullptr;
        if (validateDevice(device) != AVBD2D_OK)
            return AVBD2D_ERR_INVALID_ARG;
        return createImpl(nullptr, device, out_world);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_world_with_vulkan(const Avbd2dWorldDef *def, const Avbd2dVulkanDevice *device,
                                                            Avbd2dWorld **out_world)
    {
        NEED(out_world);
        *out_world = nullptr;
        if (!def || validateDevice(device) != AVBD2D_OK)
            return AVBD2D_ERR_INVALID_ARG;
        return createImpl(def, device, out_world);
    }

    AVBD2D_API Avbd2dResult avbd2d_destroy(Avbd2dWorld *world)
    {
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            if (!registry().erase(world))
                return AVBD2D_ERR_NULL_HANDLE;
            g_worldTable[world->id].store(nullptr, std::memory_order_release);
        }
        try
        {
            delete world; // ~VkWorld2D tears the device down; a lost device must not take the host with it
        }
        catch (...)
        {
        }
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_set_memory_budget(Avbd2dWorld *world, uint64_t bytes)
    {
        AVBD2D_GET_WORLD(world);
        w->world->device().setMemoryBudget(bytes);
        w->def.memoryBudget = bytes;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_set_world_def(Avbd2dWorld *world, const Avbd2dWorldDef *def)
    {
        AVBD2D_GET_WORLD(world);
        if (!validWorldDef(def))
            return AVBD2D_ERR_INVALID_ARG;
        const uint64_t budget = w->def.memoryBudget;
        w->def = *def;
        w->def.memoryBudget = budget;
        w->world->params() = toParams(*def);
        w->world->setHitEventThreshold(def->hitEventThreshold);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_get_world_def(Avbd2dWorld *world, Avbd2dWorldDef *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        *out = w->def;
        fromParams(w->world->params(), *out);
        out->hitEventThreshold = w->def.hitEventThreshold;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_set_gravity(Avbd2dWorld *world, Avbd2dVec2 gravity)
    {
        AVBD2D_GET_WORLD(world);
        if (!finite2(gravity))
            return AVBD2D_ERR_INVALID_ARG;
        w->world->params().gravity = V(gravity);
        w->def.gravity = gravity;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_gravity(Avbd2dWorld *world, Avbd2dVec2 *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        *out = A(w->world->params().gravity);
        return AVBD2D_OK;
    }

    // ----- Stepping --------------------------------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_step_begin(Avbd2dWorld *world)
    {
        AVBD2D_GET_WORLD(world);
        return stepBeginImpl(w);
    }

    AVBD2D_API Avbd2dResult avbd2d_step_end(Avbd2dWorld *world)
    {
        AVBD2D_GET_WORLD(world);
        if (!w->inFlight)
        {
            const Avbd2dResult r = w->deferred;
            w->deferred = AVBD2D_OK;
            return r;
        }
        return stepEndImpl(w);
    }

    AVBD2D_API Avbd2dResult avbd2d_step(Avbd2dWorld *world)
    {
        AVBD2D_GET_WORLD(world);
        Avbd2dResult r = stepBeginImpl(w);
        if (r != AVBD2D_OK)
            return r;
        w->deferred = AVBD2D_OK;
        return stepEndImpl(w);
    }

    AVBD2D_API Avbd2dResult avbd2d_wake_all(Avbd2dWorld *world)
    {
        AVBD2D_GET_WORLD(world);
        w->world->wakeAll();
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_get_stats(Avbd2dWorld *world, Avbd2dStats *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        *out = Avbd2dStats{};
        out->bodies = w->bodies.live();
        out->joints = w->joints.live();
        out->pairs = w->last.pairs;
        out->contacts = w->last.contacts;
        out->colors = w->last.colors;
        out->capacity_reruns = w->last.capacityReruns;
        out->submits = w->last.submits;
        out->ok = w->last.ok ? 1 : 0;
        out->step_ms = w->last.stepMs;
        if (w->world->bodyCount() > 0)
        {
            const WorldStats2D ws = w->world->stats();
            out->awake = ws.awake;
            out->asleep = ws.asleep;
            out->removed = ws.removed;
            out->broken_joints = ws.brokenJoints;
            out->gpu_ms = w->world->timing().valid ? w->world->timing().gpuMs : 0.0f;
        }
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_get_capabilities(Avbd2dWorld *world, Avbd2dCapabilities *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        if (out->struct_size < sizeof(Avbd2dCapabilities))
            return AVBD2D_ERR_INVALID_ARG;
        Avbd2dCapabilities c{};
        c.struct_size = sizeof(Avbd2dCapabilities);
        c.restitution = 0;
        c.determinism = 0;
        c.continuous = 1;
        c.sensors = 1;
        c.contact_events = 1;
        c.hit_events = 1;
        c.joint_events = 1;
        c.move_events = 1;
        c.joint_forces = 1;
        c.explosions = 1;
        c.queries = 1;
        c.chains = 1;
        c.live_body_create_destroy = 1;
        c.live_shape_add = 1;
        c.per_body_sleep_disable = 0;
        c.set_body_type = 0;
        c.kinematic_bodies = 1;
        c.grab = 1;
        c.max_polygon_vertices = AVBD2D_MAX_POLYGON_VERTICES;
        *out = c;
        return AVBD2D_OK;
    }

    // ===== ABI 3.2 ==================================================================================

    // ----- Layout check ----------------------------------------------------------------------------
    AVBD2D_API int32_t avbd2d_struct_size(int32_t kind)
    {
        switch (kind)
        {
        case AVBD2D_STRUCT_VEC2:
            return (int32_t)sizeof(Avbd2dVec2);
        case AVBD2D_STRUCT_ROT:
            return (int32_t)sizeof(Avbd2dRot);
        case AVBD2D_STRUCT_TRANSFORM:
            return (int32_t)sizeof(Avbd2dTransform);
        case AVBD2D_STRUCT_AABB:
            return (int32_t)sizeof(Avbd2dAABB);
        case AVBD2D_STRUCT_BODY_ID:
            return (int32_t)sizeof(Avbd2dBodyId);
        case AVBD2D_STRUCT_SHAPE_ID:
            return (int32_t)sizeof(Avbd2dShapeId);
        case AVBD2D_STRUCT_JOINT_ID:
            return (int32_t)sizeof(Avbd2dJointId);
        case AVBD2D_STRUCT_CHAIN_ID:
            return (int32_t)sizeof(Avbd2dChainId);
        case AVBD2D_STRUCT_WORLD_DEF:
            return (int32_t)sizeof(Avbd2dWorldDef);
        case AVBD2D_STRUCT_VULKAN_DEVICE:
            return (int32_t)sizeof(Avbd2dVulkanDevice);
        case AVBD2D_STRUCT_STATS:
            return (int32_t)sizeof(Avbd2dStats);
        case AVBD2D_STRUCT_CAPABILITIES:
            return (int32_t)sizeof(Avbd2dCapabilities);
        case AVBD2D_STRUCT_BODY_DEF:
            return (int32_t)sizeof(Avbd2dBodyDef);
        case AVBD2D_STRUCT_FILTER:
            return (int32_t)sizeof(Avbd2dFilter);
        case AVBD2D_STRUCT_SHAPE_DEF:
            return (int32_t)sizeof(Avbd2dShapeDef);
        case AVBD2D_STRUCT_POLYGON:
            return (int32_t)sizeof(Avbd2dPolygon);
        case AVBD2D_STRUCT_CIRCLE:
            return (int32_t)sizeof(Avbd2dCircle);
        case AVBD2D_STRUCT_CAPSULE:
            return (int32_t)sizeof(Avbd2dCapsule);
        case AVBD2D_STRUCT_SEGMENT:
            return (int32_t)sizeof(Avbd2dSegment);
        case AVBD2D_STRUCT_CHAIN_DEF:
            return (int32_t)sizeof(Avbd2dChainDef);
        case AVBD2D_STRUCT_REVOLUTE_JOINT_DEF:
            return (int32_t)sizeof(Avbd2dRevoluteJointDef);
        case AVBD2D_STRUCT_WELD_JOINT_DEF:
            return (int32_t)sizeof(Avbd2dWeldJointDef);
        case AVBD2D_STRUCT_PRISMATIC_JOINT_DEF:
            return (int32_t)sizeof(Avbd2dPrismaticJointDef);
        case AVBD2D_STRUCT_WHEEL_JOINT_DEF:
            return (int32_t)sizeof(Avbd2dWheelJointDef);
        case AVBD2D_STRUCT_DISTANCE_JOINT_DEF:
            return (int32_t)sizeof(Avbd2dDistanceJointDef);
        case AVBD2D_STRUCT_MOTOR_JOINT_DEF:
            return (int32_t)sizeof(Avbd2dMotorJointDef);
        case AVBD2D_STRUCT_FILTER_JOINT_DEF:
            return (int32_t)sizeof(Avbd2dFilterJointDef);
        case AVBD2D_STRUCT_BODY_MOVE_EVENT:
            return (int32_t)sizeof(Avbd2dBodyMoveEvent);
        case AVBD2D_STRUCT_BODY_EVENTS:
            return (int32_t)sizeof(Avbd2dBodyEvents);
        case AVBD2D_STRUCT_CONTACT_BEGIN_TOUCH_EVENT:
            return (int32_t)sizeof(Avbd2dContactBeginTouchEvent);
        case AVBD2D_STRUCT_CONTACT_END_TOUCH_EVENT:
            return (int32_t)sizeof(Avbd2dContactEndTouchEvent);
        case AVBD2D_STRUCT_CONTACT_HIT_EVENT:
            return (int32_t)sizeof(Avbd2dContactHitEvent);
        case AVBD2D_STRUCT_CONTACT_EVENTS:
            return (int32_t)sizeof(Avbd2dContactEvents);
        case AVBD2D_STRUCT_SENSOR_BEGIN_TOUCH_EVENT:
            return (int32_t)sizeof(Avbd2dSensorBeginTouchEvent);
        case AVBD2D_STRUCT_SENSOR_END_TOUCH_EVENT:
            return (int32_t)sizeof(Avbd2dSensorEndTouchEvent);
        case AVBD2D_STRUCT_SENSOR_EVENTS:
            return (int32_t)sizeof(Avbd2dSensorEvents);
        case AVBD2D_STRUCT_JOINT_EVENT:
            return (int32_t)sizeof(Avbd2dJointEvent);
        case AVBD2D_STRUCT_JOINT_EVENTS:
            return (int32_t)sizeof(Avbd2dJointEvents);
        case AVBD2D_STRUCT_BODY_TRANSFORM:
            return (int32_t)sizeof(Avbd2dBodyTransform);
        case AVBD2D_STRUCT_EXPLOSION_DEF:
            return (int32_t)sizeof(Avbd2dExplosionDef);
        case AVBD2D_STRUCT_QUERY_FILTER:
            return (int32_t)sizeof(Avbd2dQueryFilter);
        case AVBD2D_STRUCT_RAY_RESULT:
            return (int32_t)sizeof(Avbd2dRayResult);
        case AVBD2D_STRUCT_SHAPE_PROXY:
            return (int32_t)sizeof(Avbd2dShapeProxy);
        case AVBD2D_STRUCT_RAY_QUERY:
            return (int32_t)sizeof(Avbd2dRayQuery);
        case AVBD2D_STRUCT_BODY_VELOCITY:
            return (int32_t)sizeof(Avbd2dBodyVelocity);
        case AVBD2D_STRUCT_SHAPE_GEOMETRY:
            return (int32_t)sizeof(Avbd2dShapeGeometry);
        case AVBD2D_STRUCT_BODY_BATCH:
            return (int32_t)sizeof(Avbd2dBodyBatch);
        case AVBD2D_STRUCT_BODY_WRENCH:
            return (int32_t)sizeof(Avbd2dBodyWrench);
        case AVBD2D_STRUCT_BODY_STATE_VIEW:
            return (int32_t)sizeof(Avbd2dBodyStateView);
        case AVBD2D_STRUCT_REMOVED_BODIES:
            return (int32_t)sizeof(Avbd2dRemovedBodies);
        case AVBD2D_STRUCT_CONTACT_DEBUG:
            return (int32_t)sizeof(Avbd2dContactDebug);
        case AVBD2D_STRUCT_JOINT_DEBUG:
            return (int32_t)sizeof(Avbd2dJointDebug);
        case AVBD2D_STRUCT_STEP_INFO:
            return (int32_t)sizeof(Avbd2dStepInfo);
        case AVBD2D_STRUCT_GRAB_OPTIONS:
            return (int32_t)sizeof(Avbd2dGrabOptions);
        case AVBD2D_STRUCT_STEP_DIAGNOSTICS:
            return (int32_t)sizeof(Avbd2dStepDiagnostics);
        default:
            return -1;
        }
    }
} // extern "C"
