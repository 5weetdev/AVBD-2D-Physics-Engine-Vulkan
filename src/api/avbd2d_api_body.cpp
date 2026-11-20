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


// Bodies of the 2D C ABI: creation, destruction, getters and setters.

#include "avbd2d_world.h"

extern "C"
{

    // ----- Bodies ----------------------------------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_create_body(Avbd2dWorld *world, const Avbd2dBodyDef *def, Avbd2dBodyId *out_id)
    {
        AVBD2D_GET_WORLD(world);
        NEED(def);
        NEED(out_id);
        if (!finite2(def->position) || !finite1(def->rotation.c) || !finite1(def->rotation.s) ||
            !finite2(def->linearVelocity) || !finite1(def->angularVelocity) || def->type < AVBD2D_STATIC_BODY ||
            def->type > AVBD2D_DYNAMIC_BODY)
            return AVBD2D_ERR_INVALID_ARG;
        return guarded(w, true,
                       [&]() -> Avbd2dResult
                       {
                           const int slot = w->bodies.acquire();
                           BodyRec &b = w->bodies.items[(size_t)slot];
                           b.def = *def;
                           w->pendingBodies.push_back(slot);
                           markDirty(w, slot);
                           *out_id = bodyIdOf(w, slot);
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_destroy_body(Avbd2dBodyId id)
    {
        BODY_CTX(id);
        (void)b;
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           destroyBodySlot(w, slot, true);
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_position(Avbd2dBodyId id, Avbd2dVec2 *out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = A(poseOf(w, *b).origin());
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_rotation(Avbd2dBodyId id, Avbd2dRot *out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = live(*b) ? rotOf(poseOf(w, *b).angle) : b->def.rotation;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_angle(Avbd2dBodyId id, float *out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = poseOf(w, *b).angle;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_transform(Avbd2dBodyId id, Avbd2dTransform *out)
    {
        BODY_CTX(id);
        NEED(out);
        const Pose p = poseOf(w, *b);
        out->p = A(p.origin());
        out->q = live(*b) ? rotOf(p.angle) : b->def.rotation;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_world_center(Avbd2dBodyId id, Avbd2dVec2 *out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = A(poseOf(w, *b).com);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_local_center(Avbd2dBodyId id, Avbd2dVec2 *out)
    {
        BODY_CTX(id);
        NEED(out);
        ensureProps(w, *b);
        *out = A(b->lc);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_linear_velocity(Avbd2dBodyId id, Avbd2dVec2 *out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = live(*b) ? A(w->world->velocities()[(size_t)b->solver]) : b->def.linearVelocity;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_angular_velocity(Avbd2dBodyId id, float *out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = live(*b) ? w->world->angularVelocities()[(size_t)b->solver] : b->def.angularVelocity;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_mass(Avbd2dBodyId id, float *out)
    {
        BODY_CTX(id);
        NEED(out);
        ensureProps(w, *b);
        *out = b->mass;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_rotational_inertia(Avbd2dBodyId id, float *out)
    {
        BODY_CTX(id);
        NEED(out);
        ensureProps(w, *b);
        *out = b->inertia;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_type(Avbd2dBodyId id, int32_t *out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = b->def.type;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_is_awake(Avbd2dBodyId id, int32_t *out)
    {
        BODY_CTX(id);
        NEED(out);
        if (live(*b))
            *out = w->world->awakeFlags()[(size_t)b->solver] ? 1 : 0;
        else
            *out = (b->def.isAwake && b->def.isEnabled && b->def.type != AVBD2D_STATIC_BODY) ? 1 : 0;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_is_enabled(Avbd2dBodyId id, int32_t *out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = b->def.isEnabled ? 1 : 0;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_get_user_data(Avbd2dBodyId id, void **out)
    {
        BODY_CTX(id);
        NEED(out);
        *out = b->def.userData;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_set_user_data(Avbd2dBodyId id, void *userData)
    {
        BODY_CTX(id);
        b->def.userData = userData;
        markDirty(w, slot); // the move events read it from the view
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_set_transform(Avbd2dBodyId id, Avbd2dVec2 position, Avbd2dRot rotation)
    {
        BODY_CTX(id);
        if (!finite2(position) || !finiteRot(rotation))
            return AVBD2D_ERR_INVALID_ARG;
        setTransformOn(w, *b, slot, position, rotation);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_set_linear_velocity(Avbd2dBodyId id, Avbd2dVec2 v)
    {
        BODY_CTX(id);
        if (!finite2(v))
            return AVBD2D_ERR_INVALID_ARG;
        setVelocityOn(w, *b, slot, &v, nullptr);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_set_angular_velocity(Avbd2dBodyId id, float av)
    {
        BODY_CTX(id);
        if (!finite1(av))
            return AVBD2D_ERR_INVALID_ARG;
        setVelocityOn(w, *b, slot, nullptr, &av);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_apply_force(Avbd2dBodyId id, Avbd2dVec2 force, Avbd2dVec2 worldPoint,
                                                    int32_t wake)
    {
        BODY_CTX(id);
        if (!finite2(force) || !finite2(worldPoint))
            return AVBD2D_ERR_INVALID_ARG;
        const float dt = w->world->params().dt;
        const Vec2 p = V(worldPoint);
        impulseOn(w, *b, V(force) * dt, &p, 0.0f, wake);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_apply_force_to_center(Avbd2dBodyId id, Avbd2dVec2 force, int32_t wake)
    {
        BODY_CTX(id);
        if (!finite2(force))
            return AVBD2D_ERR_INVALID_ARG;
        impulseOn(w, *b, V(force) * w->world->params().dt, nullptr, 0.0f, wake);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_apply_torque(Avbd2dBodyId id, float torque, int32_t wake)
    {
        BODY_CTX(id);
        if (!finite1(torque))
            return AVBD2D_ERR_INVALID_ARG;
        impulseOn(w, *b, Vec2(), nullptr, torque * w->world->params().dt, wake);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_apply_linear_impulse(Avbd2dBodyId id, Avbd2dVec2 impulse, Avbd2dVec2 worldPoint,
                                                             int32_t wake)
    {
        BODY_CTX(id);
        if (!finite2(impulse) || !finite2(worldPoint))
            return AVBD2D_ERR_INVALID_ARG;
        const Vec2 p = V(worldPoint);
        impulseOn(w, *b, V(impulse), &p, 0.0f, wake);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_apply_linear_impulse_to_center(Avbd2dBodyId id, Avbd2dVec2 impulse,
                                                                       int32_t wake)
    {
        BODY_CTX(id);
        if (!finite2(impulse))
            return AVBD2D_ERR_INVALID_ARG;
        impulseOn(w, *b, V(impulse), nullptr, 0.0f, wake);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_apply_angular_impulse(Avbd2dBodyId id, float impulse, int32_t wake)
    {
        BODY_CTX(id);
        if (!finite1(impulse))
            return AVBD2D_ERR_INVALID_ARG;
        impulseOn(w, *b, Vec2(), nullptr, impulse, wake);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_set_awake(Avbd2dBodyId id, int32_t awake)
    {
        BODY_CTX(id);
        setAwakeOn(w, *b, slot, awake);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_enable(Avbd2dBodyId id)
    {
        BODY_CTX(id);
        setEnabledOn(w, *b, slot, 1);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_body_disable(Avbd2dBodyId id)
    {
        BODY_CTX(id);
        setEnabledOn(w, *b, slot, 0);
        return AVBD2D_OK;
    }
} // extern "C"
