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


// Mouse grab of the 2D C ABI.

#include "avbd2d_world.h"

extern "C"
{

    // ----- Mouse grab ------------------------------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_grab_begin(Avbd2dWorld *world, Avbd2dVec2 point, Avbd2dBodyId *out_body)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out_body);
        if (!finite2(point))
            return AVBD2D_ERR_INVALID_ARG;
        *out_body = Avbd2dBodyId{};
        settle(w);
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           flush(w);
                           if (w->world->bodyCount() == 0)
                               return AVBD2D_OK;
                           if (w->world->grabBegin(V(point)))
                           {
                               const int sb = w->world->grabbedBody();
                               if (sb >= 0 && sb < (int)w->solverBody.size() && w->solverBody[(size_t)sb] >= 0)
                                   *out_body = bodyIdOf(w, w->solverBody[(size_t)sb]);
                           }
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_grab_move(Avbd2dWorld *world, Avbd2dVec2 point)
    {
        AVBD2D_GET_WORLD(world);
        if (!finite2(point))
            return AVBD2D_ERR_INVALID_ARG;
        if (w->world->bodyCount() > 0)
            w->world->grabMove(V(point));
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_grab_end(Avbd2dWorld *world)
    {
        AVBD2D_GET_WORLD(world);
        settle(w);
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           if (w->world->bodyCount() > 0)
                               w->world->grabEnd();
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_grab_set_angle(Avbd2dWorld *world, float angle)
    {
        AVBD2D_GET_WORLD(world);
        if (!finite1(angle))
            return AVBD2D_ERR_INVALID_ARG;
        if (w->world->bodyCount() > 0)
            w->world->grabSetAngle(angle);
        return AVBD2D_OK;
    }

    // ----- Grab options ----------------------------------------------------------------------------
    AVBD2D_API Avbd2dGrabOptions avbd2d_default_grab_options(void)
    {
        const VkWorld2D::GrabOptions o;
        Avbd2dGrabOptions d{};
        d.struct_size = sizeof(Avbd2dGrabOptions);
        d.radius = o.radius;
        d.followJoints = o.followJoints ? 1 : 0;
        d.falloff = o.falloff;
        d.minStrength = o.minStrength;
        d.angularStrength = o.angularStrength;
        return d;
    }

    AVBD2D_API Avbd2dResult avbd2d_grab_begin_ex(Avbd2dWorld *world, Avbd2dVec2 point, const Avbd2dGrabOptions *options,
                                                 Avbd2dBodyId *out_body, int32_t *out_count)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out_body);
        NEED(options);
        NEED(options->struct_size >=
             offsetof(Avbd2dGrabOptions, angularStrength)); // pre-3.3 callers end at minStrength
        const bool hasAng = options->struct_size >= sizeof(Avbd2dGrabOptions);
        if (hasAng && !(options->angularStrength >= 0.0f && finite1(options->angularStrength)))
            return AVBD2D_ERR_INVALID_ARG;
        if (!finite2(point) || !(options->radius >= 0.0f) || !finite1(options->radius) || !(options->falloff >= 0.0f) ||
            !(options->minStrength >= 0.0f) || !(options->minStrength <= 1.0f))
            return AVBD2D_ERR_INVALID_ARG;
        *out_body = Avbd2dBodyId{};
        if (out_count)
            *out_count = 0;
        settle(w);
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           flush(w);
                           if (w->world->bodyCount() == 0)
                               return AVBD2D_OK;
                           VkWorld2D::GrabOptions o;
                           o.radius = options->radius;
                           o.followJoints = options->followJoints != 0;
                           o.falloff = options->falloff;
                           o.minStrength = options->minStrength;
                           if (hasAng)
                               o.angularStrength = options->angularStrength;
                           if (w->world->grabBegin(V(point), o))
                           {
                               *out_body = bodyIdOfSolver(w, w->world->grabbedBody());
                               if (out_count)
                                   *out_count = w->world->grabCount();
                           }
                           return AVBD2D_OK;
                       });
    }
} // extern "C"
