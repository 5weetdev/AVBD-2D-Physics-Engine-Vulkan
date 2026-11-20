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


// Debug views, step info and diagnostics of the 2D C ABI.

#include "avbd2d_world.h"

extern "C"
{

    // ----- Body state view -------------------------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_world_get_body_state_view(Avbd2dWorld *world, Avbd2dBodyStateView *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        NEED(out->struct_size >= sizeof(Avbd2dBodyStateView));
        out->struct_size = sizeof(Avbd2dBodyStateView);
        out->slotCount = (int32_t)w->viewFlags.size();
        out->transforms = w->viewXf.empty() ? nullptr : w->viewXf.data();
        out->velocities = w->viewVel.empty() ? nullptr : w->viewVel.data();
        out->flags = w->viewFlags.empty() ? nullptr : w->viewFlags.data();
        out->generations = w->viewGen.empty() ? nullptr : w->viewGen.data();
        out->version = w->viewVersion;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_removed_bodies(Avbd2dWorld *world, Avbd2dRemovedBodies *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        out->bodyIds = w->removedBodies.empty() ? nullptr : w->removedBodies.data();
        out->count = (int32_t)w->removedBodies.size();
        return AVBD2D_OK;
    }

    // ----- Debug and diagnostics -------------------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_world_set_debug_flags(Avbd2dWorld *world, uint32_t flags)
    {
        AVBD2D_GET_WORLD(world);
        NEED((flags & ~(uint32_t)(AVBD2D_DEBUG_BODY_EXTRAS | AVBD2D_DEBUG_CONTACTS)) == 0);
        w->debugFlags = flags;
        w->world->setExtraReadback((flags & AVBD2D_DEBUG_BODY_EXTRAS) != 0);
        w->world->setContactDebug((flags & AVBD2D_DEBUG_CONTACTS) != 0); // latched on in the world
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_set_events_enabled(Avbd2dWorld *world, int32_t enabled)
    {
        AVBD2D_GET_WORLD(world);
        w->world->setEventsEnabled(enabled != 0);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_body_colors(Avbd2dWorld *world, const int32_t **out_colors,
                                                         int32_t *out_count)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out_colors);
        NEED(out_count);
        *out_colors = w->slotColors.empty() ? nullptr : w->slotColors.data();
        *out_count = (int32_t)w->slotColors.size();
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_contact_debug(Avbd2dWorld *world, const Avbd2dContactDebug **out,
                                                           int32_t *out_count)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        NEED(out_count);
        settle(w);
        const Avbd2dResult r = guarded(w, false,
                                       [&]() -> Avbd2dResult
                                       {
                                           if (w->contactDebugAt == w->stepsEnded)
                                               return AVBD2D_OK;
                                           w->contactDebugAt = w->stepsEnded;
                                           w->contactDebug.clear();
                                           if (!(w->debugFlags & AVBD2D_DEBUG_CONTACTS) || w->stepsEnded == 0 ||
                                               w->world->bodyCount() == 0)
                                               return AVBD2D_OK;
                                           std::vector<ContactDebug2D> raw;
                                           w->world->downloadContacts(raw);
                                           w->contactDebug.reserve(raw.size());
                                           for (const ContactDebug2D &c : raw)
                                           {
                                               Avbd2dContactDebug d{};
                                               d.point = A(c.p);
                                               d.anchorA = A(c.pA);
                                               d.anchorB = A(c.pB);
                                               d.normal = A(c.n);
                                               d.normalForce = c.force;
                                               d.tangentForce = c.tangent;
                                               d.separation = c.sep;
                                               d.friction = c.mu;
                                               d.bodyIdA = bodyIdOfSolver(w, c.bodyA);
                                               d.bodyIdB = bodyIdOfSolver(w, c.bodyB);
                                               d.penalty = A(c.penalty);
                                               d.sticking = c.stick ? 1 : 0;
                                               w->contactDebug.push_back(d);
                                           }
                                           return AVBD2D_OK;
                                       });
        if (r != AVBD2D_OK)
            return r;
        *out = w->contactDebug.empty() ? nullptr : w->contactDebug.data();
        *out_count = (int32_t)w->contactDebug.size();
        return AVBD2D_OK;
    }

    static int32_t abiJointType(JointType t)
    {
        switch (t)
        {
        case JointType::Revolute:
            return AVBD2D_REVOLUTE_JOINT;
        case JointType::Weld:
            return AVBD2D_WELD_JOINT;
        case JointType::Prismatic:
            return AVBD2D_PRISMATIC_JOINT;
        case JointType::Wheel:
            return AVBD2D_WHEEL_JOINT;
        case JointType::Distance:
            return AVBD2D_DISTANCE_JOINT;
        case JointType::Motor:
            return AVBD2D_MOTOR_JOINT;
        case JointType::Filter:
            return AVBD2D_FILTER_JOINT;
        }
        return AVBD2D_FILTER_JOINT;
    }

    AVBD2D_API Avbd2dResult avbd2d_joint_get_type(Avbd2dJointId id, int32_t *out)
    {
        JOINT_CTX(id);
        NEED(out);
        *out = abiJointType(j->d.type);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_joint_get_bodies(Avbd2dJointId id, Avbd2dBodyId *out_a, Avbd2dBodyId *out_b)
    {
        JOINT_CTX(id);
        NEED(out_a);
        NEED(out_b);
        *out_a = bodyIdOf(w, j->bodyA);
        *out_b = bodyIdOf(w, j->bodyB);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_joint_debug(Avbd2dWorld *world, const Avbd2dJointDebug **out,
                                                         int32_t *out_count)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        NEED(out_count);
        settle(w); // the constraint forces are copied by the step in flight
        if (w->jointDebugAt != w->stepsEnded)
        {
            w->jointDebugAt = w->stepsEnded;
            w->jointDebug.clear();
            for (size_t i = 0; i < w->joints.items.size(); i++)
            {
                const JointRec &j = w->joints.items[i];
                if (!j.alive)
                    continue;
                BodyRec &ba = w->bodies.items[(size_t)j.bodyA];
                BodyRec &bb = w->bodies.items[(size_t)j.bodyB];
                const Pose pa = poseOf(w, ba), pb = poseOf(w, bb);
                Avbd2dJointDebug d{};
                d.jointId = jointIdOf(w, (int)i);
                d.anchorA = A(pa.origin() + rotate(pa.angle, j.d.localAnchorA));
                d.anchorB = A(pb.origin() + rotate(pb.angle, j.d.localAnchorB));
                if (j.flushed && j.solver >= 0)
                {
                    float f[3];
                    w->world->jointForce(j.solver, f);
                    d.force = Avbd2dVec2{f[0], f[1]};
                    d.torque = f[2];
                }
                d.type = abiJointType(j.d.type);
                w->jointDebug.push_back(d);
            }
        }
        *out = w->jointDebug.empty() ? nullptr : w->jointDebug.data();
        *out_count = (int32_t)w->jointDebug.size();
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_get_step_info(Avbd2dWorld *world, Avbd2dStepInfo *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        NEED(out->struct_size >= sizeof(Avbd2dStepInfo));
        Avbd2dStepInfo s{};
        s.struct_size = sizeof(Avbd2dStepInfo);
        const StepResult2D &r = w->last;
        s.ok = r.ok ? 1 : 0;
        s.truncated = r.truncated ? 1 : 0;
        s.capacityReruns = r.capacityReruns;
        s.submits = r.submits;
        s.pairs = r.pairs;
        s.contacts = r.contacts;
        s.colors = r.colors;
        s.removedThisStep = r.removed;
        s.brokenJointsThisStep = r.brokenJoints;
        s.eventsTruncatedMask = r.eventsTruncated;
        s.stepMs = r.stepMs;
        s.bodies = w->bodies.live();
        s.joints = w->joints.live();
        if (w->world->bodyCount() > 0)
        {
            const WorldStats2D ws = w->world->stats();
            s.awake = ws.awake;
            s.asleep = ws.asleep;
            s.removedTotal = ws.removed;
            s.brokenJointsTotal = ws.brokenJoints;
            s.statics = ws.statics;
            s.dynamic = ws.dynamic;
            s.pairCapacity = ws.pairCap;
            s.contactCapacity = ws.contactCap;
            s.bodyCapacity = ws.bodyCap;
            s.deviceBytes = ws.deviceBytes;
            const CcdStats2D cs = w->world->ccdStats();
            s.ccdClamped = cs.clamped;
            s.ccdTruncated = cs.truncated;
            const StepTiming2D &t = w->world->timing();
            s.timingValid = t.valid ? 1 : 0;
            if (t.valid)
            {
                s.gpuMs = t.gpuMs;
                s.integrateMs = t.integrateMs;
                s.broadMs = t.broadMs;
                s.narrowMs = t.narrowMs;
                s.buildMs = t.buildMs;
                s.primalMs = t.primalMs;
                s.dualMs = t.dualMs;
                s.tailMs = t.tailMs;
            }
        }
        *out = s;
        return AVBD2D_OK;
    }

    // ----- ABI 3.6: substeps, diagnostics, poll ------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_world_set_substeps(Avbd2dWorld *world, int32_t substeps)
    {
        AVBD2D_GET_WORLD(world);
        w->world->setSubsteps(substeps);
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_world_get_substeps(Avbd2dWorld *world, int32_t *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        *out = w->world->substeps();
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_get_step_diagnostics(Avbd2dWorld *world, Avbd2dStepDiagnostics *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        NEED(out->struct_size >= sizeof(Avbd2dStepDiagnostics));
        Avbd2dStepDiagnostics d{};
        d.struct_size = sizeof(Avbd2dStepDiagnostics);
        const StepDiagnostics2D &s = w->world->diagnostics();
        d.substeps = s.substeps;
        d.contacts = s.contacts;
        d.colors = s.colors;
        d.colorReruns = s.colorReruns;
        d.contactCflMax = s.contactCflMax;
        d.penetrationRatioMax = s.penetrationRatioMax;
        d.speedMax = s.speedMax;
        *out = d;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_step_poll(Avbd2dWorld *world, int32_t *out_done)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out_done);
        *out_done = (!w->inFlight || w->world->stepDone()) ? 1 : 0;
        return AVBD2D_OK;
    }
} // extern "C"
