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


// Joints of the 2D C ABI.

#include "avbd2d_world.h"

extern "C"
{

    // ----- Joints ----------------------------------------------------------------------------------
    static Avbd2dResult makeJoint(Avbd2dWorld *world, Avbd2dBodyId ida, Avbd2dBodyId idb, JointDef d, float breakForce,
                                  void *userData, Avbd2dJointId *out_id)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out_id);
        if (ida.world0 != w->id || idb.world0 != w->id)
            return AVBD2D_ERR_INVALID_ARG;
        if (!w->bodies.find(ida.index1, ida.generation) || !w->bodies.find(idb.index1, idb.generation) ||
            ida.index1 == idb.index1)
            return AVBD2D_ERR_INVALID_ARG;
        if (!(breakForce > 0.0f))
            return AVBD2D_ERR_INVALID_ARG;
        d.breakForce = breakForce >= 1.0e38f ? kStiffInf : breakForce;
        return guarded(w, true,
                       [&]() -> Avbd2dResult
                       {
                           const int slot = w->joints.acquire();
                           JointRec &j = w->joints.items[(size_t)slot];
                           j.d = d;
                           j.bodyA = ida.index1 - 1;
                           j.bodyB = idb.index1 - 1;
                           j.userData = userData;
                           w->bodies.items[(size_t)j.bodyA].joints.push_back(slot);
                           w->bodies.items[(size_t)j.bodyB].joints.push_back(slot);
                           w->pendingJoints.push_back(slot);
                           *out_id = jointIdOf(w, slot);
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_create_revolute_joint(Avbd2dWorld *world, const Avbd2dRevoluteJointDef *def,
                                                         Avbd2dJointId *out_id)
    {
        NEED(def);
        if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !finite1(def->referenceAngle))
            return AVBD2D_ERR_INVALID_ARG;
        JointDef d;
        d.type = JointType::Revolute;
        d.localAnchorA = V(def->localAnchorA);
        d.localAnchorB = V(def->localAnchorB);
        d.referenceAngle = def->referenceAngle;
        d.enableSpring = def->enableSpring != 0;
        d.hertz = def->hertz;
        d.enableLimit = def->enableLimit != 0;
        d.lower = def->lowerAngle;
        d.upper = def->upperAngle;
        d.enableMotor = def->enableMotor != 0;
        d.motorSpeed = def->motorSpeed;
        d.maxMotorTorque = def->maxMotorTorque;
        d.collideConnected = def->collideConnected != 0;
        return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_weld_joint(Avbd2dWorld *world, const Avbd2dWeldJointDef *def,
                                                     Avbd2dJointId *out_id)
    {
        NEED(def);
        if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !finite1(def->referenceAngle))
            return AVBD2D_ERR_INVALID_ARG;
        JointDef d;
        d.type = JointType::Weld;
        d.localAnchorA = V(def->localAnchorA);
        d.localAnchorB = V(def->localAnchorB);
        d.referenceAngle = def->referenceAngle;
        d.linearHertz = def->linearHertz;
        d.angularHertz = def->angularHertz;
        d.collideConnected = def->collideConnected != 0;
        return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_prismatic_joint(Avbd2dWorld *world, const Avbd2dPrismaticJointDef *def,
                                                          Avbd2dJointId *out_id)
    {
        NEED(def);
        if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !finite2(def->localAxisA) ||
            (def->localAxisA.x == 0.0f && def->localAxisA.y == 0.0f) || !finite1(def->referenceAngle))
            return AVBD2D_ERR_INVALID_ARG;
        JointDef d;
        d.type = JointType::Prismatic;
        d.localAnchorA = V(def->localAnchorA);
        d.localAnchorB = V(def->localAnchorB);
        d.localAxisA = V(def->localAxisA);
        d.referenceAngle = def->referenceAngle;
        d.enableSpring = def->enableSpring != 0;
        d.hertz = def->hertz;
        d.enableLimit = def->enableLimit != 0;
        d.lower = def->lowerTranslation;
        d.upper = def->upperTranslation;
        d.enableMotor = def->enableMotor != 0;
        d.motorSpeed = def->motorSpeed;
        d.maxMotorForce = def->maxMotorForce;
        d.collideConnected = def->collideConnected != 0;
        return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_wheel_joint(Avbd2dWorld *world, const Avbd2dWheelJointDef *def,
                                                      Avbd2dJointId *out_id)
    {
        NEED(def);
        if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !finite2(def->localAxisA) ||
            (def->localAxisA.x == 0.0f && def->localAxisA.y == 0.0f))
            return AVBD2D_ERR_INVALID_ARG;
        JointDef d;
        d.type = JointType::Wheel;
        d.localAnchorA = V(def->localAnchorA);
        d.localAnchorB = V(def->localAnchorB);
        d.localAxisA = V(def->localAxisA);
        d.enableSpring = def->enableSpring != 0;
        d.hertz = def->hertz;
        d.enableLimit = def->enableLimit != 0;
        d.lower = def->lowerTranslation;
        d.upper = def->upperTranslation;
        d.enableMotor = def->enableMotor != 0;
        d.motorSpeed = def->motorSpeed;
        d.maxMotorTorque = def->maxMotorTorque;
        d.collideConnected = def->collideConnected != 0;
        return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_distance_joint(Avbd2dWorld *world, const Avbd2dDistanceJointDef *def,
                                                         Avbd2dJointId *out_id)
    {
        NEED(def);
        if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !(def->length >= 0.0f))
            return AVBD2D_ERR_INVALID_ARG;
        JointDef d;
        d.type = JointType::Distance;
        d.localAnchorA = V(def->localAnchorA);
        d.localAnchorB = V(def->localAnchorB);
        d.length = def->length;
        d.enableSpring = def->enableSpring != 0;
        d.hertz = def->hertz;
        d.enableLimit = def->enableLimit != 0;
        d.minLength = def->minLength;
        d.maxLength = def->maxLength;
        d.enableMotor = def->enableMotor != 0;
        d.motorSpeed = def->motorSpeed;
        d.maxMotorForce = def->maxMotorForce;
        d.collideConnected = def->collideConnected != 0;
        return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_motor_joint(Avbd2dWorld *world, const Avbd2dMotorJointDef *def,
                                                      Avbd2dJointId *out_id)
    {
        NEED(def);
        if (!finite2(def->linearOffset) || !finite1(def->angularOffset))
            return AVBD2D_ERR_INVALID_ARG;
        JointDef d;
        d.type = JointType::Motor;
        d.linearOffset = V(def->linearOffset);
        d.angularOffset = def->angularOffset;
        d.maxMotorForce = def->maxForce;
        d.maxMotorTorque = def->maxTorque;
        d.collideConnected = def->collideConnected != 0;
        return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_create_filter_joint(Avbd2dWorld *world, const Avbd2dFilterJointDef *def,
                                                       Avbd2dJointId *out_id)
    {
        NEED(def);
        JointDef d;
        d.type = JointType::Filter;
        return makeJoint(world, def->bodyIdA, def->bodyIdB, d, 3.4e38f, def->userData, out_id);
    }

    AVBD2D_API Avbd2dResult avbd2d_destroy_joint(Avbd2dJointId id)
    {
        JOINT_CTX(id);
        return guarded(w, false,
                       [&]() -> Avbd2dResult
                       {
                           destroyJointSlot(w, slot);
                           return AVBD2D_OK;
                       });
    }

    AVBD2D_API Avbd2dResult avbd2d_joint_get_user_data(Avbd2dJointId id, void **out)
    {
        JOINT_CTX(id);
        NEED(out);
        *out = j->userData;
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_joint_get_constraint_force(Avbd2dJointId id, Avbd2dVec2 *out)
    {
        JOINT_CTX(id);
        NEED(out);
        float f[3] = {};
        if (j->flushed && j->solver >= 0)
            w->world->jointForce(j->solver, f);
        *out = Avbd2dVec2{f[0], f[1]};
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_joint_get_constraint_torque(Avbd2dJointId id, float *out)
    {
        JOINT_CTX(id);
        NEED(out);
        float f[3] = {};
        if (j->flushed && j->solver >= 0)
            w->world->jointForce(j->solver, f);
        *out = f[2];
        return AVBD2D_OK;
    }

    AVBD2D_API Avbd2dResult avbd2d_joint_wake_bodies(Avbd2dJointId id)
    {
        JOINT_CTX(id);
        if (j->flushed && j->solver >= 0)
            w->world->queueJointWake(j->solver);
        return AVBD2D_OK;
    }

    static void syncLimit(Avbd2dWorld *w, const JointRec &j)
    {
        if (j.flushed && j.solver >= 0)
            w->world->queueJointLimit(j.solver, limitAxis(j.d.type), j.d.enableLimit, j.d.lower, j.d.upper);
    }

    static void syncLockScale(Avbd2dWorld *w, const JointRec &j)
    {
        if (j.flushed && j.solver >= 0)
            w->world->queueJointLockScale(j.solver, j.lockScale);
    }

#define JOINT_SETTER(name, TYPE, ARGDECL, BODY, SYNC)                                                                  \
    AVBD2D_API Avbd2dResult name(Avbd2dJointId id ARGDECL)                                                             \
    {                                                                                                                  \
        JOINT_CTX(id);                                                                                                 \
        if (j->d.type != JointType::TYPE)                                                                              \
            return AVBD2D_ERR_INVALID_ARG;                                                                             \
        BODY;                                                                                                          \
        SYNC(w, *j);                                                                                                   \
        return AVBD2D_OK;                                                                                              \
    }

#define COMMA ,
    JOINT_SETTER(avbd2d_revolute_joint_enable_limit, Revolute, COMMA int32_t flag, j->d.enableLimit = flag != 0,
                 syncLimit)
    JOINT_SETTER(avbd2d_revolute_joint_set_limits, Revolute, COMMA float lower COMMA float upper,
                 (j->d.lower = lower, j->d.upper = upper), syncLimit)
    JOINT_SETTER(avbd2d_revolute_joint_enable_motor, Revolute, COMMA int32_t flag, j->d.enableMotor = flag != 0,
                 syncMotor)
    JOINT_SETTER(avbd2d_revolute_joint_set_motor_speed, Revolute, COMMA float speed,
                 (j->d.motorMode = JOINT_MOTOR_VELOCITY, j->d.motorSpeed = speed), syncMotor)
    JOINT_SETTER(avbd2d_revolute_joint_set_max_motor_torque, Revolute, COMMA float torque, j->d.maxMotorTorque = torque,
                 syncMotor)

    // Not a JOINT_SETTER: the scale is validated and clamped, and it is not stored in the scene def.
    AVBD2D_API Avbd2dResult avbd2d_revolute_joint_set_lock_scale(Avbd2dJointId id, float scale)
    {
        if (!std::isfinite(scale) || scale < 1.0f)
            return AVBD2D_ERR_INVALID_ARG;
        JOINT_CTX(id);
        if (j->d.type != JointType::Revolute)
            return AVBD2D_ERR_INVALID_ARG;
        j->lockScale = scale > 64.0f ? 64.0f : scale;
        syncLockScale(w, *j);
        return AVBD2D_OK;
    }

    JOINT_SETTER(avbd2d_prismatic_joint_enable_limit, Prismatic, COMMA int32_t flag, j->d.enableLimit = flag != 0,
                 syncLimit)
    JOINT_SETTER(avbd2d_prismatic_joint_set_limits, Prismatic, COMMA float lower COMMA float upper,
                 (j->d.lower = lower, j->d.upper = upper), syncLimit)
    JOINT_SETTER(avbd2d_prismatic_joint_enable_motor, Prismatic, COMMA int32_t flag, j->d.enableMotor = flag != 0,
                 syncMotor)
    JOINT_SETTER(avbd2d_prismatic_joint_set_motor_speed, Prismatic, COMMA float speed, j->d.motorSpeed = speed,
                 syncMotor)
    JOINT_SETTER(avbd2d_prismatic_joint_set_max_motor_force, Prismatic, COMMA float force, j->d.maxMotorForce = force,
                 syncMotor)

    JOINT_SETTER(avbd2d_wheel_joint_enable_limit, Wheel, COMMA int32_t flag, j->d.enableLimit = flag != 0, syncLimit)
    JOINT_SETTER(avbd2d_wheel_joint_set_limits, Wheel, COMMA float lower COMMA float upper,
                 (j->d.lower = lower, j->d.upper = upper), syncLimit)
    JOINT_SETTER(avbd2d_wheel_joint_enable_motor, Wheel, COMMA int32_t flag, j->d.enableMotor = flag != 0, syncMotor)
    JOINT_SETTER(avbd2d_wheel_joint_set_motor_speed, Wheel, COMMA float speed, j->d.motorSpeed = speed, syncMotor)
    JOINT_SETTER(avbd2d_wheel_joint_set_max_motor_torque, Wheel, COMMA float torque, j->d.maxMotorTorque = torque,
                 syncMotor)

    JOINT_SETTER(avbd2d_distance_joint_enable_limit, Distance, COMMA int32_t flag, j->d.enableLimit = flag != 0,
                 syncLimit)
    JOINT_SETTER(avbd2d_distance_joint_set_length_range, Distance, COMMA float minLength COMMA float maxLength,
                 (j->d.minLength = minLength, j->d.maxLength = maxLength), syncLimit)
    JOINT_SETTER(avbd2d_distance_joint_enable_motor, Distance, COMMA int32_t flag, j->d.enableMotor = flag != 0,
                 syncMotor)
    JOINT_SETTER(avbd2d_distance_joint_set_motor_speed, Distance, COMMA float speed, j->d.motorSpeed = speed, syncMotor)
    JOINT_SETTER(avbd2d_distance_joint_set_max_motor_force, Distance, COMMA float force, j->d.maxMotorForce = force,
                 syncMotor)

    // ----- Events ----------------------------------------------------------------------------------
    AVBD2D_API Avbd2dResult avbd2d_world_get_body_events(Avbd2dWorld *world, Avbd2dBodyEvents *out)
    {
        AVBD2D_GET_WORLD(world);
        NEED(out);
        out->moveEvents = w->bodyEvents.empty() ? nullptr : w->bodyEvents.data();
        out->moveCount = (int32_t)w->bodyEvents.size();
        return AVBD2D_OK;
    }
} // extern "C"
