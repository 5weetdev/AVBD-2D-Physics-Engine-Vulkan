#include "vk_world2d.h"
#include "vk_world2d_internal.h"

#include "vk_capacity.h"
#include "vk_util.h"

#include "broadphase2d_spirv.h"
#include "ccd2d_spirv.h"
#include "constraints_spirv.h"
#include "constraints2d_spirv.h"
#include "events2d_spirv.h"
#include "integrate2d_spirv.h"
#include "narrowphase2d_spirv.h"
#include "query2d_spirv.h"
#include "sleep2d_spirv.h"
#include "solver2d_spirv.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace avbdvk;

namespace avbd2d
{


// ---------------------------------------------------------------------------
// Mutation queue
// ---------------------------------------------------------------------------
VkWorld2D::BodyPatch &VkWorld2D::bodyPatch(int body)
{
    if ((int)m_bodyPatchSlot.size() < m_bodyCount)
        m_bodyPatchSlot.resize((size_t)m_bodyCount, -1);
    int &slot = m_bodyPatchSlot[(size_t)body];
    if (slot < 0)
    {
        slot = (int)m_bodyPatches.size();
        m_bodyPatches.emplace_back();
        m_bodyPatches.back().body = (uint32_t)body;
    }
    return m_bodyPatches[(size_t)slot];
}


VkWorld2D::JointPatch &VkWorld2D::jointPatch(int joint)
{
    // A lock scale is queued while a new joint is flushed, before m_jointCount is updated. Grow to the
    // requested index so that early patch is not dropped (the device jointCount is final by the step).
    const int need = joint + 1;
    if ((int)m_jointPatchSlot.size() < need)
        m_jointPatchSlot.resize((size_t)need, -1);
    int &slot = m_jointPatchSlot[(size_t)joint];
    if (slot < 0)
    {
        slot = (int)m_jointPatches.size();
        m_jointPatches.emplace_back();
        m_jointPatches.back().slot = (uint32_t)(kGrabSlotCount + joint);
    }
    return m_jointPatches[(size_t)slot];
}


void VkWorld2D::queueSetPose(int body, Vec2 pos, float angle)
{
    if (body < 0 || body >= m_bodyCount)
        return;
    BodyPatch &p = bodyPatch(body);
    p.flags |= BP_SET_POSE;
    p.pose[0] = pos.x;
    p.pose[1] = pos.y;
    p.pose[2] = angle;
    m_hostPos[(size_t)body] = pos;
    m_hostAng[(size_t)body] = angle;
    m_query.invalidate();
}


void VkWorld2D::queueSetVelocity(int body, Vec2 vel, float angVel)
{
    if (body < 0 || body >= m_bodyCount)
        return;
    BodyPatch &p = bodyPatch(body);
    p.flags = (p.flags | BP_SET_VEL) & ~BP_ADD_VEL;
    p.vel[0] = vel.x;
    p.vel[1] = vel.y;
    p.vel[2] = angVel;
    p.dv[0] = p.dv[1] = p.dv[2] = 0.0f;
    m_hostVel[(size_t)body] = vel;
    m_hostAngVel[(size_t)body] = angVel;
}


void VkWorld2D::queueImpulse(int body, Vec2 impulse, const Vec2 *worldPoint)
{
    if (body < 0 || body >= m_bodyCount || m_scene.mass[(size_t)body] <= 0.0f)
        return;
    const float invM = 1.0f / m_scene.mass[(size_t)body];
    BodyPatch &p = bodyPatch(body);
    p.flags |= BP_ADD_VEL;
    p.dv[0] += impulse.x * invM;
    p.dv[1] += impulse.y * invM;
    if (worldPoint && m_scene.moment[(size_t)body] > 0.0f)
    {
        const Vec2 r = *worldPoint - m_hostPos[(size_t)body];
        p.dv[2] += (r.x * impulse.y - r.y * impulse.x) / m_scene.moment[(size_t)body];
    }
}


void VkWorld2D::queueAngularImpulse(int body, float impulse)
{
    if (body < 0 || body >= m_bodyCount || m_scene.mass[(size_t)body] <= 0.0f || m_scene.moment[(size_t)body] <= 0.0f)
        return;
    BodyPatch &p = bodyPatch(body);
    p.flags |= BP_ADD_VEL;
    p.dv[2] += impulse / m_scene.moment[(size_t)body];
}


void VkWorld2D::queueWake(int body)
{
    if (body >= 0 && body < m_bodyCount)
        bodyPatch(body).flags = (bodyPatch(body).flags | BP_WAKE) & ~BP_SLEEP;
}


void VkWorld2D::queueSleep(int body)
{
    if (body >= 0 && body < m_bodyCount)
        bodyPatch(body).flags = (bodyPatch(body).flags | BP_SLEEP) & ~BP_WAKE;
}


void VkWorld2D::queueDisable(int body)
{
    if (body >= 0 && body < m_bodyCount)
        bodyPatch(body).flags = (bodyPatch(body).flags | BP_DISABLE) & ~BP_ENABLE;
}


void VkWorld2D::queueEnable(int body)
{
    if (body >= 0 && body < m_bodyCount)
        bodyPatch(body).flags = (bodyPatch(body).flags | BP_ENABLE) & ~BP_DISABLE;
}


void VkWorld2D::queueJointMotor(int joint, int axis, uint32_t mode, float speed, float maxForce)
{
    if (joint < 0 || joint >= m_jointCount)
        return;
    JointPatch &p = jointPatch(joint);
    p.flags |= JP_MOTOR;
    p.motorAxis = (uint32_t)axis;
    p.motorMode = mode;
    p.speed = speed;
    p.maxForce = maxForce;
}


void VkWorld2D::queueJointLimit(int joint, int axis, bool on, float lower, float upper)
{
    if (joint < 0 || joint >= m_jointCount)
        return;
    JointPatch &p = jointPatch(joint);
    p.flags = (p.flags | JP_LIMIT) & ~JP_LIMIT_ON;
    if (on)
        p.flags |= JP_LIMIT_ON;
    p.limitAxis = (uint32_t)axis;
    p.lower = lower;
    p.upper = upper;
}


void VkWorld2D::queueJointLockScale(int joint, float scale)
{
    // Deliberately not gated on m_jointCount: flushJoint() queues this before the authored count has
    // been committed, and the patch is applied by the step once jointCount is final.
    if (joint < 0)
        return;
    JointPatch &p = jointPatch(joint);
    p.flags |= JP_LOCK_SCALE;
    p.lockScale = scale;
}


void VkWorld2D::queueJointWake(int joint)
{
    if (joint >= 0 && joint < m_jointCount)
        jointPatch(joint).flags |= JP_WAKE;
}


void VkWorld2D::jointForce(int joint, float out[3]) const
{
    out[0] = out[1] = out[2] = 0.0f;
    if (joint < 0 || joint >= m_jointCount || (size_t)(kGrabSlotCount + joint + 1) * 12 > (size_t)m_jForceHost.size())
        return;
    std::memcpy(out, m_jForceHost.mappedAs<uint8_t>() + (size_t)(kGrabSlotCount + joint) * 12, 12);
}


// Copies the queued records into their host-visible buffers (grown first, nothing is in flight)
// and empties the queue.
void VkWorld2D::uploadPatches()
{
    m_bPatchCount = (int)m_bodyPatches.size();
    m_jPatchCount = (int)m_jointPatches.size();
    if ((size_t)m_bPatchCount > m_bPatchCap)
    {
        m_bPatchCap = (size_t)m_bPatchCount + (size_t)m_bPatchCount / 2 + 64;
        m_bPatchBuf.destroy();
        m_bPatchBuf.init(m_dev, m_bPatchCap * sizeof(BodyPatch), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         MappedBuffer::HostAccess::Write);
    }
    if ((size_t)m_jPatchCount > m_jPatchCap)
    {
        m_jPatchCap = (size_t)m_jPatchCount + (size_t)m_jPatchCount / 2 + 64;
        m_jPatchBuf.destroy();
        m_jPatchBuf.init(m_dev, m_jPatchCap * sizeof(JointPatch), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         MappedBuffer::HostAccess::Write);
    }
    if (m_bPatchCount > 0)
        std::memcpy(m_bPatchBuf.mapped(), m_bodyPatches.data(), (size_t)m_bPatchCount * sizeof(BodyPatch));
    if (m_jPatchCount > 0)
        std::memcpy(m_jPatchBuf.mapped(), m_jointPatches.data(), (size_t)m_jPatchCount * sizeof(JointPatch));
    clearPatches();
}


void VkWorld2D::clearPatches()
{
    for (const BodyPatch &p : m_bodyPatches)
        if (p.body < m_bodyPatchSlot.size())
            m_bodyPatchSlot[p.body] = -1;
    for (const JointPatch &p : m_jointPatches)
        if (p.slot >= (uint32_t)kGrabSlotCount && p.slot - kGrabSlotCount < m_jointPatchSlot.size())
            m_jointPatchSlot[p.slot - kGrabSlotCount] = -1;
    m_bodyPatches.clear();
    m_jointPatches.clear();
}


int VkWorld2D::commitAuthored(int firstBody, int jointsBefore, int ignoreBefore)
{
    if (!m_init)
        init();
    if (m_bodyCount == 0)
    {
        Scene2D s = m_scene;
        s.params = m_params;
        build(s, &m_params);
        return firstBody;
    }
    return commitSpawn(firstBody, jointsBefore, ignoreBefore);
}


// ---------------------------------------------------------------------------
// Poses and the grab
// ---------------------------------------------------------------------------
void VkWorld2D::putBody(int body, const uint8_t awake)
{
    flushUploadsNow();
    const int zero = 0;
    putAt(m_awake, body, 1, &awake);
    if (awake)
        putAt(m_sleepTimer, body, 4, &zero);
}


void VkWorld2D::setBodyState(int body, Vec2 pos, float angle, Vec2 vel, float angVel)
{
    if (body < 0 || body >= m_bodyCount)
        return;
    flushUploadsNow();
    putAt(m_pos, body, 8, &pos);
    putAt(m_ang, body, 4, &angle);
    putAt(m_vel, body, 8, &vel);
    putAt(m_prevVel, body, 8, &vel);
    putAt(m_angVel, body, 4, &angVel);
    if (m_scene.mass[body] > 0.0f)
        putBody(body, 1);
    m_hostPos[body] = pos;
    m_hostAng[body] = angle;
    m_query.invalidate();
}


void VkWorld2D::clearBodyPatch(int body, uint32_t keepMask)
{
    if (body < 0 || (size_t)body >= m_bodyPatchSlot.size() || m_bodyPatchSlot[(size_t)body] < 0)
        return;
    BodyPatch &p = m_bodyPatches[(size_t)m_bodyPatchSlot[(size_t)body]];
    p.flags &= keepMask;
    if (keepMask == 0)
        p.dv[0] = p.dv[1] = p.dv[2] = 0.0f;
}


void VkWorld2D::clearJointPatch(int joint)
{
    if (joint < 0 || (size_t)joint >= m_jointPatchSlot.size() || m_jointPatchSlot[(size_t)joint] < 0)
        return;
    m_jointPatches[(size_t)m_jointPatchSlot[(size_t)joint]].flags = 0;
}


void VkWorld2D::setHostAwake(int body, bool awake)
{
    if (body >= 0 && (size_t)body < m_hostAwake.size())
        m_hostAwake[(size_t)body] = awake ? 1 : 0;
}

} // namespace avbd2d
