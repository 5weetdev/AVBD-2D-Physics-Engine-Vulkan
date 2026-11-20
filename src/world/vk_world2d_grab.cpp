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


int VkWorld2D::pick(Vec2 p) const
{
    int best = -1;
    float bestDist = 1e30f;
    for (int i = 0; i < m_bodyCount; i++)
    {
        if (m_scene.mass[i] <= 0.0f)
            continue;
        const Vec2 local = rotate(-m_hostAng[i], p - m_hostPos[i]);
        if (!m_scene.contains(i, local))
            continue;
        const float d = lengthSq(local);
        if (d < bestDist)
        {
            bestDist = d;
            best = i;
        }
    }
    return best;
}


float VkWorld2D::grabWeight(float dist, const GrabOptions &opt)
{
    if (opt.radius <= 0.0f)
        return 1.0f;
    const float t = std::min(std::max(dist / opt.radius, 0.0f), 1.0f);
    // Smoothstep: its zero slope at both ends keeps the grip from stepping as a body crosses
    // the radius.
    const float sm = 1.0f - t * t * (3.0f - 2.0f * t);
    const float w = std::pow(sm, std::max(opt.falloff, 0.0f));
    return std::min(std::max(opt.minStrength + (1.0f - opt.minStrength) * w, 0.0f), 1.0f);
}


bool VkWorld2D::isGrabbed(int body) const
{
    return std::binary_search(m_grabSorted.begin(), m_grabSorted.end(), body);
}


// Rewrites every grab slot from m_grabGroup / m_grabOffset (unused slots become inactive).
void VkWorld2D::uploadGrabSlots()
{
    const int K = kGrabSlotCount;
    const int first = 0;
    m_grabLocal.assign(m_grabGroup.size(), Vec2());
    std::vector<int> jB((size_t)K, 0);
    std::vector<Vec2> jRA((size_t)K), jRB((size_t)K);
    std::vector<float> kLin((size_t)K, 0.0f), kAng((size_t)K, 0.0f), rest((size_t)K, 0.0f), arm((size_t)K, 0.0f),
        zero((size_t)K * 9, 0.0f);
    std::vector<uint8_t> broken((size_t)K, 1);
    for (size_t g = 0; g < m_grabGroup.size(); g++)
    {
        const int body = m_grabGroup[g];
        const float w = grabWeight(length(m_hostPos[(size_t)body] - m_grabClick), m_grabOpt);
        jB[g] = body;
        // The clicked body is held at the click point, the rest at their centres.
        jRB[g] = body == m_grabBody ? rotate(-m_hostAng[(size_t)body], m_grabClick - m_hostPos[(size_t)body]) : Vec2();
        jRA[g] = m_grabClick + m_grabOffset[g];
        m_grabLocal[g] = jRB[g];
        kLin[g] = 5000.0f * m_scene.mass[(size_t)body] * m_params.grabStrength * w;
        kAng[g] = 5000.0f * m_scene.moment[(size_t)body] * m_params.grabStrength * m_grabOpt.angularStrength * w;
        rest[g] = m_grabStartAng[g];
        arm[g] = kAng[g] > 0.0f ? 1.0f : 0.0f;
        broken[g] = 0;
    }
    const VkDeviceSize f = (VkDeviceSize)first;
    m_jB.upload(jB.data(), (size_t)K * 4, f * 4);
    m_jRA.upload(jRA.data(), (size_t)K * 8, f * 8);
    m_jRB.upload(jRB.data(), (size_t)K * 8, f * 8);
    // The grab: x and y are ramped springs (no frequency) on the world frame; the angle is a
    // spring towards jRest (arm 1 enables the row) or free when angularStrength is 0.
    std::vector<JointAxis2> axes((size_t)K * 3);
    for (size_t g = 0; g < (size_t)K; g++)
    {
        for (size_t k = 0; k < 2; k++)
        {
            JointAxis2 &a = axes[g * 3 + k];
            a.pos = JOINT_AXIS_SPRING;
            a.stiff = kLin[g];
            a.flags = k == 0 ? JOINT_FLAG_GRAB : 0u;
        }
        if (kAng[g] > 0.0f)
        {
            JointAxis2 &a = axes[g * 3 + 2];
            a.pos = JOINT_AXIS_SPRING;
            a.stiff = kAng[g];
        }
    }
    m_jAxis.upload(axes.data(), (size_t)K * 3 * sizeof(JointAxis2), f * 3 * sizeof(JointAxis2));
    m_jFrameA.upload(zero.data(), (size_t)K * 4, f * 4);
    m_jRest.upload(rest.data(), (size_t)K * 4, f * 4);
    m_jArm.upload(arm.data(), (size_t)K * 4, f * 4);
    m_jC0.upload(zero.data(), (size_t)K * 36, f * 36);
    m_jPen.upload(zero.data(), (size_t)K * 36, f * 36);
    m_jLam.upload(zero.data(), (size_t)K * 36, f * 36);
    m_jForce.upload(zero.data(), (size_t)K * 12, f * 12);
    m_jBroken.upload(broken.data(), (size_t)K, f);
}


bool VkWorld2D::grabBegin(Vec2 p)
{
    return grabBegin(p, GrabOptions());
}


bool VkWorld2D::grabBegin(Vec2 p, const GrabOptions &opt)
{
    if (!m_init || m_bodyCount == 0)
        return false;
    grabEnd();
    const int body = pick(p);
    if (body < 0)
        return false;

    // The group: the clicked body, then (with a radius) either what its joints reach inside the
    // radius, or -- for a body with no joints, or with joint-following off -- every dynamic body
    // inside the radius. Nearest first, at most kGrabSlots.
    std::vector<int> group{body};
    if (opt.radius > 0.0f)
    {
        const float r2 = opt.radius * opt.radius;
        std::vector<char> seen((size_t)m_bodyCount, 0);
        seen[(size_t)body] = 1;
        bool traced = false;
        if (opt.followJoints && m_jointCount > 0)
        {
            std::vector<uint8_t> broken((size_t)m_jointCount, 0);
            m_jBroken.readback(broken.data(), (VkDeviceSize)m_jointCount, (VkDeviceSize)kGrabSlotCount);
            std::vector<std::vector<int>> adj;
            std::vector<int> ids((size_t)m_bodyCount, -1);
            auto idOf = [&](int b)
            {
                if (ids[(size_t)b] < 0)
                {
                    ids[(size_t)b] = (int)adj.size();
                    adj.emplace_back();
                }
                return ids[(size_t)b];
            };
            for (int j = 0; j < m_jointCount; j++)
            {
                const int a = m_scene.jointA[(size_t)j], b = m_scene.jointB[(size_t)j];
                if (broken[(size_t)j] || m_jointDead[(size_t)j] || a < 0 || b < 0 || m_scene.mass[(size_t)a] <= 0.0f ||
                    m_scene.mass[(size_t)b] <= 0.0f)
                    continue;
                const int ia = idOf(a), ib = idOf(b);
                adj[(size_t)ia].push_back(b);
                adj[(size_t)ib].push_back(a);
            }
            if (ids[(size_t)body] >= 0)
            {
                traced = true;
                for (size_t head = 0; head < group.size() && (int)group.size() < kGrabSlotCount; head++)
                {
                    const int cur = group[head];
                    if (ids[(size_t)cur] < 0)
                        continue;
                    for (int nb : adj[(size_t)ids[(size_t)cur]])
                    {
                        if (seen[(size_t)nb] || (int)group.size() >= kGrabSlotCount)
                            continue;
                        if (lengthSq(m_hostPos[(size_t)nb] - p) > r2)
                            continue;
                        seen[(size_t)nb] = 1;
                        group.push_back(nb);
                    }
                }
            }
        }
        if (!traced)
        {
            std::vector<std::pair<float, int>> inRadius;
            for (int i = 0; i < m_bodyCount; i++)
            {
                if (seen[(size_t)i] || m_scene.mass[(size_t)i] <= 0.0f)
                    continue;
                const float d2 = lengthSq(m_hostPos[(size_t)i] - p);
                if (d2 <= r2)
                    inRadius.emplace_back(d2, i);
            }
            const size_t room = (size_t)(kGrabSlotCount - 1);
            if (inRadius.size() > room)
            {
                std::nth_element(inRadius.begin(), inRadius.begin() + (std::ptrdiff_t)room, inRadius.end());
                inRadius.resize(room);
            }
            for (const auto &e : inRadius)
                group.push_back(e.second);
        }
    }

    m_grabOpt = opt;
    m_grabClick = p;
    m_grabCursor = p;
    m_grabBody = body;
    m_grabGroup = group;
    m_grabOffset.assign(group.size(), Vec2());
    m_grabStartAng.assign(group.size(), 0.0f);
    m_grabAngle = 0.0f;
    m_grabApplied = 0.0f;
    for (size_t g = 0; g < group.size(); g++)
    {
        m_grabStartAng[g] = m_hostAng[(size_t)group[g]];
        if (group[g] != body)
            m_grabOffset[g] = m_hostPos[(size_t)group[g]] - p;
    }
    m_grabSorted = group;
    std::sort(m_grabSorted.begin(), m_grabSorted.end());
    uploadGrabSlots();
    putBody(body, 1);
    return true;
}


void VkWorld2D::grabMove(Vec2 p)
{
    if (m_grabBody < 0)
        return;
    m_grabCursor = p;
}


void VkWorld2D::grabSetAngle(float angle)
{
    if (m_grabBody < 0)
        return;
    m_grabAngle = angle;
}


// Aims each grab spring for this step (recorded into the step's own submission, so moving the
// cursor costs no upload). The target is the cursor (plus the slot's offset), but no further
// than grabMaxSpeed * dt from where the slot's anchor is now. A spring aimed straight at a
// cursor that has run ahead stores the whole gap: the body is pulled at whatever speed closes
// it in a few steps, and a body that was blocked is flung when it slips free. Bounding the
// stretch bounds both the speed and the force the grab can apply.
void VkWorld2D::recordGrabTargets()
{
    if (m_grabBody < 0 || m_grabGroup.empty())
        return;
    const float maxStep = std::max(m_params.grabMaxSpeed, 0.0f) * m_params.dt;
    // The rotation eases towards its target (kGrabTurnRate rad/s) so a quarter turn is not a snap.
    const float maxTurn = kGrabTurnRate * m_params.dt;
    m_grabApplied += std::min(std::max(m_grabAngle - m_grabApplied, -maxTurn), maxTurn);
    Vec2 *t = (Vec2 *)m_grabStage.mapped();
    float *ang = (float *)((char *)m_grabStage.mapped() + (size_t)kGrabSlotCount * sizeof(Vec2));
    for (size_t g = 0; g < m_grabGroup.size(); g++)
    {
        const size_t b = (size_t)m_grabGroup[g];
        const Vec2 anchor = m_hostPos[b] + rotate(m_hostAng[b], m_grabLocal[g]);
        Vec2 d = m_grabCursor + rotate(m_grabApplied, m_grabOffset[g]) - anchor;
        const float len = length(d);
        if (len > maxStep)
            d = d * (maxStep / len);
        t[g] = anchor + d;
        ang[g] = m_grabStartAng[g] + m_grabApplied;
    }
    m_cmd.copy(m_grabStage, m_jRA, (VkDeviceSize)m_grabGroup.size() * sizeof(Vec2));
    if (m_grabOpt.angularStrength > 0.0f)
        m_cmd.copy(m_grabStage, m_jRest, (VkDeviceSize)m_grabGroup.size() * sizeof(float),
                   (VkDeviceSize)kGrabSlotCount * sizeof(Vec2));
}


void VkWorld2D::grabEnd()
{
    if (m_grabBody < 0)
        return;
    m_grabBody = -1;
    m_grabGroup.clear();
    m_grabOffset.clear();
    m_grabSorted.clear();
    uploadGrabSlots();
}

} // namespace avbd2d
