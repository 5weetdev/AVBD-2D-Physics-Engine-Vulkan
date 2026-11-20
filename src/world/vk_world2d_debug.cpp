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
// Stats and debug read-backs
// ---------------------------------------------------------------------------
WorldStats2D VkWorld2D::stats() const
{
    WorldStats2D s;
    s.bodies = m_bodyCount - (int)m_freeBodies.size();
    s.statics = m_statCount;
    s.dynamic = std::max(m_dynCount - m_removedTotal, 0);
    s.awake = m_awakeNow < 0 ? s.dynamic : std::min(m_awakeNow, s.dynamic);
    s.asleep = s.dynamic - s.awake;
    s.removed = m_removedTotal;
    s.joints = m_jointCount - (int)m_freeJoints.size();
    s.brokenJoints = m_brokenTotal;
    s.pairCap = m_pairCap;
    s.contactCap = m_contactCap;
    s.bodyCap = m_bodyCap;

    uint64_t bytes = 0;
    // buffers() is non-const only because Buffer::init is; sizes are read-only here.
    for (const Buffer *b : const_cast<VkWorld2D *>(this)->buffers())
        bytes += (uint64_t)b->size();
    bytes += m_man[0].bytes() + m_man[1].bytes();
    bytes += m_constraints.bytesAllocated() + m_radix.bytesAllocated();
    for (const MappedBuffer *b : {&m_posHost, &m_angHost, &m_velHost, &m_awakeHost, &m_colorHost, &m_upStage})
        bytes += (uint64_t)b->size();
    s.deviceBytes = bytes;
    return s;
}


void VkWorld2D::downloadLiveFlags(std::vector<uint8_t> &bodyRemoved, std::vector<uint8_t> &jointBroken) const
{
    bodyRemoved.assign((size_t)std::max(m_bodyCount, 0), 0);
    jointBroken.assign((size_t)std::max(m_jointCount, 0), 0);
    if (!m_init || m_stepActive)
        return;
    if (m_bodyCount > 0)
    {
        std::vector<int> timer((size_t)m_bodyCount);
        m_sleepTimer.readback(timer.data(), (VkDeviceSize)m_bodyCount * 4);
        for (size_t i = 0; i < timer.size(); i++)
            bodyRemoved[i] = timer[i] < 0 ? 1 : 0;
    }
    if (m_jointCount > 0)
        m_jBroken.readback(jointBroken.data(), (VkDeviceSize)m_jointCount, (VkDeviceSize)kGrabSlotCount);
    for (int j = 0; j < m_jointCount; j++)
        if (m_jointDead[(size_t)j])
            jointBroken[(size_t)j] = 1;
}


void VkWorld2D::downloadContacts(std::vector<ContactDebug2D> &out) const
{
    out.clear();
    const int pairs = m_prevPairs;
    const int contacts = std::min(m_result.contacts, m_contactCap);
    if (!m_init || pairs <= 0 || contacts <= 0)
        return;
    if (m_contactDebug && m_manDebugHost.size() >= dbgBytes(m_pairCap, m_contactCap))
    {
        // The fast path: the step copied the manifold arrays into this mapped buffer inside
        // its own submission, so the data is already on the host -- no staging, no submit,
        // no wait. The layout is the kDbg* one from the recorder.
        const uint8_t *base = m_manDebugHost.mappedAs<uint8_t>();
        const VkDeviceSize P = (VkDeviceSize)m_pairCap, C = (VkDeviceSize)m_contactCap;
        const VkDeviceSize cb = dbgContactBase(m_pairCap);
        const uint64_t *key = reinterpret_cast<const uint64_t *>(base);
        const int32_t *bodyA = reinterpret_cast<const int32_t *>(base + P * 8);
        const int32_t *bodyB = reinterpret_cast<const int32_t *>(base + P * 12);
        const int32_t *cnt = reinterpret_cast<const int32_t *>(base + P * 16);
        const int32_t *off = reinterpret_cast<const int32_t *>(base + P * 20);
        const Vec2 *nrm = reinterpret_cast<const Vec2 *>(base + P * 24);
        const float *fric = reinterpret_cast<const float *>(base + P * 32);
        const Vec2 *rA = reinterpret_cast<const Vec2 *>(base + cb);
        const Vec2 *rB = reinterpret_cast<const Vec2 *>(base + cb + C * 8);
        const Vec2 *lam = reinterpret_cast<const Vec2 *>(base + cb + C * 16);
        const Vec2 *pen = reinterpret_cast<const Vec2 *>(base + cb + C * 24);
        const uint8_t *stk = base + cb + C * 32;
        out.reserve((size_t)contacts);
        for (int t = 0; t < pairs; t++)
        {
            const int a = bodyA[t], b = bodyB[t];
            if (cnt[t] <= 0 || a < 0 || b < 0 || a >= m_bodyCount || b >= m_bodyCount)
                continue;
            for (int c = 0; c < cnt[t]; c++)
            {
                const int o = off[t] + c;
                if (o < 0 || o >= contacts)
                    continue;
                Vec2 xA = m_hostPos[a] + rotate(m_hostAng[a], rA[o]);
                Vec2 xB = m_hostPos[b] + rotate(m_hostAng[b], rB[o]);
                // A rounded shape's anchor is its core point; its surface is the radius further on.
                // (the manifold key is the pair of shapes)
                const int sA = (int)(key[t] >> 32), sB = (int)(key[t] & 0xFFFFFFFFu);
                if (sA < m_scene.shapeTotal() && m_scene.coreAnchored(sA))
                    xA = xA - nrm[t] * m_scene.shRad[(size_t)sA];
                if (sB < m_scene.shapeTotal() && m_scene.coreAnchored(sB))
                    xB = xB + nrm[t] * m_scene.shRad[(size_t)sB];
                ContactDebug2D d;
                d.p = (xA + xB) * 0.5f;
                d.pA = xA;
                d.pB = xB;
                d.n = nrm[t];
                d.force = std::fabs(lam[o].x);
                d.tangent = std::fabs(lam[o].y);
                d.sep = dot(nrm[t], xA - xB);
                d.mu = fric[t];
                d.bodyA = a;
                d.bodyB = b;
                d.stick = stk[o];
                d.penalty = pen[o];
                out.push_back(d);
            }
        }
        return;
    }
    downloadContactsSync(out);
}


void VkWorld2D::downloadContactsSync(std::vector<ContactDebug2D> &out) const
{
    out.clear();
    const int pairs = m_prevPairs;
    const int contacts = std::min(m_result.contacts, m_contactCap);
    if (!m_init || pairs <= 0 || contacts <= 0)
        return;
    const ManBufs &m = m_man[1 - m_curIdx]; // the last committed step
    std::vector<int> bodyA((size_t)pairs), bodyB((size_t)pairs), cnt((size_t)pairs), off((size_t)pairs);
    std::vector<uint64_t> key((size_t)pairs);
    std::vector<Vec2> nrm((size_t)pairs), rA((size_t)contacts), rB((size_t)contacts), lam((size_t)contacts),
        pen((size_t)contacts);
    std::vector<float> fric((size_t)pairs);
    std::vector<uint8_t> stk((size_t)contacts);
    m.key.readback(key.data(), (VkDeviceSize)pairs * 8);
    m.bodyA.readback(bodyA.data(), (VkDeviceSize)pairs * 4);
    m.bodyB.readback(bodyB.data(), (VkDeviceSize)pairs * 4);
    m.count.readback(cnt.data(), (VkDeviceSize)pairs * 4);
    m.offset.readback(off.data(), (VkDeviceSize)pairs * 4);
    m.normal.readback(nrm.data(), (VkDeviceSize)pairs * 8);
    m.friction.readback(fric.data(), (VkDeviceSize)pairs * 4);
    m.rA.readback(rA.data(), (VkDeviceSize)contacts * 8);
    m.rB.readback(rB.data(), (VkDeviceSize)contacts * 8);
    m.lambda.readback(lam.data(), (VkDeviceSize)contacts * 8);
    m.penalty.readback(pen.data(), (VkDeviceSize)contacts * 8);
    m.stick.readback(stk.data(), (VkDeviceSize)contacts);
    out.reserve((size_t)contacts);
    for (int t = 0; t < pairs; t++)
    {
        const int a = bodyA[t], b = bodyB[t];
        if (cnt[t] <= 0 || a < 0 || b < 0 || a >= m_bodyCount || b >= m_bodyCount)
            continue;
        for (int c = 0; c < cnt[t]; c++)
        {
            const int o = off[t] + c;
            if (o < 0 || o >= contacts)
                continue;
            Vec2 xA = m_hostPos[a] + rotate(m_hostAng[a], rA[o]);
            Vec2 xB = m_hostPos[b] + rotate(m_hostAng[b], rB[o]);
            // A rounded shape's anchor is its core point; its surface is the radius further on.
            // (the manifold key is the pair of shapes)
            const int sA = (int)(key[t] >> 32), sB = (int)(key[t] & 0xFFFFFFFFu);
            if (sA < m_scene.shapeTotal() && m_scene.coreAnchored(sA))
                xA = xA - nrm[t] * m_scene.shRad[(size_t)sA];
            if (sB < m_scene.shapeTotal() && m_scene.coreAnchored(sB))
                xB = xB + nrm[t] * m_scene.shRad[(size_t)sB];
            ContactDebug2D d;
            d.p = (xA + xB) * 0.5f;
            d.pA = xA;
            d.pB = xB;
            d.n = nrm[t];
            d.force = std::fabs(lam[o].x);
            d.tangent = std::fabs(lam[o].y);
            d.sep = dot(nrm[t], xA - xB);
            d.mu = fric[t];
            d.bodyA = a;
            d.bodyB = b;
            d.stick = stk[o];
            d.penalty = pen[o];
            out.push_back(d);
        }
    }
}


// ---------------------------------------------------------------------------
// Events, explosions and queries
// ---------------------------------------------------------------------------
void VkWorld2D::sizeEvents()
{
    static const uint32_t stride[EVENT2D_KINDS] = {2, 2, 2, 2, 8, 1};
    uint32_t off = 8; // ints: the six counters, padded
    for (int k = 0; k < EVENT2D_KINDS; k++)
    {
        m_evOff[k] = off;
        off += m_evCap[k] * stride[k];
    }
    m_evBytes = (size_t)off * 4;
    sizeBuf(m_events, m_evBytes, false);
    m_eventsHost.destroy();
    m_eventsHost.init(m_dev, m_evBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
}


void VkWorld2D::collectEvents()
{
    static const uint32_t stride[EVENT2D_KINDS] = {2, 2, 2, 2, 8, 1};
    Events2D &e = m_events2d;
    e.contactBegin.clear();
    e.contactEnd.clear();
    e.sensorBegin.clear();
    e.sensorEnd.clear();
    e.hit.clear();
    e.jointBreak.clear();
    for (int k = 0; k < EVENT2D_KINDS; k++)
    {
        e.truncated[k] = false;
        e.required[k] = 0;
    }
    if (!m_eventsOn || m_bodyCount == 0 || m_eventsHost.size() < m_evBytes)
        return;
    const int32_t *ev = m_eventsHost.mappedAs<int32_t>();
    bool grow = false;
    for (int k = 0; k < EVENT2D_KINDS; k++)
    {
        const int req = ev[k];
        e.required[k] = req;
        const int n = std::min<int>(req, (int)m_evCap[k]);
        const int32_t *p = ev + m_evOff[k];
        for (int i = 0; i < n; i++, p += stride[k])
        {
            switch (k)
            {
            case EVENT2D_CONTACT_BEGIN:
                e.contactBegin.push_back({p[0], p[1]});
                break;
            case EVENT2D_CONTACT_END:
                e.contactEnd.push_back({p[0], p[1]});
                break;
            case EVENT2D_SENSOR_BEGIN:
                e.sensorBegin.push_back({p[0], p[1]});
                break;
            case EVENT2D_SENSOR_END:
                e.sensorEnd.push_back({p[0], p[1]});
                break;
            case EVENT2D_HIT:
            {
                HitEvent2D h;
                float f[5];
                memcpy(f, p + 2, sizeof(f));
                h.shapeA = p[0];
                h.shapeB = p[1];
                h.point = Vec2(f[0], f[1]);
                h.normal = Vec2(f[2], f[3]);
                h.approachSpeed = f[4];
                e.hit.push_back(h);
                break;
            }
            default:
                e.jointBreak.push_back({p[0]});
                break;
            }
        }
        if (req > (int)m_evCap[k])
        {
            e.truncated[k] = true;
            const uint32_t want = (uint32_t)cappedGrow(req, 64, 1 << 22);
            // Growth honours Device::setMemoryBudget (device buffer + host copy); a refused one
            // leaves the kind truncated.
            const uint64_t delta = 2ull * 4ull * stride[k] * (want - m_evCap[k]);
            if (m_dev.memoryBudget() != 0 && m_dev.allocatedBytes() + delta > m_dev.memoryBudget())
                continue;
            m_evCap[k] = want;
            grow = true;
        }
    }
    if (grow)
        sizeEvents();
}


void VkWorld2D::explode(Vec2 pos, float radius, float falloff, float impulsePerLength)
{
    if (!m_init || m_bodyCount == 0)
        return;
    std::vector<ExplosionImpulse2D> out;
    m_query.explosion(queryView(), pos, radius, falloff, impulsePerLength, out);
    m_pendingImpulses.insert(m_pendingImpulses.end(), out.begin(), out.end());
    m_impulseCount = (int)m_pendingImpulses.size();
    if (m_impulseCount == 0)
        return;
    if (m_impulseCount > m_impulseCap)
    {
        m_impulseCap = cappedGrow(m_impulseCount, 64, 1 << 22);
        sizeBuf(m_impulses, (size_t)m_impulseCap * 16, false);
    }
    std::vector<float> raw((size_t)m_impulseCount * 4);
    for (int i = 0; i < m_impulseCount; i++)
    {
        const ExplosionImpulse2D &im = m_pendingImpulses[(size_t)i];
        memcpy(&raw[(size_t)i * 4], &im.body, 4);
        raw[(size_t)i * 4 + 1] = im.dv.x;
        raw[(size_t)i * 4 + 2] = im.dv.y;
        raw[(size_t)i * 4 + 3] = im.dw;
    }
    m_impulses.upload(raw.data(), raw.size() * 4);
}

} // namespace avbd2d
