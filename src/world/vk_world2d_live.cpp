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
// Live slot edits
// ---------------------------------------------------------------------------
namespace
{
// Sorts and merges [off, off + n) ranges that touch or overlap.
std::vector<std::pair<int, int>> mergeRanges(const std::vector<std::pair<int, int>> &in)
{
    std::vector<std::pair<int, int>> v = in, out;
    std::sort(v.begin(), v.end());
    for (const auto &r : v)
    {
        if (r.second <= 0)
            continue;
        if (!out.empty() && r.first <= out.back().first + out.back().second)
            out.back().second = std::max(out.back().second, r.first + r.second - out.back().first);
        else
            out.push_back(r);
    }
    return out;
}

// Runs of consecutive ids: (first, count).
std::vector<std::pair<int, int>> idRuns(std::vector<int> ids)
{
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::vector<std::pair<int, int>> out;
    for (int id : ids)
    {
        if (!out.empty() && out.back().first + out.back().second == id)
            out.back().second++;
        else
            out.push_back({id, 1});
    }
    return out;
}
} // namespace


void VkWorld2D::liveReserve(const LiveNeeds &n)
{
    if (!m_liveBuilt)
        return; // build() sizes everything
    flushUploadsNow();
    const Scene2D &S = m_scene;
    const int nb = std::max(n.bodies, 0), ns = std::max(n.shapes, 0), nv = std::max(n.verts, 0),
              nj = std::max(n.joints, 0);
    const int needBodies = S.bodyCount() + std::max(0, nb - (int)m_freeBodies.size());
    const int needShapes = S.shapeTotal() + ns;
    const size_t needVerts = S.verts.size() + (size_t)nv;
    const int needSlots = m_jointSlots + std::max(0, nj - (int)m_freeJoints.size());
    const int ignoreNeed = (int)S.ignore.size() + 2 * nj + 64;

    int bodyCap = m_bodyCap, shapeCap = m_shapeCap, jointCap = m_jointCap;
    if (needBodies > bodyCap)
        bodyCap = needBodies + needBodies / 2 + 1024;
    if (needShapes > shapeCap)
        shapeCap = needShapes + needShapes / 2 + 1024;
    if (needSlots > jointCap)
        jointCap = needSlots + needSlots / 2 + 1024;
    const size_t vertCap = (size_t)m_verts.size() / 8;
    size_t newVertCap = vertCap;
    if (needVerts > vertCap)
        newVertCap = needVerts + needVerts / 2 + 64;
    const size_t stageNeed = 192 * (size_t)nb + 128 * (size_t)ns + 16 * (size_t)nv + 384 * (size_t)nj +
                             12 * (size_t)needShapes + 4 * (size_t)needBodies + 8 * (size_t)ignoreNeed + 65536;
    const bool growStage = (size_t)m_upStage.size() < stageNeed;
    const size_t stageSize = growStage ? stageNeed + stageNeed / 4 : (size_t)m_upStage.size();
    size_t tableBytes = 1024;
    while ((int)tableBytes < 2 * shapeCap)
        tableBytes <<= 1;
    tableBytes *= 4;
    const bool growIgnore = (size_t)m_ignore.size() < (size_t)ignoreNeed * 8;

    // Pre-flight against Device::setMemoryBudget before touching anything (a preserved growth holds
    // the old and the new buffer at once, hence the 2x).
    if (m_dev.memoryBudget() != 0)
    {
        size_t perBody = 60 + 64, perShape = 82 + 4 + 16 + 4, perJoint = 400 + 3 * sizeof(JointAxis2);
        for (const BodyBuf &b : bodyBuffers())
            perBody += b.elem;
        uint64_t delta = 0;
        delta += 2ull * (uint64_t)(bodyCap - m_bodyCap) * perBody;
        delta += 2ull * (uint64_t)(shapeCap - m_shapeCap) * perShape;
        delta += 2ull * (uint64_t)(jointCap - m_jointCap) * perJoint;
        delta += 2ull * (uint64_t)(newVertCap - vertCap) * 16;
        if (growStage)
            delta += (uint64_t)stageSize;
        if (growIgnore)
            delta += (uint64_t)ignoreNeed * 8;
        {
            // The constraint graph is reallocated whole when its sizes change.
            const int rb = std::max(bodyCap, 1), rf = std::max(jointCap, m_jointSlots);
            if (m_csRes[0] != rb || m_csRes[1] != m_pairCap || m_csRes[2] != rf)
                delta += 2ull * ((uint64_t)(m_pairCap + rf) * 2 * 24 + (uint64_t)rb * 32);
        }
        if (delta > 0)
            m_dev.checkBudget(delta);
    }

    bool listsLost = false;
    if (bodyCap != m_bodyCap)
    {
        sizeBodyBuffers(bodyCap, true);
        m_bodyCap = bodyCap;
        listsLost = true;
    }
    if (shapeCap != m_shapeCap)
    {
        sizeShapeBuffers(shapeCap, true);
        m_shapeCap = shapeCap;
        listsLost = true;
    }
    if (newVertCap != vertCap)
    {
        sizeBuf(m_verts, newVertCap * 8, true);
        sizeBuf(m_norms, newVertCap * 8, true);
    }
    if (jointCap != m_jointCap)
    {
        sizeJointBuffers(jointCap, true);
        m_jointCap = jointCap;
    }
    {
        const int rb = std::max(m_bodyCap, 1), rf = std::max(m_jointCap, m_jointSlots);
        if (m_csRes[0] != rb || m_csRes[1] != m_pairCap || m_csRes[2] != rf)
        {
            m_constraints.reserve(rb, m_pairCap, rf);
            m_csRes[0] = rb;
            m_csRes[1] = m_pairCap;
            m_csRes[2] = rf;
        }
    }
    if ((size_t)m_statIdx.size() < (size_t)shapeCap * 4 || (size_t)m_gridList.size() < (size_t)shapeCap * 4 ||
        (size_t)m_overList.size() < (size_t)shapeCap * 4 || (size_t)m_gridHead.size() < tableBytes)
        listsLost = true;
    fitBuf(m_dynIdx, (size_t)m_bodyCap * 4);
    fitBuf(m_statIdx, (size_t)shapeCap * 4);
    fitBuf(m_gridList, (size_t)shapeCap * 4);
    fitBuf(m_overList, (size_t)shapeCap * 4);
    fitBuf(m_gridHead, tableBytes);
    if (growIgnore)
    {
        sizeBuf(m_ignore, (size_t)ignoreNeed * 8, false);
        if (!S.ignore.empty())
            m_ignore.upload(S.ignore.data(), S.ignore.size() * 8);
    }
    if (growStage)
    {
        m_upStage.destroy();
        m_upStage.init(m_dev, stageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MappedBuffer::HostAccess::Write);
    }
    if ((size_t)m_posHost.size() < (size_t)m_bodyCap * 8)
        sizeHostMirrors(m_bodyCap);
    if (listsLost)
        uploadLists(); // the broadphase lists were reallocated without their contents
}


int VkWorld2D::liveAddBody(const Scene2D &tmp, int ti)
{
    Scene2D &S = m_scene;
    int slot = -1;
    if (!m_freeBodies.empty())
    {
        slot = m_freeBodies.back();
        m_freeBodies.pop_back();
    }
    const int ns = tmp.shapeCount[(size_t)ti], nv = tmp.vertTotal(ti);
    const int so = ns > 0 ? m_shapeAlloc.alloc(ns, S.shapeTotal()) : 0;
    const int vo = nv > 0 ? m_vertAlloc.alloc(nv, (int)S.verts.size()) : 0;
    const int id = S.installBody(tmp, ti, slot, so, vo);
    if (ns > 0)
        m_dirtyShapeR.push_back({so, ns});
    if (nv > 0)
        m_dirtyVertR.push_back({vo, nv});
    m_dirtyBodiesNew.push_back(id);
    m_listsDirty = true;
    if (m_liveBuilt)
    {
        if (id >= m_bodyCount)
            m_bodyCount = id + 1;
        if ((int)m_hostPos.size() < m_bodyCount)
        {
            m_hostPos.resize((size_t)m_bodyCount);
            m_hostAng.resize((size_t)m_bodyCount);
            m_hostVel.resize((size_t)m_bodyCount);
            m_hostAngVel.resize((size_t)m_bodyCount);
            m_hostAwake.resize((size_t)m_bodyCount, 1);
            m_hostColor.resize((size_t)m_bodyCount, 0);
        }
        m_hostPos[(size_t)id] = S.pos[(size_t)id];
        m_hostAng[(size_t)id] = S.ang[(size_t)id];
        m_hostVel[(size_t)id] = S.vel[(size_t)id];
        m_hostAngVel[(size_t)id] = S.angVel[(size_t)id];
        m_hostAwake[(size_t)id] = 1;
        m_hostColor[(size_t)id] = 0;
        clearBodyPatch(id, 0);
        m_query.invalidate();
    }
    else
        m_liveDirty = true;
    return id;
}


void VkWorld2D::liveReplaceBody(int body, const Scene2D &tmp, int ti)
{
    Scene2D &S = m_scene;
    const size_t b = (size_t)body;
    if (isGrabbed(body))
        grabEnd();
    const Vec2 lcOld = S.localCenter[b];
    // The old shapes are retired (their rows stay readable until the next step commits).
    const int of = S.shapeFirst[b], on = S.shapeCount[b];
    int ovOff = 0, ovN = 0;
    for (int s = of; s < of + on; s++)
        if (S.shVCnt[(size_t)s] > 0)
        {
            if (ovN == 0)
                ovOff = S.shVOff[(size_t)s];
            ovN += S.shVCnt[(size_t)s];
        }
    if (on > 0)
        m_retired.push_back({false, of, on});
    if (ovN > 0)
        m_retired.push_back({true, ovOff, ovN});
    S.releaseShapes(of, on);
    if (on > 0)
        m_dirtyShapeR.push_back({of, on});

    const int ns = tmp.shapeCount[(size_t)ti], nv = tmp.vertTotal(ti);
    const int so = ns > 0 ? m_shapeAlloc.alloc(ns, S.shapeTotal()) : 0;
    const int vo = nv > 0 ? m_vertAlloc.alloc(nv, (int)S.verts.size()) : 0;
    S.installBody(tmp, ti, body, so, vo);
    if (ns > 0)
        m_dirtyShapeR.push_back({so, ns});
    if (nv > 0)
        m_dirtyVertR.push_back({vo, nv});
    m_dirtyBodiesNew.push_back(body);
    m_listsDirty = true;

    // Joint anchors are in the centre-of-mass frame: keep them on the same material point.
    const Vec2 shift = S.localCenter[b] - lcOld;
    for (int j = 0; j < S.jointCount(); j++)
    {
        const size_t q = (size_t)j;
        if (q < m_jointDead.size() && m_jointDead[q])
            continue;
        bool hit = false;
        if (S.jointA[q] == body)
        {
            S.jointRA[q] = S.jointRA[q] - shift;
            hit = true;
        }
        if (S.jointB[q] == body)
        {
            S.jointRB[q] = S.jointRB[q] - shift;
            hit = true;
        }
        if (!hit)
            continue;
        const Vec2 sizeA = S.jointA[q] >= 0 ? S.half[(size_t)S.jointA[q]] * 2.0f : Vec2();
        const Vec2 sizeB = S.half[(size_t)S.jointB[q]] * 2.0f;
        S.jointArm[q] = lengthSq(sizeA + sizeB);
        m_dirtyJoints.push_back(j);
    }

    if (m_liveBuilt)
    {
        m_hostPos[b] = S.pos[b];
        m_hostAng[b] = S.ang[b];
        m_hostVel[b] = S.vel[b];
        m_hostAngVel[b] = S.angVel[b];
        m_hostAwake[b] = 1;
        clearBodyPatch(body, ~BP_SET_POSE);
        m_query.invalidate();
    }
    else
        m_liveDirty = true;
}


void VkWorld2D::liveRemoveBody(int body)
{
    Scene2D &S = m_scene;
    if (body < 0 || body >= S.bodyCount())
        return;
    const size_t b = (size_t)body;
    if (m_liveBuilt && isGrabbed(body))
        grabEnd();
    const int f = S.shapeFirst[b], n = S.shapeCount[b];
    int vOff = 0, vN = 0;
    for (int s = f; s < f + n; s++)
        if (S.shVCnt[(size_t)s] > 0)
        {
            if (vN == 0)
                vOff = S.shVOff[(size_t)s];
            vN += S.shVCnt[(size_t)s];
        }
    if (n > 0)
        m_retired.push_back({false, f, n});
    if (vN > 0)
        m_retired.push_back({true, vOff, vN});
    S.releaseShapes(f, n);
    if (n > 0)
        m_dirtyShapeR.push_back({f, n});
    S.shapeFirst[b] = 0;
    S.shapeCount[b] = 0;
    S.mass[b] = 0.0f;
    S.moment[b] = 0.0f;
    S.vel[b] = Vec2();
    S.angVel[b] = 0.0f;
    S.bodyType[b] = BODY2D_STATIC;
    S.localCenter[b] = Vec2();
    S.isBullet[b] = 0;
    if (S.eraseIgnore(body))
        m_dirtyIgnore = true;
    m_freeBodies.push_back(body);
    m_dirtyBodiesNew.push_back(body);
    m_listsDirty = true;
    if (m_liveBuilt)
    {
        if ((size_t)body < m_hostVel.size())
        {
            m_hostVel[b] = Vec2();
            m_hostAngVel[b] = 0.0f;
        }
        clearBodyPatch(body, 0);
        m_query.invalidate();
    }
    else
        m_liveDirty = true;
}


int VkWorld2D::liveAddJoint(const Scene2D::JointDef2D &d)
{
    Scene2D &S = m_scene;
    const size_t ignoreBefore = S.ignore.size();
    int idx = S.addJoint(d);
    if (S.ignore.size() != ignoreBefore)
        m_dirtyIgnore = true;
    if (!m_liveBuilt)
        m_liveDirty = true;
    if (idx < 0)
        return -1;
    if (!m_freeJoints.empty())
    {
        const int slot = m_freeJoints.back();
        m_freeJoints.pop_back();
        S.moveJoint(idx, slot);
        S.popJoint();
        idx = slot;
        m_jointDead.resize((size_t)S.jointCount(), 0);
        m_jointDead[(size_t)slot] = 0;
    }
    else
    {
        m_jointDead.resize((size_t)S.jointCount(), 0);
        if (m_liveBuilt)
        {
            m_jointCount = S.jointCount();
            m_jointSlots = kGrabSlotCount + m_jointCount;
        }
    }
    m_dirtyJoints.push_back(idx);
    if (m_liveBuilt)
        clearJointPatch(idx);
    return idx;
}


void VkWorld2D::liveRemoveJoint(int joint)
{
    if (joint < 0 || joint >= m_scene.jointCount() || m_jointDead[(size_t)joint])
        return;
    m_jointDead[(size_t)joint] = 1;
    m_freeJoints.push_back(joint);
    m_dirtyJoints.push_back(joint);
    if (m_liveBuilt)
        clearJointPatch(joint);
    else
        m_liveDirty = true;
}


// Queues everything the live edits changed (or, for a world that was never built, builds it).
void VkWorld2D::liveFinish()
{
    Scene2D &S = m_scene;
    if (!m_liveBuilt)
    {
        if (m_liveDirty && S.bodyCount() > 0)
        {
            Scene2D copy = S;
            m_keepLive = true;
            try
            {
                build(copy, &m_params);
            }
            catch (...)
            {
                m_keepLive = false;
                throw; // m_liveDirty stays set: the next flush retries
            }
            m_keepLive = false;
        }
        m_dirtyJoints.clear();
        m_dirtyBodiesNew.clear();
        m_dirtyShapeR.clear();
        m_dirtyVertR.clear();
        m_listsDirty = m_dirtyIgnore = false;
        m_liveDirty = S.bodyCount() > 0 && !m_liveBuilt;
        return;
    }
    if (m_dirtyJoints.empty() && m_dirtyBodiesNew.empty() && m_dirtyShapeR.empty() && m_dirtyVertR.empty() &&
        !m_listsDirty && !m_dirtyIgnore)
        return;

    struct Defer
    {
        VkWorld2D &w;
        explicit Defer(VkWorld2D &x) : w(x) { w.m_deferUp = true; }
        ~Defer() { w.m_deferUp = false; }
    } defer(*this);

    if (m_dirtyIgnore)
    {
        S.finalize();
        m_ignoreCount = (int)S.ignore.size();
        fitBuf(m_ignore, (S.ignore.size() + S.ignore.size() / 2 + 64) * 8);
        up(m_ignore, S.ignore.data(), S.ignore.size() * 8);
    }
    {
        std::vector<std::pair<int, int>> in;
        for (const Range &r : m_dirtyShapeR)
            in.push_back({r.off, r.n});
        for (const auto &r : mergeRanges(in))
            uploadShapes(r.first, r.first + r.second);
    }
    {
        std::vector<std::pair<int, int>> in;
        for (const Range &r : m_dirtyVertR)
            in.push_back({r.off, r.n});
        for (const auto &r : mergeRanges(in))
            uploadVerts(r.first, r.second);
    }
    for (const auto &r : idRuns(m_dirtyBodiesNew))
        uploadBodies(r.first, r.second);
    for (const auto &r : idRuns(m_dirtyJoints))
        uploadAuthoredJoints(r.first, r.first + r.second);
    if (m_listsDirty || m_dirtyIgnore)
        uploadLists();
    m_dirtyJoints.clear();
    m_dirtyBodiesNew.clear();
    m_dirtyShapeR.clear();
    m_dirtyVertR.clear();
    m_listsDirty = m_dirtyIgnore = false;
}

} // namespace avbd2d
