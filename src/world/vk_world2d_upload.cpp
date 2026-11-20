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


// Per-body type, gravity scale, damping and motion locks, packed as the shaders read them.
void VkWorld2D::uploadBodyProps(size_t first, size_t count)
{
    const Scene2D &S = m_scene;
    std::vector<float> p(4 * count);
    for (size_t i = 0; i < count; i++)
    {
        const size_t b = first + i;
        const uint32_t flags = (S.bodyType[b] == BODY2D_KINEMATIC ? 1u : 0u) | (S.lockX[b] ? 2u : 0u) |
                               (S.lockY[b] ? 4u : 0u) | (S.lockRot[b] ? 8u : 0u);
        p[4 * i] = S.gravityScale[b];
        p[4 * i + 1] = S.linearDamping[b];
        p[4 * i + 2] = S.angularDamping[b];
        std::memcpy(&p[4 * i + 3], &flags, 4);
    }
    up(m_bProps, p.data(), count * 16, first * 16);
}


// The dynamic body list, the broadphase shape lists, the grid cell size and the table size,
// from the host scene. The grid holds the dynamic shapes, then the statics no bigger than the
// largest of them; the cell is sized to that largest dynamic shape. A few dynamic shapes far
// bigger than the rest (a long plank in a pile of pebbles) would blow the cell up for all, so
// up to kMaxOversize of them are kept out of the grid and tested against everything by brute
// force (csOversizePairs2D). Static shapes bigger than the cell (the ground) stay in `stat`,
// which every grid shape tests by brute force. The dyn list is every dynamic body in index
// order, as the constraint build expects; the other lists are of shapes.
void VkWorld2D::uploadLists()
{
    constexpr int kMaxOversize = 64;
    constexpr float kOversizeRatio = 4.0f; // radius over the median dynamic radius
    const Scene2D &S = m_scene;
    // How far a shape reaches from its body's centre of mass (the half extents' diagonal, plus
    // the offset of the shape; the offset is zero for a one-shape body).
    auto reach = [&](int s) { return length(S.shHalf[(size_t)s]) + length(S.shOff[(size_t)s]); };
    // A dead shape (its body was destroyed, shBody < 0) is in no list.
    auto liveShape = [&](int s) { return S.shBody[(size_t)s] >= 0; };
    auto dynamicShape = [&](int s) { return S.mass[(size_t)S.shBody[(size_t)s]] > 0.0f; };
    const int shapes = S.shapeTotal();
    std::vector<int> dyn, stat, grid, over, dynShapes;
    for (int i = 0; i < m_bodyCount; i++)
        if (S.mass[i] > 0.0f)
            dyn.push_back(i);
    for (int s = 0; s < shapes; s++)
        if (liveShape(s) && dynamicShape(s))
            dynShapes.push_back(s);
    m_dynCount = (int)dyn.size();

    float limit = 1e30f;
    if (!dynShapes.empty())
    {
        std::vector<float> r;
        r.reserve(dynShapes.size());
        for (int s : dynShapes)
            r.push_back(reach(s));
        std::nth_element(r.begin(), r.begin() + (std::ptrdiff_t)(r.size() / 2), r.end());
        limit = std::max(kOversizeRatio * r[r.size() / 2], 1.0f);
        int big = 0;
        for (int s : dynShapes)
            big += reach(s) > limit ? 1 : 0;
        if (big > kMaxOversize)
            limit = 1e30f; // too many to brute-force: grid them all
    }
    float maxRadius = 0.1f;
    for (int s : dynShapes)
    {
        if (reach(s) > limit)
            over.push_back(s);
        else
        {
            grid.push_back(s);
            maxRadius = std::max(maxRadius, reach(s));
        }
    }
    m_gridDyn = (int)grid.size();
    for (int s = 0; s < shapes; s++)
        if (liveShape(s) && !dynamicShape(s))
            (reach(s) <= maxRadius ? grid : stat).push_back(s);
    m_statCount = (int)stat.size();
    m_gridCount = (int)grid.size();
    m_overCount = (int)over.size();
    for (std::vector<int> *v : {&dyn, &stat, &grid, &over})
        if (v->empty())
            v->push_back(0); // never read (count 0), but the upload needs a byte
    // Grow-only: a live edit queues these uploads for the step's own command buffer, so a list
    // buffer must already be large enough (liveReserve sizes them to the caps).
    fitBuf(m_dynIdx, dyn.size() * 4);
    fitBuf(m_statIdx, stat.size() * 4);
    fitBuf(m_gridList, grid.size() * 4);
    fitBuf(m_overList, over.size() * 4);
    up(m_dynIdx, dyn.data(), dyn.size() * 4);
    up(m_statIdx, stat.data(), stat.size() * 4);
    up(m_gridList, grid.data(), grid.size() * 4);
    up(m_overList, over.data(), over.size() * 4);

    m_cellSize = std::max(2.0f * maxRadius + 0.1f, 0.5f);
    m_tableSize = 1024;
    while (m_tableSize < 2 * m_gridCount)
        m_tableSize <<= 1;
    fitBuf(m_gridHead, (size_t)m_tableSize * 4);
}


void VkWorld2D::fitBuf(Buffer &b, size_t bytes)
{
    if (b.size() < std::max<size_t>(bytes, 16))
        sizeBuf(b, bytes, false);
}


// Per-body continuous-collision data for bodies [from, bodyCount): the shape range, the smallest
// and largest extent (what "fast" and the swept bound are measured against) and the bullet flag.
void VkWorld2D::uploadCcdBodies(int from, int to)
{
    const Scene2D &S = m_scene;
    if (to < 0)
        to = S.bodyCount();
    const size_t n = (size_t)std::max(to - from, 0);
    if (n == 0)
        return;
    std::vector<int> range(n * 2);
    std::vector<Vec2> ext(n);
    std::vector<uint8_t> bullet(n);
    for (size_t k = 0; k < n; k++)
    {
        const size_t b = (size_t)from + k;
        float lo = 1e30f, hi = 0.0f;
        for (int s = S.shapeFirst[b]; s < S.shapeFirst[b] + S.shapeCount[b]; s++)
        {
            const size_t q = (size_t)s;
            const Vec2 h = S.shHalf[q];
            const bool disc = S.shType[q] == SHAPE2D_CIRCLE;
            lo = std::min(lo, disc ? (S.shRad[q] > 0.0f ? S.shRad[q] : h.x) : std::min(h.x, h.y));
            hi = std::max(hi, std::sqrt(h.x * h.x + h.y * h.y) +
                                  std::sqrt(S.shOff[q].x * S.shOff[q].x + S.shOff[q].y * S.shOff[q].y));
        }
        const bool none = S.shapeCount[b] == 0; // a shapeless or destroyed body: nothing to sweep
        range[2 * k] = none ? 0 : S.shapeFirst[b];
        range[2 * k + 1] = S.shapeCount[b];
        ext[k] = none ? Vec2(1e-3f, 1e-3f) : Vec2(std::max(lo, 1e-3f), std::max(hi, 1e-3f));
        bullet[k] = S.isBullet[b];
    }
    up(m_bShape, range.data(), n * 8, (size_t)from * 8);
    up(m_bExt, ext.data(), n * 8, (size_t)from * 8);
    up(m_bBullet, bullet.data(), n, (size_t)from);
}


void VkWorld2D::setBullet(int body, bool bullet)
{
    if (body < 0 || body >= m_bodyCount)
        return;
    flushUploadsNow();
    m_scene.isBullet[(size_t)body] = bullet ? 1 : 0;
    const uint8_t v = bullet ? 1 : 0;
    m_bBullet.upload(&v, 1, (size_t)body);
}


// Uploads shapes [from, shapeTotal) of the host table, and the whole vertex/normal pool. Called
// at build and after a spawn (the pool is small; spawning is not per step).
void VkWorld2D::uploadShapes(int from, int to)
{
    const Scene2D &S = m_scene;
    const bool wholePool = to < 0; // a spawn / build uploads the vertex pool too; a live edit does its own range
    if (to < 0)
        to = S.shapeTotal();
    const size_t f = (size_t)from, n = (size_t)std::max(to - from, 0);
    if (n > 0)
    {
        std::vector<int> range(2 * n);
        for (size_t i = 0; i < n; i++)
        {
            range[2 * i] = S.shVOff[f + i];
            range[2 * i + 1] = S.shVCnt[f + i];
        }
        up(m_sBody, S.shBody.data() + f, n * 4, f * 4);
        up(m_sType, S.shType.data() + f, n, f);
        up(m_sOff, S.shOff.data() + f, n * 8, f * 8);
        up(m_sAng, S.shAng.data() + f, n * 4, f * 4);
        up(m_sRad, S.shRad.data() + f, n * 4, f * 4);
        up(m_sHalf, S.shHalf.data() + f, n * 8, f * 8);
        up(m_sVR, range.data(), n * 8, f * 8);
        up(m_sFric, S.shFric.data() + f, n * 4, f * 4);
        up(m_sCat, S.shCat.data() + f, n * 8, f * 8);
        up(m_sMask, S.shMask.data() + f, n * 8, f * 8);
        up(m_sGroup, S.shGroup.data() + f, n * 4, f * 4);
        std::vector<float> mat(4 * n);
        for (size_t i = 0; i < n; i++)
        {
            mat[4 * i] = S.shTangentSpeed[f + i];
            mat[4 * i + 1] = S.shRolling[f + i];
            mat[4 * i + 2] = S.shRestitution[f + i];
            std::memcpy(&mat[4 * i + 3], &S.shUserId[f + i], 4);
        }
        up(m_sMat, mat.data(), n * 16, f * 16);
        up(m_sFlags, S.shFlags.data() + f, n, f);
    }
    if (wholePool)
    {
        const size_t v = std::max<size_t>(S.verts.size(), 1);
        if (m_verts.size() < v * 8)
        {
            sizeBuf(m_verts, (v + v / 2 + 64) * 8, false);
            sizeBuf(m_norms, (v + v / 2 + 64) * 8, false);
        }
        uploadVerts(0, (int)S.verts.size());
    }
}


// Vertices and normals [off, off + n) of the host pool (the buffers are already large enough).
void VkWorld2D::uploadVerts(int off, int n)
{
    const Scene2D &S = m_scene;
    if (n <= 0)
        return;
    up(m_verts, S.verts.data() + off, (size_t)n * 8, (size_t)off * 8);
    up(m_norms, S.norms.data() + off, (size_t)n * 8, (size_t)off * 8);
}


// Uploads authored joints [from, to) of the scene with fresh dual state.
void VkWorld2D::uploadAuthoredJoints(int from, int to)
{
    if (to <= from)
        return;
    const Scene2D &S = m_scene;
    const size_t n = (size_t)(to - from), f = (size_t)(kGrabSlotCount + from);
    // A freed joint slot is uploaded as broken: inert in every kernel.
    std::vector<uint8_t> live(n, 0);
    for (size_t i = 0; i < n; i++)
        if ((size_t)from + i < m_jointDead.size() && m_jointDead[(size_t)from + i])
            live[i] = 1;
    const std::vector<float> zero(n * 9, 0.0f);
    up(m_jA, S.jointA.data() + from, n * 4, f * 4);
    up(m_jB, S.jointB.data() + from, n * 4, f * 4);
    up(m_jRA, S.jointRA.data() + from, n * 8, f * 8);
    up(m_jRB, S.jointRB.data() + from, n * 8, f * 8);
    up(m_jAxis, S.jointAxis.data() + (size_t)from * 3, (size_t)n * 3 * sizeof(JointAxis2),
       (size_t)f * 3 * sizeof(JointAxis2));
    up(m_jFrameA, S.jointFrameA.data() + from, n * 4, f * 4);
    up(m_jRest, S.jointRest.data() + from, n * 4, f * 4);
    up(m_jArm, S.jointArm.data() + from, n * 4, f * 4);
    up(m_jFrac, S.jointFrac.data() + from, n * 4, f * 4);
    up(m_jBroken, live.data(), n, f);
    up(m_jC0, zero.data(), n * 36, f * 36);
    up(m_jPen, zero.data(), n * 36, f * 36);
    up(m_jLam, zero.data(), n * 36, f * 36);
    up(m_jForce, zero.data(), n * 12, f * 12);
}


// ---------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------
// Authors one SpawnDesc2D into a scene.
int addSpawnDesc(Scene2D &s, const SpawnDesc2D &d)
{
    switch (d.shape)
    {
    case SHAPE2D_CIRCLE:
        return s.addCircle(d.pos, d.size.x, d.density, d.friction, d.vel, d.angVel, d.color);
    case SHAPE2D_CAPSULE:
    {
        const float r = 0.5f * d.size.y;
        const float h = 0.5f * std::max(d.size.x - d.size.y, 0.0f);
        const Vec2 ax = rotate(d.angle, Vec2(h, 0.0f));
        return s.addCapsule(d.pos - ax, d.pos + ax, r, d.density, d.friction, d.vel, d.angVel, d.color);
    }
    case SHAPE2D_POLYGON:
    {
        const int n = std::min(std::max(d.sides, 3), kMaxPolygonVertices);
        Vec2 pts[kMaxPolygonVertices];
        for (int i = 0; i < n; i++)
        {
            const float a = 2.0f * kPi * (float)i / (float)n;
            pts[i] = Vec2(d.size.x * std::cos(a), d.size.x * std::sin(a));
        }
        const int id = s.addPolygon(d.pos, d.angle, pts, n, d.radius, d.density, d.friction, d.vel, d.angVel, d.color);
        if (id >= 0)
            return id;
        return s.addCircle(d.pos, std::max(d.size.x, kLinearSlop), d.density, d.friction, d.vel, d.angVel, d.color);
    }
    default:
        return s.addBox(d.pos, d.angle, d.size, d.density, d.friction, d.vel, d.angVel, d.color);
    }
}


int VkWorld2D::spawn(const SpawnDesc2D *d, int count)
{
    if (!m_init || !d || count <= 0)
        return -1;
    flushUploadsNow();
    const int first = m_bodyCount;

    if (m_bodyCount == 0)
    {
        // An empty world has no buffers yet: author the bodies into a scene and build it.
        Scene2D s = m_scene;
        for (int k = 0; k < count; k++)
            addSpawnDesc(s, d[k]);
        s.params = m_params;
        build(s);
        return 0;
    }

    for (int k = 0; k < count; k++)
        addSpawnDesc(m_scene, d[k]);
    return commitSpawn(first, m_jointCount, (int)m_scene.ignore.size());
}


int VkWorld2D::spawnScene(const Scene2D &frag, Vec2 offset)
{
    if (!m_init || frag.bodyCount() == 0)
        return -1;
    flushUploadsNow();
    Scene2D &S = m_scene;
    const int first = S.bodyCount();
    const int jointsBefore = S.jointCount();
    const int ignoreBefore = (int)S.ignore.size();
    if (m_bodyCount == 0)
    {
        // An empty world has no buffers yet: author into a scene copy and build it.
        Scene2D s = m_scene;
        s.params = m_params;
        for (int i = 0; i < frag.bodyCount(); i++)
            s.appendBody(frag, i, offset);
        for (int j = 0; j < frag.jointCount(); j++)
        {
            const size_t k = (size_t)j;
            s.jointA.push_back(frag.jointA[k]);
            s.jointB.push_back(frag.jointB[k]);
            s.jointRA.push_back(frag.jointA[k] < 0 ? frag.jointRA[k] + offset : frag.jointRA[k]);
            s.jointRB.push_back(frag.jointRB[k]);
            s.jointFrameA.push_back(frag.jointFrameA[k]);
            for (int ax = 0; ax < 3; ax++)
                s.jointAxis.push_back(frag.jointAxis[k * 3 + (size_t)ax]);
            s.jointRest.push_back(frag.jointRest[k]);
            s.jointArm.push_back(frag.jointArm[k]);
            s.jointFrac.push_back(frag.jointFrac[k]);
        }
        s.ignore.insert(s.ignore.end(), frag.ignore.begin(), frag.ignore.end());
        build(s, &m_params);
        return 0;
    }

    for (int i = 0; i < frag.bodyCount(); i++)
        S.appendBody(frag, i, offset);
    for (int j = 0; j < frag.jointCount(); j++)
    {
        const size_t k = (size_t)j;
        const int a = frag.jointA[k], b = frag.jointB[k];
        S.jointA.push_back(a < 0 ? -1 : a + first);
        S.jointB.push_back(b + first);
        S.jointRA.push_back(a < 0 ? frag.jointRA[k] + offset : frag.jointRA[k]);
        S.jointRB.push_back(frag.jointRB[k]);
        S.jointFrameA.push_back(frag.jointFrameA[k]);
        for (int ax = 0; ax < 3; ax++)
            S.jointAxis.push_back(frag.jointAxis[k * 3 + (size_t)ax]);
        S.jointRest.push_back(frag.jointRest[k]);
        S.jointArm.push_back(frag.jointArm[k]);
        S.jointFrac.push_back(frag.jointFrac[k]);
    }
    for (uint64_t key : frag.ignore)
    {
        const uint32_t lo = (uint32_t)(key >> 32) + (uint32_t)first;
        const uint32_t hi = (uint32_t)(key & 0xFFFFFFFFu) + (uint32_t)first;
        S.ignore.push_back(((uint64_t)lo << 32) | hi);
    }
    return commitSpawn(first, jointsBefore, ignoreBefore);
}


// Uploads what spawn() / spawnScene() appended to the host scene: bodies [first, bodyCount),
// joints [jointsBefore, ...) and any new ignore pairs.
int VkWorld2D::commitSpawn(int first, int jointsBefore, int ignoreBefore)
{
    flushUploadsNow();
    const Scene2D &S = m_scene;
    const int newCount = S.bodyCount();
    const int count = newCount - first;
    const bool grow = newCount > m_bodyCap;
    if (grow)
    {
        m_bodyCap = newCount + newCount / 2 + 1024;
        sizeBodyBuffers(m_bodyCap, true);
        sizeHostMirrors(m_bodyCap);
    }
    m_bodyCount = newCount;
    const int firstShape = S.shapeFirst[(size_t)first];
    if (S.shapeTotal() > m_shapeCap)
    {
        m_shapeCap = S.shapeTotal() + S.shapeTotal() / 2 + 1024;
        sizeShapeBuffers(m_shapeCap, true);
    }

    const size_t n = (size_t)count, f = (size_t)first;
    m_pos.upload(S.pos.data() + f, n * 8, f * 8);
    m_ang.upload(S.ang.data() + f, n * 4, f * 4);
    m_initPos.upload(S.pos.data() + f, n * 8, f * 8);
    m_initAng.upload(S.ang.data() + f, n * 4, f * 4);
    m_inertPos.upload(S.pos.data() + f, n * 8, f * 8);
    m_inertAng.upload(S.ang.data() + f, n * 4, f * 4);
    m_vel.upload(S.vel.data() + f, n * 8, f * 8);
    m_angVel.upload(S.angVel.data() + f, n * 4, f * 4);
    m_prevVel.upload(S.vel.data() + f, n * 8, f * 8);
    m_moment.upload(S.moment.data() + f, n * 4, f * 4);
    m_mass.upload(S.mass.data() + f, n * 4, f * 4);
    uploadBodyProps(f, n);
    uploadShapes(firstShape);
    uploadCcdBodies(first);
    {
        std::vector<uint8_t> ones(n, 1);
        std::vector<int> zeros(n, 0);
        std::vector<int> none(n, -1);
        std::vector<uint8_t> zeroB(n, 0);
        m_awake.upload(ones.data(), n, f);
        m_sleepTimer.upload(zeros.data(), n * 4, f * 4);
        m_restPos.upload(S.pos.data() + f, n * 8, f * 8);
        m_restAng.upload(S.ang.data() + f, n * 4, f * 4);
        m_cand.upload(zeroB.data(), n, f);
        m_snapCount.upload(none.data(), n * 4, f * 4);
        m_snapHash.upload(zeros.data(), n * 4, f * 4);
    }
    m_hostPos.insert(m_hostPos.end(), S.pos.begin() + first, S.pos.end());
    m_hostAng.insert(m_hostAng.end(), S.ang.begin() + first, S.ang.end());
    m_hostVel.insert(m_hostVel.end(), S.vel.begin() + first, S.vel.end());
    m_hostAngVel.insert(m_hostAngVel.end(), S.angVel.begin() + first, S.angVel.end());
    m_hostAwake.resize(newCount, 1);
    m_hostColor.resize(newCount, 0);

    // Joints and the no-collide list.
    bool jointsGrew = false;
    if (S.jointCount() > jointsBefore)
    {
        m_jointCount = S.jointCount();
        m_jointDead.resize((size_t)m_jointCount, 0);
        m_jointSlots = kGrabSlotCount + m_jointCount;
        if (m_jointSlots > m_jointCap)
        {
            m_jointCap = m_jointSlots + m_jointSlots / 2 + 1024;
            sizeJointBuffers(m_jointCap, true);
        }
        uploadAuthoredJoints(jointsBefore, m_jointCount);
        jointsGrew = true;
    }
    if ((int)S.ignore.size() > ignoreBefore)
    {
        m_scene.finalize();
        m_ignoreCount = (int)S.ignore.size();
        fitBuf(m_ignore, S.ignore.size() * 8);
        m_ignore.upload(S.ignore.data(), S.ignore.size() * 8);
    }

    uploadLists();
    if (grow || jointsGrew)
    {
        m_constraints.reserve(std::max(m_bodyCap, 1), m_pairCap, std::max(m_jointCap, m_jointSlots));
        m_csRes[0] = std::max(m_bodyCap, 1);
        m_csRes[1] = m_pairCap;
        m_csRes[2] = std::max(m_jointCap, m_jointSlots);
    }
    return first;
}


// ---------------------------------------------------------------------------
// Deferred uploads
// ---------------------------------------------------------------------------
void VkWorld2D::up(Buffer &b, const void *src, size_t bytes, size_t off)
{
    if (bytes == 0 || !src)
        return;
    if (!m_deferUp)
    {
        b.upload(src, bytes, off);
        return;
    }
    const size_t at = m_upData.size();
    m_upData.resize(at + bytes);
    std::memcpy(m_upData.data() + at, src, bytes);
    m_upOps.push_back({&b, at, bytes, off});
}


// Records the queued uploads into the step's own command buffer (a copy from the persistent
// staging buffer). Pass 0 only: a rerun restores the state copied after them.
void VkWorld2D::recordUploads()
{
    if (m_upOps.empty())
        return;
    if (m_upData.size() > (size_t)m_upStage.size())
    {
        flushUploadsNow(); // cannot happen after stepBegin's check; never record a copy past the end
        return;
    }
    std::memcpy(m_upStage.mapped(), m_upData.data(), m_upData.size());
    for (const UpOp &op : m_upOps)
        m_cmd.copy(m_upStage, *op.dst, op.bytes, op.src, op.dstOff);
    m_upOps.clear();
    m_upData.clear();
}


// The synchronous path: used when nothing is recording (no dynamic body yet, an oversized batch,
// or a call that uploads immediately and must see the queued data first).
void VkWorld2D::flushUploadsNow()
{
    for (const UpOp &op : m_upOps)
        op.dst->upload(m_upData.data() + op.src, op.bytes, op.dstOff);
    m_upOps.clear();
    m_upData.clear();
}


// Releases the retired shape and vertex ranges the step that just committed covered.
void VkWorld2D::releaseRetired()
{
    const size_t n = std::min(m_retiredMark, m_retired.size());
    for (size_t i = 0; i < n; i++)
    {
        const Retired &r = m_retired[i];
        (r.verts ? m_vertAlloc : m_shapeAlloc).release(r.off, r.n);
    }
    m_retired.erase(m_retired.begin(), m_retired.begin() + (std::ptrdiff_t)n);
    m_retiredMark = 0;
}


int VkWorld2D::RangeAlloc::alloc(int n, int tableEnd)
{
    if (n <= 0)
        return 0;
    for (size_t i = 0; i < freeR.size(); i++)
    {
        Range &r = freeR[i];
        if (r.n < n)
            continue;
        const int off = r.off;
        if (r.n == n)
            freeR.erase(freeR.begin() + (std::ptrdiff_t)i);
        else
        {
            r.off += n;
            r.n -= n;
        }
        return off;
    }
    if (!freeR.empty() && freeR.back().off + freeR.back().n == tableEnd)
    {
        const int off = freeR.back().off; // a short free tail: the table grows past its end
        freeR.pop_back();
        return off;
    }
    return tableEnd;
}


void VkWorld2D::RangeAlloc::release(int off, int n)
{
    if (n <= 0)
        return;
    size_t i = 0;
    while (i < freeR.size() && freeR[i].off < off)
        i++;
    freeR.insert(freeR.begin() + (std::ptrdiff_t)i, Range{off, n});
    if (i + 1 < freeR.size() && freeR[i].off + freeR[i].n == freeR[i + 1].off)
    {
        freeR[i].n += freeR[i + 1].n;
        freeR.erase(freeR.begin() + (std::ptrdiff_t)i + 1);
    }
    if (i > 0 && freeR[i - 1].off + freeR[i - 1].n == freeR[i].off)
    {
        freeR[i - 1].n += freeR[i].n;
        freeR.erase(freeR.begin() + (std::ptrdiff_t)i);
    }
}


// Body state, mass and shape ranges of bodies [first, first + count) from the host scene: a plain
// fresh body (awake, no sleep timer). A destroyed slot is a shapeless static.
void VkWorld2D::uploadBodies(int first, int count)
{
    const Scene2D &S = m_scene;
    const size_t n = (size_t)count, f = (size_t)first;
    up(m_pos, S.pos.data() + f, n * 8, f * 8);
    up(m_ang, S.ang.data() + f, n * 4, f * 4);
    up(m_initPos, S.pos.data() + f, n * 8, f * 8);
    up(m_initAng, S.ang.data() + f, n * 4, f * 4);
    up(m_inertPos, S.pos.data() + f, n * 8, f * 8);
    up(m_inertAng, S.ang.data() + f, n * 4, f * 4);
    up(m_vel, S.vel.data() + f, n * 8, f * 8);
    up(m_angVel, S.angVel.data() + f, n * 4, f * 4);
    up(m_prevVel, S.vel.data() + f, n * 8, f * 8);
    up(m_moment, S.moment.data() + f, n * 4, f * 4);
    up(m_mass, S.mass.data() + f, n * 4, f * 4);
    uploadBodyProps(f, n);
    uploadCcdBodies(first, first + count);
    std::vector<uint8_t> ones(n, 1), zeroB(n, 0);
    std::vector<int> zeros(n, 0), none(n, -1);
    up(m_awake, ones.data(), n, f);
    up(m_sleepTimer, zeros.data(), n * 4, f * 4);
    up(m_restPos, S.pos.data() + f, n * 8, f * 8);
    up(m_restAng, S.ang.data() + f, n * 4, f * 4);
    up(m_cand, zeroB.data(), n, f);
    up(m_snapCount, none.data(), n * 4, f * 4);
    up(m_snapHash, zeros.data(), n * 4, f * 4);
}

} // namespace avbd2d
