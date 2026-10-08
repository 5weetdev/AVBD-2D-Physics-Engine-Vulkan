#include "vk_world2d.h"

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

namespace
{
using Clock = std::chrono::steady_clock;

constexpr int kMaxColors = AVBD_MAX_COLORS_VK;
constexpr int kMaxPasses = 64;
constexpr int kAbsMaxPairs = 16 * 1000 * 1000;
constexpr int kMinPairCeiling = 65536;
constexpr int kPairsPerBodyCeiling = 12; // a real pile needs ~4 per body; this is the runaway cap
constexpr int kMaxTimedIterations = 32;  // the command list carries 80 timestamps
constexpr int kJointWakeRounds = 6;
constexpr int kGrabSlotCount = VkWorld2D::kGrabSlots;

uint32_t divUp(int n, int d)
{
    return n <= 0 ? 0u : (uint32_t)((n + d - 1) / d);
}

// need + 25% + slack, clamped to a ceiling so a runaway count cannot ask for the moon.
int cappedGrow(int need, int slack, int ceiling)
{
    if (need < 0)
        return ceiling;
    long long grown = (long long)need + need / 4 + slack;
    return (int)std::min<long long>(grown, ceiling);
}

void sizeBuf(Buffer &b, size_t bytes, bool preserve, VkBufferUsageFlags extra = 0)
{
    bytes = std::max<size_t>(bytes, 16);
    if (preserve)
        b.resizePreserve(bytes, extra);
    else
        b.resize(bytes, extra);
}

// Uploads one element of an array of `elemSize`-byte elements.
void putAt(Buffer &b, int index, size_t elemSize, const void *src)
{
    b.upload(src, elemSize, (VkDeviceSize)index * elemSize);
}
} // namespace

// ---------------------------------------------------------------------------
// ManBufs
// ---------------------------------------------------------------------------
void VkWorld2D::ManBufs::init(Device &dev)
{
    for (Buffer *b : {&key, &bodyA, &bodyB, &normal, &friction, &offset, &count, &feature, &rA, &rB, &c0, &penalty,
                      &lambda, &stick, &roll, &flags})
        b->init(dev);
}

void VkWorld2D::ManBufs::destroy()
{
    for (Buffer *b : {&key, &bodyA, &bodyB, &normal, &friction, &offset, &count, &feature, &rA, &rB, &c0, &penalty,
                      &lambda, &stick, &roll, &flags})
        b->destroy();
}

void VkWorld2D::ManBufs::size(int pairCap, int contactCap, bool preserve)
{
    const size_t P = (size_t)pairCap, C = (size_t)contactCap;
    sizeBuf(key, P * 8, preserve);
    sizeBuf(bodyA, P * 4, preserve);
    sizeBuf(bodyB, P * 4, preserve);
    sizeBuf(normal, P * 8, preserve);
    sizeBuf(friction, P * 4, preserve);
    sizeBuf(offset, P * 4, preserve);
    sizeBuf(count, P * 4, preserve);
    sizeBuf(feature, C * 4, preserve);
    sizeBuf(rA, C * 8, preserve);
    sizeBuf(rB, C * 8, preserve);
    sizeBuf(c0, C * 8, preserve);
    sizeBuf(penalty, C * 8, preserve);
    sizeBuf(lambda, C * 8, preserve);
    sizeBuf(stick, C * 1, preserve);
    sizeBuf(roll, P * 16, preserve);
    sizeBuf(flags, P * 1, preserve);
}

ManSet2Gpu VkWorld2D::ManBufs::gpu() const
{
    ManSet2Gpu m{};
    m.key = key.address();
    m.bodyA = bodyA.address();
    m.bodyB = bodyB.address();
    m.normal = normal.address();
    m.friction = friction.address();
    m.offset = offset.address();
    m.count = count.address();
    m.feature = feature.address();
    m.rA = rA.address();
    m.rB = rB.address();
    m.c0 = c0.address();
    m.penalty = penalty.address();
    m.lambda = lambda.address();
    m.stick = stick.address();
    m.roll = roll.address();
    m.flags = flags.address();
    return m;
}

uint64_t VkWorld2D::ManBufs::bytes() const
{
    uint64_t t = 0;
    for (const Buffer *b : {&key, &bodyA, &bodyB, &normal, &friction, &offset, &count, &feature, &rA, &rB, &c0,
                            &penalty, &lambda, &stick, &roll, &flags})
        t += (uint64_t)b->size();
    return t;
}

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------
std::vector<Buffer *> VkWorld2D::buffers()
{
    return {&m_pos,       &m_ang,        &m_initPos,   &m_initAng,    &m_inertPos,    &m_inertAng,
            &m_vel,       &m_angVel,     &m_prevVel,   &m_moment,
            &m_mass,      &m_awake,     &m_sleepTimer, &m_blocked,     &m_dynIdx,
            &m_statIdx,   &m_savePos,    &m_saveAng,   &m_saveVel,    &m_saveAngVel,  &m_savePrevVel,
            &m_saveAwake, &m_saveTimer,  &m_saveJPen,  &m_saveJLam,   &m_saveJBroken, &m_gridHead,
            &m_gridNext,  &m_pairsRaw,   &m_pairsSorted, &m_ignore,   &m_counters,   &m_jA,
            &m_jB,        &m_jRA,        &m_jRB,       &m_jC0,        &m_jPen,        &m_jLam,
            &m_jAxis,     &m_jFrameA,    &m_jRest,     &m_jArm,       &m_jFrac,       &m_jForce,      &m_jBroken,
            &m_restPos,   &m_restAng,    &m_cand,      &m_touchCount, &m_touchHash,   &m_snapCount,
            &m_snapHash,  &m_wakeReq,    &m_saveRestPos, &m_saveRestAng, &m_saveCand,
            &m_colorIndirect, &m_worldDev, &m_sBody,  &m_sType, &m_sOff, &m_sAng, &m_sRad, &m_sHalf, &m_sVR,
            &m_sFric,     &m_verts,       &m_norms, &m_sFlags, &m_events, &m_impulses,   &m_bProps, &m_sCat, &m_sMask, &m_sGroup, &m_sMat,
            &m_gridList,  &m_overList,
            &m_ccdFlag,   &m_ccdList,    &m_ccdCount,  &m_ccdStat,    &m_ccdArgs,     &m_bShape,
            &m_bExt,      &m_bBullet,    &m_movedCount, &m_movedDev, &m_qBox, &m_qHead, &m_qNext, &m_qBig, &m_qCount,
            &m_qOutHitDev, &m_qOutIdsDev, &m_qOutCountsDev};
}

// Every buffer indexed by body, with its element size. Growth keeps the ones that carry state.
std::vector<VkWorld2D::BodyBuf> VkWorld2D::bodyBuffers()
{
    return {{&m_pos, 8, true},       {&m_ang, 4, true},       {&m_initPos, 8, true},  {&m_initAng, 4, true},
            {&m_inertPos, 8, true},  {&m_inertAng, 4, true},  {&m_vel, 8, true},      {&m_angVel, 4, true},
            {&m_prevVel, 8, true},   {&m_moment, 4, true},
            {&m_mass, 4, true},      {&m_bProps, 16, true},  {&m_awake, 1, true},    {&m_sleepTimer, 4, true},
            {&m_blocked, 4, false},  {&m_dynIdx, 4, false},   {&m_statIdx, 4, false}, {&m_savePos, 8, false},
            {&m_saveAng, 4, false},  {&m_saveVel, 8, false},  {&m_saveAngVel, 4, false},
            {&m_savePrevVel, 8, false}, {&m_saveAwake, 1, false}, {&m_saveTimer, 4, false},
            {&m_restPos, 8, true},   {&m_restAng, 4, true},   {&m_cand, 1, true},     {&m_touchCount, 4, false},
            {&m_touchHash, 4, false}, {&m_snapCount, 4, true}, {&m_snapHash, 4, true}, {&m_wakeReq, 1, false},
            {&m_saveRestPos, 8, false}, {&m_saveRestAng, 4, false}, {&m_saveCand, 1, false},
            {&m_ccdFlag, 1, false},  {&m_ccdList, 4, false},  {&m_bShape, 8, true},   {&m_bExt, 8, true},
            {&m_bBullet, 1, true}};
}

void VkWorld2D::createPipelines()
{
    auto mk = [&](Pipeline &p, const uint32_t *spv, size_t words, const char *entry) {
        p.init(m_dev, m_layout, spv, words, entry, entry);
    };
    mk(m_integrate, integrate2d_spirv, integrate2d_spirv_words, "csIntegrate2D");
    mk(m_velocity, integrate2d_spirv, integrate2d_spirv_words, "csVelocity2D");
    mk(m_guard, integrate2d_spirv, integrate2d_spirv_words, "csGuard2D");
    mk(m_velocityGuard, integrate2d_spirv, integrate2d_spirv_words, "csVelocityGuard2D");
    mk(m_gravityX, integrate2d_spirv, integrate2d_spirv_words, "csGravityX2D");
    mk(m_ccdFlagPl, ccd2d_spirv, ccd2d_spirv_words, "csCcdFlag2D");
    mk(m_ccdArgsPl, ccd2d_spirv, ccd2d_spirv_words, "csCcdArgs2D");
    mk(m_ccdPl, ccd2d_spirv, ccd2d_spirv_words, "csCcd2D");
    mk(m_gridClear, broadphase2d_spirv, broadphase2d_spirv_words, "csGridClear2D");
    mk(m_gridInsert, broadphase2d_spirv, broadphase2d_spirv_words, "csGridInsert2D");
    mk(m_findPairs, broadphase2d_spirv, broadphase2d_spirv_words, "csFindPairs2D");
    mk(m_oversizePairs, broadphase2d_spirv, broadphase2d_spirv_words, "csOversizePairs2D");
    mk(m_narrowphase, narrowphase2d_spirv, narrowphase2d_spirv_words, "csNarrowphase2D");
    mk(m_jointInit, solver2d_spirv, solver2d_spirv_words, "csJointInit2D");
    mk(m_integrateJoint, solver2d_spirv, solver2d_spirv_words, "csIntegrateJointInit2D");
    mk(m_jointDual, solver2d_spirv, solver2d_spirv_words, "csJointDual2D");
    mk(m_colorArgs, solver2d_spirv, solver2d_spirv_words, "csColorArgs2D");
    mk(m_primal, solver2d_spirv, solver2d_spirv_words, "csPrimal2D");
    mk(m_dual, solver2d_spirv, solver2d_spirv_words, "csDual2D");
    mk(m_dualAll, solver2d_spirv, solver2d_spirv_words, "csDualAll2D");
    mk(m_iterate, solver2d_spirv, solver2d_spirv_words, "csIterate2D");
    mk(m_sortSmall, broadphase2d_spirv, broadphase2d_spirv_words, "csSortPairsSmall2D");
    mk(m_buildSmall, constraints2d_spirv, constraints2d_spirv_words, "csBuildSmall2D");
    mk(m_sleepPrep, sleep2d_spirv, sleep2d_spirv_words, "csSleepPrep2D");
    mk(m_sleepBlockPre, sleep2d_spirv, sleep2d_spirv_words, "csSleepBlockPre2D");
    mk(m_sleepMark, sleep2d_spirv, sleep2d_spirv_words, "csSleepMark2D");
    mk(m_sleepNbrC, sleep2d_spirv, sleep2d_spirv_words, "csSleepNbrContacts2D");
    mk(m_sleepNbrJ, sleep2d_spirv, sleep2d_spirv_words, "csSleepNbrJoints2D");
    mk(m_sleepCommit, sleep2d_spirv, sleep2d_spirv_words, "csSleepCommit2D");
    mk(m_touchClear, sleep2d_spirv, sleep2d_spirv_words, "csTouchClear2D");
    mk(m_touchAccumC, sleep2d_spirv, sleep2d_spirv_words, "csTouchAccumContacts2D");
    mk(m_touchAccumJ, sleep2d_spirv, sleep2d_spirv_words, "csTouchAccumJoints2D");
    mk(m_wakeReqC, sleep2d_spirv, sleep2d_spirv_words, "csWakeReqContacts2D");
    mk(m_wakeReqJ, sleep2d_spirv, sleep2d_spirv_words, "csWakeReqJoints2D");
    mk(m_wakeLost, sleep2d_spirv, sleep2d_spirv_words, "csWakeLost2D");
    mk(m_wakeApply, sleep2d_spirv, sleep2d_spirv_words, "csWakeApply2D");
    mk(m_wakeAll, sleep2d_spirv, sleep2d_spirv_words, "csWakeAll2D");
    mk(m_queryClear, query2d_spirv, query2d_spirv_words, "csQueryClear2D");
    mk(m_queryInsert, query2d_spirv, query2d_spirv_words, "csQueryInsert2D");
    mk(m_queryRays, query2d_spirv, query2d_spirv_words, "csQueryRays2D");
    mk(m_queryAabbs, query2d_spirv, query2d_spirv_words, "csQueryAabbs2D");
    mk(m_eventBegin, events2d_spirv, events2d_spirv_words, "csEventBegin2D");
    mk(m_eventEnd, events2d_spirv, events2d_spirv_words, "csEventEnd2D");
    mk(m_eventHit, events2d_spirv, events2d_spirv_words, "csEventHit2D");
    mk(m_eventJoint, events2d_spirv, events2d_spirv_words, "csEventJoint2D");
    mk(m_applyImpulses, events2d_spirv, events2d_spirv_words, "csApplyImpulses2D");
    mk(m_applyBodyPatches, events2d_spirv, events2d_spirv_words, "csApplyBodyPatches2D");
    mk(m_applyJointPatches, events2d_spirv, events2d_spirv_words, "csApplyJointPatches2D");
    mk(m_compactMoved, events2d_spirv, events2d_spirv_words, "csCompactMoved2D");
}

void VkWorld2D::init()
{
    if (m_init)
        return;
    if (const char *e = std::getenv("AVBD2D_SMALL_WORLD")) // bodies; 0 = multi-dispatch path only
        m_smallLimit = std::atoi(e);
    try
    {
        if (!m_dev.adopted())
        {
            configureDevice();
            m_dev.init();
        }
        m_layout.init(m_dev);
        createPipelines();
        m_csPl.init(m_dev, m_layout, constraints_spirv, constraints_spirv_words);
        m_primPl.init(m_dev, m_layout);
        m_select.init(m_dev);
        m_radix.init(m_dev);
        m_constraints.init(m_dev);
        m_cmd.init(m_dev, 80);
        m_ring.init(m_dev, 1 << 23);
        for (Buffer *b : buffers())
            b->init(m_dev);
        m_man[0].init(m_dev);
        m_man[1].init(m_dev);
        m_worldDev.resize(sizeof(World2Gpu));
        m_worldStage.init(m_dev, sizeof(World2Gpu), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          MappedBuffer::HostAccess::Write);
        m_grabStage.init(m_dev, (VkDeviceSize)kGrabSlotCount * sizeof(Vec2), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         MappedBuffer::HostAccess::Write);
    }
    catch (...)
    {
        // Half-built: release what exists so the caller can retry or report; destroy() is
        // written to tolerate objects that never initialised.
        m_init = true;
        destroy();
        throw;
    }
    m_init = true;
}

// The 2D kernels (and the shared radix/constraints ones) are written for subgroup size 32 or
// 64 and never use float atomics; see Device::SubgroupPolicy. Call before Device::init/adopt.
void VkWorld2D::configureDevice()
{
    m_dev.setSubgroupPolicy(avbdvk::Device::SubgroupPolicy::Generic3264);
    m_dev.setRequireAtomicFloat(false);
}

void VkWorld2D::destroy()
{
    if (!m_init)
        return;
    if (m_dev.device())
        vkDeviceWaitIdle(m_dev.device());
    m_worldStage.destroy();
    m_grabStage.destroy();
    for (MappedBuffer *b : {&m_statusHost, &m_posHost, &m_angHost, &m_velHost, &m_awakeHost, &m_colorHost, &m_eventsHost, &m_ccdHost,
                            &m_qRays, &m_qAabbs, &m_qOutHit, &m_qOutIds, &m_qOutCounts, &m_bPatchBuf, &m_jPatchBuf, &m_movedHost, &m_movedCountHost, &m_jForceHost, &m_upStage})
        b->destroy();
    m_upOps.clear();
    m_upData.clear();
    m_bPatchCap = m_jPatchCap = 0;
    for (Buffer *b : buffers())
        b->destroy();
    m_man[0].destroy();
    m_man[1].destroy();
    m_radix.destroy();
    m_constraints.destroy();
    m_cmd.destroy();
    m_ring.destroy();
    m_select.destroy();
    m_primPl.destroy();
    m_csPl.destroy();
    for (Pipeline *p : {&m_integrate, &m_gravityX, &m_velocity, &m_guard, &m_velocityGuard, &m_ccdFlagPl, &m_ccdArgsPl, &m_ccdPl, &m_gridClear, &m_gridInsert, &m_findPairs, &m_oversizePairs,
                        &m_narrowphase, &m_jointInit, &m_integrateJoint, &m_jointDual, &m_colorArgs, &m_primal, &m_dual, &m_dualAll, &m_iterate, &m_sortSmall, &m_buildSmall,
                        &m_sleepPrep, &m_sleepBlockPre, &m_sleepMark, &m_sleepNbrC, &m_sleepNbrJ,
                        &m_sleepCommit, &m_touchClear, &m_touchAccumC, &m_touchAccumJ, &m_wakeReqC, &m_wakeReqJ,
                        &m_wakeLost, &m_wakeApply, &m_wakeAll, &m_eventBegin, &m_eventEnd, &m_eventHit,
                        &m_eventJoint, &m_applyImpulses, &m_queryClear, &m_queryInsert, &m_queryRays, &m_queryAabbs, &m_applyBodyPatches, &m_applyJointPatches, &m_compactMoved})
        p->destroy();
    m_layout.destroy();
    m_dev.shutdown();
    m_init = false;
}

// ---------------------------------------------------------------------------
// Capacity and sizing
// ---------------------------------------------------------------------------
int VkWorld2D::pairCeiling() const
{
    return std::min(kAbsMaxPairs, std::max(kMinPairCeiling, kPairsPerBodyCeiling * m_bodyCount));
}

void VkWorld2D::setCapacity(int pairCap, int contactCap, bool preserve)
{
    // The cur set is regenerated every pass; the prev set is the warm start and must survive.
    // The caps are committed last: a throw (memory budget, device error) leaves the old ones, and
    // every buffer is then at least that large.
    m_man[m_curIdx].size(pairCap, contactCap, false);
    m_man[1 - m_curIdx].size(pairCap, contactCap, preserve);
    sizeBuf(m_pairsRaw, (size_t)pairCap * 8, false);
    sizeBuf(m_pairsSorted, (size_t)pairCap * 8, false);
    m_constraints.reserve(std::max(m_bodyCap, 1), pairCap, std::max(m_jointCap, m_jointSlots));
    m_csRes[0] = std::max(m_bodyCap, 1);
    m_csRes[1] = pairCap;
    m_csRes[2] = std::max(m_jointCap, m_jointSlots);
    m_pairCap = pairCap;
    m_contactCap = contactCap;
}

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

void VkWorld2D::sizeBodyBuffers(int cap, bool growing)
{
    for (const BodyBuf &b : bodyBuffers())
        sizeBuf(*b.buf, (size_t)cap * b.elem, growing && b.preserve);
}

// Every buffer indexed by shape. The table keeps its contents on growth; the grid's link array
// is rebuilt by every pass.
void VkWorld2D::sizeShapeBuffers(int cap, bool growing)
{
    const size_t c = (size_t)cap;
    sizeBuf(m_sBody, c * 4, growing);
    sizeBuf(m_sType, c, growing);
    sizeBuf(m_sOff, c * 8, growing);
    sizeBuf(m_sAng, c * 4, growing);
    sizeBuf(m_sRad, c * 4, growing);
    sizeBuf(m_sHalf, c * 8, growing);
    sizeBuf(m_sVR, c * 8, growing);
    sizeBuf(m_sFric, c * 4, growing);
    sizeBuf(m_sCat, c * 8, growing);
    sizeBuf(m_sMask, c * 8, growing);
    sizeBuf(m_sGroup, c * 4, growing);
    sizeBuf(m_sMat, c * 16, growing);
    sizeBuf(m_sFlags, c, growing);
    sizeBuf(m_gridNext, c * 4, false);
}

void VkWorld2D::sizeHostMirrors(int cap)
{
    for (MappedBuffer *b : {&m_posHost, &m_angHost, &m_velHost, &m_awakeHost, &m_colorHost, &m_movedHost})
        b->destroy();
    const VkBufferUsageFlags dst = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const size_t c = (size_t)std::max(cap, 1);
    m_movedHost.init(m_dev, c * 32, dst, MappedBuffer::HostAccess::Read);
    sizeBuf(m_movedDev, c * 32, false);
    m_posHost.init(m_dev, c * 8, dst, MappedBuffer::HostAccess::Read);
    m_angHost.init(m_dev, c * 4, dst, MappedBuffer::HostAccess::Read);
    m_velHost.init(m_dev, c * 8, dst, MappedBuffer::HostAccess::Read);
    m_awakeHost.init(m_dev, (c + 3) & ~size_t(3), dst, MappedBuffer::HostAccess::Read);
    m_colorHost.init(m_dev, c * 4, dst, MappedBuffer::HostAccess::Read);
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

// ---------------------------------------------------------------------------
// Joint storage: slots [0, kGrabSlots) are the grab, authored joint j is slot kGrabSlots + j.
// ---------------------------------------------------------------------------
void VkWorld2D::sizeJointBuffers(int slotCap, bool preserve)
{
    const size_t J = (size_t)slotCap;
    struct E
    {
        Buffer *b;
        size_t elem;
        bool keep;
    };
    for (const E &e : {E{&m_jA, 4, true}, E{&m_jB, 4, true}, E{&m_jRA, 8, true}, E{&m_jRB, 8, true},
                       E{&m_jC0, 36, true}, E{&m_jPen, 36, true}, E{&m_jLam, 36, true},
                       E{&m_jAxis, 3 * sizeof(JointAxis2), true}, E{&m_jFrameA, 4, true}, E{&m_jRest, 4, true},
                       E{&m_jArm, 4, true}, E{&m_jFrac, 4, true}, E{&m_jForce, 12, true},
                       E{&m_jBroken, 1, true}, E{&m_saveJPen, 36, false}, E{&m_saveJLam, 36, false},
                       E{&m_saveJBroken, 1, false}})
        sizeBuf(*e.b, J * e.elem, preserve && e.keep);
    m_jForceHost.destroy();
    m_jForceHost.init(m_dev, J * 12, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    std::memset(m_jForceHost.mapped(), 0, J * 12);
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
    up(m_jAxis, S.jointAxis.data() + from * 3, n * 3 * sizeof(JointAxis2), f * 3 * sizeof(JointAxis2));
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
// Build
// ---------------------------------------------------------------------------
void VkWorld2D::build(const Scene2D &sceneIn, const SolverParams2D *keepParams)
{
    if (!m_init)
        init();

    m_scene = sceneIn;
    m_scene.finalize();
    m_params = keepParams ? *keepParams : m_scene.params;
    m_scene.params = m_params;
    const Scene2D &S = m_scene;

    m_grabBody = -1;
    m_grabGroup.clear();
    m_grabOffset.clear();
    m_grabSorted.clear();
    m_result = StepResult2D();
    m_timing = StepTiming2D();
    // liveFinish() of a world that was never built replays its edits through here.
    const bool keepLive = m_keepLive;
    std::vector<uint8_t> keepDead;
    std::vector<int> keepFB, keepFJ;
    std::vector<Retired> keepRet;
    if (keepLive)
    {
        keepDead = m_jointDead;
        keepFB = m_freeBodies;
        keepFJ = m_freeJoints;
        keepRet = m_retired;
    }
    m_upOps.clear();
    m_upData.clear();
    m_shapeAlloc.freeR.clear();
    m_vertAlloc.freeR.clear();
    m_retired.clear();
    m_retiredMark = 0;
    m_freeBodies.clear();
    m_freeJoints.clear();
    m_dirtyJoints.clear();
    m_dirtyBodiesNew.clear();
    m_dirtyShapeR.clear();
    m_dirtyVertR.clear();
    m_listsDirty = m_dirtyIgnore = m_liveDirty = false;
    m_liveBuilt = false;
    m_bodyCount = S.bodyCount();
    m_jointCount = S.jointCount();
    m_jointDead.assign((size_t)m_jointCount, 0);
    if (keepLive)
    {
        m_jointDead = keepDead;
        m_jointDead.resize((size_t)m_jointCount, 0);
    }
    m_jointSlots = m_jointCount + kGrabSlotCount; // + the grab slots
    m_prevPairs = 0;
    m_lastColors = 0;
    m_curIdx = 0;
    m_removedTotal = m_brokenTotal = 0;
    m_awakeNow = -1;
    m_wakeAllPending = false;
    m_sleepWasOn = sleepActive();

    m_statusHost.destroy();
    m_eventsHost.destroy();
    m_stepActive = false;
    m_bodyPatches.clear();
    m_jointPatches.clear();
    m_bodyPatchSlot.clear();
    m_jointPatchSlot.clear();
    m_bPatchCount = m_jPatchCount = 0;
    m_moved2d.clear();
    if (m_bodyCount == 0)
    {
        m_dynCount = m_statCount = 0;
        m_hostPos.clear();
        m_hostAng.clear();
        m_hostVel.clear();
        m_hostAngVel.clear();
        m_hostAwake.clear();
        m_hostColor.clear();
        return;
    }
    const size_t N = (size_t)m_bodyCount;

    // --- Bodies ---------------------------------------------------------------------------
    m_bodyCap = m_bodyCount + m_bodyCount / 8 + 1024; // headroom for spawning
    sizeBodyBuffers(m_bodyCap, false);
    m_pos.upload(S.pos.data(), N * 8);
    m_ang.upload(S.ang.data(), N * 4);
    m_initPos.upload(S.pos.data(), N * 8);
    m_initAng.upload(S.ang.data(), N * 4);
    m_inertPos.upload(S.pos.data(), N * 8);
    m_inertAng.upload(S.ang.data(), N * 4);
    m_vel.upload(S.vel.data(), N * 8);
    m_angVel.upload(S.angVel.data(), N * 4);
    m_prevVel.upload(S.vel.data(), N * 8);
    m_moment.upload(S.moment.data(), N * 4);
    m_mass.upload(S.mass.data(), N * 4);
    uploadBodyProps(0, N);
    m_shapeCap = S.shapeTotal() + S.shapeTotal() / 8 + 1024;
    sizeShapeBuffers(m_shapeCap, false);
    uploadShapes(0);
    uploadCcdBodies(0);
    {
        std::vector<uint8_t> ones(N, 1);
        std::vector<int> zeros(N, 0);
        std::vector<int> none(N, -1);
        std::vector<uint8_t> zeroB(N, 0);
        m_awake.upload(ones.data(), N);
        m_sleepTimer.upload(zeros.data(), N * 4);
        m_restPos.upload(S.pos.data(), N * 8);
        m_restAng.upload(S.ang.data(), N * 4);
        m_cand.upload(zeroB.data(), N);
        m_snapCount.upload(none.data(), N * 4);
        m_snapHash.upload(zeros.data(), N * 4);
    }
    uploadLists();

    // --- Broadphase -----------------------------------------------------------------------
    sizeBuf(m_counters, 32, false);
    sizeBuf(m_ccdCount, 16, false);
    sizeBuf(m_ccdStat, 16, false);
    sizeBuf(m_ccdArgs, 16, false, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    sizeBuf(m_movedCount, 16, false);
    m_movedCountHost.destroy();
    m_movedCountHost.init(m_dev, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    *m_movedCountHost.mappedAs<int32_t>() = 0;
    m_ignoreCount = (int)S.ignore.size();
    if (m_ignoreCount > 0)
    {
        sizeBuf(m_ignore, S.ignore.size() * 8, false);
        m_ignore.upload(S.ignore.data(), S.ignore.size() * 8);
    }
    else
        sizeBuf(m_ignore, 16, false);

    // --- Joints (the grab slots, then the authored joints) ---------------------------------
    m_jointCap = m_jointSlots + 1024; // headroom for spawned joints
    sizeJointBuffers(m_jointCap, false);
    {
        const std::vector<int> noWorld((size_t)kGrabSlotCount, -1);
        const std::vector<float> inf((size_t)kGrabSlotCount, kStiffInf);
        m_jA.upload(noWorld.data(), (size_t)kGrabSlotCount * 4);
        m_jFrac.upload(inf.data(), (size_t)kGrabSlotCount * 4);
        uploadGrabSlots();
        uploadAuthoredJoints(0, m_jointCount);
    }

    sizeBuf(m_colorIndirect, (size_t)kMaxColors * 3 * 4, false, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    // --- Capacities ----------------------------------------------------------------------
    int pairCap = std::min(std::max(2048, 4 * m_bodyCount), pairCeiling());
    int contactCap = 2 * pairCap;
    if (m_forcedPairCap > 0)
    {
        pairCap = m_forcedPairCap;
        contactCap = std::max(m_forcedContactCap, 1);
    }
    setCapacity(pairCap, contactCap, false);
    m_man[0].count.zero();
    m_man[1].count.zero();

    // --- Host copies of the poses --------------------------------------------------------
    m_statusHost.init(m_dev, 32, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    m_events2d = Events2D();
    m_pendingImpulses.clear();
    m_impulseCount = 0;
    m_impulseCap = 16;
    sizeBuf(m_impulses, (size_t)m_impulseCap * 16, false);
    for (int k = 0; k < EVENT2D_KINDS; k++)
        m_evCap[k] = m_forcedEventCap > 0 ? (uint32_t)m_forcedEventCap : (k == EVENT2D_JOINT_BREAK ? 256u : 1024u);
    sizeEvents();
    m_query.invalidate();
    m_ccdHost.destroy();
    m_ccdHost.init(m_dev, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    m_ccdStats = CcdStats2D();
    sizeHostMirrors(m_bodyCap);
    m_hostPos = S.pos;
    m_hostAng = S.ang;
    m_hostVel = S.vel;
    m_hostAngVel = S.angVel;
    m_hostAwake.assign(N, 1);
    m_hostColor.assign(N, 0);
    m_liveBuilt = true;
    if (keepLive)
    {
        m_freeBodies = keepFB;
        m_freeJoints = keepFJ;
        m_retired = keepRet;
        m_retiredMark = m_retired.size();
        releaseRetired();
    }
}

void VkWorld2D::debugForceTinyCapacity(int pairCap, int contactCap)
{
    m_forcedPairCap = pairCap;
    m_forcedContactCap = contactCap;
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
// Step
// ---------------------------------------------------------------------------
World2Gpu VkWorld2D::makeWorld(int colorBound) const
{
    World2Gpu W{};
    W.pos = m_pos.address();
    W.ang = m_ang.address();
    W.initPos = m_initPos.address();
    W.initAng = m_initAng.address();
    W.inertPos = m_inertPos.address();
    W.inertAng = m_inertAng.address();
    W.vel = m_vel.address();
    W.angVel = m_angVel.address();
    W.prevVel = m_prevVel.address();
    W.moment = m_moment.address();
    W.mass = m_mass.address();
    W.awake = m_awake.address();
    W.sleepTimer = m_sleepTimer.address();
    W.blocked = m_blocked.address();
    W.restPos = m_restPos.address();
    W.restAng = m_restAng.address();
    W.cand = m_cand.address();
    W.touchCount = m_touchCount.address();
    W.touchHash = m_touchHash.address();
    W.snapCount = m_snapCount.address();
    W.snapHash = m_snapHash.address();
    W.wakeReq = m_wakeReq.address();
    W.dyn = m_dynIdx.address();
    W.stat = m_statIdx.address();
    W.sBody = m_sBody.address();
    W.sType = m_sType.address();
    W.sOff = m_sOff.address();
    W.sAng = m_sAng.address();
    W.sRad = m_sRad.address();
    W.sHalf = m_sHalf.address();
    W.sVR = m_sVR.address();
    W.sFric = m_sFric.address();
    W.sCat = m_sCat.address();
    W.sMask = m_sMask.address();
    W.sGroup = m_sGroup.address();
    W.sMat = m_sMat.address();
    W.bProps = m_bProps.address();
    W.verts = m_verts.address();
    W.norms = m_norms.address();
    W.grid = m_gridList.address();
    W.over = m_overList.address();

    W.gridHead = m_gridHead.address();
    W.gridNext = m_gridNext.address();
    W.pairsRaw = m_pairsRaw.address();
    W.pairsSorted = m_pairsSorted.address();
    W.ignore = m_ignore.address();
    W.counters = m_counters.address();
    W.ccdFlag = m_ccdFlag.address();
    W.ccdList = m_ccdList.address();
    W.ccdCount = m_ccdCount.address();
    W.ccdStat = m_ccdStat.address();
    W.ccdArgs = m_ccdArgs.address();
    W.bShape = m_bShape.address();
    W.bExt = m_bExt.address();
    W.bBullet = m_bBullet.address();

    W.cur = m_man[m_curIdx].gpu();
    W.prev = m_man[1 - m_curIdx].gpu();

    const VkConstraintGraph g = m_constraints.graph();
    W.colorBodies = m_constraints.colorBodies().address();
    W.colorStart = m_constraints.colorStartAddr();
    W.colorEnd = m_constraints.colorEndAddr();
    W.gStart = g.start;
    W.gEnd = g.end;
    W.gKey = g.key;
    W.gIsA = g.isA;
    W.colorIndirect = m_colorIndirect.address();

    W.jA = m_jA.address();
    W.jB = m_jB.address();
    W.jRA = m_jRA.address();
    W.jRB = m_jRB.address();
    W.jC0 = m_jC0.address();
    W.jPen = m_jPen.address();
    W.jLam = m_jLam.address();
    W.jAxis = m_jAxis.address();
    W.jFrameA = m_jFrameA.address();
    W.jForce = m_jForce.address();
    W.jRest = m_jRest.address();
    W.jArm = m_jArm.address();
    W.jFrac = m_jFrac.address();
    W.jBroken = m_jBroken.address();

    W.bodyCount = (uint32_t)m_bodyCount;
    W.dynCount = (uint32_t)m_dynCount;
    W.statCount = (uint32_t)m_statCount;
    W.tableMask = (uint32_t)(m_tableSize - 1);
    W.ignoreCount = (uint32_t)m_ignoreCount;
    W.pairCap = (uint32_t)m_pairCap;
    W.contactCap = (uint32_t)m_contactCap;
    W.prevCount = (uint32_t)m_prevPairs;
    W.jointCount = (uint32_t)m_jointSlots;
    W.dt = m_params.dt;
    W.gravityX = m_params.gravity.x;
    W.alpha = m_params.alpha;
    W.betaLin = m_params.betaLin;
    W.betaAng = m_params.betaAng;
    W.gamma = m_params.gamma;
    W.dualDamping = m_params.dualDamping;
    W.dualDampVel = m_params.dualDampVel;
    W.cellSize = m_cellSize;
    W.killY = m_params.killY;
    W.autoK = m_params.autoKStart ? 1u : 0u;
    W.killOn = m_params.killFallenEnabled ? 1u : 0u;
    W.colorBound = colorBound;
    W.sleepOn = sleepActive() ? 1u : 0u;
    W.sleepFrames = (uint32_t)std::max(m_params.sleepFrames, 1);
    W.sleepLin = m_params.sleepLinVel;
    W.sleepAng = m_params.sleepAngVel;
    W.maxSpeed = m_params.maxSpeed;
    W.sleepDisp = m_params.sleepDisp;
    W.sleepFrac = m_params.sleepFracture;
    W.contactCapPts = (uint32_t)std::max(m_params.contactCap, 0);
    W.gridCount = (uint32_t)m_gridCount;
    W.gridDyn = (uint32_t)m_gridDyn;
    W.overCount = (uint32_t)m_overCount;
    W.frictionMix = (uint32_t)std::max(m_params.frictionMix, 0);
    W.sFlags = m_sFlags.address();
    W.ev = m_events.address();
    W.jBrokenPrev = m_saveJBroken.address();
    W.impulses = m_impulses.address();
    for (int k = 0; k < EVENT2D_KINDS; k++)
    {
        W.evOff[k] = m_evOff[k];
        W.evCap[k] = m_evCap[k];
    }
    W.hitThreshold = m_hitThreshold;
    W.evOn = m_eventsOn ? 1u : 0u;
    W.impulseCount = (uint32_t)m_impulseCount;
    W.ccdOn = m_params.enableContinuous ? 1u : 0u;
    W.ccdSafety = 0.5f;
    W.gravityY = m_params.gravity.y;
    W.bPatch = m_bPatchBuf.address();
    W.jPatch = m_jPatchBuf.address();
    W.moved = m_movedDev.address();
    W.movedCount = m_movedCount.address();
    W.saveAwake = m_saveAwake.address();
    W.savePos = m_savePos.address();
    W.saveAng = m_saveAng.address();
    W.bPatchCount = (uint32_t)m_bPatchCount;
    W.jPatchCount = (uint32_t)m_jPatchCount;
    return W;
}

void VkWorld2D::saveState()
{
    const VkDeviceSize n = (VkDeviceSize)m_bodyCount;
    m_cmd.copy(m_pos, m_savePos, n * 8);
    m_cmd.copy(m_ang, m_saveAng, n * 4);
    m_cmd.copy(m_vel, m_saveVel, n * 8);
    m_cmd.copy(m_angVel, m_saveAngVel, n * 4);
    m_cmd.copy(m_prevVel, m_savePrevVel, n * 8);
    m_cmd.copy(m_awake, m_saveAwake, n);
    m_cmd.copy(m_sleepTimer, m_saveTimer, n * 4);
    m_cmd.copy(m_restPos, m_saveRestPos, n * 8);
    m_cmd.copy(m_restAng, m_saveRestAng, n * 4);
    m_cmd.copy(m_cand, m_saveCand, n);
    m_cmd.copy(m_jPen, m_saveJPen, (VkDeviceSize)m_jointSlots * 36);
    m_cmd.copy(m_jLam, m_saveJLam, (VkDeviceSize)m_jointSlots * 36);
    m_cmd.copy(m_jBroken, m_saveJBroken, (VkDeviceSize)m_jointSlots);
}

void VkWorld2D::restoreState()
{
    const VkDeviceSize n = (VkDeviceSize)m_bodyCount;
    m_cmd.copy(m_savePos, m_pos, n * 8);
    m_cmd.copy(m_saveAng, m_ang, n * 4);
    m_cmd.copy(m_saveVel, m_vel, n * 8);
    m_cmd.copy(m_saveAngVel, m_angVel, n * 4);
    m_cmd.copy(m_savePrevVel, m_prevVel, n * 8);
    m_cmd.copy(m_saveAwake, m_awake, n);
    m_cmd.copy(m_saveTimer, m_sleepTimer, n * 4);
    m_cmd.copy(m_saveRestPos, m_restPos, n * 8);
    m_cmd.copy(m_saveRestAng, m_restAng, n * 4);
    m_cmd.copy(m_saveCand, m_cand, n);
    m_cmd.copy(m_saveJPen, m_jPen, (VkDeviceSize)m_jointSlots * 36);
    m_cmd.copy(m_saveJLam, m_jLam, (VkDeviceSize)m_jointSlots * 36);
    m_cmd.copy(m_saveJBroken, m_jBroken, (VkDeviceSize)m_jointSlots);
}

void VkWorld2D::recordPass(int pass, int colorBound, int jpRounds)
{
    const SolverParams2D &P = m_params;
    const int iters = std::max(P.iterations, 1);

    // Everything that can reallocate is grown before the first recorded command.
    reserveRadix(m_radix, (uint32_t)m_pairCap, 8, 0);
    m_constraints.prepare(m_radix, m_dynCount);
    const bool ccdOn = P.enableContinuous && m_dynCount > 0;
    if (ccdOn)
    {
        reserveScan(m_select.scan, (uint32_t)m_dynCount);
        if (m_select.flagsU32.size() < (VkDeviceSize)m_dynCount * 4)
            m_select.flagsU32.resize((VkDeviceSize)m_dynCount * 4);
    }

    m_ring.reset();
    m_cmd.begin();
    m_cmd.writeTimestamp(0);

    const World2Gpu W = makeWorld(colorBound);
    memcpy(m_worldStage.mapped(), &W, sizeof(W));
    m_cmd.copy(m_worldStage, m_worldDev, sizeof(W));
    recordGrabTargets();
    m_cmd.barrier();
    // Queued live edits (created / destroyed bodies, shapes, joints) land before anything reads them.
    if (pass == 0)
        recordUploads();
    const VkDeviceAddress wAddr = m_worldDev.address();

    auto args = [&](uint32_t a) {
        DispatchArgs2Gpu da{wAddr, a, 0u, 0u, 0u};
        return m_ring.write(&da, sizeof(da));
    };
    auto disp = [&](Pipeline &pl, int threads, int group, uint32_t a = 0) {
        const uint32_t groups = divUp(threads, group);
        if (groups == 0)
            return;
        m_cmd.dispatch(pl, groups, args(a));
        m_cmd.barrier();
    };

    // The step mutates body and joint state in place, so a pass that overflowed must be
    // undone: the first pass takes a copy, every rerun starts from it. A pending "wake all"
    // lands before the copy so a rerun keeps it.
    if (pass == 0)
    {
        if (m_wakeAllPending)
            disp(m_wakeAll, m_dynCount, 256);
        if (m_bPatchCount > 0)
            disp(m_applyBodyPatches, m_bPatchCount, 128);
        if (m_jPatchCount > 0)
            disp(m_applyJointPatches, m_jPatchCount, 128);
        if (m_impulseCount > 0)
            disp(m_applyImpulses, m_impulseCount, 128);
        saveState();
    }
    else
        restoreState();
    m_cmd.fill(m_counters, 0u);
    if (m_eventsOn)
        m_cmd.fill(m_events, 0u, 32);
    m_cmd.fill(m_ccdStat, 0u);
    m_cmd.fill(m_movedCount, 0u);
    m_cmd.barrier();

    // 1. Joint warm start, THEN integrate: a hard joint's C0 is its error at x^t. Taken at the
    //    warm-started pose it would hold the error the joint exists to remove.
    //    One dispatch: the joint part reads the pre-step pose copy (csIntegrateJointInit2D).
    disp(m_integrateJoint, std::max(m_bodyCount, m_jointSlots), 256);
    if (m_params.gravity.x != 0.0f)
        disp(m_gravityX, m_bodyCount, 256);
    m_cmd.writeTimestamp(1);

    // 2. Broadphase: grid, pairs, canonical order.
    disp(m_gridClear, m_tableSize, 256);
    disp(m_gridInsert, m_gridCount, 256);
    disp(m_findPairs, m_gridDyn, 128);
    // Each oversize body against every grid member, every big static and the oversize bodies after it.
    disp(m_oversizePairs, m_overCount * (m_gridCount + m_statCount + m_overCount), 128);
    if (m_smallLimit > 0 && m_pairCap <= 4096)
        m_cmd.dispatch(m_sortSmall, 1, args(0)); // one workgroup; same sorted keys as the radix sort
    else
        radixSortKeysU64(m_cmd, m_ring, m_primPl, m_radix, m_pairsRaw, m_pairsSorted, (uint32_t)m_pairCap,
                         m_counters.address());
    m_cmd.barrier();
    m_cmd.writeTimestamp(2);

    // 3. Narrowphase and the persistent contact cache, then wake what the contacts and joints
    //    say must wake (before the graph is built, which leaves sleepers out).
    disp(m_narrowphase, m_pairCap, 128);
    if (m_eventsOn)
    {
        disp(m_eventBegin, m_pairCap, 128);
        disp(m_eventEnd, m_pairCap, 128);
    }
    const bool sleepOn = sleepActive();
    if (sleepOn)
    {
        // Fingerprints of this step's contacts and joints, then the wake requests: a mover
        // touching a sleeper, a sleeper whose fingerprint changed, an awake joint partner (a
        // few links per step), the grab.
        disp(m_touchClear, m_dynCount, 256);
        disp(m_touchAccumC, m_pairCap, 128);
        disp(m_touchAccumJ, m_jointSlots, 128);
        disp(m_wakeReqC, m_pairCap, 128);
        disp(m_wakeLost, m_dynCount, 256);
        const int rounds = m_jointCount > 0 ? kJointWakeRounds : 1; // 1: the grab only
        for (int r = 0; r < rounds; r++)
        {
            disp(m_wakeReqJ, m_jointSlots, 128);
            disp(m_wakeApply, m_dynCount, 256);
        }
    }
    m_cmd.writeTimestamp(3);

    // 4. Constraint graph and colouring. A small world builds all of it in one workgroup
    //    (constraints2d.slang, same result as the shared build); a large one uses the shared
    //    multi-dispatch build. The choice is host-side, from the dynamic body count.
    const bool fused = m_smallLimit > 0 && m_dynCount <= m_smallLimit;
    m_lastFused = fused;
    m_lastSmallBuild = fused && m_bodyCount <= kSmallBuildMaxBodies && m_dynCount <= 1024;
    const ManSet2Gpu cur = W.cur;
    if (m_lastSmallBuild)
    {
        const VkConstraintGraph g = m_constraints.graph();
        BuildArgs2Gpu ba{};
        ba.mBodyA = cur.bodyA;
        ba.mBodyB = cur.bodyB;
        ba.mCount = cur.count;
        ba.jBodyA = m_jA.address();
        ba.jBodyB = m_jB.address();
        ba.jBroken = m_jBroken.address();
        ba.mass = m_mass.address();
        ba.awake = m_awake.address();
        ba.dyn = m_dynIdx.address();
        ba.keys = g.key;
        ba.isA = g.isA;
        ba.rowStart = g.start;
        ba.rowEnd = g.end;
        ba.color = m_constraints.bodyColor().address();
        ba.colorBodies = m_constraints.colorBodies().address();
        ba.colorStart = m_constraints.colorStartAddr();
        ba.counters = m_counters.address();
        ba.slots = (uint32_t)m_pairCap;
        ba.jointSlots = (uint32_t)m_jointSlots;
        ba.bodyCount = (uint32_t)m_bodyCount;
        ba.dynCount = (uint32_t)m_dynCount;
        ba.capacity = m_constraints.entryCapacity();
        m_cmd.dispatch(m_buildSmall, 1, m_ring.write(&ba, sizeof(ba)));
        m_cmd.barrier();
    }
    else
    {
        VkConstraintsBodyRefs bodyRefs;
        bodyRefs.mass = m_mass.address();
        bodyRefs.awake = m_awake.address();
        VkConstraintsManifoldRefs manRefs;
        manRefs.bodyA = cur.bodyA;
        manRefs.bodyB = cur.bodyB;
        manRefs.contactCount = cur.count;
        manRefs.slots = m_pairCap;
        VkConstraintsJointRefs jointRefs;
        jointRefs.bodyA = m_jA.address();
        jointRefs.bodyB = m_jB.address();
        jointRefs.broken = m_jBroken.address();
        jointRefs.count = m_jointSlots;
        VkConstraintsSpringRefs springRefs;
        m_constraints.recordBuild(m_cmd, m_ring, m_csPl, m_primPl, m_radix, bodyRefs, manRefs, jointRefs, springRefs,
                                  m_dynIdx, m_dynCount, 0, jpRounds);
    }
    const int timed = std::min(iters, kMaxTimedIterations);
    if (fused)
    {
        m_cmd.barrier();
        m_cmd.writeTimestamp(4);
        m_cmd.dispatch(m_iterate, 1, args((uint32_t)iters));
        m_cmd.barrier();
        for (int it = 0; it < timed; it++)
        {
            m_cmd.writeTimestamp((uint32_t)(5 + 2 * it));
            m_cmd.writeTimestamp((uint32_t)(6 + 2 * it));
        }
    }
    else
    {
    m_cmd.fill(m_colorIndirect, 0u);
    m_cmd.barrier();
    disp(m_colorArgs, colorBound, 128);
    m_cmd.barrierIndirect();
    m_cmd.writeTimestamp(4);

    // 5. Iterations: Gauss-Seidel over the colours, then the dual update.
    for (int it = 0; it < iters; it++)
    {
        for (int c = 0; c < colorBound; c++)
        {
            m_cmd.dispatchIndirect(m_primal, m_colorIndirect, (VkDeviceSize)c * 12, args((uint32_t)c));
            m_cmd.barrier();
        }
        if (it < kMaxTimedIterations)
            m_cmd.writeTimestamp((uint32_t)(5 + 2 * it));
        disp(m_dualAll, std::max(m_pairCap, m_jointSlots), 128);
        if (it < kMaxTimedIterations)
            m_cmd.writeTimestamp((uint32_t)(6 + 2 * it));
    }
    }

    // 6. Velocities, the guard, then sleep.
    if (m_eventsOn)
    {
        disp(m_eventHit, m_pairCap, 128);
        disp(m_eventJoint, m_jointSlots, 128, (uint32_t)kGrabSlotCount);
    }
    disp(m_velocityGuard, m_bodyCount, 256); // velocities, then the guard per body
    // Continuous collision, a pose clamp after the solve (the solve itself is untouched): flag the
    // fast bodies, compact them, sweep only those (indirect). Non-bullets go first against statics;
    // bullets then also see the dynamic bodies at their final poses.
    if (ccdOn)
    {
        disp(m_ccdFlagPl, m_dynCount, 256);
        selectFlagged(m_cmd, m_ring, m_primPl, m_select, m_dynIdx, m_ccdFlag, m_ccdList, m_ccdCount,
                      (uint32_t)m_dynCount);
        m_cmd.barrier();
        disp(m_ccdArgsPl, 1, 1);
        m_cmd.barrierIndirect();
        for (uint32_t bulletPass = 0; bulletPass < 2; bulletPass++)
        {
            m_cmd.dispatchIndirect(m_ccdPl, m_ccdArgs, 0, args(bulletPass));
            m_cmd.barrier();
        }
    }
    if (sleepOn)
    {
        disp(m_sleepPrep, m_dynCount, 256);
        disp(m_sleepBlockPre, m_jointSlots, 128);
        disp(m_sleepMark, m_dynCount, 256);
        disp(m_sleepNbrC, m_pairCap, 128);
        disp(m_sleepNbrJ, m_jointSlots, 128);
    }
    disp(m_sleepCommit, m_dynCount, 256); // also counts the awake bodies
    disp(m_compactMoved, m_bodyCount, 256);

    // 7. Read-backs, in the same submission.
    m_cmd.copyToHost(m_counters, m_statusHost, 32);
    if (m_eventsOn)
        m_cmd.copyToHost(m_events, m_eventsHost, (VkDeviceSize)m_evBytes);
    m_cmd.copyToHost(m_ccdStat, m_ccdHost, 8);
    m_cmd.copyToHost(m_movedCount, m_movedCountHost, 4);
    m_cmd.copyToHost(m_movedDev, m_movedHost, (VkDeviceSize)m_bodyCount * 32);
    if (m_jointForceReadback)
        m_cmd.copyToHost(m_jForce, m_jForceHost, (VkDeviceSize)m_jointSlots * 12);
    if (m_fullReadback)
    {
        m_cmd.copyToHost(m_pos, m_posHost, (VkDeviceSize)m_bodyCount * 8);
        m_cmd.copyToHost(m_ang, m_angHost, (VkDeviceSize)m_bodyCount * 4);
    }
    if (m_extras)
    {
        m_cmd.copyToHost(m_vel, m_velHost, (VkDeviceSize)m_bodyCount * 8);
        m_cmd.copyToHost(m_awake, m_awakeHost, (VkDeviceSize)m_bodyCount);
        m_cmd.copyToHost(m_constraints.bodyColor(), m_colorHost, (VkDeviceSize)m_bodyCount * 4);
    }
    m_cmd.writeTimestamp((uint32_t)(5 + 2 * timed));

    if (m_ring.wrapped())
    {
        fprintf(stderr, "[avbd2d] step: argument ring wrapped within one submission\n");
        throw avbdvk::VkError(VK_ERROR_UNKNOWN, "step: argument ring wrapped within one submission");
    }
}

void VkWorld2D::refreshHostCopies()
{
    const size_t n = (size_t)m_bodyCount;
    // The moved list first: it is the only body read-back when the full copies are off.
    m_moved2d.clear();
    const int count = std::min(*m_movedCountHost.mappedAs<int32_t>(), m_bodyCount);
    const uint32_t *rec = m_movedHost.mappedAs<uint32_t>();
    for (int k = 0; k < count; k++, rec += 8)
    {
        float f[6];
        std::memcpy(f, rec + 2, sizeof(f));
        MovedBody2D m;
        m.body = (int)rec[0];
        if (m.body < 0 || m.body >= m_bodyCount)
            continue;
        m.awake = (rec[1] & 1u) != 0;
        m.fellAsleep = (rec[1] & 2u) != 0;
        m.pos = Vec2(f[0], f[1]);
        m.angle = f[2];
        m.vel = Vec2(f[3], f[4]);
        m.angVel = f[5];
        m_moved2d.push_back(m);
        const size_t b = (size_t)m.body;
        m_hostPos[b] = m.pos;
        m_hostAng[b] = m.angle;
        m_hostVel[b] = m.vel;
        m_hostAngVel[b] = m.angVel;
        m_hostAwake[b] = m.awake ? 1 : 0;
    }
    if (m_fullReadback)
    {
        memcpy(m_hostPos.data(), m_posHost.mapped(), n * sizeof(Vec2));
        memcpy(m_hostAng.data(), m_angHost.mapped(), n * sizeof(float));
    }
    if (m_extras)
    {
        memcpy(m_hostVel.data(), m_velHost.mapped(), n * sizeof(Vec2));
        memcpy(m_hostAwake.data(), m_awakeHost.mapped(), n);
        memcpy(m_hostColor.data(), m_colorHost.mapped(), n * sizeof(int));
    }
    // A pose or velocity queued while the step was in flight stays what the getters report.
    for (const BodyPatch &p : m_bodyPatches)
    {
        const size_t b = p.body;
        if (b >= n)
            continue;
        if (p.flags & 1u)
        {
            m_hostPos[b] = Vec2(p.pose[0], p.pose[1]);
            m_hostAng[b] = p.pose[2];
        }
        if (p.flags & 2u)
        {
            m_hostVel[b] = Vec2(p.vel[0], p.vel[1]);
            m_hostAngVel[b] = p.vel[2];
        }
    }
}

void VkWorld2D::readTimings(int iterations)
{
    const int timed = std::min(std::max(iterations, 1), kMaxTimedIterations);
    auto ms = [&](uint32_t a, uint32_t b) {
        const double ns = m_cmd.timestampDeltaNs(a, b);
        return ns < 0.0 ? -1.0f : (float)(ns * 1e-6);
    };
    StepTiming2D t;
    t.gpuMs = ms(0, (uint32_t)(5 + 2 * timed));
    t.valid = t.gpuMs >= 0.0f;
    if (!t.valid)
    {
        m_timing = t;
        return;
    }
    t.integrateMs = ms(0, 1);
    t.broadMs = ms(1, 2);
    t.narrowMs = ms(2, 3);
    t.buildMs = ms(3, 4);
    float primal = 0.0f, dual = 0.0f;
    for (int it = 0; it < timed; it++)
    {
        const uint32_t before = it == 0 ? 4u : (uint32_t)(6 + 2 * (it - 1));
        primal += ms(before, (uint32_t)(5 + 2 * it));
        dual += ms((uint32_t)(5 + 2 * it), (uint32_t)(6 + 2 * it));
    }
    t.primalMs = primal;
    t.dualMs = dual;
    t.tailMs = ms((uint32_t)(4 + 2 * timed), (uint32_t)(5 + 2 * timed));
    m_timing = t;
}

StepResult2D VkWorld2D::step()
{
    stepBegin();
    return stepEnd();
}

void VkWorld2D::stepBegin()
{
    if (m_stepActive)
        return;
    m_stepResult = StepResult2D();
    if (m_bodyCount == 0 || m_dynCount == 0)
    {
        // Nothing to solve. Queued mutations stay queued until there is a dynamic body.
        flushUploadsNow();
        if (m_prevPairs == 0)
        {
            m_retiredMark = m_retired.size();
            releaseRetired();
        }
        m_moved2d.clear();
        m_result = m_stepResult;
        return;
    }
    // The staging buffer is sized by liveReserve; an edit that outgrew it goes up synchronously.
    if (m_upData.size() > (size_t)m_upStage.size())
        flushUploadsNow();
    m_retiredMark = m_retired.size();
    m_stepT0 = Clock::now();
    m_stepSubmits0 = Device::submitCount();

    // Switching sleep off must not leave anything frozen.
    if (m_sleepWasOn && !sleepActive())
        m_wakeAllPending = true;
    m_sleepWasOn = sleepActive();

    m_stepColorBound = std::min(std::max(4, m_lastColors + 2), kMaxColors);
    m_stepJpRounds = m_constraints.jpBound();
    uploadPatches();
    recordPass(0, m_stepColorBound, m_stepJpRounds);
    m_wakeAllPending = false;
    // The patches and the explosion impulses are applied by pass 0 only: a rerun restores the
    // state copied after them.
    m_bPatchCount = m_jPatchCount = 0;
    m_stepFirst = true;
    m_cmd.submit();
    m_stepActive = true;
}

StepResult2D VkWorld2D::stepEnd()
{
    if (!m_stepActive)
        return m_result;
    m_stepActive = false;
    StepResult2D r;
    int colorBound = m_stepColorBound;
    int jpRounds = m_stepJpRounds;
    const Clock::time_point t0 = m_stepT0;
    const uint64_t submits0 = m_stepSubmits0;

    for (int pass = 0;; pass++)
    {
        if (pass > kMaxPasses)
        {
            fprintf(stderr, "[avbd2d] step: %d reruns without converging\n", pass);
            throw avbdvk::VkError(VK_ERROR_UNKNOWN, "step: reruns without converging");
        }
        if (pass > 0)
        {
            recordPass(pass, colorBound, jpRounds);
            m_cmd.submit();
        }
        m_cmd.wait();

        const int32_t *st = m_statusHost.mappedAs<int32_t>();
        const int pairsReq = st[0];
        const int contactsReq = st[1];
        m_constraints.finishRecorded();
        const int builtColors = m_lastSmallBuild ? st[5] : m_constraints.colorCount();

        const bool pairShort = pairsReq > m_pairCap;
        const bool contactShort = contactsReq > m_contactCap;
        const bool jpShort = !m_lastSmallBuild && !m_constraints.jpConverged() && jpRounds < VkConstraints::kMaxJpRounds;
        const bool colorShort = colorBound < kMaxColors && builtColors > colorBound;

        if (pairShort || contactShort)
        {
            const int newPairs = pairShort ? cappedGrow(pairsReq, 256, pairCeiling()) : m_pairCap;
            const int newContacts = std::min(
                std::max(contactShort ? cappedGrow(contactsReq, 256, contactCeiling()) : m_contactCap, 2 * newPairs),
                contactCeiling());
            bool grew = newPairs > m_pairCap || newContacts > m_contactCap;
            const char *refusedBy = nullptr;
            if (grew && m_dev.memoryBudget() != 0)
            {
                // Pre-flight against Device::setMemoryBudget (conservative bytes per pair/contact
                // across both manifold sets, the sorted pair lists and the constraint graph). A
                // refused growth is the ceiling: the pass commits clamped, nothing is allocated.
                const uint64_t delta = 160ull * (uint64_t)std::max(newPairs - m_pairCap, 0) +
                                       100ull * (uint64_t)std::max(newContacts - m_contactCap, 0);
                if (m_dev.allocatedBytes() + delta > m_dev.memoryBudget())
                {
                    grew = false;
                    refusedBy = "memory budget";
                }
            }
            if (grew)
            {
                setCapacity(newPairs, newContacts, true);
                r.capacityReruns++;
                continue;
            }
            // At the ceiling: another identical pass would overflow again. Commit clamped.
            r.ok = false;
            r.truncated = true;
            r.failResource = refusedBy ? refusedBy : pairShort ? "pairs" : "contacts";
        }
        if (jpShort || colorShort)
        {
            if (jpShort)
            {
                jpRounds = std::min(2 * jpRounds, (int)VkConstraints::kMaxJpRounds);
                r.jpReruns++;
            }
            if (colorShort)
            {
                colorBound = std::min(builtColors + 2, kMaxColors);
                r.colorReruns++;
            }
            continue;
        }
        if (r.truncated || (!pairShort && !contactShort))
        {
            // Commit: this pass's manifolds become the next step's warm start.
            m_prevPairs = std::min(pairsReq, m_pairCap);
            m_lastColors = builtColors;
            r.pairs = m_prevPairs;
            r.contacts = std::min(contactsReq, m_contactCap);
            r.colors = m_lastColors;
            r.awake = st[2];
            r.removed = st[3];
            r.brokenJoints = st[4];
            m_awakeNow = st[2];
            m_removedTotal += st[3];
            m_brokenTotal += st[4];
            const int32_t *cs = m_ccdHost.mappedAs<int32_t>();
            m_ccdStats.clamped = cs[0];
            m_ccdStats.truncated = cs[1];
            m_curIdx ^= 1;
            break;
        }
    }

    m_impulseCount = 0;
    m_pendingImpulses.clear();
    m_query.invalidate();
    releaseRetired();
    collectEvents();
    r.eventsTruncated = 0;
    for (int k = 0; k < EVENT2D_KINDS; k++)
        if (m_events2d.truncated[k])
            r.eventsTruncated |= 1 << k;

    readTimings(m_params.iterations);
    refreshHostCopies();
    r.submits = (int)(Device::submitCount() - submits0);
    r.stepMs = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    m_result = r;
    return r;
}

// ---------------------------------------------------------------------------
// Mutation queue
// ---------------------------------------------------------------------------
namespace
{
constexpr uint32_t BP_SET_POSE = 1, BP_SET_VEL = 2, BP_ADD_VEL = 4, BP_WAKE = 8, BP_DISABLE = 16, BP_ENABLE = 32,
                   BP_SLEEP = 64;
constexpr uint32_t JP_MOTOR = 1, JP_LIMIT = 2, JP_LIMIT_ON = 4, JP_BREAK = 8, JP_WAKE = 16;
} // namespace

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
    if ((int)m_jointPatchSlot.size() < m_jointCount)
        m_jointPatchSlot.resize((size_t)m_jointCount, -1);
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

void VkWorld2D::queueJointBreak(int joint)
{
    if (joint >= 0 && joint < m_jointCount)
        jointPatch(joint).flags |= JP_BREAK;
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
    const ManBufs &m = m_man[1 - m_curIdx]; // the last committed step
    std::vector<int> bodyA((size_t)pairs), bodyB((size_t)pairs), cnt((size_t)pairs), off((size_t)pairs);
    std::vector<uint64_t> key((size_t)pairs);
    std::vector<Vec2> nrm((size_t)pairs), rA((size_t)contacts), rB((size_t)contacts), lam((size_t)contacts);
    m.key.readback(key.data(), (VkDeviceSize)pairs * 8);
    m.bodyA.readback(bodyA.data(), (VkDeviceSize)pairs * 4);
    m.bodyB.readback(bodyB.data(), (VkDeviceSize)pairs * 4);
    m.count.readback(cnt.data(), (VkDeviceSize)pairs * 4);
    m.offset.readback(off.data(), (VkDeviceSize)pairs * 4);
    m.normal.readback(nrm.data(), (VkDeviceSize)pairs * 8);
    m.rA.readback(rA.data(), (VkDeviceSize)contacts * 8);
    m.rB.readback(rB.data(), (VkDeviceSize)contacts * 8);
    m.lambda.readback(lam.data(), (VkDeviceSize)contacts * 8);
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
            d.n = nrm[t];
            d.force = std::fabs(lam[o].x);
            d.sep = dot(nrm[t], xA - xB);
            d.bodyA = a;
            d.bodyB = b;
            out.push_back(d);
        }
    }
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
    std::vector<float> kLin((size_t)K, 0.0f), zero((size_t)K * 9, 0.0f);
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
        broken[g] = 0;
    }
    const VkDeviceSize f = (VkDeviceSize)first;
    m_jB.upload(jB.data(), (size_t)K * 4, f * 4);
    m_jRA.upload(jRA.data(), (size_t)K * 8, f * 8);
    m_jRB.upload(jRB.data(), (size_t)K * 8, f * 8);
    // The grab: x and y are ramped springs (no frequency) on the world frame, the angle is free.
    std::vector<JointAxis2> axes((size_t)K * 3);
    for (size_t g = 0; g < (size_t)K; g++)
        for (size_t k = 0; k < 2; k++)
        {
            JointAxis2 &a = axes[g * 3 + k];
            a.pos = JOINT_AXIS_SPRING;
            a.stiff = kLin[g];
            a.flags = k == 0 ? JOINT_FLAG_GRAB : 0u;
        }
    m_jAxis.upload(axes.data(), (size_t)K * 3 * sizeof(JointAxis2), f * 3 * sizeof(JointAxis2));
    m_jFrameA.upload(zero.data(), (size_t)K * 4, f * 4);
    m_jRest.upload(zero.data(), (size_t)K * 4, f * 4);
    m_jArm.upload(zero.data(), (size_t)K * 4, f * 4);
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
            auto idOf = [&](int b) {
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
    for (size_t g = 0; g < group.size(); g++)
        if (group[g] != body)
            m_grabOffset[g] = m_hostPos[(size_t)group[g]] - p;
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
    Vec2 *t = (Vec2 *)m_grabStage.mapped();
    for (size_t g = 0; g < m_grabGroup.size(); g++)
    {
        const size_t b = (size_t)m_grabGroup[g];
        const Vec2 anchor = m_hostPos[b] + rotate(m_hostAng[b], m_grabLocal[g]);
        Vec2 d = m_grabCursor + m_grabOffset[g] - anchor;
        const float len = length(d);
        if (len > maxStep)
            d = d * (maxStep / len);
        t[g] = anchor + d;
    }
    m_cmd.copy(m_grabStage, m_jRA, (VkDeviceSize)m_grabGroup.size() * sizeof(Vec2));
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

QueryView2D VkWorld2D::queryView()
{
    QueryView2D v;
    v.scene = &m_scene;
    v.pos = m_hostPos.data();
    v.ang = m_hostAng.data();
    return v;
}

bool VkWorld2D::castRayClosest(Vec2 origin, Vec2 translation, RayHit2D &out)
{
    return m_init && m_query.castRayClosest(queryView(), origin, translation, out);
}
void VkWorld2D::castRay(Vec2 origin, Vec2 translation, const Query2D::CastFn &fn)
{
    if (m_init)
        m_query.castRay(queryView(), origin, translation, fn);
}
void VkWorld2D::overlapAABB(Vec2 lo, Vec2 hi, const Query2D::ShapeFn &fn)
{
    if (m_init)
        m_query.overlapAABB(queryView(), lo, hi, fn);
}
void VkWorld2D::overlapShape(const Proxy2D &proxy, const Query2D::ShapeFn &fn)
{
    if (m_init)
        m_query.overlapShape(queryView(), proxy, fn);
}
bool VkWorld2D::castShapeClosest(const Proxy2D &proxy, Vec2 translation, RayHit2D &out)
{
    return m_init && m_query.castShapeClosest(queryView(), proxy, translation, out);
}
void VkWorld2D::castShape(const Proxy2D &proxy, Vec2 translation, const Query2D::CastFn &fn)
{
    if (m_init)
        m_query.castShape(queryView(), proxy, translation, fn);
}

// ---------------------------------------------------------------------------
// Batched queries
// ---------------------------------------------------------------------------
namespace
{
void growMapped(MappedBuffer &b, Device &dev, VkDeviceSize bytes, MappedBuffer::HostAccess access)
{
    bytes = std::max<VkDeviceSize>(bytes, 256);
    if (b.size() >= bytes)
        return;
    b.destroy();
    b.init(dev, bytes + bytes / 2, 0, access);
}
} // namespace

// Runs the clear / insert / (rays|boxes) kernels in one submission and waits. The caller has
// checked that no step is in flight. Results land in m_qOutHit / m_qOutIds / m_qOutCounts.
bool VkWorld2D::runQueryBatch(bool rays, const float *in, int count, uint64_t cat, uint64_t mask, int maxPer)
{
    const int shapes = m_scene.shapeTotal();
    const uint32_t insertCount = (uint32_t)(m_gridCount + m_overCount + m_statCount);
    uint32_t table = 64;
    while (table < 2u * (uint32_t)std::max(m_gridCount, 1))
        table <<= 1;

    auto sz = [&](Buffer &b, size_t bytes) {
        if (b.size() < bytes)
            b.resize(std::max<size_t>(bytes + bytes / 2, 256));
    };
    sz(m_qBox, (size_t)shapes * 16);
    sz(m_qHead, (size_t)table * 4);
    sz(m_qNext, (size_t)shapes * 16);
    sz(m_qBig, (size_t)shapes * 4);
    sz(m_qCount, 16);
    if (rays)
    {
        growMapped(m_qRays, m_dev, (VkDeviceSize)count * 16, MappedBuffer::HostAccess::Write);
        growMapped(m_qOutHit, m_dev, (VkDeviceSize)count * 32, MappedBuffer::HostAccess::Read);
        growMapped(m_qAabbs, m_dev, 16, MappedBuffer::HostAccess::Write);
        growMapped(m_qOutIds, m_dev, 16, MappedBuffer::HostAccess::Read);
        growMapped(m_qOutCounts, m_dev, 16, MappedBuffer::HostAccess::Read);
        memcpy(m_qRays.mapped(), in, (size_t)count * 16);
    }
    else
    {
        growMapped(m_qAabbs, m_dev, (VkDeviceSize)count * 16, MappedBuffer::HostAccess::Write);
        growMapped(m_qOutIds, m_dev, (VkDeviceSize)count * (size_t)std::max(maxPer, 1) * 4,
                   MappedBuffer::HostAccess::Read);
        growMapped(m_qOutCounts, m_dev, (VkDeviceSize)count * 4, MappedBuffer::HostAccess::Read);
        growMapped(m_qRays, m_dev, 16, MappedBuffer::HostAccess::Write);
        growMapped(m_qOutHit, m_dev, 32, MappedBuffer::HostAccess::Read);
        memcpy(m_qAabbs.mapped(), in, (size_t)count * 16);
    }
    // The kernels write device-local buffers; the copy engine moves them to the mapped ones.
    // Shader stores straight into host-cached memory cost ~1 us each (see csCompactMoved2D).
    const size_t hitBytes = rays ? (size_t)count * 32 : 32;
    const size_t idBytes = rays ? 16 : (size_t)count * (size_t)std::max(maxPer, 1) * 4;
    const size_t countBytes = rays ? 16 : (size_t)count * 4;
    sz(m_qOutHitDev, hitBytes);
    sz(m_qOutIdsDev, idBytes);
    sz(m_qOutCountsDev, countBytes);

    m_ring.reset();
    m_cmd.begin();
    const World2Gpu W = makeWorld(4);
    memcpy(m_worldStage.mapped(), &W, sizeof(W));
    m_cmd.copy(m_worldStage, m_worldDev, sizeof(W));
    m_cmd.barrier();

    QueryBatch2Gpu Q{};
    Q.w = m_worldDev.address();
    Q.qBox = m_qBox.address();
    Q.qHead = m_qHead.address();
    Q.qNext = m_qNext.address();
    Q.qBig = m_qBig.address();
    Q.qCount = m_qCount.address();
    Q.rays = m_qRays.address();
    Q.aabbs = m_qAabbs.address();
    Q.outHit = m_qOutHitDev.address();
    Q.outIds = m_qOutIdsDev.address();
    Q.outCounts = m_qOutCountsDev.address();
    Q.cat = cat;
    Q.mask = mask;
    Q.headMask = table - 1;
    Q.count = (uint32_t)count;
    Q.maxPer = (uint32_t)std::max(maxPer, 1);
    Q.insertCount = insertCount;
    const VkDeviceAddress qa = m_ring.write(&Q, sizeof(Q));
    auto groups = [](uint32_t n, uint32_t g) { return (n + g - 1) / g; };
    m_cmd.dispatch(m_queryClear, groups(table, 256), qa);
    m_cmd.barrier();
    if (insertCount > 0)
    {
        m_cmd.dispatch(m_queryInsert, groups(insertCount, 256), qa);
        m_cmd.barrier();
    }
    m_cmd.dispatch(rays ? m_queryRays : m_queryAabbs, groups((uint32_t)count, 64), qa);
    if (rays)
        m_cmd.copyToHost(m_qOutHitDev, m_qOutHit, hitBytes);
    else
    {
        m_cmd.copyToHost(m_qOutCountsDev, m_qOutCounts, countBytes);
        m_cmd.copyToHost(m_qOutIdsDev, m_qOutIds, idBytes);
    }
    m_cmd.submit();
    m_cmd.wait();
    return true;
}

bool VkWorld2D::castRaysBatch(const RayQuery2D *rays, int count, uint64_t cat, uint64_t mask, RayBatchHit2D *out)
{
    static_assert(sizeof(RayQuery2D) == 16, "RayQuery2D must pack as float4");
    for (int i = 0; i < count; i++)
        out[i] = RayBatchHit2D();
    if (!m_init || count <= 0)
        return false;
    const bool device = !m_stepActive && !uploadsPending() && m_bodyCount > 0 && m_scene.shapeTotal() > 0;
    if (!device)
    {
        for (int i = 0; i < count; i++)
        {
            RayHit2D best;
            bool found = false;
            m_query.castRay(queryView(), rays[i].origin, rays[i].translation, [&](const RayHit2D &h) {
                const size_t s = (size_t)h.shape;
                if ((m_scene.shCat[s] & mask) == 0 || (cat & m_scene.shMask[s]) == 0)
                    return -1.0f;
                best = h;
                found = true;
                return h.fraction;
            });
            if (found)
            {
                out[i].shape = best.shape;
                out[i].fraction = best.fraction;
                out[i].point = best.point;
                out[i].normal = best.normal;
            }
        }
        return false;
    }
    runQueryBatch(true, reinterpret_cast<const float *>(rays), count, cat, mask, 1);
    const float *r = m_qOutHit.mappedAs<float>();
    for (int i = 0; i < count; i++)
    {
        int shape;
        memcpy(&shape, r + (size_t)i * 8, 4);
        if (shape < 0)
            continue;
        out[i].shape = shape;
        out[i].fraction = r[(size_t)i * 8 + 1];
        out[i].point = Vec2(r[(size_t)i * 8 + 2], r[(size_t)i * 8 + 3]);
        out[i].normal = Vec2(r[(size_t)i * 8 + 4], r[(size_t)i * 8 + 5]);
    }
    return true;
}

bool VkWorld2D::overlapAabbsBatch(const Vec2 *lo, const Vec2 *hi, int count, uint64_t cat, uint64_t mask, int maxPer,
                                  int *outIds, int *outCounts)
{
    for (int i = 0; i < count; i++)
        outCounts[i] = 0;
    if (!m_init || count <= 0)
        return false;
    maxPer = std::max(maxPer, 1);
    const bool device = !m_stepActive && !uploadsPending() && m_bodyCount > 0 && m_scene.shapeTotal() > 0;
    if (!device)
    {
        for (int i = 0; i < count; i++)
        {
            int n = 0;
            m_query.overlapAABB(queryView(), lo[i], hi[i], [&](int s) {
                if ((m_scene.shCat[(size_t)s] & mask) != 0 && (cat & m_scene.shMask[(size_t)s]) != 0)
                {
                    if (n < maxPer)
                        outIds[(size_t)i * (size_t)maxPer + (size_t)n] = s;
                    n++;
                }
                return true;
            });
            outCounts[i] = n;
        }
        return false;
    }
    std::vector<float> boxes((size_t)count * 4);
    for (int i = 0; i < count; i++)
    {
        boxes[(size_t)i * 4 + 0] = lo[i].x;
        boxes[(size_t)i * 4 + 1] = lo[i].y;
        boxes[(size_t)i * 4 + 2] = hi[i].x;
        boxes[(size_t)i * 4 + 3] = hi[i].y;
    }
    runQueryBatch(false, boxes.data(), count, cat, mask, maxPer);
    memcpy(outCounts, m_qOutCounts.mapped(), (size_t)count * 4);
    const int *ids = m_qOutIds.mappedAs<int>();
    for (int i = 0; i < count; i++)
    {
        const int n = std::min(outCounts[i], maxPer);
        for (int k = 0; k < n; k++)
            outIds[(size_t)i * (size_t)maxPer + (size_t)k] = ids[(size_t)i * (size_t)maxPer + (size_t)k];
    }
    return true;
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
