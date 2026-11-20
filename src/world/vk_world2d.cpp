#include "vk_world2d.h"

#include "vk_capacity.h"
#include "vk_world2d_internal.h"
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
    return {
        &m_pos,          &m_ang,         &m_initPos,    &m_initAng,       &m_inertPos,  &m_inertAng,   &m_vel,
        &m_angVel,       &m_prevVel,     &m_moment,     &m_mass,          &m_awake,     &m_sleepTimer, &m_blocked,
        &m_dynIdx,       &m_statIdx,     &m_savePos,    &m_saveAng,       &m_subPos,    &m_subAng,     &m_saveVel,
        &m_saveAngVel,   &m_savePrevVel, &m_saveAwake,  &m_saveTimer,     &m_saveJPen,  &m_saveJLam,   &m_saveJBroken,
        &m_gridHead,     &m_gridNext,    &m_pairsRaw,   &m_pairsSorted,   &m_ignore,    &m_counters,   &m_jA,
        &m_jB,           &m_jRA,         &m_jRB,        &m_jC0,           &m_jPen,      &m_jLam,       &m_jAxis,
        &m_jFrameA,      &m_jRest,       &m_jArm,       &m_jFrac,         &m_jForce,    &m_jBroken,    &m_restPos,
        &m_restAng,      &m_cand,        &m_touchCount, &m_touchHash,     &m_snapCount, &m_snapHash,   &m_wakeReq,
        &m_saveRestPos,  &m_saveRestAng, &m_saveCand,   &m_colorIndirect, &m_worldDev,  &m_sBody,      &m_sType,
        &m_sOff,         &m_sAng,        &m_sRad,       &m_sHalf,         &m_sVR,       &m_sFric,      &m_verts,
        &m_norms,        &m_sFlags,      &m_events,     &m_impulses,      &m_bProps,    &m_sCat,       &m_sMask,
        &m_sGroup,       &m_sMat,        &m_gridList,   &m_overList,      &m_ccdFlag,   &m_ccdList,    &m_ccdCount,
        &m_ccdStat,      &m_ccdArgs,     &m_bShape,     &m_bExt,          &m_bBullet,   &m_movedCount, &m_movedDev,
        &m_qBox,         &m_qHead,       &m_qNext,      &m_qBig,          &m_qCount,    &m_qOutHitDev, &m_qOutIdsDev,
        &m_qOutCountsDev};
}

// Every buffer indexed by body, with its element size. Growth keeps the ones that carry state.
std::vector<VkWorld2D::BodyBuf> VkWorld2D::bodyBuffers()
{
    return {{&m_pos, 8, true},        {&m_ang, 4, true},         {&m_initPos, 8, true},      {&m_initAng, 4, true},
            {&m_inertPos, 8, true},   {&m_inertAng, 4, true},    {&m_vel, 8, true},          {&m_angVel, 4, true},
            {&m_prevVel, 8, true},    {&m_moment, 4, true},      {&m_mass, 4, true},         {&m_bProps, 16, true},
            {&m_awake, 1, true},      {&m_sleepTimer, 4, true},  {&m_blocked, 4, false},     {&m_dynIdx, 4, false},
            {&m_statIdx, 4, false},   {&m_savePos, 8, false},    {&m_saveAng, 4, false},     {&m_subPos, 8, false},
            {&m_subAng, 4, false},    {&m_saveVel, 8, false},    {&m_saveAngVel, 4, false},  {&m_savePrevVel, 8, false},
            {&m_saveAwake, 1, false}, {&m_saveTimer, 4, false},  {&m_restPos, 8, true},      {&m_restAng, 4, true},
            {&m_cand, 1, true},       {&m_touchCount, 4, false}, {&m_touchHash, 4, false},   {&m_snapCount, 4, true},
            {&m_snapHash, 4, true},   {&m_wakeReq, 1, false},    {&m_saveRestPos, 8, false}, {&m_saveRestAng, 4, false},
            {&m_saveCand, 1, false},  {&m_ccdFlag, 1, false},    {&m_ccdList, 4, false},     {&m_bShape, 8, true},
            {&m_bExt, 8, true},       {&m_bBullet, 1, true}};
}

// Every compute pipeline of the world: member, embedded SPIR-V module and entry point, named once.
// createPipelines() builds this table and destroy() walks it.
struct VkWorld2D::PipelineSpec2D
{
    Pipeline VkWorld2D::*member;
    const uint32_t *spirv;
    size_t words;
    const char *entry;
};

const VkWorld2D::PipelineSpec2D *VkWorld2D::pipelineSpecs(size_t &count)
{
    static const PipelineSpec2D specs[] = {
        {&VkWorld2D::m_integrate, integrate2d_spirv, integrate2d_spirv_words, "csIntegrate2D"},
        {&VkWorld2D::m_velocity, integrate2d_spirv, integrate2d_spirv_words, "csVelocity2D"},
        {&VkWorld2D::m_velocityGuard, integrate2d_spirv, integrate2d_spirv_words, "csVelocityGuard2D"},
        {&VkWorld2D::m_gravityX, integrate2d_spirv, integrate2d_spirv_words, "csGravityX2D"},
        {&VkWorld2D::m_ccdFlagPl, ccd2d_spirv, ccd2d_spirv_words, "csCcdFlag2D"},
        {&VkWorld2D::m_ccdArgsPl, ccd2d_spirv, ccd2d_spirv_words, "csCcdArgs2D"},
        {&VkWorld2D::m_ccdPl, ccd2d_spirv, ccd2d_spirv_words, "csCcd2D"},
        {&VkWorld2D::m_gridClear, broadphase2d_spirv, broadphase2d_spirv_words, "csGridClear2D"},
        {&VkWorld2D::m_gridInsert, broadphase2d_spirv, broadphase2d_spirv_words, "csGridInsert2D"},
        {&VkWorld2D::m_findPairs, broadphase2d_spirv, broadphase2d_spirv_words, "csFindPairs2D"},
        {&VkWorld2D::m_oversizePairs, broadphase2d_spirv, broadphase2d_spirv_words, "csOversizePairs2D"},
        {&VkWorld2D::m_narrowphase, narrowphase2d_spirv, narrowphase2d_spirv_words, "csNarrowphase2D"},
        {&VkWorld2D::m_jointInit, solver2d_spirv, solver2d_spirv_words, "csJointInit2D"},
        {&VkWorld2D::m_integrateJoint, solver2d_spirv, solver2d_spirv_words, "csIntegrateJointInit2D"},
        {&VkWorld2D::m_colorArgs, solver2d_spirv, solver2d_spirv_words, "csColorArgs2D"},
        {&VkWorld2D::m_primal, solver2d_spirv, solver2d_spirv_words, "csPrimal2D"},
        {&VkWorld2D::m_dualAll, solver2d_spirv, solver2d_spirv_words, "csDualAll2D"},
        {&VkWorld2D::m_iterate, solver2d_spirv, solver2d_spirv_words, "csIterate2D"},
        {&VkWorld2D::m_sortSmall, broadphase2d_spirv, broadphase2d_spirv_words, "csSortPairsSmall2D"},
        {&VkWorld2D::m_buildSmall, constraints2d_spirv, constraints2d_spirv_words, "csBuildSmall2D"},
        {&VkWorld2D::m_sleepPrep, sleep2d_spirv, sleep2d_spirv_words, "csSleepPrep2D"},
        {&VkWorld2D::m_sleepBlockPre, sleep2d_spirv, sleep2d_spirv_words, "csSleepBlockPre2D"},
        {&VkWorld2D::m_sleepMark, sleep2d_spirv, sleep2d_spirv_words, "csSleepMark2D"},
        {&VkWorld2D::m_sleepNbrC, sleep2d_spirv, sleep2d_spirv_words, "csSleepNbrContacts2D"},
        {&VkWorld2D::m_sleepNbrJ, sleep2d_spirv, sleep2d_spirv_words, "csSleepNbrJoints2D"},
        {&VkWorld2D::m_sleepCommit, sleep2d_spirv, sleep2d_spirv_words, "csSleepCommit2D"},
        {&VkWorld2D::m_touchClear, sleep2d_spirv, sleep2d_spirv_words, "csTouchClear2D"},
        {&VkWorld2D::m_touchAccumC, sleep2d_spirv, sleep2d_spirv_words, "csTouchAccumContacts2D"},
        {&VkWorld2D::m_touchAccumJ, sleep2d_spirv, sleep2d_spirv_words, "csTouchAccumJoints2D"},
        {&VkWorld2D::m_wakeReqC, sleep2d_spirv, sleep2d_spirv_words, "csWakeReqContacts2D"},
        {&VkWorld2D::m_wakeReqJ, sleep2d_spirv, sleep2d_spirv_words, "csWakeReqJoints2D"},
        {&VkWorld2D::m_wakeLost, sleep2d_spirv, sleep2d_spirv_words, "csWakeLost2D"},
        {&VkWorld2D::m_wakeApply, sleep2d_spirv, sleep2d_spirv_words, "csWakeApply2D"},
        {&VkWorld2D::m_wakeAll, sleep2d_spirv, sleep2d_spirv_words, "csWakeAll2D"},
        {&VkWorld2D::m_queryClear, query2d_spirv, query2d_spirv_words, "csQueryClear2D"},
        {&VkWorld2D::m_queryInsert, query2d_spirv, query2d_spirv_words, "csQueryInsert2D"},
        {&VkWorld2D::m_queryRays, query2d_spirv, query2d_spirv_words, "csQueryRays2D"},
        {&VkWorld2D::m_queryAabbs, query2d_spirv, query2d_spirv_words, "csQueryAabbs2D"},
        {&VkWorld2D::m_eventBegin, events2d_spirv, events2d_spirv_words, "csEventBegin2D"},
        {&VkWorld2D::m_eventEnd, events2d_spirv, events2d_spirv_words, "csEventEnd2D"},
        {&VkWorld2D::m_eventHit, events2d_spirv, events2d_spirv_words, "csEventHit2D"},
        {&VkWorld2D::m_eventJoint, events2d_spirv, events2d_spirv_words, "csEventJoint2D"},
        {&VkWorld2D::m_applyImpulses, events2d_spirv, events2d_spirv_words, "csApplyImpulses2D"},
        {&VkWorld2D::m_applyBodyPatches, events2d_spirv, events2d_spirv_words, "csApplyBodyPatches2D"},
        {&VkWorld2D::m_applyJointPatches, events2d_spirv, events2d_spirv_words, "csApplyJointPatches2D"},
        {&VkWorld2D::m_compactMoved, events2d_spirv, events2d_spirv_words, "csCompactMoved2D"},
    };
    count = sizeof(specs) / sizeof(specs[0]);
    return specs;
}

void VkWorld2D::createPipelines()
{
    size_t n = 0;
    const PipelineSpec2D *specs = pipelineSpecs(n);
    for (size_t i = 0; i < n; i++)
        (this->*specs[i].member).init(m_dev, m_layout, specs[i].spirv, specs[i].words, specs[i].entry, specs[i].entry);
}

void VkWorld2D::initDevice()
{
    if (m_dev.adopted() || m_dev.device())
        return;
    configureDevice();
    m_dev.init();
}

void VkWorld2D::init()
{
    if (m_init)
        return;
    if (const char *e = std::getenv("AVBD2D_SMALL_WORLD")) // bodies; 0 = multi-dispatch path only
        m_smallLimit = std::atoi(e);
    try
    {
        initDevice();
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
        m_worldStage.init(m_dev, sizeof(World2Gpu), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MappedBuffer::HostAccess::Write);
        m_grabStage.init(m_dev, (VkDeviceSize)kGrabSlotCount * (sizeof(Vec2) + sizeof(float)),
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MappedBuffer::HostAccess::Write);
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
    for (MappedBuffer *b :
         {&m_statusHost, &m_posHost,   &m_angHost,        &m_velHost,    &m_awakeHost, &m_colorHost,   &m_eventsHost,
          &m_ccdHost,    &m_qRays,     &m_qAabbs,         &m_qOutHit,    &m_qOutIds,   &m_qOutCounts,  &m_bPatchBuf,
          &m_jPatchBuf,  &m_movedHost, &m_movedCountHost, &m_jForceHost, &m_upStage,   &m_manDebugHost})
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
    size_t n = 0;
    const PipelineSpec2D *specs = pipelineSpecs(n);
    for (size_t i = 0; i < n; i++)
        (this->*specs[i].member).destroy();
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
    for (const E &e :
         {E{&m_jA, 4, true}, E{&m_jB, 4, true}, E{&m_jRA, 8, true}, E{&m_jRB, 8, true}, E{&m_jC0, 36, true},
          E{&m_jPen, 36, true}, E{&m_jLam, 36, true}, E{&m_jAxis, 3 * sizeof(JointAxis2), true}, E{&m_jFrameA, 4, true},
          E{&m_jRest, 4, true}, E{&m_jArm, 4, true}, E{&m_jFrac, 4, true}, E{&m_jForce, 12, true},
          E{&m_jBroken, 1, true}, E{&m_saveJPen, 36, false}, E{&m_saveJLam, 36, false}, E{&m_saveJBroken, 1, false}})
        sizeBuf(*e.b, J * e.elem, preserve && e.keep);
    m_jForceHost.destroy();
    m_jForceHost.init(m_dev, J * 12, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    std::memset(m_jForceHost.mapped(), 0, J * 12);
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
    m_grabStartAng.clear();
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
    m_prevColors = 0;
    m_diag = StepDiagnostics2D();
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
    sizeBuf(m_counters, 64, false);
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
    m_statusHost.init(m_dev, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
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
    W.subPos = m_subPos.address();
    W.subAng = m_subAng.address();
    W.bPatchCount = (uint32_t)m_bPatchCount;
    W.jPatchCount = (uint32_t)m_jPatchCount;
    W.substeps = (uint32_t)m_stepSubsteps;
    W.subDt = m_params.dt / (float)m_stepSubsteps;
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

} // namespace avbd2d
