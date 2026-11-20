#include "vk_world2d.h"
#include "vk_world2d_internal.h"
#include "vk_rerun.h"

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


// Pass state: counts and addresses fixed at record time, shared by the record* phases.
struct VkWorld2D::PassCtx
{
    int pass = 0, colorBound = 0, jpRounds = 0;
    int iters = 0, timed = 0, nSub = 1;
    bool ccdOn = false, sleepOn = false, fused = false;
    World2Gpu W{};
    ManSet2Gpu cur{};
    VkDeviceAddress wAddr = 0;
};

VkDeviceAddress VkWorld2D::ringArgs(const PassCtx &pc, uint32_t a)
{
    DispatchArgs2Gpu da{pc.wAddr, a, 0u, 0u, 0u};
    return m_ring.write(&da, sizeof(da));
}

void VkWorld2D::dispatchGrid(const PassCtx &pc, Pipeline &pl, int threads, int group, uint32_t a)
{
    const uint32_t groups = ceilDiv(threads, group);
    if (groups == 0)
        return;
    m_cmd.dispatch(pl, groups, ringArgs(pc, a));
    m_cmd.barrier();
}

void VkWorld2D::beginPass(PassCtx &pc)
{
    const SolverParams2D &P = m_params;
    pc.iters = std::max(P.iterations, 1);

    // Everything that can reallocate is grown before the first recorded command.
    reserveRadix(m_radix, (uint32_t)m_pairCap, 8, 0);
    m_constraints.prepare(m_radix, m_dynCount);
    pc.ccdOn = P.enableContinuous && m_dynCount > 0;
    if (pc.ccdOn)
    {
        reserveScan(m_select.scan, (uint32_t)m_dynCount);
        if (m_select.flagsU32.size() < (VkDeviceSize)m_dynCount * 4)
            m_select.flagsU32.resize((VkDeviceSize)m_dynCount * 4);
    }

    m_ring.reset();
    m_cmd.begin();
    m_cmd.writeTimestamp(0);

    pc.W = makeWorld(pc.colorBound);
    memcpy(m_worldStage.mapped(), &pc.W, sizeof(pc.W));
    m_cmd.copy(m_worldStage, m_worldDev, sizeof(pc.W));
    recordGrabTargets();
    m_cmd.barrier();
    // Queued live edits (created / destroyed bodies, shapes, joints) land before anything reads them.
    if (pc.pass == 0)
        recordUploads();
    pc.wAddr = m_worldDev.address();


    // The step mutates body and joint state in place, so a pc.pass that overflowed must be
    // undone: the first pc.pass takes a copy, every rerun starts from it. A pending "wake all"
    // lands before the copy so a rerun keeps it.
    if (pc.pass == 0)
    {
        if (m_wakeAllPending)
            dispatchGrid(pc, m_wakeAll, m_dynCount, 256);
        if (m_bPatchCount > 0)
            dispatchGrid(pc, m_applyBodyPatches, m_bPatchCount, 128);
        if (m_jPatchCount > 0)
            dispatchGrid(pc, m_applyJointPatches, m_jPatchCount, 128);
        if (m_impulseCount > 0)
            dispatchGrid(pc, m_applyImpulses, m_impulseCount, 128);
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

}

// Solver substeps: the contacts are found once and the solve runs nSub times with h = dt / nSub.
// A later substep starts with the joint init (C0 at the substep's pose, no extra warm-start decay)
// and the integrate, which rebuilds the inertial target from the velocity the last substep ended with.
void VkWorld2D::recordSubstepStart(PassCtx &pc, int s)
{
    dispatchGrid(pc, m_jointInit, m_jointSlots, 128, (uint32_t)s);
    dispatchGrid(pc, m_integrate, m_bodyCount, 256, (uint32_t)s);
    if (m_params.gravity.x != 0.0f)
        dispatchGrid(pc, m_gravityX, m_bodyCount, 256);
}

// Hit events read v^t, so they run once, right after substep 0's iterations, before any velocity pass.
void VkWorld2D::recordSubstepEvents(PassCtx &pc)
{
    if (m_eventsOn)
        dispatchGrid(pc, m_eventHit, m_pairCap, 128);
}

// Phases 1-3: joint warm start and integrate, broadphase, narrowphase and the wake requests.
void VkWorld2D::recordFront(PassCtx &pc)
{
    // 1. Joint warm start, THEN integrate: a hard joint's C0 is its error at x^t. Taken at the
    //    warm-started pose it would hold the error the joint exists to remove.
    //    One dispatch: the joint part reads the pre-step pose copy (csIntegrateJointInit2D).
    dispatchGrid(pc, m_integrateJoint, std::max(m_bodyCount, m_jointSlots), 256);
    if (m_params.gravity.x != 0.0f)
        dispatchGrid(pc, m_gravityX, m_bodyCount, 256);
    m_cmd.writeTimestamp(1);

    // 2. Broadphase: grid, pairs, canonical order.
    dispatchGrid(pc, m_gridClear, m_tableSize, 256);
    dispatchGrid(pc, m_gridInsert, m_gridCount, 256);
    dispatchGrid(pc, m_findPairs, m_gridDyn, 128);
    // Each oversize body against every grid member, every big static and the oversize bodies after it.
    dispatchGrid(pc, m_oversizePairs, m_overCount * (m_gridCount + m_statCount + m_overCount), 128);
    if (m_smallLimit > 0 && m_pairCap <= 4096)
        m_cmd.dispatch(m_sortSmall, 1, ringArgs(pc, 0)); // one workgroup; same sorted keys as the radix sort
    else
        radixSortKeysU64(m_cmd, m_ring, m_primPl, m_radix, m_pairsRaw, m_pairsSorted, (uint32_t)m_pairCap,
                         m_counters.address());
    m_cmd.barrier();
    m_cmd.writeTimestamp(2);

    // 3. Narrowphase and the persistent contact cache, then wake what the contacts and joints
    //    say must wake (before the graph is built, which leaves sleepers out).
    dispatchGrid(pc, m_narrowphase, m_pairCap, 128);
    if (m_eventsOn)
    {
        dispatchGrid(pc, m_eventBegin, m_pairCap, 128);
        dispatchGrid(pc, m_eventEnd, m_pairCap, 128);
    }
    pc.sleepOn = sleepActive();
    if (pc.sleepOn)
    {
        // Fingerprints of this step's contacts and joints, then the wake requests: a mover
        // touching a sleeper, a sleeper whose fingerprint changed, an awake joint partner (a
        // few links per step), the grab.
        dispatchGrid(pc, m_touchClear, m_dynCount, 256);
        dispatchGrid(pc, m_touchAccumC, m_pairCap, 128);
        dispatchGrid(pc, m_touchAccumJ, m_jointSlots, 128);
        dispatchGrid(pc, m_wakeReqC, m_pairCap, 128);
        dispatchGrid(pc, m_wakeLost, m_dynCount, 256);
        const int rounds = m_jointCount > 0 ? kJointWakeRounds : 1; // 1: the grab only
        for (int r = 0; r < rounds; r++)
        {
            dispatchGrid(pc, m_wakeReqJ, m_jointSlots, 128);
            dispatchGrid(pc, m_wakeApply, m_dynCount, 256);
        }
    }
    m_cmd.writeTimestamp(3);

}

// Phase 4: constraint graph and colouring. A small world builds all of it in one workgroup
// (constraints2d.slang, same result as the shared build); a large one uses the shared multi-dispatch
// build. The choice is host-side, from the dynamic body count.
void VkWorld2D::recordGraph(PassCtx &pc)
{
    // 4. Constraint graph and colouring. A small world builds all of it in one workgroup
    //    (constraints2d.slang, same result as the shared build); a large one uses the shared
    //    multi-dispatch build. The choice is host-side, from the dynamic body count.
    pc.fused = m_smallLimit > 0 && m_dynCount <= m_smallLimit;
    m_lastFused = pc.fused;
    m_lastSmallBuild = pc.fused && m_bodyCount <= kSmallBuildMaxBodies && m_dynCount <= 1024;
    pc.cur = pc.W.cur;
    if (m_lastSmallBuild)
    {
        const VkConstraintGraph g = m_constraints.graph();
        BuildArgs2Gpu ba{};
        ba.mBodyA = pc.cur.bodyA;
        ba.mBodyB = pc.cur.bodyB;
        ba.mCount = pc.cur.count;
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
        m_cmd.dispatchArgs(m_buildSmall, 1, m_ring, ba);
        m_cmd.barrier();
    }
    else
    {
        VkConstraintsBodyRefs bodyRefs;
        bodyRefs.mass = m_mass.address();
        bodyRefs.awake = m_awake.address();
        VkConstraintsManifoldRefs manRefs;
        manRefs.bodyA = pc.cur.bodyA;
        manRefs.bodyB = pc.cur.bodyB;
        manRefs.contactCount = pc.cur.count;
        manRefs.slots = m_pairCap;
        VkConstraintsJointRefs jointRefs;
        jointRefs.bodyA = m_jA.address();
        jointRefs.bodyB = m_jB.address();
        jointRefs.broken = m_jBroken.address();
        jointRefs.count = m_jointSlots;
        VkConstraintsSpringRefs springRefs;
        m_constraints.recordBuild(m_cmd, m_ring, m_csPl, m_primPl, m_radix, bodyRefs, manRefs, jointRefs, springRefs,
                                  m_dynIdx, m_dynCount, 0, pc.jpRounds);
    }
}

// Phase 5, small worlds: the whole iteration loop is one single-workgroup dispatch per substep.
void VkWorld2D::recordFusedSolve(PassCtx &pc)
{
    m_cmd.barrier();
    for (int s = 0; s < pc.nSub; s++)
    {
        if (s > 0)
            recordSubstepStart(pc, s);
        if (s == 0)
            m_cmd.writeTimestamp(4);
        m_cmd.dispatch(m_iterate, 1, ringArgs(pc, (uint32_t)pc.iters));
        m_cmd.barrier();
        if (s == 0)
        {
            for (int it = 0; it < pc.timed; it++)
            {
                m_cmd.writeTimestamp((uint32_t)(5 + 2 * it));
                m_cmd.writeTimestamp((uint32_t)(6 + 2 * it));
            }
            recordSubstepEvents(pc);
        }
        if (s < pc.nSub - 1)
            dispatchGrid(pc, m_velocity, m_bodyCount, 256);
    }

}

// Phase 5: Gauss-Seidel over the colours, then the dual update.
void VkWorld2D::recordColorSolve(PassCtx &pc)
{
    m_cmd.fill(m_colorIndirect, 0u);
    m_cmd.barrier();
    dispatchGrid(pc, m_colorArgs, pc.colorBound, 128);
    m_cmd.barrierIndirect();
    m_cmd.writeTimestamp(4);

    // 5. Iterations: Gauss-Seidel over the colours, then the dual update.
    for (int s = 0; s < pc.nSub; s++)
    {
        if (s > 0)
            recordSubstepStart(pc, s);
        for (int it = 0; it < pc.iters; it++)
        {
            for (int c = 0; c < pc.colorBound; c++)
            {
                m_cmd.dispatchIndirect(m_primal, m_colorIndirect, (VkDeviceSize)c * 12, ringArgs(pc, (uint32_t)c));
                m_cmd.barrier();
            }
            if (s == 0 && it < kMaxTimedIterations)
                m_cmd.writeTimestamp((uint32_t)(5 + 2 * it));
            dispatchGrid(pc, m_dualAll, std::max(m_pairCap, m_jointSlots), 128);
            if (s == 0 && it < kMaxTimedIterations)
                m_cmd.writeTimestamp((uint32_t)(6 + 2 * it));
        }
        if (s == 0)
            recordSubstepEvents(pc);
        if (s < pc.nSub - 1)
            dispatchGrid(pc, m_velocity, m_bodyCount, 256);
    }

}

void VkWorld2D::recordSolve(PassCtx &pc)
{
    pc.timed = std::min(pc.iters, kMaxTimedIterations);
    // Solver substeps: the contacts above are found once; the solve below runs pc.nSub times with
    // h = dt / pc.nSub. Substep 0 was integrated in section 1. A later one starts with the joint init
    // (C0 at the substep's pose, no extra warm-start decay) and the integrate, which rebuilds the
    // inertial target from the velocity the last substep ended with; it ends with a velocity pc.pass
    // except the last, which the guard dispatch below folds in. Only substep 0 is pc.timed per
    // iteration; the rest of the substeps fall into the tail interval.
    pc.nSub = std::max(m_stepSubsteps, 1);
    if (pc.fused)
        recordFusedSolve(pc);
    else
        recordColorSolve(pc);
}

// Phase 6: velocities, the guard, continuous collision, sleep.
void VkWorld2D::recordTail(PassCtx &pc)
{
    // 6. Velocities, the guard, then sleep.
    if (m_eventsOn)
        dispatchGrid(pc, m_eventJoint, m_jointSlots, 128, (uint32_t)kGrabSlotCount);
    dispatchGrid(pc, m_velocityGuard, m_bodyCount, 256); // velocities, then the guard per body
    // Continuous collision, a pose clamp after the solve (the solve itself is untouched): flag the
    // fast bodies, compact them, sweep only those (indirect). Non-bullets go first against statics;
    // bullets then also see the dynamic bodies at their final poses.
    if (pc.ccdOn)
    {
        dispatchGrid(pc, m_ccdFlagPl, m_dynCount, 256);
        selectFlagged(m_cmd, m_ring, m_primPl, m_select, m_dynIdx, m_ccdFlag, m_ccdList, m_ccdCount,
                      (uint32_t)m_dynCount);
        m_cmd.barrier();
        dispatchGrid(pc, m_ccdArgsPl, 1, 1);
        m_cmd.barrierIndirect();
        for (uint32_t bulletPass = 0; bulletPass < 2; bulletPass++)
        {
            m_cmd.dispatchIndirect(m_ccdPl, m_ccdArgs, 0, ringArgs(pc, bulletPass));
            m_cmd.barrier();
        }
    }
    if (pc.sleepOn)
    {
        dispatchGrid(pc, m_sleepPrep, m_dynCount, 256);
        dispatchGrid(pc, m_sleepBlockPre, m_jointSlots, 128);
        dispatchGrid(pc, m_sleepMark, m_dynCount, 256);
        dispatchGrid(pc, m_sleepNbrC, m_pairCap, 128);
        dispatchGrid(pc, m_sleepNbrJ, m_jointSlots, 128);
    }
    dispatchGrid(pc, m_sleepCommit, m_dynCount, 256); // also counts the awake bodies
    dispatchGrid(pc, m_compactMoved, m_bodyCount, 256);

}

// Phase 7: read-backs, in the same submission.
void VkWorld2D::recordReadbacks(PassCtx &pc)
{
    m_cmd.copyToHost(m_counters, m_statusHost, 64);
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
    // The contact overlays' manifold snapshot: plain copies into one mapped buffer, in this
    // same submission, so downloadContacts() never has to stage or wait on its own. Only the
    // committed pc.pass's copies survive to the host (a rerun re-records over them). The caps are
    // fixed at record time; the counts the host reads are bounded by them.
    if (m_contactDebug && m_pairCap > 0)
    {
        const size_t need = dbgBytes(m_pairCap, m_contactCap);
        if (m_manDebugHost.size() < need)
        {
            m_manDebugHost.destroy();
            m_manDebugHost.init(m_dev, need, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
        }
        const ManBufs &M = m_man[m_curIdx];
        const VkDeviceSize P = (VkDeviceSize)m_pairCap, C = (VkDeviceSize)m_contactCap, cb = dbgContactBase(m_pairCap);
        m_cmd.copyToHost(M.key, m_manDebugHost, P * 8, 0, 0);
        m_cmd.copyToHost(M.bodyA, m_manDebugHost, P * 4, 0, P * 8);
        m_cmd.copyToHost(M.bodyB, m_manDebugHost, P * 4, 0, P * 12);
        m_cmd.copyToHost(M.count, m_manDebugHost, P * 4, 0, P * 16);
        m_cmd.copyToHost(M.offset, m_manDebugHost, P * 4, 0, P * 20);
        m_cmd.copyToHost(M.normal, m_manDebugHost, P * 8, 0, P * 24);
        m_cmd.copyToHost(M.friction, m_manDebugHost, P * 4, 0, P * 32);
        m_cmd.copyToHost(M.rA, m_manDebugHost, C * 8, 0, cb);
        m_cmd.copyToHost(M.rB, m_manDebugHost, C * 8, 0, cb + C * 8);
        m_cmd.copyToHost(M.lambda, m_manDebugHost, C * 8, 0, cb + C * 16);
        m_cmd.copyToHost(M.penalty, m_manDebugHost, C * 8, 0, cb + C * 24);
        m_cmd.copyToHost(M.stick, m_manDebugHost, C, 0, cb + C * 32);
    }
    m_cmd.writeTimestamp((uint32_t)(5 + 2 * pc.timed));

}

void VkWorld2D::recordPass(int pass, int colorBound, int jpRounds)
{
    PassCtx pc;
    pc.pass = pass;
    pc.colorBound = colorBound;
    pc.jpRounds = jpRounds;
    beginPass(pc);
    recordFront(pc);
    recordGraph(pc);
    recordSolve(pc);
    recordTail(pc);
    recordReadbacks(pc);
    if (m_ring.wrapped())
    {
        fprintf(stderr, "[avbd2d] step: argument ring wrapped within one submission\n");
        throw avbdvk::VkError(VK_ERROR_UNKNOWN, "step: argument ring wrapped within one submission");
    }
}



void VkWorld2D::refreshHostCopies()
{
    const size_t n = (size_t)m_bodyCount;
    // The moved list first: it is the only body read-back when the full copies are off. Each
    // body appears at most once, so the chunks scatter into the mirrors independently.
    const int count = std::max(0, std::min(*m_movedCountHost.mappedAs<int32_t>(), m_bodyCount));
    const uint32_t *recs = m_movedHost.mappedAs<uint32_t>();
    m_moved2d.resize((size_t)count);
    std::atomic<bool> invalid{false};
    m_hostPool->run((size_t)count, kMovedChunk,
                    [&](size_t, size_t begin, size_t end)
                    {
                        for (size_t k = begin; k < end; k++)
                        {
                            const uint32_t *rec = recs + k * 8;
                            float f[6];
                            std::memcpy(f, rec + 2, sizeof(f));
                            MovedBody2D &m = m_moved2d[k];
                            m.body = (int)rec[0];
                            if (m.body < 0 || m.body >= m_bodyCount)
                            {
                                m.body = -1;
                                invalid.store(true, std::memory_order_relaxed);
                                continue;
                            }
                            m.awake = (rec[1] & 1u) != 0;
                            m.fellAsleep = (rec[1] & 2u) != 0;
                            m.removed = (rec[1] & 4u) != 0;
                            m.pos = Vec2(f[0], f[1]);
                            m.angle = f[2];
                            m.vel = Vec2(f[3], f[4]);
                            m.angVel = f[5];
                            const size_t b = (size_t)m.body;
                            m_hostPos[b] = m.pos;
                            m_hostAng[b] = m.angle;
                            m_hostVel[b] = m.vel;
                            m_hostAngVel[b] = m.angVel;
                            m_hostAwake[b] = m.awake ? 1 : 0;
                        }
                    });
    if (invalid.load(std::memory_order_relaxed))
        m_moved2d.erase(
            std::remove_if(m_moved2d.begin(), m_moved2d.end(), [](const MovedBody2D &m) { return m.body < 0; }),
            m_moved2d.end());
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
    auto ms = [&](uint32_t a, uint32_t b)
    {
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

    // Headroom over the last step's colours: a quarter of them, plus how much they grew since the step
    // before. In a collapse the count climbs fast and a bound that tracks it saves the rerun; the
    // spare colours cost only empty indirect dispatches (zero groups).
    const int growth = std::max(0, m_lastColors - m_prevColors);
    m_stepColorBound = std::min(std::max(4, m_lastColors + std::max(2, m_lastColors / 4) + growth), kMaxColors);
    m_stepSubsteps = m_substeps;
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
        if (pass > kMaxRerunPasses)
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
        const bool jpShort =
            !m_lastSmallBuild && jpBoundShort(m_constraints.jpConverged(), jpRounds, VkConstraints::kMaxJpRounds);
        const bool colorShort = colorBoundShort(colorBound, builtColors, kMaxColors);

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
                jpRounds = widenJpRounds(jpRounds, (int)VkConstraints::kMaxJpRounds);
                r.jpReruns++;
            }
            if (colorShort)
            {
                colorBound = widenColorBound(builtColors, kMaxColors);
                r.colorReruns++;
            }
            continue;
        }
        if (r.truncated || (!pairShort && !contactShort))
        {
            // Commit: this pass's manifolds become the next step's warm start.
            m_prevPairs = std::min(pairsReq, m_pairCap);
            m_prevColors = m_lastColors;
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
            m_diag.substeps = m_stepSubsteps;
            m_diag.contacts = r.contacts;
            m_diag.colors = r.colors;
            m_diag.colorReruns = r.colorReruns;
            std::memcpy(&m_diag.contactCflMax, st + 8, 4);
            std::memcpy(&m_diag.penetrationRatioMax, st + 9, 4);
            std::memcpy(&m_diag.speedMax, st + 10, 4);
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

} // namespace avbd2d
