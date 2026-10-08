#pragma once

// Host mirrors of shaders/types2d.slang. Field order and types must match the
// shader's structs exactly (scalar layout: every pointer first, 8 bytes each, then an even
// number of 4-byte scalars, so there is no implied padding). The static_asserts pin the
// total sizes; the gate for a layout change is a changed assert plus a changed shader.

#include <cstdint>

namespace avbd2d
{

using DevPtr = uint64_t; // a VkDeviceAddress; 0 only for buffers the kernels never touch

struct ManSet2Gpu
{
    DevPtr key, bodyA, bodyB, normal, friction, offset, count, feature;
    DevPtr rA, rB, c0, penalty, lambda, stick, roll, flags;
};

struct World2Gpu
{
    // Bodies.
    DevPtr pos, ang, initPos, initAng, inertPos, inertAng, vel, angVel, prevVel;
    DevPtr moment, mass, awake, sleepTimer, blocked, restPos, restAng, cand, touchCount,
        touchHash, snapCount, snapHash, wakeReq, bProps, dyn, stat;
    // Broadphase.
    DevPtr gridHead, gridNext, pairsRaw, pairsSorted, ignore, counters;
    // Manifolds.
    ManSet2Gpu cur, prev;
    // Constraint graph and colour sweep.
    DevPtr colorBodies, colorStart, colorEnd, gStart, gEnd, gKey, gIsA, colorIndirect;
    // Joints.
    DevPtr jA, jB, jRA, jRB, jC0, jPen, jLam, jAxis, jFrameA, jRest, jArm, jFrac, jForce, jBroken;
    // Shape table and broadphase lists.
    DevPtr sBody, sType, sOff, sAng, sRad, sHalf, sVR, sFric, sFlags, verts, norms, grid, over, sCat, sMask, sGroup, sMat;
    // Events.
    DevPtr ev, jBrokenPrev, impulses;
    // Continuous collision.
    DevPtr ccdFlag, ccdList, ccdCount, ccdStat, ccdArgs, bShape, bExt, bBullet;
    // Host mutations and the moved-body list.
    DevPtr bPatch, jPatch, moved, movedCount, saveAwake, savePos, saveAng;
    // Scalars.
    uint32_t bodyCount, dynCount, statCount, tableMask, ignoreCount, pairCap, contactCap, prevCount, jointCount;
    float dt, gravityX, alpha, betaLin, betaAng, gamma, dualDamping, dualDampVel, cellSize, killY;
    uint32_t autoK, killOn;
    int32_t colorBound;
    uint32_t sleepOn, sleepFrames;
    float sleepLin, sleepAng, maxSpeed, sleepDisp, sleepFrac;
    uint32_t contactCapPts;
    uint32_t gridCount, gridDyn, overCount, frictionMix;
    uint32_t evOff[6], evCap[6];
    float hitThreshold;
    uint32_t evOn, impulseCount;
    uint32_t ccdOn;
    float ccdSafety;
    float gravityY;
    uint32_t bPatchCount, jPatchCount;
};

// What a dispatch's push constant points at (Args2 in types2d.slang).
struct DispatchArgs2Gpu
{
    DevPtr w; // device address of a World2Gpu
    uint32_t a, b, c, d;
};

// Arguments of csBuildSmall2D (constraints2d.slang), the one-workgroup constraint build.
struct BuildArgs2Gpu
{
    DevPtr mBodyA, mBodyB, mCount, jBodyA, jBodyB, jBroken, mass, awake, dyn, keys, isA, rowStart, rowEnd, color,
        colorBodies, colorStart, counters;
    uint32_t slots, jointSlots, bodyCount, dynCount;
    int32_t capacity, pad0;
};
static_assert(sizeof(BuildArgs2Gpu) == 17 * 8 + 6 * 4, "BuildArgs2Gpu drifted from constraints2d.slang");

// What a batched query dispatch's push constant points at (QueryBatch2 in types2d.slang).
struct QueryBatch2Gpu
{
    DevPtr w, qBox, qHead, qNext, qBig, qCount, rays, aabbs, outHit, outIds, outCounts;
    uint64_t cat, mask;
    uint32_t headMask, count, maxPer, insertCount, pad0, pad1;
};

static_assert(sizeof(ManSet2Gpu) == 16 * 8, "ManSet2Gpu drifted from types2d.slang");
static_assert(sizeof(World2Gpu) == 120 * 8 + 54 * 4, "World2Gpu drifted from types2d.slang");
static_assert(sizeof(QueryBatch2Gpu) == 13 * 8 + 6 * 4, "QueryBatch2Gpu drifted from types2d.slang");
static_assert(sizeof(DispatchArgs2Gpu) == 24, "DispatchArgs2Gpu drifted from types2d.slang");

} // namespace avbd2d
