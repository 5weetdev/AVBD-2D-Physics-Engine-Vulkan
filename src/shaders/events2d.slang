// Device event streams for the 2D solver, and the explosion impulse apply.
//
// Events are derived each pass by comparing this step's manifold set (cur) with last step's
// (prev), so a rerun regenerates them from scratch (the counters are zeroed per pass). Each
// kind has an append region in w->ev: the counter always counts, and a record is written only
// while it fits, so the host sees both how many there were and how many it can read. An
// overflow never reruns the step; the host grows the region for the next one.
//
// Sleeping pairs keep their manifolds verbatim, so a pair that falls asleep produces no end
// event. A body that is disabled or destroyed leaves the broadphase, its manifolds vanish and
// the key-set comparison reports the end (sensor ends included). Hit events fire once per
// approach: flags bit 1 marks a manifold that carried load, and a hit needs a loaded manifold
// whose previous-step counterpart was not loaded.

module events2d;

import maths2d;
import types2d;
import collide2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const int EV_CONTACT_BEGIN = 0;
static const int EV_CONTACT_END = 1;
static const int EV_SENSOR_BEGIN = 2;
static const int EV_SENSOR_END = 3;
static const int EV_HIT = 4;
static const int EV_JOINT_BREAK = 5;

// Index (in ints) of a new record of `kind`, or -1 when the region is full.
int evReserve(World2 *w, int kind, int stride)
{
    int idx;
    InterlockedAdd(w->ev[kind], 1, idx);
    if ((uint)idx >= w->evCap[kind])
        return -1;
    return (int)w->evOff[kind] + idx * stride;
}

int findIn(uint64_t *keys, int n, uint64_t want)
{
    int lo = 0;
    int hi = n - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        uint64_t k = keys[mid];
        if (k == want)
            return mid;
        if (k < want)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

void emitPair(World2 *w, int kind, int a, int b)
{
    int o = evReserve(w, kind, 2);
    if (o >= 0)
    {
        w->ev[o] = a;
        w->ev[o + 1] = b;
    }
}

// Sensor shape first, then the visitor.
void emitSensor(World2 *w, int kind, int sa, int sb)
{
    if (((uint)w->sFlags[sa] & SHAPE_FLAG_SENSOR) != 0u)
        emitPair(w, kind, sa, sb);
    else
        emitPair(w, kind, sb, sa);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csEventBegin2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (w->evOn == 0u || t >= min((uint)w->counters[0], w->pairCap))
        return;
    uint64_t key = w->cur.key[t];
    int sa = (int)(uint)(key >> 32);
    int sb = (int)(uint)(key & uint64_t(0xFFFFFFFFu));
    int pj = findIn(w->prev.key, (int)w->prevCount, key);
    uint fl = (uint)w->sFlags[sa] | (uint)w->sFlags[sb];
    if (w->cur.count[t] > 0 && (fl & SHAPE_FLAG_CONTACT_EVENTS) != 0u)
    {
        if (pj < 0 || w->prev.count[pj] == 0)
            emitPair(w, EV_CONTACT_BEGIN, sa, sb);
    }
    if ((w->cur.flags[t] & 1) != 0)
    {
        if (pj < 0 || (w->prev.flags[pj] & 1) == 0)
            emitSensor(w, EV_SENSOR_BEGIN, sa, sb);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csEventEnd2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (w->evOn == 0u || t >= w->prevCount)
        return;
    uint64_t key = w->prev.key[t];
    int sa = (int)(uint)(key >> 32);
    int sb = (int)(uint)(key & uint64_t(0xFFFFFFFFu));
    int live = (int)min((uint)w->counters[0], w->pairCap);
    int cj = findIn(w->cur.key, live, key);
    uint fl = (uint)w->sFlags[sa] | (uint)w->sFlags[sb];
    if (w->prev.count[t] > 0 && (fl & SHAPE_FLAG_CONTACT_EVENTS) != 0u)
    {
        if (cj < 0 || w->cur.count[cj] == 0)
            emitPair(w, EV_CONTACT_END, sa, sb);
    }
    if ((w->prev.flags[t] & 1) != 0)
    {
        if (cj < 0 || (w->cur.flags[cj] & 1) == 0)
            emitSensor(w, EV_SENSOR_END, sa, sb);
    }
}

// After the iterations, before csVelocity: vel / angVel still hold v^t, the speed the pair
// arrived with.
[shader("compute")]
[numthreads(128, 1, 1)]
void csEventHit2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (w->evOn == 0u || t >= min((uint)w->counters[0], w->pairCap))
        return;
    int cnt = w->cur.count[t];
    if (cnt <= 0)
        return;
    uint64_t key = w->cur.key[t];
    int sa = (int)(uint)(key >> 32);
    int sb = (int)(uint)(key & uint64_t(0xFFFFFFFFu));
    if ((((uint)w->sFlags[sa] | (uint)w->sFlags[sb]) & SHAPE_FLAG_HIT_EVENTS) == 0u)
        return;
    int ia = w->cur.bodyA[t];
    int ib = w->cur.bodyB[t];
    vec2 n = w->cur.normal[t];
    float radiusA = w->sRad[sa];
    float radiusB = w->sRad[sb];
    int modeA = anchorModeOf((int)w->sType[sa], w->sVR[sa].y, radiusA);
    int modeB = anchorModeOf((int)w->sType[sb], w->sVR[sb].y, radiusB);
    vec2 offA = modeA == ANCHOR_CORE ? v2mul(n, -radiusA) : vec2(0.0f, 0.0f);
    vec2 offB = modeB == ANCHOR_CORE ? v2mul(n, radiusB) : vec2(0.0f, 0.0f);
    int o = w->cur.offset[t];
    float best = 0.0f;
    vec2 bestP = vec2(0.0f, 0.0f);
    bool loaded = false;
    for (int i = 0; i < cnt; i++)
    {
        if (w->cur.lambda[o + i].x >= 0.0f)
            continue;
        loaded = true;
        vec2 rAW = v2add(v2rot(w->initAng[ia], w->cur.rA[o + i]), offA);
        vec2 rBW = v2add(v2rot(w->initAng[ib], w->cur.rB[o + i]), offB);
        vec2 vA = v2add(w->vel[ia], v2mul(v2perp(rAW), w->angVel[ia]));
        vec2 vB = v2add(w->vel[ib], v2mul(v2perp(rBW), w->angVel[ib]));
        float approach = -v2dot(n, v2sub(vA, vB));
        if (approach > best)
        {
            best = approach;
            bestP = v2mul(v2add(v2add(w->initPos[ia], rAW), v2add(w->initPos[ib], rBW)), 0.5f);
        }
    }
    if (loaded)
        w->cur.flags[t] = w->cur.flags[t] | (uint8_t)2;
    int pj = findIn(w->prev.key, (int)w->prevCount, key);
    bool prevLoaded = pj >= 0 && (w->prev.flags[pj] & 2) != 0;
    if (!loaded || prevLoaded || best <= w->hitThreshold)
        return;
    int e = evReserve(w, EV_HIT, 8);
    if (e < 0)
        return;
    w->ev[e] = sa;
    w->ev[e + 1] = sb;
    w->ev[e + 2] = asint(bestP.x);
    w->ev[e + 3] = asint(bestP.y);
    w->ev[e + 4] = asint(n.x);
    w->ev[e + 5] = asint(n.y);
    w->ev[e + 6] = asint(best);
    w->ev[e + 7] = 0;
}

// Dispatch arg a = the first authored joint slot (the grab slots before it never report).
[shader("compute")]
[numthreads(128, 1, 1)]
void csEventJoint2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint j = tid.x;
    if (w->evOn == 0u || j < args->a || j >= w->jointCount)
        return;
    if (w->jBroken[j] != 0 && w->jBrokenPrev[j] == 0)
    {
        int e = evReserve(w, EV_JOINT_BREAK, 1);
        if (e >= 0)
            w->ev[e] = (int)(j - args->a);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csApplyImpulses2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->impulseCount)
        return;
    Impulse2 im = w->impulses[t];
    int b = im.body;
    if (b < 0 || b >= (int)w->bodyCount || w->mass[b] <= 0.0f || w->sleepTimer[b] < 0)
        return;
    w->vel[b] = v2add(w->vel[b], vec2(im.dvx, im.dvy));
    w->angVel[b] = w->angVel[b] + im.dw;
    w->awake[b] = (uint8_t)1;
    w->sleepTimer[b] = 0;
}

// ---------------------------------------------------------------------------
// Host mutations (the ABI's mutation queue), applied at the top of the step before the state copy.
// One BodyPatch2 record per body (merged on the host), 12 words:
//   [0] body  [1] flags  [2..4] pose x y angle  [5..7] velocity x y w  [8..10] velocity change dx dy dw
// ---------------------------------------------------------------------------
static const uint BP_SET_POSE = 1u;
static const uint BP_SET_VEL = 2u;
static const uint BP_ADD_VEL = 4u;
static const uint BP_WAKE = 8u;
static const uint BP_DISABLE = 16u;
static const uint BP_ENABLE = 32u;
static const uint BP_SLEEP = 64u;

[shader("compute")]
[numthreads(128, 1, 1)]
void csApplyBodyPatches2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->bPatchCount)
        return;
    uint *r = w->bPatch + t * 12u;
    int b = (int)r[0];
    if (b < 0 || b >= (int)w->bodyCount)
        return;
    uint f = r[1];
    bool dynamic = w->mass[b] > 0.0f;
    bool removed = w->sleepTimer[b] < 0;
    if ((f & BP_ENABLE) != 0u)
    {
        w->awake[b] = (uint8_t)1;
        w->sleepTimer[b] = 0;
        removed = false;
    }
    if ((f & BP_SET_POSE) != 0u)
    {
        vec2 p = vec2(asfloat(r[2]), asfloat(r[3]));
        float a = asfloat(r[4]);
        w->pos[b] = p;
        w->ang[b] = a;
        w->restPos[b] = p;
        w->restAng[b] = a;
        if (dynamic && !removed)
        {
            w->awake[b] = (uint8_t)1;
            w->sleepTimer[b] = 0;
        }
    }
    if ((f & BP_SET_VEL) != 0u)
    {
        vec2 v = vec2(asfloat(r[5]), asfloat(r[6]));
        w->vel[b] = v;
        w->prevVel[b] = v;
        w->angVel[b] = asfloat(r[7]);
    }
    if ((f & BP_ADD_VEL) != 0u && dynamic)
    {
        w->vel[b] = v2add(w->vel[b], vec2(asfloat(r[8]), asfloat(r[9])));
        w->angVel[b] = w->angVel[b] + asfloat(r[10]);
    }
    if ((f & (BP_WAKE | BP_ADD_VEL)) != 0u && dynamic && !removed)
    {
        w->awake[b] = (uint8_t)1;
        w->sleepTimer[b] = 0;
    }
    if ((f & BP_SLEEP) != 0u && dynamic && !removed)
    {
        // Born asleep: no contact fingerprint yet, so the first step takes the baseline.
        w->awake[b] = (uint8_t)0;
        w->sleepTimer[b] = 0;
        w->cand[b] = (uint8_t)2;
        w->snapCount[b] = -1;
    }
    if ((f & BP_DISABLE) != 0u)
    {
        w->awake[b] = (uint8_t)0;
        w->sleepTimer[b] = -1;
        w->vel[b] = vec2(0.0f, 0.0f);
        w->prevVel[b] = vec2(0.0f, 0.0f);
        w->angVel[b] = 0.0f;
    }
}

// One JointPatch2 record per joint slot, 12 words:
//   [0] slot  [1] flags  [2] motor axis  [3] motor mode  [4] motor speed  [5] max force
//   [6] limit axis  [7] lower  [8] upper
static const uint JP_MOTOR = 1u;
static const uint JP_LIMIT = 2u;
static const uint JP_LIMIT_ON = 4u;
static const uint JP_BREAK = 8u;
static const uint JP_WAKE = 16u;

[shader("compute")]
[numthreads(128, 1, 1)]
void csApplyJointPatches2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->jPatchCount)
        return;
    uint *r = w->jPatch + t * 12u;
    int j = (int)r[0];
    if (j < 0 || j >= (int)w->jointCount)
        return;
    uint f = r[1];
    if ((f & JP_MOTOR) != 0u)
    {
        int axis = (int)min(r[2], 2u);
        w->jAxis[j * 3 + axis].motor = r[3];
        w->jAxis[j * 3 + axis].mtarget = asfloat(r[4]);
        w->jAxis[j * 3 + axis].maxF = asfloat(r[5]);
    }
    if ((f & JP_LIMIT) != 0u)
    {
        int axis = (int)min(r[6], 2u);
        w->jAxis[j * 3 + axis].limit = (f & JP_LIMIT_ON) != 0u ? 1u : 0u;
        w->jAxis[j * 3 + axis].lo = asfloat(r[7]);
        w->jAxis[j * 3 + axis].hi = asfloat(r[8]);
    }
    if ((f & JP_BREAK) != 0u)
        w->jBroken[j] = (uint8_t)1;
    if ((f & JP_WAKE) != 0u)
    {
        int a = w->jA[j];
        int b = w->jB[j];
        if (a >= 0 && w->mass[a] > 0.0f && w->sleepTimer[a] >= 0)
        {
            w->awake[a] = (uint8_t)1;
            w->sleepTimer[a] = 0;
        }
        if (b >= 0 && w->mass[b] > 0.0f && w->sleepTimer[b] >= 0)
        {
            w->awake[b] = (uint8_t)1;
            w->sleepTimer[b] = 0;
        }
    }
}

// The moved-body list: every dynamic body that was awake at the start of the step or still is, and
// every kinematic body with a velocity. Eight words per record, built in device-local memory and
// copied to the host after the step: [0] body  [1] bit 0 awake now, bit 1 fell asleep  [2..4] pose  [5..7] velocity.
// Shader stores straight into host memory cost ~1 us per body (Pile 10k: 10 ms) even when
// coalesced; the copy engine moves the same bytes in microseconds. One atomic per wave, and each
// record is two 16-byte stores.
[shader("compute")]
[numthreads(256, 1, 1)]
void csCompactMoved2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint i = tid.x;
    bool report = false;
    bool awakeNow = false, wasAwake = false;
    if (i < w->bodyCount)
    {
        bool live = w->sleepTimer[i] >= 0;
        awakeNow = w->awake[i] != 0 && live;
        wasAwake = w->saveAwake[i] != 0;
        if (w->mass[i] > 0.0f)
            report = awakeNow || wasAwake;
        else if ((asuint(w->bProps[i].w) & BODY_KINEMATIC) != 0u)
            report = w->vel[i].x != 0.0f || w->vel[i].y != 0.0f || w->angVel[i] != 0.0f;
    }
    // No early return above: every lane takes part in the wave ops.
    uint rank = WavePrefixCountBits(report);
    uint total = WaveActiveCountBits(report);
    int base = 0;
    if (total != 0u && WaveIsFirstLane())
        InterlockedAdd(w->movedCount[0], (int)total, base);
    base = WaveReadLaneFirst(base);
    if (!report)
        return;
    uint4 *o = (uint4 *)(w->moved + (uint)(base + (int)rank) * 8u);
    uint flags = (awakeNow ? 1u : 0u) | ((wasAwake && !awakeNow && w->mass[i] > 0.0f) ? 2u : 0u);
    o[0] = uint4(i, flags, asuint(w->pos[i].x), asuint(w->pos[i].y));
    o[1] = uint4(asuint(w->ang[i]), asuint(w->vel[i].x), asuint(w->vel[i].y), asuint(w->angVel[i]));
}
