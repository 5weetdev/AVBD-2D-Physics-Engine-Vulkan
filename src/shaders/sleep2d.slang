// Per-body sleeping for the 2D solver: the sleep and wake rule.
//
// A sleeping body (awake = 0, sleepTimer >= 0) is frozen: the integrate and solve skip it and
// the constraint graph leaves it out, so to its neighbours it is a fixed anchor. The broadphase
// and narrowphase still see it (its manifolds are carried unchanged while nothing around it
// moves), so a sleeper keeps a contact fingerprint to be woken by.
//
// Going to sleep (tail of the step):
//   * csSleepMark2D runs the rest test. A body moving faster than twice the sleep speed is not
//     resting; otherwise it is observed for a window of sleepFrames steps and, at the end of
//     the window, is a candidate if its net movement over the whole window (position plus
//     gyration-weighted rotation) is under sleepDisp. The velocity gate is only a fast reject:
//     the displacement is the test, so a slow creep is not mistaken for rest.
//   * a body held by the grab, or by a joint loaded past sleepFrac of its break force, is
//     never a candidate;
//   * csSleepNbr*2D then blocks every body that touches or is jointed to a non-candidate, and
//     csSleepCommit2D sleeps the candidates left, so touching bodies sleep together.
//
// Waking (after the narrowphase, before the graph is built):
//   * an awake body moving at sleepLin / sleepAng or faster wakes a sleeper it touches;
//   * a sleeper whose contacts changed since it fell asleep (fewer, or different partners)
//     has lost its support and wakes;
//   * an awake joint partner wakes a sleeper, one link per round.

import maths2d;
import types2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

bool isDynamic(World2 *w, int i)
{
    return i >= 0 && w->mass[i] > 0.0f;
}

// A kinematic body that is moving: it wakes the sleepers it touches and keeps its own resting
// partners awake.
bool isMovingKinematic(World2 *w, int i)
{
    if (i < 0 || w->mass[i] > 0.0f || (asuint(w->bProps[i].w) & BODY_KINEMATIC) == 0u)
        return false;
    return v2lenSq(w->vel[i]) > 0.0f || w->angVel[i] != 0.0f;
}

// ---------------------------------------------------------------------------
// Tail: the rest test
// ---------------------------------------------------------------------------

// Clears the block flag the next kernels set.
[shader("compute")]
[numthreads(256, 1, 1)]
void csSleepPrep2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    w->blocked[w->dyn[t]] = 0;
}

// Bodies that may not sleep whatever their motion: the grabbed one, and the endpoints of a
// joint carrying more than sleepFrac of its break force.
[shader("compute")]
[numthreads(128, 1, 1)]
void csSleepBlockPre2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint j = tid.x;
    if (j >= w->jointCount || w->jBroken[j] != 0)
        return;
    int a = w->jA[j];
    int b = w->jB[j];
    if (a < 0)
    {
        // A soft world joint is the mouse grab: the held body stays awake.
        if ((w->jAxis[j * 3].flags & 1u) != 0u && isDynamic(w, b))
            w->blocked[b] = 1;
        return;
    }
    float frac = w->jFrac[j];
    if (isinf(frac))
        return;
    float limit = frac * w->sleepFrac;
    vec3 force = w->jForce[j]; // constraint force from the last dual (x, y: N)
    if (force.x * force.x + force.y * force.y <= limit * limit)
        return;
    if (isDynamic(w, a))
        w->blocked[a] = 1;
    if (isDynamic(w, b))
        w->blocked[b] = 1;
}

// The rest test. cand: 0 not a candidate, 1 rested through the last full window, 2 asleep.
[shader("compute")]
[numthreads(256, 1, 1)]
void csSleepMark2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    int i = w->dyn[t];

    if (w->sleepTimer[i] < 0)
    {
        w->cand[i] = 2; // removed: a valid neighbour, never a streak of its own
        return;
    }
    if (w->blocked[i] != 0)
    {
        w->sleepTimer[i] = 0;
        w->cand[i] = 0;
        return;
    }
    if (w->awake[i] == 0)
    {
        w->cand[i] = 2;
        return;
    }

    if (v2len(w->vel[i]) > w->sleepLin * 2.0f || abs(w->angVel[i]) > w->sleepAng * 2.0f)
    {
        w->sleepTimer[i] = 0;
        w->cand[i] = 0;
        return;
    }

    int timer = w->sleepTimer[i];
    if (timer == 0)
    {
        w->restPos[i] = w->pos[i];
        w->restAng[i] = w->ang[i];
    }
    timer = timer + 1;
    if (timer < (int)w->sleepFrames)
    {
        // Inside a window the flag from the last completed window stands, so an island whose
        // bodies' windows are out of phase can still commit. A 2 is left over from sleep: this
        // body was woken, and committing it again at once would sleep it in mid-air.
        w->sleepTimer[i] = timer;
        if (w->cand[i] == 2)
            w->cand[i] = 0;
        return;
    }

    float gyration = w->moment[i] > 0.0f ? sqrt(w->moment[i] / w->mass[i]) : 0.0f;
    float d = v2len(v2sub(w->pos[i], w->restPos[i])) + gyration * abs(w->ang[i] - w->restAng[i]);
    w->cand[i] = (d <= w->sleepDisp) ? (uint8_t)1 : (uint8_t)0;
    w->sleepTimer[i] = 0;
    w->restPos[i] = w->pos[i];
    w->restAng[i] = w->ang[i];
}

// A body that is not a candidate blocks every dynamic body it touches.
[shader("compute")]
[numthreads(128, 1, 1)]
void csSleepNbrContacts2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->pairCap || w->cur.count[t] <= 0)
        return;
    int a = w->cur.bodyA[t];
    int b = w->cur.bodyB[t];
    if (isDynamic(w, a) && isMovingKinematic(w, b))
        w->blocked[a] = 1;
    if (isDynamic(w, b) && isMovingKinematic(w, a))
        w->blocked[b] = 1;
    if (!isDynamic(w, a) || !isDynamic(w, b))
        return;
    if (w->cand[a] == 0)
        w->blocked[b] = 1;
    if (w->cand[b] == 0)
        w->blocked[a] = 1;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csSleepNbrJoints2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint j = tid.x;
    if (j >= w->jointCount || w->jBroken[j] != 0)
        return;
    int a = w->jA[j];
    int b = w->jB[j];
    if (!isDynamic(w, a) || !isDynamic(w, b))
        return;
    if (w->cand[a] == 0)
        w->blocked[b] = 1;
    if (w->cand[b] == 0)
        w->blocked[a] = 1;
}

// Puts the ready candidates to sleep, remembering their contact fingerprint, and counts the
// bodies still awake in counters[2].
[shader("compute")]
[numthreads(256, 1, 1)]
void csSleepCommit2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    int i = w->dyn[t];
    if (w->awake[i] == 0)
        return;
    if (w->sleepOn != 0 && w->cand[i] == 1 && w->blocked[i] == 0)
    {
        w->awake[i] = 0;
        w->vel[i] = vec2(0.0f, 0.0f);
        w->angVel[i] = 0.0f;
        w->prevVel[i] = vec2(0.0f, 0.0f);
        // To its neighbours a sleeper is a fixed anchor: no displacement since x^t.
        w->initPos[i] = w->pos[i];
        w->initAng[i] = w->ang[i];
        w->snapCount[i] = w->touchCount[i];
        w->snapHash[i] = w->touchHash[i];
        return;
    }
    int old;
    InterlockedAdd(w->counters[2], 1, old);
}

// ---------------------------------------------------------------------------
// Wake
// ---------------------------------------------------------------------------

// Zeroes the fingerprint and the wake requests of every dynamic body.
[shader("compute")]
[numthreads(256, 1, 1)]
void csTouchClear2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    int i = w->dyn[t];
    w->touchCount[i] = 0;
    w->touchHash[i] = 0;
    w->wakeReq[i] = 0;
}

void touch(World2 *w, int i, int other, uint mul)
{
    int old;
    InterlockedAdd(w->touchCount[i], 1, old);
    InterlockedXor(w->touchHash[i], (int)((uint)other * mul + 0x85EBCA6Bu));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csTouchAccumContacts2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->pairCap || w->cur.count[t] <= 0)
        return;
    int a = w->cur.bodyA[t];
    int b = w->cur.bodyB[t];
    if (isDynamic(w, a))
        touch(w, a, b, 0x9E3779B1u);
    if (isDynamic(w, b))
        touch(w, b, a, 0x9E3779B1u);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csTouchAccumJoints2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint j = tid.x;
    if (j >= w->jointCount || w->jBroken[j] != 0)
        return;
    int a = w->jA[j];
    int b = w->jB[j];
    if (isDynamic(w, a))
        touch(w, a, b, 0x27D4EB2Du);
    if (isDynamic(w, b))
        touch(w, b, a, 0x27D4EB2Du);
}

// A body is "moving" for waking purposes at the sleep speed itself (not twice it).
bool movingFast(World2 *w, int i)
{
    return w->awake[i] != 0 && (v2len(w->vel[i]) >= w->sleepLin || abs(w->angVel[i]) >= w->sleepAng);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csWakeReqContacts2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->pairCap || w->cur.count[t] <= 0)
        return;
    int a = w->cur.bodyA[t];
    int b = w->cur.bodyB[t];
    bool dynA = isDynamic(w, a);
    bool dynB = isDynamic(w, b);
    if (dynA && dynB)
    {
        if (movingFast(w, a) && w->awake[b] == 0)
            w->wakeReq[b] = 1;
        if (movingFast(w, b) && w->awake[a] == 0)
            w->wakeReq[a] = 1;
    }
    else if (dynA && isMovingKinematic(w, b) && w->awake[a] == 0)
        w->wakeReq[a] = 1;
    else if (dynB && isMovingKinematic(w, a) && w->awake[b] == 0)
        w->wakeReq[b] = 1;
}

// One joint link per dispatch; the host records a few rounds per step (request, then apply).
[shader("compute")]
[numthreads(128, 1, 1)]
void csWakeReqJoints2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint j = tid.x;
    if (j >= w->jointCount || w->jBroken[j] != 0)
        return;
    int a = w->jA[j];
    int b = w->jB[j];
    if (a < 0)
    {
        if ((w->jAxis[j * 3].flags & 1u) != 0u && isDynamic(w, b))
            w->wakeReq[b] = 1; // the mouse grab
        return;
    }
    if (!isDynamic(w, a) || !isDynamic(w, b))
        return;
    if (w->awake[a] != 0 && w->awake[b] == 0)
        w->wakeReq[b] = 1;
    if (w->awake[b] != 0 && w->awake[a] == 0)
        w->wakeReq[a] = 1;
}

// A sleeper whose contact fingerprint no longer matches the one it fell asleep with has lost
// (or changed) what held it.
[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeLost2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    int i = w->dyn[t];
    if (w->awake[i] != 0 || w->sleepTimer[i] < 0)
        return;
    if (w->snapCount[i] < 0)
    {
        // Born asleep (or put to sleep from outside): take the baseline now.
        w->snapCount[i] = w->touchCount[i];
        w->snapHash[i] = w->touchHash[i];
        return;
    }
    bool lost = w->touchCount[i] < w->snapCount[i] ||
                (w->touchCount[i] == w->snapCount[i] && w->touchHash[i] != w->snapHash[i]);
    if (lost)
        w->wakeReq[i] = 1;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeApply2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    int i = w->dyn[t];
    if (w->wakeReq[i] == 0 || w->awake[i] != 0 || w->sleepTimer[i] < 0)
        return;
    w->awake[i] = 1;
    w->sleepTimer[i] = 0;
}

// Wakes everything that is asleep (the panel's "Wake all").
[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeAll2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    int i = w->dyn[t];
    if (w->sleepTimer[i] < 0)
        return; // removed
    w->awake[i] = 1;
    w->sleepTimer[i] = 0;
}
