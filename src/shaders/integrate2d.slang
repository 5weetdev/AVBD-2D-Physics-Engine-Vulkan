// The 2D integrate stage: csIntegrate2D (inertial target y, adaptive warm start),
// csVelocity2D (BDF1 velocity reconstruction) and csGuard2D (kill-fallen plus the
// last-resort explosion circuit breaker).

import maths2d;
import types2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const float WORLD_BOUND = 100000.0f;

// One body of csIntegrate2D; csIntegrateJointInit2D (solver2d.slang) runs it too.
public void integrateBody(World2 *w, uint i)
{
    float dt = w->dt;
    float m = w->mass[i];
    vec2 p = w->pos[i];
    float th = w->ang[i];
    vec2 v = w->vel[i];
    float om = w->angVel[i];
    float4 props = w->bProps[i];
    uint flags = asuint(props.w);
    // Only gravity's y part is here; a sideways part is added by csGravityX2D (dispatched only when
    // gravity.x != 0), because a branch on it in this kernel changes the rounding of the default case.
    float grav = w->gravityY * props.x;
    vec2 g = vec2(0.0f, grav);
    if (flags != 0u || props.y > 0.0f || props.z > 0.0f)
    {
        // Motion locks: a locked degree of freedom has no velocity, so nothing carries it away.
        if ((flags & BODY_LOCK_X) != 0u)
            v.x = 0.0f;
        if ((flags & BODY_LOCK_Y) != 0u)
            v.y = 0.0f;
        if ((flags & BODY_LOCK_ROT) != 0u)
            om = 0.0f;
        // Damping the Box2D way, on the velocity that forms the inertial target.
        if (props.y > 0.0f)
            v = v2mul(v, 1.0f / (1.0f + dt * props.y));
        if (props.z > 0.0f)
            om = om * (1.0f / (1.0f + dt * props.z));
    }

    // x^t, and the inertial target y (Eq. 2).
    w->initPos[i] = p;
    w->initAng[i] = th;
    vec2 inert = v2add(p, v2mul(v, dt));
    if (m > 0.0f)
        inert = v2add(inert, v2mul(g, dt * dt));
    w->inertPos[i] = inert;
    w->inertAng[i] = th + om * dt;

    // A kinematic body takes the pose its own velocity gives it. It is never solved, so this is
    // where it ends the step; the contacts read the motion as pos - initPos.
    if (m <= 0.0f && (flags & BODY_KINEMATIC) != 0u)
    {
        w->pos[i] = inert;
        w->ang[i] = th + om * dt;
    }

    if (m > 0.0f && w->awake[i] != 0)
    {
        // Adaptive warm start: only as much gravity as the body was actually accelerating by.
        float gmag = abs(grav);
        float accelWeight = 0.0f;
        if (gmag > 0.0f)
        {
            vec2 accel = v2mul(v2sub(v, w->prevVel[i]), 1.0f / dt);
            accelWeight = clampf(accel.y * signf(grav) / gmag, 0.0f, 1.0f);
        }
        if (!finitef(accelWeight))
            accelWeight = 0.0f;

        w->pos[i] = v2add(v2add(p, v2mul(v, dt)), v2mul(g, accelWeight * dt * dt));
        w->ang[i] = th + om * dt;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csIntegrate2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    if (tid.x >= w->bodyCount)
        return;
    integrateBody(w, tid.x);
}

// The x part of gravity, run right after csIntegrate2D when gravity.x != 0: the same inertial
// target and adaptive warm start as the y part, along x.
[shader("compute")]
[numthreads(256, 1, 1)]
void csGravityX2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint i = tid.x;
    if (i >= w->bodyCount)
        return;
    float m = w->mass[i];
    if (m <= 0.0f)
        return;

    float dt = w->dt;
    float gx = w->gravityX * w->bProps[i].x;
    if (gx == 0.0f)
        return;
    w->inertPos[i].x += gx * dt * dt;
    if (w->awake[i] != 0)
    {
        float accelWeight = clampf((w->vel[i].x - w->prevVel[i].x) / dt * signf(gx) / abs(gx), 0.0f, 1.0f);
        if (!finitef(accelWeight))
            accelWeight = 0.0f;
        w->pos[i].x += gx * accelWeight * dt * dt;
    }
}

void velocityBody(World2 *w, uint i)
{
    w->prevVel[i] = w->vel[i];
    if (w->mass[i] > 0.0f && w->awake[i] != 0)
    {
        float invDt = 1.0f / w->dt;
        w->vel[i] = v2mul(v2sub(w->pos[i], w->initPos[i]), invDt);
        w->angVel[i] = (w->ang[i] - w->initAng[i]) * invDt;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csVelocity2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    if (tid.x >= w->bodyCount)
        return;
    velocityBody(w, tid.x);
}

void guardBody(World2 *w, int i)
{
    if (w->awake[i] == 0)
        return;

    vec2 p = w->pos[i];
    float th = w->ang[i];
    vec2 v = w->vel[i];

    bool fell = w->killOn != 0 && p.y < w->killY;
    bool finite = finitef(p.x) && finitef(p.y) && finitef(th) && finitef(v.x) && finitef(v.y);
    bool inBounds = finite && abs(p.x) <= WORLD_BOUND && abs(p.y) <= WORLD_BOUND;
    bool tooFast = finite && v2lenSq(v) > w->maxSpeed * w->maxSpeed;
    if (!fell && inBounds && !tooFast)
        return;

    if (!finite)
    {
        p = vec2(0.0f, 0.0f);
        th = 0.0f;
    }
    w->pos[i] = vec2(clampf(p.x, -WORLD_BOUND, WORLD_BOUND), clampf(p.y, -WORLD_BOUND, WORLD_BOUND));
    w->ang[i] = th;
    w->vel[i] = vec2(0.0f, 0.0f);
    w->angVel[i] = 0.0f;
    w->prevVel[i] = vec2(0.0f, 0.0f);
    w->awake[i] = 0;
    w->sleepTimer[i] = -1;
    int old;
    InterlockedAdd(w->counters[3], 1, old);
}

// Last-resort circuit breaker, run at the end of every step over the dynamic bodies: a body
// below killY, with a non-finite or out-of-range pose, or faster than maxSpeed is removed
// from the simulation (awake = 0, sleepTimer = -1: it never wakes and the broadphase ignores
// it). Counts removals in counters[3]. Without this a diverged joint or pile would hand the
// broadphase thousands of bodies on one point.
[shader("compute")]
[numthreads(256, 1, 1)]
void csGuard2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    guardBody(w, w->dyn[t]);
}

// csVelocity2D then csGuard2D for one body, in one dispatch over all bodies. Both touch only that
// body's own data (guard writes the removal counter), so doing the two in order per body equals
// doing all the velocities and then all the guards. The dynamic list is the bodies with mass > 0.
[shader("compute")]
[numthreads(256, 1, 1)]
void csVelocityGuard2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint i = tid.x;
    if (i >= w->bodyCount)
        return;
    velocityBody(w, i);
    if (w->mass[i] > 0.0f)
        guardBody(w, (int)i);
}
