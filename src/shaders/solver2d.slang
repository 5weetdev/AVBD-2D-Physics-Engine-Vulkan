// The 2D AVBD iteration: the per-colour primal sweep (assemble the 3x3 body system from the
// body's constraint row in the shared CSR graph, solve it, move the body), and the dual
// stages for contacts and joints (force clamping, friction cone, multiplier update, penalty
// ramp). The body block is (x, y, theta):
// a scalar angle replaces the quaternion, one tangent replaces two, and each body is one
// thread -- a 3x3 solve does not need the cooperative-lane trick that pays for 6x6.
//
// Joints are per-axis rows (lock, spring, motor, limit; see the Joints section below). A joint
// whose bodyA is negative is anchored to the world; its rA is then a world point. A soft row is
// a plain spring (the mouse grab is two of them).

import maths2d;
import types2d;
import collide2d;
import integrate2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const int FORCE_MANIFOLD = 0;
static const int FORCE_JOINT = 1;

static const float PENALTY_MIN = 1.0f;
static const float PENALTY_MAX = 10000000000.0f;
static const float JOINT_PENALTY_MAX = 500000.0f;
static const float STICK_THRESH = 0.00001f;

// Graph key layout: (body << 32) | (type << 29) | index, as built by constraints.slang.
int graphType(uint64_t k) { return (int)((uint)(k & 0xFFFFFFFFu) >> 29); }
int graphIndex(uint64_t k) { return (int)((uint)(k & 0xFFFFFFFFu) & ((1u << 29) - 1u)); }

// The symmetric 3x3 system of one body: [l00 l01 c0; l01 l11 c1; c0 c1 a] dx = -r.
struct Sys3
{
    float l00, l01, l11;
    float c0, c1;
    float a;
    float r0, r1, ra;
};

// ---------------------------------------------------------------------------
// Colour dispatch arguments: one VkDispatchIndirectCommand per colour.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)]
void csColorArgs2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    int c = (int)tid.x;
    if (c >= w->colorBound)
        return;
    int n = max(w->colorEnd[c] - w->colorStart[c], 0);
    w->colorIndirect[c * 3 + 0] = (n + 127) / 128;
    w->colorIndirect[c * 3 + 1] = 1;
    w->colorIndirect[c * 3 + 2] = 1;
}

// ---------------------------------------------------------------------------
// Contacts
// ---------------------------------------------------------------------------
struct CEval
{
    vec2 jA;      // angular Jacobian of the (normal, tangent) rows for body A
    vec2 jB;
    vec2 C;       // constraint value: (normal, tangent)
    vec2 F;       // force after normal and friction-cone clamping
    float fscale; // magnitude of the unclamped friction force
    float bounds; // friction cone limit
};

// How far one body's contact anchor has moved along `nrm` and `tg` since x^t, and the angular
// Jacobian of the two rows. The anchors were found at x^t (narrowphase2d.slang), so a body
// that turns during the step carries them by its whole rotation, not just the last
// iteration's. A box corner is a material point: its displacement is taken exactly,
// R(ang) r - R(ang0) r, so a spinning box's anchor does not drift off its true path. A circle's
// surface rolls under the contact instead: its anchor stays where it was found, facing along
// the normal, so the normal row has no angular term and the tangent row the rolling arc
// r dtheta. Turning a disc's anchor with the disc lifts it off the floor by r (1 - cos dtheta)
// per step -- a spinning disc sinks through the ground by that much every step.
// A rounded shape (ANCHOR_CORE, collide2d.slang) combines the two: its anchor is a material point
// of the core, taken exactly, and the surface sits at the world offset `off` from it along the
// normal, which rolls: no normal term, and the tangent row's rolling arc (off x tg) dtheta.
void anchorMotion(vec2 r, float ang, float dAng, int mode, vec2 off, vec2 nrm, vec2 tg, out vec2 jac,
                  out vec2 disp)
{
    vec2 r0 = v2rot(ang - dAng, r);
    if (mode == ANCHOR_DISC)
    {
        jac = vec2(v2cross(r0, nrm), v2cross(r0, tg));
        disp = v2mul(jac, dAng);
        return;
    }
    vec2 rW = v2rot(ang, r);
    jac = vec2(v2cross(rW, nrm), v2cross(rW, tg));
    vec2 u = v2sub(rW, r0);
    disp = vec2(v2dot(nrm, u), v2dot(tg, u));
    if (mode == ANCHOR_CORE)
    {
        vec2 jo = vec2(v2cross(off, nrm), v2cross(off, tg));
        jac = v2add(jac, jo);
        disp = v2add(disp, v2mul(jo, dAng));
    }
}

// Anchor mode of shape i (a manifold's shapes are the two halves of its key).
int shapeAnchorMode(World2 *w, int i)
{
    return anchorModeOf((int)w->sType[i], w->sVR[i].y, w->sRad[i]);
}

CEval evalContact(vec2 nrm, float mu, vec2 rA, vec2 rB, float angA, float angB, int modeA, int modeB,
                  float radA, float radB, vec2 dqALin, float dqAAng, vec2 dqBLin, float dqBAng,
                  vec2 c0, vec2 pen, vec2 lam, float alpha, float tanShift)
{
    CEval e;
    vec2 tg = v2perp(nrm);

    // d(xA - xB)/d(angle) for each row: A moves with +rA x, B with -rB x.
    vec2 uA, uB;
    anchorMotion(rA, angA, dqAAng, modeA, v2mul(nrm, -radA), nrm, tg, e.jA, uA);
    anchorMotion(rB, angB, dqBAng, modeB, v2mul(nrm, radB), nrm, tg, e.jB, uB);
    e.jB = v2neg(e.jB);

    // Eq. 18 forgives alpha of the error the step started with. Only an overlap is an error: a
    // speculative contact's gap (c0.x > 0) is kept whole, or the body would stop short of the
    // surface, alpha of the way across the gap.
    float a1 = 1.0f - alpha;
    e.C = vec2(c0.x - alpha * min(c0.x, 0.0f) + v2dot(nrm, dqALin) - v2dot(nrm, dqBLin) + uA.x - uB.x,
               c0.y * a1 + v2dot(tg, dqALin) - v2dot(tg, dqBLin) + uA.y - uB.y + tanShift);

    e.F = vec2(pen.x * e.C.x + lam.x, pen.y * e.C.y + lam.y);
    e.F.x = min(e.F.x, 0.0f); // repulsion only

    // Coulomb cone: with one tangent it is an interval.
    e.bounds = abs(e.F.x) * mu;
    e.fscale = abs(e.F.y);
    if (e.fscale > e.bounds && e.fscale > 0.0f)
        e.F.y = e.F.y * (e.bounds / e.fscale);
    return e;
}

void stampManifold(World2 *w, int mi, bool isA, vec2 dqSelfLin, float dqSelfAng, float angSelf, inout Sys3 s)
{
    int cnt = w->cur.count[mi];
    if (cnt <= 0)
        return;
    int ia = w->cur.bodyA[mi];
    int ib = w->cur.bodyB[mi];
    vec2 nrm = w->cur.normal[mi];
    vec2 tg = v2perp(nrm);
    float mu = w->cur.friction[mi];
    int off = w->cur.offset[mi];
    float sLin = isA ? 1.0f : -1.0f;
    int shA = (int)(uint)(w->cur.key[mi] >> 32);
    int shB = (int)(uint)(w->cur.key[mi] & uint64_t(0xFFFFFFFFu));
    int modeA = shapeAnchorMode(w, shA);
    int modeB = shapeAnchorMode(w, shB);
    float radA = w->sRad[shA];
    float radB = w->sRad[shB];
    // A conveyor's surface speed: the tangent row aims at a relative slide of that much per step.
    float tanShift = (w->sMat[shA].x + w->sMat[shB].x) * w->dt;

    int other = isA ? ib : ia;
    float angOther = w->ang[other];
    vec2 dqOtherLin = v2sub(w->pos[other], w->initPos[other]);
    float dqOtherAng = angOther - w->initAng[other];

    float angA = angSelf, angB = angOther;
    vec2 dqALin = dqSelfLin, dqBLin = dqOtherLin;
    float dqAAng = dqSelfAng, dqBAng = dqOtherAng;
    if (!isA)
    {
        angA = angOther;
        angB = angSelf;
        dqALin = dqOtherLin;
        dqBLin = dqSelfLin;
        dqAAng = dqOtherAng;
        dqBAng = dqSelfAng;
    }

    float sumN = 0.0f;
    for (int c = 0; c < cnt; c++)
    {
        int o = off + c;
        vec2 pen = w->cur.penalty[o];
        CEval e = evalContact(nrm, mu, w->cur.rA[o], w->cur.rB[o], angA, angB, modeA, modeB, radA, radB, dqALin, dqAAng,
                              dqBLin, dqBAng,
                              w->cur.c0[o], pen, w->cur.lambda[o], w->alpha, tanShift);
        vec2 jAng = e.jA;
        if (!isA)
            jAng = e.jB;

        s.l00 += pen.x * nrm.x * nrm.x + pen.y * tg.x * tg.x;
        s.l01 += pen.x * nrm.x * nrm.y + pen.y * tg.x * tg.y;
        s.l11 += pen.x * nrm.y * nrm.y + pen.y * tg.y * tg.y;
        s.a += pen.x * jAng.x * jAng.x + pen.y * jAng.y * jAng.y;
        s.c0 += sLin * (pen.x * jAng.x * nrm.x + pen.y * jAng.y * tg.x);
        s.c1 += sLin * (pen.x * jAng.x * nrm.y + pen.y * jAng.y * tg.y);
        s.r0 += sLin * (e.F.x * nrm.x + e.F.y * tg.x);
        s.r1 += sLin * (e.F.x * nrm.y + e.F.y * tg.y);
        s.ra += jAng.x * e.F.x + jAng.y * e.F.y;
        sumN += abs(e.F.x);
    }

    // Rolling resistance: a bounded angular row on the relative rotation since x^t.
    float4 rl = w->cur.roll[mi];
    if (rl.y > 0.0f)
    {
        float bound = rl.z * sumN;
        float F = clampf(rl.y * (dqAAng - dqBAng) + rl.x, -bound, bound);
        s.a += rl.y;
        s.ra += (isA ? 1.0f : -1.0f) * F;
    }
}

// ---------------------------------------------------------------------------
// Joints: per-axis rows (JointAxis2 in types2d.slang, model in core/joint2d.h).
//
// A joint has three axes -- translation along frame-A x and y (a distance joint: along the line
// between the anchors) and the relative angle -- and each axis up to three rows: sub 0 the
// position row (lock = hard with a multiplier, spring = soft), sub 1 the motor row, sub 2 the
// limit row. Row (axis, sub) of joint j lives at index j * 9 + sub * 3 + axis in jC0/jPen/jLam.
// Angular rows are scaled by the joint's arm so every row is in newtons / metres.
//
// jointRow() is the one place that turns a coordinate into (error, bounded force, active):
// the primal stamps its result, the dual stores it back and ramps the penalty from it.
// ---------------------------------------------------------------------------
static const float J_BIG = 3.0e38f;

struct JGeo
{
    vec2 rAW;
    vec2 rBW;
    vec2 d; // anchor B minus anchor A
    vec2 u0;
    vec2 u1;
    float s0;
    float s1;
    float s2;
    float arm;
    bool dist;
};

JGeo jointGeoAt(World2 *w, int j, vec2 *P, float *A)
{
    int ia = w->jA[j];
    int ib = w->jB[j];
    JGeo g;
    g.rAW = vec2(0.0f, 0.0f);
    vec2 aA = w->jRA[j]; // a world anchor is already a world point
    float angA = 0.0f;
    if (ia >= 0)
    {
        angA = A[ia];
        g.rAW = v2rot(angA, aA);
        aA = v2add(P[ia], g.rAW);
    }
    g.rBW = v2rot(A[ib], w->jRB[j]);
    g.d = v2sub(v2add(P[ib], g.rBW), aA);
    g.dist = (w->jAxis[j * 3].flags & 2u) != 0u;
    if (g.dist)
    {
        float l = v2len(g.d);
        g.u0 = l > 1.0e-9f ? v2mul(g.d, 1.0f / l) : vec2(1.0f, 0.0f);
        g.u1 = v2perp(g.u0);
        g.s0 = l;
        g.s1 = 0.0f;
    }
    else
    {
        g.u0 = v2rot(angA + w->jFrameA[j], vec2(1.0f, 0.0f));
        g.u1 = v2perp(g.u0);
        g.s0 = v2dot(g.u0, g.d);
        g.s1 = v2dot(g.u1, g.d);
    }
    g.s2 = A[ib] - angA - w->jRest[j];
    g.arm = w->jArm[j];
    return g;
}

JGeo jointGeo(World2 *w, int j)
{
    return jointGeoAt(w, j, w->pos, w->ang);
}

float jointCoord(JGeo g, int ax)
{
    return ax == 0 ? g.s0 : (ax == 1 ? g.s1 : g.s2);
}

struct JRow
{
    bool exists;
    bool hard;   // carries a multiplier
    bool stamp;  // adds curvature: the unclamped force is strictly inside its bounds
    float C;     // error after Eq. 18 stabilisation, scaled units
    float F;     // force, clamped to [lo, hi]
    float lo;
    float hi;
    float pen;
};

// The row (ax, sub) of joint j at axis coordinate s. Force bounds apply to pen * C + lambda, not to
// lambda alone: a motor saturates on the force it would apply, and a limit is one-sided so a
// multiplier warm-started on the other side clamps to zero instead of walking the joint to the
// opposite stop.
JRow jointRow(World2 *w, int j, int ax, int sub, JointAxis2 a, float s)
{
    JRow r;
    r.exists = false;
    r.hard = false;
    r.stamp = false;
    r.C = 0.0f;
    r.F = 0.0f;
    r.lo = -J_BIG;
    r.hi = J_BIG;
    r.pen = 0.0f;
    float sc = 1.0f;
    if (ax == 2)
    {
        sc = w->jArm[j];
        if (sc <= 0.0f)
            return r;
    }
    int idx = j * 9 + sub * 3 + ax;
    float c0 = w->jC0[idx];
    float alpha = w->alpha;

    if (sub == 0)
    {
        if (a.pos == 1u)
        {
            r.exists = true;
            r.hard = true;
            r.C = sc * ((s - a.rest) - alpha * (c0 - a.rest));
        }
        else if (a.pos == 2u)
        {
            r.exists = true;
            r.C = sc * (s - a.rest);
        }
    }
    else if (sub == 1)
    {
        if (a.motor == 1u)
        {
            r.exists = true;
            r.hard = true;
            r.C = sc * (s - c0 - a.mtarget * w->dt); // no stabilisation: a velocity, not a position
        }
        else if (a.motor == 2u)
        {
            r.exists = true;
            r.hard = true;
            r.C = sc * ((s - a.mtarget) - alpha * (c0 - a.mtarget));
        }
        if (r.exists)
        {
            float b = max(a.maxF, 0.0f) / sc;
            r.lo = -b;
            r.hi = b;
        }
    }
    else if (a.limit != 0u)
    {
        r.exists = true;
        r.hard = true;
        if (s > 0.5f * (a.lo + a.hi))
        {
            r.C = sc * ((s - a.hi) - alpha * max(c0 - a.hi, 0.0f));
            r.lo = 0.0f;
        }
        else
        {
            r.C = sc * ((s - a.lo) - alpha * min(c0 - a.lo, 0.0f));
            r.hi = 0.0f;
        }
    }
    if (!r.exists)
        return r;

    r.pen = w->jPen[idx];
    if (r.hard)
    {
        float fu = r.pen * r.C + w->jLam[idx];
        r.F = clamp(fu, r.lo, r.hi);
        r.stamp = r.pen > 0.0f && fu > r.lo && fu < r.hi;
    }
    else
    {
        r.F = r.pen * r.C;
        r.stamp = r.pen > 0.0f;
    }
    return r;
}

// d(coordinate)/d(x, y, theta) of axis ax for one body, scaled like the row.
void jointJac(JGeo g, int ax, bool isA, out vec2 jl, out float ja)
{
    if (ax == 2)
    {
        jl = vec2(0.0f, 0.0f);
        ja = isA ? -g.arm : g.arm;
        return;
    }
    vec2 u = ax == 0 ? g.u0 : g.u1;
    if (isA)
    {
        jl = v2neg(u);
        ja = -v2dot(u, v2perp(g.rAW));
        if (!g.dist)
            ja += v2dot(v2perp(u), g.d); // the axis turns with body A
    }
    else
    {
        jl = u;
        ja = v2dot(u, v2perp(g.rBW));
    }
}

void stampJoint(World2 *w, int j, bool isA, inout Sys3 s)
{
    if (w->jBroken[j] != 0)
        return;
    JGeo g = jointGeo(w, j);
    vec2 rW = isA ? g.rAW : g.rBW;
    vec2 fLin = vec2(0.0f, 0.0f);
    for (int ax = 0; ax < 3; ax++)
    {
        JointAxis2 a = w->jAxis[j * 3 + ax];
        float coord = jointCoord(g, ax);
        for (int sub = 0; sub < 3; sub++)
        {
            JRow r = jointRow(w, j, ax, sub, a, coord);
            if (!r.exists)
                continue;
            vec2 jl;
            float ja;
            jointJac(g, ax, isA, jl, ja);
            if (r.stamp)
            {
                float p = r.pen;
                s.l00 += p * jl.x * jl.x;
                s.l01 += p * jl.x * jl.y;
                s.l11 += p * jl.y * jl.y;
                s.c0 += p * jl.x * ja;
                s.c1 += p * jl.y * ja;
                s.a += p * ja * ja;
            }
            s.r0 += r.F * jl.x;
            s.r1 += r.F * jl.y;
            s.ra += r.F * ja;
            if (ax < 2)
                fLin = v2add(fLin, v2mul(ax == 0 ? g.u0 : g.u1, r.F));
        }
    }
    // Geometric stiffness, column-norm diagonal approximation: |F . r|.
    s.a += abs(v2dot(fLin, rW));
}

bool jointDormant(World2 *w, int ia, int ib)
{
    bool stillA = ia < 0 || w->mass[ia] <= 0.0f || w->awake[ia] == 0;
    bool stillB = ib < 0 || w->mass[ib] <= 0.0f || w->awake[ib] == 0;
    return stillA && stillB;
}

// Inverse effective mass along a world axis through the joint anchors (J M^-1 J^T).
float jointInvMass(World2 *w, int ia, int ib, vec2 rAW, vec2 rBW, vec2 axis)
{
    float inv = 0.0f;
    if (ia >= 0)
    {
        if (w->mass[ia] > 0.0f)
            inv += 1.0f / w->mass[ia];
        float iA = w->moment[ia];
        float cA = v2cross(rAW, axis);
        if (iA > 0.0f)
            inv += cA * cA / iA;
    }
    if (w->mass[ib] > 0.0f)
        inv += 1.0f / w->mass[ib];
    float iB = w->moment[ib];
    float cB = v2cross(rBW, axis);
    if (iB > 0.0f)
        inv += cB * cB / iB;
    return inv;
}

// The same as a penalty (m_eff / dt^2).
float jointEffK(World2 *w, int ia, int ib, vec2 rAW, vec2 rBW, vec2 axis)
{
    float inv = jointInvMass(w, ia, ib, rAW, rBW, axis);
    float dt = w->dt;
    return inv > 0.0f ? clampf(1.0f / (inv * dt * dt), PENALTY_MIN, PENALTY_MAX) : PENALTY_MIN;
}

void jointInitBody(World2 *w, int j, vec2 *P, float *A)
{
    if ((uint)j >= w->jointCount || w->jBroken[j] != 0)
        return;
    int ia = w->jA[j];
    int ib = w->jB[j];
    if (jointDormant(w, ia, ib))
        return;

    JGeo g = jointGeoAt(w, j, P, A);
    float alpha = w->alpha;
    float gamma = w->gamma;

    // Clamp to the material stiffness and to the effective-mass baseline, which stops a
    // warm-started penalty ratcheting past what the bodies involved can support.
    float kx = jointEffK(w, ia, ib, g.rAW, g.rBW, g.u0);
    float ky = jointEffK(w, ia, ib, g.rAW, g.rBW, g.u1);
    float invI = 0.0f;
    if (ia >= 0 && w->moment[ia] > 0.0f)
        invI += 1.0f / w->moment[ia];
    if (w->moment[ib] > 0.0f)
        invI += 1.0f / w->moment[ib];
    float wAng = invI * g.arm * g.arm;
    float kAng = wAng > 0.0f ? clampf(1.0f / (wAng * w->dt * w->dt), PENALTY_MIN, PENALTY_MAX) : PENALTY_MIN;
    float maxKLin = min(max(10.0f * max(kx, ky), 1000.0f), JOINT_PENALTY_MAX);
    float maxKAng = min(max(10.0f * kAng, 1000.0f), JOINT_PENALTY_MAX);

    for (int ax = 0; ax < 3; ax++)
    {
        JointAxis2 a = w->jAxis[j * 3 + ax];
        float coord = jointCoord(g, ax);
        float maxK = ax == 2 ? maxKAng : maxKLin;
        for (int sub = 0; sub < 3; sub++)
        {
            int idx = j * 9 + sub * 3 + ax;
            w->jC0[idx] = coord;
            JRow r = jointRow(w, j, ax, sub, a, coord);
            if (!r.exists)
            {
                w->jPen[idx] = 0.0f;
                w->jLam[idx] = 0.0f;
                continue;
            }
            // Warm start; a multiplier on the wrong side of its bounds is dropped here.
            w->jLam[idx] = r.hard ? clamp(w->jLam[idx] * alpha * gamma, r.lo, r.hi) : 0.0f;
            if (sub == 0 && a.pos == 2u && a.hertz > 0.0f)
            {
                // A spring with a frequency: k = m_eff (2 pi f)^2, fixed for the step. Angular
                // rows are in arm-scaled units, so divide by arm^2.
                float w2 = 6.28318531f * a.hertz;
                w2 = w2 * w2;
                float k;
                if (ax == 2)
                    k = invI > 0.0f ? w2 / (invI * g.arm * g.arm) : PENALTY_MIN;
                else
                {
                    float inv = jointInvMass(w, ia, ib, g.rAW, g.rBW, ax == 0 ? g.u0 : g.u1);
                    k = inv > 0.0f ? w2 / inv : PENALTY_MIN;
                }
                w->jPen[idx] = clampf(k, 0.0001f, JOINT_PENALTY_MAX);
                continue;
            }
            float pen = clampf(w->jPen[idx] * gamma, PENALTY_MIN, JOINT_PENALTY_MAX);
            float cap = maxK;
            if (!r.hard)
                cap = min(a.stiff, maxK);
            w->jPen[idx] = min(pen, cap);
        }
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csJointInit2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    jointInitBody(w, (int)tid.x, w->pos, w->ang);
}

// csJointInit2D then csIntegrate2D in one dispatch. The joint warm start must see x^t, the pose
// before integrate moves it; integrate rewrites pos/ang in the same dispatch, so the joint part
// reads the pre-step copy (savePos/saveAng, taken by saveState after the patches and identical to
// pos/ang at this point of every pass). The two parts touch disjoint data otherwise: jointInit
// writes only joint arrays, integrate only body arrays, and both only read mass/moment/awake.
[shader("compute")]
[numthreads(256, 1, 1)]
void csIntegrateJointInit2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    if (tid.x < w->jointCount)
        jointInitBody(w, (int)tid.x, w->savePos, w->saveAng);
    if (tid.x < w->bodyCount)
        integrateBody(w, tid.x);
}

void jointDualBody(World2 *w, int j)
{
    if ((uint)j >= w->jointCount || w->jBroken[j] != 0)
        return;
    int ia = w->jA[j];
    int ib = w->jB[j];
    if (jointDormant(w, ia, ib))
        return;

    JGeo g = jointGeo(w, j);
    vec2 fB = vec2(0.0f, 0.0f); // force on body B
    float tqB = 0.0f;
    float lockSq = 0.0f;
    for (int ax = 0; ax < 3; ax++)
    {
        JointAxis2 a = w->jAxis[j * 3 + ax];
        float coord = jointCoord(g, ax);
        float beta = ax < 2 ? w->betaLin : w->betaAng;
        for (int sub = 0; sub < 3; sub++)
        {
            JRow r = jointRow(w, j, ax, sub, a, coord);
            if (!r.exists)
                continue;
            int idx = j * 9 + sub * 3 + ax;
            float pen = r.pen;
            if (r.hard)
            {
                w->jLam[idx] = r.F;
                // A saturated motor or an inactive limit is at a bound: freeze its penalty.
                if (r.stamp)
                    pen = min(pen + beta * abs(r.C), JOINT_PENALTY_MAX);
                if (sub == 0 && ax < 2)
                    lockSq += r.F * r.F;
            }
            else if (a.hertz <= 0.0f)
                pen = min(pen + beta * abs(r.C), min(a.stiff, JOINT_PENALTY_MAX));
            w->jPen[idx] = pen;

            if (ax == 2)
                tqB -= r.F * g.arm;
            else
                fB = v2sub(fB, v2mul(ax == 0 ? g.u0 : g.u1, r.F));
        }
    }

    // Fracture: a joint whose hard linear multipliers (a force, in N) exceed its break force is
    // cut for good. Both ends are now free bodies, so they stay awake.
    if (lockSq > w->jFrac[j] * w->jFrac[j])
    {
        w->jBroken[j] = 1;
        for (int k = 0; k < 9; k++)
        {
            w->jPen[j * 9 + k] = 0.0f;
            w->jLam[j * 9 + k] = 0.0f;
        }
        w->jForce[j] = vec3(0.0f, 0.0f, 0.0f);
        int old;
        InterlockedAdd(w->counters[4], 1, old);
        return;
    }
    w->jForce[j] = vec3(fB.x, fB.y, tqB);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csJointDual2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    jointDualBody(args->w, (int)tid.x);
}

// ---------------------------------------------------------------------------
// Primal: one dispatch per colour, one body per thread.
// ---------------------------------------------------------------------------
void primalBody(World2 *w, int color, int t)
{
    int start = w->colorStart[color];
    int count = w->colorEnd[color] - start;
    if (t >= count)
        return;
    int i = w->colorBodies[start + t];
    float m = w->mass[i];
    if (m <= 0.0f || w->awake[i] == 0)
        return;

    float invDt2 = 1.0f / (w->dt * w->dt);
    float inertia = w->moment[i];
    vec2 pos = w->pos[i];
    float th = w->ang[i];
    vec2 dqLin = v2sub(pos, w->initPos[i]);
    float dqAng = th - w->initAng[i];
    vec2 inert = w->inertPos[i];

    Sys3 s;
    s.l00 = m * invDt2;
    s.l01 = 0.0f;
    s.l11 = m * invDt2;
    s.c0 = 0.0f;
    s.c1 = 0.0f;
    s.a = inertia * invDt2;
    s.r0 = m * invDt2 * (pos.x - inert.x);
    s.r1 = m * invDt2 * (pos.y - inert.y);
    s.ra = inertia * invDt2 * (th - w->inertAng[i]);

    int e0 = w->gStart[i];
    int e1 = w->gEnd[i];
    for (int e = e0; e < e1; e++)
    {
        uint64_t key = w->gKey[e];
        int idx = graphIndex(key);
        bool isA = w->gIsA[e] != 0;
        if (graphType(key) == FORCE_JOINT)
            stampJoint(w, idx, isA, s);
        else
            stampManifold(w, idx, isA, dqLin, dqAng, th, s);
    }

    // Motion locks: drop the locked degree of freedom from the system (identity row, no coupling)
    // so the others solve as if it were not there and it takes no update.
    uint flags = asuint(w->bProps[i].w);
    if ((flags & (BODY_LOCK_X | BODY_LOCK_Y | BODY_LOCK_ROT)) != 0u)
    {
        if ((flags & BODY_LOCK_X) != 0u)
        {
            s.l00 = 1.0f;
            s.l01 = 0.0f;
            s.c0 = 0.0f;
            s.r0 = 0.0f;
        }
        if ((flags & BODY_LOCK_Y) != 0u)
        {
            s.l11 = 1.0f;
            s.l01 = 0.0f;
            s.c1 = 0.0f;
            s.r1 = 0.0f;
        }
        if ((flags & BODY_LOCK_ROT) != 0u)
        {
            s.a = 1.0f;
            s.c0 = 0.0f;
            s.c1 = 0.0f;
            s.ra = 0.0f;
        }
    }

    float dx, dy, dth;
    solve3(s.l00, s.l01, s.c0, s.l11, s.c1, s.a, -s.r0, -s.r1, -s.ra, dx, dy, dth);
    w->pos[i] = vec2(pos.x + dx, pos.y + dy);
    w->ang[i] = th + dth;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csPrimal2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    primalBody(args->w, (int)args->a, (int)tid.x);
}

// ---------------------------------------------------------------------------
// Dual for contacts: one thread per manifold slot.
// ---------------------------------------------------------------------------
void dualBody(World2 *w, uint t)
{
    if (t >= w->pairCap)
        return;
    int cnt = w->cur.count[t];
    if (cnt <= 0)
        return;

    int ia = w->cur.bodyA[t];
    int ib = w->cur.bodyB[t];
    // A manifold between bodies that cannot move (asleep, removed, static) is carried as is.
    if ((w->mass[ia] <= 0.0f || w->awake[ia] == 0) && (w->mass[ib] <= 0.0f || w->awake[ib] == 0))
        return;
    vec2 nrm = w->cur.normal[t];
    float mu = w->cur.friction[t];
    int off = w->cur.offset[t];
    float angA = w->ang[ia];
    float angB = w->ang[ib];
    vec2 dqALin = v2sub(w->pos[ia], w->initPos[ia]);
    vec2 dqBLin = v2sub(w->pos[ib], w->initPos[ib]);
    float dqAAng = angA - w->initAng[ia];
    float dqBAng = angB - w->initAng[ib];
    int shA = (int)(uint)(w->cur.key[t] >> 32);
    int shB = (int)(uint)(w->cur.key[t] & uint64_t(0xFFFFFFFFu));
    int modeA = shapeAnchorMode(w, shA);
    int modeB = shapeAnchorMode(w, shB);
    float radA = w->sRad[shA];
    float radB = w->sRad[shB];
    float betaLin = w->betaLin;
    float tanShift = (w->sMat[shA].x + w->sMat[shB].x) * w->dt;
    float sumN = 0.0f;

    for (int c = 0; c < cnt; c++)
    {
        int o = off + c;
        vec2 pen = w->cur.penalty[o];
        CEval e = evalContact(nrm, mu, w->cur.rA[o], w->cur.rB[o], angA, angB, modeA, modeB, radA, radB, dqALin, dqAAng,
                              dqBLin, dqBAng,
                              w->cur.c0[o], pen, w->cur.lambda[o], w->alpha, tanShift);
        w->cur.lambda[o] = e.F;
        sumN += abs(e.F.x);
        uint stk = (uint)w->cur.stick[o];

        if (e.F.x < 0.0f)
            pen.x = min(pen.x + betaLin * abs(e.C.x), PENALTY_MAX);
        if (e.fscale <= e.bounds)
        {
            pen.y = min(pen.y + betaLin * abs(e.C.y), PENALTY_MAX);
            stk = abs(e.C.y) < STICK_THRESH ? 1u : 0u;
        }
        w->cur.stick[o] = (uint8_t)stk;
        w->cur.penalty[o] = pen;
    }

    // Rolling resistance row: the multiplier is a bounded torque, the penalty ramps while it is
    // inside the bound.
    float4 rl = w->cur.roll[t];
    if (rl.y > 0.0f)
    {
        float C = dqAAng - dqBAng;
        float bound = rl.z * sumN;
        float F = rl.y * C + rl.x;
        float pr = rl.y;
        if (abs(F) <= bound)
            pr = min(pr + w->betaAng * abs(C), PENALTY_MAX);
        else
            F = clampf(F, -bound, bound);
        w->cur.roll[t] = float4(F, pr, rl.z, 0.0f);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csDual2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    dualBody(args->w, tid.x);
}

// csDual2D and csJointDual2D in one dispatch: contact duals on threads below pairCap, joint duals
// on threads below jointCount. The two write disjoint state and only read poses, so one dispatch
// equals the pair of them.
[shader("compute")]
[numthreads(128, 1, 1)]
void csDualAll2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    dualBody(w, tid.x);
    jointDualBody(w, (int)tid.x);
}

// ---------------------------------------------------------------------------
// The whole iteration loop in ONE workgroup, for worlds small enough that the per-dispatch and
// per-barrier floor dominates. Same bodies, same order as the multi-dispatch path: for each
// iteration, every colour's primal in colour order, then the contact dual, then the joint dual.
// A device-memory barrier with group sync replaces each pipeline barrier. Every thread runs every
// loop bound (no early return), so the barriers are uniform.
// ---------------------------------------------------------------------------
static const int ITER_GROUP = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void csIterate2D(uint3 gtid: SV_GroupThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    int lane = (int)gtid.x;
    int iters = (int)args->a;
    int colors = w->colorBound;
    for (int it = 0; it < iters; it++)
    {
        for (int c = 0; c < colors; c++)
        {
            int count = max(w->colorEnd[c] - w->colorStart[c], 0);
            for (int t = lane; t < count; t += ITER_GROUP)
                primalBody(w, c, t);
            DeviceMemoryBarrierWithGroupSync();
        }
        // Slots at or past the live pair count carry count 0 (csNarrowphase2D clears them), which
        // dualBody skips, so bounding the loop by it visits the same manifolds.
        uint live = min((uint)w->counters[0], w->pairCap);
        for (uint t = (uint)lane; t < live; t += (uint)ITER_GROUP)
            dualBody(w, t);
        // No barrier between the two duals: they write disjoint state (manifold vs joint
        // multipliers) and both only read poses, which no one writes here.
        for (int j = lane; j < (int)w->jointCount; j += ITER_GROUP)
            jointDualBody(w, j);
        DeviceMemoryBarrierWithGroupSync();
    }
}
