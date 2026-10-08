// 2D narrowphase: one thread per sorted pair. Collides the two shapes, merges last step's
// dual state by feature key (the persistent contact cache), computes C0, and applies the
// alpha/gamma warm-start decay (Eq. 19), the effective-mass starting penalty and the dual damping rule -- with one tangent.
//
// Shapes are collided at the x^t positions with the warm-started angles, and the margin is widened
// by the pair's relative sweep (|dpA - dpB| this step) plus |g| dt^2. Colliding at x^t means a
// fast body cannot skip past a neighbour it is about to land on (at 100k the warm pose of a
// 50 m/s body was 0.8 m inside the pile and the pair was found on the far side); the warm angles
// mean a body turning fast still meets the features it is turning into. Round shapes' anchors are
// rotated back into the x^t frame. C0 is measured at x^t with the same body-local anchors, so
// C(x) = C0 + (motion since x^t) is consistent with the primal's dq. A point still apart at x^t
// gets C0 > 0, which the solver keeps whole (solver2d.slang evalContact); alpha only forgives
// overlap. The broadphase (broadphase2d.slang) reports pairs on the same relative sweep.

import maths2d;
import types2d;
import collide2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const float PENALTY_MIN_NP = 1.0f;
static const float PENALTY_MAX_NP = 10000000000.0f;

int findPrevManifold(World2 *w, uint64_t want)
{
    int lo = 0;
    int hi = (int)w->prevCount - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        uint64_t k = w->prev.key[mid];
        if (k == want)
            return mid;
        if (k < want)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

// k = 1 / (J M^-1 J^T dt^2) along `axis`: the XPBD generalised inverse mass.
// A contact's friction from its two shapes': the geometric mean unless the world asks otherwise.
float mixFriction(World2 *w, float a, float b)
{
    uint m = w->frictionMix;
    if (m == 2u)
        return min(a, b);
    if (m == 3u)
        return max(a, b);
    if (m == 4u)
        return 0.5f * (a + b);
    return sqrt(a * b);
}

float effectiveK(World2 *w, int ia, int ib, vec2 rAW, vec2 rBW, vec2 axis)
{
    float mA = w->mass[ia];
    float mB = w->mass[ib];
    float iA = w->moment[ia];
    float iB = w->moment[ib];
    float inv = (mA > 0.0f ? 1.0f / mA : 0.0f) + (mB > 0.0f ? 1.0f / mB : 0.0f);
    float cA = v2cross(rAW, axis);
    float cB = v2cross(rBW, axis);
    if (iA > 0.0f)
        inv += cA * cA / iA;
    if (iB > 0.0f)
        inv += cB * cB / iB;
    float dt = w->dt;
    return inv > 0.0f ? clampf(1.0f / (inv * dt * dt), PENALTY_MIN_NP, PENALTY_MAX_NP) : PENALTY_MIN_NP;
}

// Shape i placed relative to the body's centre of mass, in the body's world orientation `ang`.
vec2 shapeArm(World2 *w, int i, float ang)
{
    vec2 o = w->sOff[i];
    if (o.x == 0.0f && o.y == 0.0f)
        return o;
    return v2rot(ang, o);
}

// A shape that sits at its body's centre with the body's orientation can take the legacy
// box/circle path.
bool shapeAtBody(World2 *w, int i)
{
    vec2 o = w->sOff[i];
    return o.x == 0.0f && o.y == 0.0f && w->sAng[i] == 0.0f;
}

// Shape i as a Box2D-port shape centred at `p` (relative to the collision origin), with world
// orientation `ang`.
Shape2 buildShape(World2 *w, int i, vec2 p, float ang)
{
    Shape2 s;
    for (int k = 0; k < MAX_POLY; k++)
    {
        s.v[k] = vec2(0.0f, 0.0f);
        s.n[k] = vec2(0.0f, 0.0f);
    }
    s.centroid = p;
    s.radius = w->sRad[i];
    int shape = (int)w->sType[i];
    int2 vr = w->sVR[i];
    if (shape == SHAPE_BOX)
    {
        vec2 h = w->sHalf[i];
        s.kind = KIND_HULL;
        s.count = 4;
        s.radius = 0.0f;
        s.v[0] = v2add(p, v2rot(ang, vec2(-h.x, -h.y)));
        s.v[1] = v2add(p, v2rot(ang, vec2(h.x, -h.y)));
        s.v[2] = v2add(p, v2rot(ang, vec2(h.x, h.y)));
        s.v[3] = v2add(p, v2rot(ang, vec2(-h.x, h.y)));
        s.n[0] = v2rot(ang, vec2(0.0f, -1.0f));
        s.n[1] = v2rot(ang, vec2(1.0f, 0.0f));
        s.n[2] = v2rot(ang, vec2(0.0f, 1.0f));
        s.n[3] = v2rot(ang, vec2(-1.0f, 0.0f));
        return s;
    }
    if (shape == SHAPE_CIRCLE)
    {
        s.kind = KIND_ROUND;
        s.count = 1;
        s.v[0] = vr.y > 0 ? v2add(p, v2rot(ang, w->verts[vr.x])) : p;
        if (vr.y == 0)
            s.radius = w->sHalf[i].x; // the legacy disc's radius
        return s;
    }
    s.kind = shape == SHAPE_CHAIN ? KIND_CHAIN : KIND_HULL;
    s.count = min(vr.y, MAX_POLY);
    for (int k = 0; k < s.count; k++)
    {
        s.v[k] = v2add(p, v2rot(ang, w->verts[vr.x + k]));
        s.n[k] = v2rot(ang, w->norms[vr.x + k]);
    }
    // A capsule shorter than the slop is a circle (Box2D's capsule routines divide by its length).
    if (s.kind == KIND_HULL && (s.count < 2 || (s.count == 2 && v2lenSq(v2sub(s.v[1], s.v[0])) <
                                                                    LINEAR_SLOP * LINEAR_SLOP)))
    {
        s.kind = KIND_ROUND;
        s.v[0] = s.count == 2 ? v2mul(v2add(s.v[0], s.v[1]), 0.5f) : (s.count == 1 ? s.v[0] : p);
        s.count = 1;
    }
    return s;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csNarrowphase2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->pairCap)
        return;

    // A dead slot must read as "no contacts" to every consumer sized by capacity.
    w->cur.count[t] = 0;
    w->cur.flags[t] = 0;
    uint live = min((uint)w->counters[0], w->pairCap);
    if (t >= live)
        return;

    uint64_t key = w->pairsSorted[t];
    // The key is a pair of shapes; the manifold belongs to their bodies.
    int sa = (int)(uint)(key >> 32);
    int sb = (int)(uint)(key & uint64_t(0xFFFFFFFFu));
    int ia = w->sBody[sa];
    int ib = w->sBody[sb];
    w->cur.key[t] = key;
    w->cur.roll[t] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    w->cur.bodyA[t] = ia;
    w->cur.bodyB[t] = ib;

    // A sensor pair never makes a manifold: it only records whether the shapes overlap now
    // (flags bit 0, read by events2d.slang), and only for a visitor that wants sensor events.
    uint fa = (uint)w->sFlags[sa];
    uint fb = (uint)w->sFlags[sb];
    if (((fa | fb) & SHAPE_FLAG_SENSOR) != 0u)
    {
        bool aSensor = (fa & SHAPE_FLAG_SENSOR) != 0u;
        bool bSensor = (fb & SHAPE_FLAG_SENSOR) != 0u;
        uint visitor = aSensor ? fb : fa;
        if (aSensor != bSensor && (visitor & SHAPE_FLAG_SENSOR_EVENTS) != 0u)
        {
            vec2 pA = w->pos[ia];
            vec2 pB = w->pos[ib];
            float gA = w->ang[ia];
            float gB = w->ang[ib];
            vec2 aA = shapeArm(w, sa, gA);
            vec2 aB = shapeArm(w, sb, gB);
            Shape2 qA = buildShape(w, sa, aA, gA + w->sAng[sa]);
            Shape2 qB = buildShape(w, sb, v2add(v2sub(pB, pA), aB), gB + w->sAng[sb]);
            vec2 sn, x0, y0, x1, y1;
            int j0, j1;
            if (collideShapes(qA, qB, 0.0f, sn, x0, y0, j0, x1, y1, j1) > 0)
                w->cur.flags[t] = 1;
        }
        return;
    }

    // Two bodies that cannot move this step (asleep, removed or static) keep last step's
    // manifold verbatim: nothing about it changed, and re-colliding it would only decay the
    // warm start and the contact fingerprint a sleeper is woken by. No previous manifold
    // means nothing to carry, and the pair is collided normally below.
    if (w->sleepOn != 0 && w->prevCount > 0 && (w->mass[ia] <= 0.0f || w->awake[ia] == 0) &&
        (w->mass[ib] <= 0.0f || w->awake[ib] == 0))
    {
        int pj = findPrevManifold(w, key);
        int pn = pj >= 0 ? w->prev.count[pj] : 0;
        if (pn > 0)
        {
            int po = w->prev.offset[pj];
            int coff;
            InterlockedAdd(w->counters[1], pn, coff);
            if ((uint)(coff + pn) > w->contactCap)
                return; // Overflow: the host sees the required total and grows.
            w->cur.normal[t] = w->prev.normal[pj];
            w->cur.friction[t] = w->prev.friction[pj];
            w->cur.roll[t] = w->prev.roll[pj];
            w->cur.flags[t] = w->prev.flags[pj] & (uint8_t)2; // hit-once marker (events2d.slang)
            w->cur.offset[t] = coff;
            w->cur.count[t] = pn;
            for (int i = 0; i < pn; i++)
            {
                w->cur.feature[coff + i] = w->prev.feature[po + i];
                w->cur.rA[coff + i] = w->prev.rA[po + i];
                w->cur.rB[coff + i] = w->prev.rB[po + i];
                w->cur.c0[coff + i] = w->prev.c0[po + i];
                w->cur.penalty[coff + i] = w->prev.penalty[po + i];
                w->cur.lambda[coff + i] = w->prev.lambda[po + i];
                w->cur.stick[coff + i] = w->prev.stick[po + i];
            }
            return;
        }
    }

    // Collide at the x^t positions with the warm-started angles. The positions make the contact
    // speculative: a point still apart at x^t is kept up to this step's relative sweep past the
    // margin, so a falling body is stopped at the surface instead of being found deep inside
    // its neighbour. The angles meet a body turning fast with the features it turns into.
    vec2 posA0 = w->initPos[ia];
    vec2 posB0 = w->initPos[ib];
    float angA0 = w->initAng[ia];
    float angB0 = w->initAng[ib];
    float angA = w->ang[ia];
    float angB = w->ang[ib];
    int shapeA = (int)w->sType[sa];
    int shapeB = (int)w->sType[sb];

    // The margin also covers the part of gravity the adaptive warm start left out.
    vec2 sweep = v2sub(v2sub(w->pos[ia], posA0), v2sub(w->pos[ib], posB0));
    float margin = COLLISION_MARGIN2 + v2len(sweep) + (abs(w->gravityY) + abs(w->gravityX)) * w->dt * w->dt;
    int2 vrA = w->sVR[sa];
    int2 vrB = w->sVR[sb];
    float radiusA = w->sRad[sa];
    float radiusB = w->sRad[sb];
    int modeA = anchorModeOf(shapeA, vrA.y, radiusA);
    int modeB = anchorModeOf(shapeB, vrB.y, radiusB);
    vec2 n;
    Contact2 c0, c1;
    int cnt;
    if (legacyShape(shapeA, vrA.y) && legacyShape(shapeB, vrB.y) && shapeAtBody(w, sa) && shapeAtBody(w, sb))
        cnt = collide2D(posA0, angA, w->sHalf[sa], shapeA, posB0, angB, w->sHalf[sb], shapeB, margin, n, c0, c1);
    else
    {
        // The Box2D port, relative to A's x^t position. Each surface point becomes an anchor
        // in the warm-angle frame, as collide2D's are; a rounded shape's anchor is its core
        // point, the surface point moved back by the radius along the normal.
        vec2 armA = shapeArm(w, sa, angA);
        vec2 armB = shapeArm(w, sb, angB);
        Shape2 sA = buildShape(w, sa, armA, angA + w->sAng[sa]);
        Shape2 sB = buildShape(w, sb, v2add(v2sub(posB0, posA0), armB), angB + w->sAng[sb]);
        vec2 xA0, xB0, xA1, xB1;
        int k0, k1;
        cnt = collideShapes(sA, sB, margin, n, xA0, xB0, k0, xA1, xB1, k1);
        vec2 coreA = modeA == ANCHOR_CORE ? v2mul(n, radiusA) : vec2(0.0f, 0.0f);
        vec2 coreB = modeB == ANCHOR_CORE ? v2mul(n, -radiusB) : vec2(0.0f, 0.0f);
        c0.rA = v2rot(-angA, v2add(xA0, coreA));
        c0.rB = v2rot(-angB, v2add(v2sub(xB0, v2sub(posB0, posA0)), coreB));
        c0.key = k0;
        c1.rA = v2rot(-angA, v2add(xA1, coreA));
        c1.rB = v2rot(-angB, v2add(v2sub(xB1, v2sub(posB0, posA0)), coreB));
        c1.key = k1;
    }
    if (cnt <= 0)
        return;

    // A disc's anchor is where its surface faces the contact, not a material point (the solver
    // rolls it, solver2d.slang anchorMotion): keep that world direction, re-expressed in the
    // disc's x^t frame.
    if (modeA == ANCHOR_DISC)
    {
        c0.rA = v2rot(angA - angA0, c0.rA);
        c1.rA = v2rot(angA - angA0, c1.rA);
    }
    if (modeB == ANCHOR_DISC)
    {
        c0.rB = v2rot(angB - angB0, c0.rB);
        c1.rB = v2rot(angB - angB0, c1.rB);
    }
    // A rounded shape's surface is its core anchor plus this world offset (solver2d.slang).
    vec2 offA = modeA == ANCHOR_CORE ? v2mul(n, -radiusA) : vec2(0.0f, 0.0f);
    vec2 offB = modeB == ANCHOR_CORE ? v2mul(n, radiusB) : vec2(0.0f, 0.0f);

    Contact2 cs[2];
    cs[0] = c0;
    cs[1] = c1;

    // Contact cap (a fidelity trade): keep the deepest point only.
    if (w->contactCapPts == 1u && cnt > 1)
    {
        // The offsets are the same for both points and do not change which is deeper.
        float sep0 = v2dot(n, v2sub(v2add(posA0, v2rot(angA0, cs[0].rA)), v2add(posB0, v2rot(angB0, cs[0].rB))));
        float sep1 = v2dot(n, v2sub(v2add(posA0, v2rot(angA0, cs[1].rA)), v2add(posB0, v2rot(angB0, cs[1].rB))));
        if (sep1 < sep0)
            cs[0] = cs[1];
        cnt = 1;
    }
    vec2 pen[2];
    vec2 lam[2];
    bool stick[2];
    for (int i = 0; i < 2; i++)
    {
        pen[i] = vec2(0.0f, 0.0f);
        lam[i] = vec2(0.0f, 0.0f);
        stick[i] = false;
    }

    // A conveyor contact moves along the surface every step, so a kept anchor (static friction)
    // would stretch with the travel and resist it.
    float tanSpeed = w->sMat[sa].x + w->sMat[sb].x;
    bool noStick = tanSpeed != 0.0f;

    // Warm start: carry the dual state of a contact with the same feature key.
    int pi = w->prevCount > 0 ? findPrevManifold(w, key) : -1;
    if (pi >= 0)
    {
        int po = w->prev.offset[pi];
        int pn = w->prev.count[pi];
        // A disc's anchor always follows the contact. A rounded shape's core anchor is kept only
        // on a two-point (face) manifold: a single point on a rounded surface is a rolling
        // contact, and a kept core point would have rolled away from under it.
        bool roundA = modeA == ANCHOR_DISC || (modeA == ANCHOR_CORE && cnt < 2);
        bool roundB = modeB == ANCHOR_DISC || (modeB == ANCHOR_CORE && cnt < 2);
        for (int i = 0; i < cnt; i++)
        {
            for (int j = 0; j < pn; j++)
            {
                int o = po + j;
                if (w->prev.feature[o] != cs[i].key)
                    continue;
                vec2 newRA = cs[i].rA;
                vec2 newRB = cs[i].rB;
                pen[i] = w->prev.penalty[o];
                lam[i] = w->prev.lambda[o];
                stick[i] = w->prev.stick[o] != 0 && !noStick;
                // A sticking contact keeps its old anchors so static friction stays anchored.
                cs[i].rA = w->prev.rA[o];
                cs[i].rB = w->prev.rB[o];
                if (!stick[i] || roundA)
                    cs[i].rA = newRA;
                if (!stick[i] || roundB)
                    cs[i].rB = newRB;
                break;
            }
        }
    }

    int off;
    InterlockedAdd(w->counters[1], cnt, off);
    if ((uint)(off + cnt) > w->contactCap)
        return; // Overflow: the host sees the required total and grows.

    w->cur.normal[t] = n;
    w->cur.friction[t] = mixFriction(w, w->sFric[sa], w->sFric[sb]);
    w->cur.offset[t] = off;
    w->cur.count[t] = cnt;

    vec2 tgt = v2perp(n);
    float alpha = w->alpha;
    float gamma = w->gamma;

    // Rolling resistance: one angular row on the manifold when a shape asks for it and one of
    // the two is round. Penalty 0 = no row. The force bound (rr * radius * the normal force) is
    // formed by the solver from the contact's multipliers.
    float4 roll = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float rr = max(w->sMat[sa].y, w->sMat[sb].y);
    float rad = max(radiusA, radiusB);
    if (rr > 0.0f && rad > 0.0f)
    {
        float invI = 0.0f;
        if (w->mass[ia] > 0.0f && w->awake[ia] != 0 && w->moment[ia] > 0.0f)
            invI += 1.0f / w->moment[ia];
        if (w->mass[ib] > 0.0f && w->awake[ib] != 0 && w->moment[ib] > 0.0f)
            invI += 1.0f / w->moment[ib];
        if (invI > 0.0f)
        {
            float kRoll = clampf(1.0f / (invI * w->dt * w->dt), PENALTY_MIN_NP, PENALTY_MAX_NP);
            roll = float4(0.0f, kRoll, rr * rad, 0.0f);
            if (pi >= 0)
            {
                float4 pr = w->prev.roll[pi];
                if (pr.y > 0.0f)
                    roll = float4(pr.x * alpha * gamma, clampf(pr.y * gamma, PENALTY_MIN_NP, PENALTY_MAX_NP), rr * rad,
                                  0.0f);
            }
        }
    }
    w->cur.roll[t] = roll;
    for (int i = 0; i < cnt; i++)
    {
        int o = off + i;
        vec2 rAW = v2rot(angA0, cs[i].rA);
        vec2 rBW = v2rot(angB0, cs[i].rB);
        // From here rAW, rBW are the surface points' levers.
        if (modeA == ANCHOR_CORE)
            rAW = v2add(rAW, offA);
        if (modeB == ANCHOR_CORE)
            rBW = v2add(rBW, offB);
        vec2 d = v2sub(v2add(posA0, rAW), v2add(posB0, rBW));
        vec2 C0 = vec2(v2dot(n, d) + COLLISION_MARGIN2, v2dot(tgt, d));

        vec2 penPrev = pen[i];
        vec2 lamI = v2mul(lam[i], alpha * gamma);
        vec2 penI = vec2(clampf(penPrev.x * gamma, PENALTY_MIN_NP, PENALTY_MAX_NP),
                         clampf(penPrev.y * gamma, PENALTY_MIN_NP, PENALTY_MAX_NP));

        vec2 kAuto = vec2(0.0f, 0.0f);
        if (w->autoK != 0 || w->dualDamping > 0.0f)
        {
            kAuto = vec2(effectiveK(w, ia, ib, rAW, rBW, n), effectiveK(w, ia, ib, rAW, rBW, tgt));
            if (w->autoK != 0 && penPrev.x <= 0.0f)
                penI = kAuto;
        }

        if (w->dualDamping > 0.0f)
        {
            vec2 vA = v2add(w->vel[ia], v2mul(v2perp(rAW), w->angVel[ia]));
            vec2 vB = v2add(w->vel[ib], v2mul(v2perp(rBW), w->angVel[ib]));
            float vRel = v2len(v2sub(vA, vB));
            float gate = w->dualDampVel > 0.0f ? max(0.0f, 1.0f - vRel / w->dualDampVel) : 1.0f;
            float strength = w->dualDamping * gate;
            if (strength > 0.0f)
            {
                if (penI.x > kAuto.x && kAuto.x > 0.0f)
                    lamI.x *= 1.0f - strength * (1.0f - kAuto.x / penI.x);
                if (penI.y > kAuto.y && kAuto.y > 0.0f)
                    lamI.y *= 1.0f - strength * (1.0f - kAuto.y / penI.y);
            }
        }

        w->cur.feature[o] = cs[i].key;
        w->cur.rA[o] = cs[i].rA;
        w->cur.rB[o] = cs[i].rB;
        w->cur.c0[o] = C0;
        w->cur.penalty[o] = penI;
        w->cur.lambda[o] = lamI;
        w->cur.stick[o] = (uint8_t)(stick[i] ? 1u : 0u);
    }
}
