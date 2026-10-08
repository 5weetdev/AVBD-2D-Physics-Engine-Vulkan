// Continuous collision for the 2D solver: after the solve, a body that moved more than a fraction
// of its smallest extent is swept against the geometry it may have tunnelled through, and clamped
// to the pose of the earliest time of impact (the same scheme as Box2D v3's b2SolveContinuous).
//
// Why a pose clamp after csVelocity: the solver's velocity (pos - x^t) / dt is left alone, so
// the body keeps its momentum and the next step's contact resolves the tangential motion, exactly
// as Box2D keeps the velocity and moves the body to the TOI pose. x^t (initPos / initAng) is not
// touched either; the next step integrates from the clamped pose, so the primal never sees it.
//
// A hit is only acted on when the solver's final pose really is on the far side: the fast shape's
// centroid must end past the contact point along the TOI normal, within the core distance. An
// ordinary landing or stack ends well on the near side and is left to the solver. (Box2D clamps
// on any hit; here speculative contacts and the penalty solve already handle those, and clamping
// them would only change results that are fine.)

import maths2d;
import types2d;
import collide2d;
import distance2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const float CORE_FRACTION = 0.25f;  // core distance, as a fraction of the body's min extent
static const int CCD_MAX_RECT = 16;        // widest cell walk, in cells per axis
static const int CCD_MAX_VISITS = 2048;    // candidates one body examines
static const int CCD_MAX_SHAPES = 8;       // shapes of the fast body that are swept
static const float CCD_PAD = 0.02f;

// One flag per dynamic body (indexed like `dyn`): awake, and moved more than ccdSafety times its
// smallest extent this step (translation plus the rim's travel from rotation).
[shader("compute")]
[numthreads(256, 1, 1)]
void csCcdFlag2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->dynCount)
        return;
    int i = w->dyn[t];
    uint8_t flag = 0;
    if (w->ccdOn != 0 && w->mass[i] > 0.0f && w->awake[i] != 0 && w->sleepTimer[i] >= 0 && w->bShape[i].y > 0)
    {
        vec2 ext = w->bExt[i];
        float move = v2len(v2sub(w->pos[i], w->initPos[i])) + abs(w->ang[i] - w->initAng[i]) * ext.y;
        if (finitef(move) && move > w->ccdSafety * ext.x)
            flag = 1;
    }
    w->ccdFlag[t] = flag;
}

// The indirect dispatch size of csCcd2D, from the compacted count.
[shader("compute")]
[numthreads(1, 1, 1)]
void csCcdArgs2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    int n = max(w->ccdCount[0], 0);
    w->ccdArgs[0] = (n + 63) / 64;
    w->ccdArgs[1] = 1;
    w->ccdArgs[2] = 1;
}

int ccdCell(float x, float cell) { return (int)floor(x / cell); }

uint ccdSlot(int cx, int cy, uint mask)
{
    uint h = ((uint)cx * 73856093u) ^ ((uint)cy * 19349663u);
    h ^= h >> 15;
    return h & mask;
}

bool ccdIgnored(World2 *w, uint64_t key)
{
    int lo = 0;
    int hi = (int)w->ignoreCount - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        uint64_t k = w->ignore[mid];
        if (k == key)
            return true;
        if (k < key)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return false;
}

// Box2D's shape filter, as filterAllows in broadphase2d.slang.
bool ccdFilterAllows(World2 *w, int a, int b)
{
    int ga = w->sGroup[a];
    if (ga != 0 && ga == w->sGroup[b])
        return ga > 0;
    return (w->sMask[b] & w->sCat[a]) != 0 && (w->sMask[a] & w->sCat[b]) != 0;
}

vec2 qOf(float a) { return vec2(cos(a), sin(a)); }

// Shape i as a convex proxy in its body's frame (the solver's body position is the centre of
// mass). Mirrors buildShape in narrowphase2d.slang.
Proxy ccdProxy(World2 *w, int i)
{
    Proxy pr;
    pr.p[0] = vec2(0.0f, 0.0f);
    pr.count = 0;
    pr.radius = w->sRad[i];
    int shape = (int)w->sType[i];
    int2 vr = w->sVR[i];
    vec2 off = w->sOff[i];
    float sa = w->sAng[i];
    if (shape == SHAPE_BOX)
    {
        vec2 h = w->sHalf[i];
        pr.count = 4;
        pr.radius = 0.0f;
        pr.p[0] = v2add(off, v2rot(sa, vec2(-h.x, -h.y)));
        pr.p[1] = v2add(off, v2rot(sa, vec2(h.x, -h.y)));
        pr.p[2] = v2add(off, v2rot(sa, vec2(h.x, h.y)));
        pr.p[3] = v2add(off, v2rot(sa, vec2(-h.x, h.y)));
        return pr;
    }
    if (shape == SHAPE_CIRCLE)
    {
        pr.count = 1;
        pr.p[0] = vr.y > 0 ? v2add(off, v2rot(sa, w->verts[vr.x])) : off;
        if (vr.y == 0)
            pr.radius = w->sHalf[i].x;
        return pr;
    }
    if (shape == SHAPE_CHAIN)
    {
        pr.count = 2;
        pr.radius = 0.0f;
        pr.p[0] = v2add(off, v2rot(sa, w->verts[vr.x + 1]));
        pr.p[1] = v2add(off, v2rot(sa, w->verts[vr.x + 2]));
        return pr;
    }
    pr.count = min(vr.y, PROXY_MAX);
    for (int k = 0; k < pr.count; k++)
        pr.p[k] = v2add(off, v2rot(sa, w->verts[vr.x + k]));
    if (pr.count == 0)
    {
        pr.count = 1;
        pr.p[0] = off;
    }
    return pr;
}

vec2 proxyCentroid(Proxy pr)
{
    vec2 c = vec2(0.0f, 0.0f);
    for (int i = 0; i < pr.count; i++)
        c = v2add(c, pr.p[i]);
    return v2mul(c, 1.0f / (float)pr.count);
}

struct Fast
{
    int body;
    int first;
    int count;
    float minExt;
    float maxExt;
    bool bullet;
    Sweep sw;
    vec2 lo;     // swept AABB of the body's bounding circle
    vec2 hi;
    float best;  // earliest accepted fraction; 1 = none
};

// Sweep fast body f against shape j (a static, or for a bullet a non-bullet dynamic at its final
// pose), lowering f.best when the solver's pose is on the far side of an earlier time of impact.
void ccdTestShape(World2 *w, inout Fast f, int j)
{
    int bB = w->sBody[j];
    if (bB == f.body || w->sleepTimer[bB] < 0 || ((uint)w->sFlags[j] & SHAPE_FLAG_SENSOR) != 0u)
        return;
    if (w->mass[bB] > 0.0f)
    {
        if (!f.bullet || w->bBullet[bB] != 0)
            return;
    }
    if (w->ignoreCount > 0 &&
        ccdIgnored(w, (uint64_t(min((uint)f.body, (uint)bB)) << 32) | uint64_t(max((uint)f.body, (uint)bB))))
        return;

    vec2 pB = w->pos[bB];
    float aB = w->ang[bB];
    float rB = w->bExt[bB].y + CCD_PAD;
    if (pB.x + rB < f.lo.x || pB.x - rB > f.hi.x || pB.y + rB < f.lo.y || pB.y - rB > f.hi.y)
        return;

    Proxy pb = ccdProxy(w, j);
    Sweep swB;
    swB.c1 = pB;
    swB.c2 = pB;
    swB.q1 = qOf(aB);
    swB.q2 = swB.q1;
    bool chain = (int)w->sType[j] == SHAPE_CHAIN;
    vec2 cp1 = v2add(pB, v2rot(aB, pb.p[0]));
    vec2 cp2 = v2add(pB, v2rot(aB, pb.p[min(1, pb.count - 1)]));

    float core = CORE_FRACTION * f.minExt;
    int n = min(f.count, CCD_MAX_SHAPES);
    for (int k = 0; k < n; k++)
    {
        int a = f.first + k;
        if (((uint)w->sFlags[a] & SHAPE_FLAG_SENSOR) != 0u || !ccdFilterAllows(w, a, j))
            continue;
        Proxy pa = ccdProxy(w, a);
        vec2 cen = proxyCentroid(pa);
        vec2 cs = v2add(f.sw.c1, rotQ(f.sw.q1, cen));
        vec2 ce = v2add(f.sw.c2, rotQ(f.sw.q2, cen));

        if (chain)
        {
            // One-sided: ignore a body that started behind the chain, or that did not close in
            // and ended well in front (Box2D's early-out).
            vec2 e = v2sub(cp2, cp1);
            float s1 = v2cross(v2sub(cs, cp1), e);
            float s2 = v2cross(v2sub(ce, cp1), e);
            float el = v2len(e);
            if (el > 0.0f)
            {
                s1 /= el;
                s2 /= el;
            }
            if (s1 < 0.0f || (s1 - s2 < core && s2 > core))
                continue;
        }

        ToiOut o = timeOfImpact(pa, pb, f.sw, swB, f.best);
        if (o.state == TOI_SEPARATED)
            continue;
        if (!(o.state == TOI_HIT && o.fraction > 0.0f))
        {
            // Overlapping at the start of the sweep: retry with a small core circle at the
            // shape's centre, so only deep penetration counts.
            Proxy pc2;
            pc2.p[0] = cen;
            pc2.count = 1;
            pc2.radius = core;
            o = timeOfImpact(pc2, pb, f.sw, swB, f.best);
            if (o.state != TOI_HIT)
                continue;
        }
        if (!(o.fraction > 0.0f && o.fraction < f.best))
            continue;
        // Only a final pose that is past the contact (within the core distance of it, or
        // beyond) counts as tunnelled.
        if (v2dot(v2sub(ce, o.point), o.normal) > -core)
            f.best = o.fraction;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCcd2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if ((int)t >= w->ccdCount[0])
        return;
    int b = w->ccdList[t];
    bool bullet = w->bBullet[b] != 0;
    if ((uint)(bullet ? 1 : 0) != args->a)
        return;
    int2 sh = w->bShape[b];
    if (sh.y <= 0)
        return;

    Fast f;
    f.body = b;
    f.first = sh.x;
    f.count = sh.y;
    f.minExt = w->bExt[b].x;
    f.maxExt = w->bExt[b].y;
    f.bullet = bullet;
    f.sw.c1 = w->initPos[b];
    f.sw.c2 = w->pos[b];
    f.sw.q1 = qOf(w->initAng[b]);
    f.sw.q2 = qOf(w->ang[b]);
    f.best = 1.0f;
    float r = f.maxExt + CCD_PAD;
    f.lo = vec2(min(f.sw.c1.x, f.sw.c2.x) - r, min(f.sw.c1.y, f.sw.c2.y) - r);
    f.hi = vec2(max(f.sw.c1.x, f.sw.c2.x) + r, max(f.sw.c1.y, f.sw.c2.y) + r);

    // Candidates: the grid cells over the swept box (plus a ring, since a body is binned by
    // its own sweep midpoint), then the big statics and, for a bullet, the oversize dynamics.
    float cell = w->cellSize;
    int x0 = ccdCell(f.lo.x, cell) - 1;
    int x1 = ccdCell(f.hi.x, cell) + 1;
    int y0 = ccdCell(f.lo.y, cell) - 1;
    int y1 = ccdCell(f.hi.y, cell) + 1;
    bool cut = false;
    if (x1 - x0 >= CCD_MAX_RECT)
    {
        x1 = x0 + CCD_MAX_RECT - 1;
        cut = true;
    }
    if (y1 - y0 >= CCD_MAX_RECT)
    {
        y1 = y0 + CCD_MAX_RECT - 1;
        cut = true;
    }
    int visits = 0;
    for (int cy = y0; cy <= y1 && visits <= CCD_MAX_VISITS; cy++)
    {
        for (int cx = x0; cx <= x1 && visits <= CCD_MAX_VISITS; cx++)
        {
            for (int j = w->gridHead[ccdSlot(cx, cy, w->tableMask)]; j >= 0; j = w->gridNext[j])
            {
                if (++visits > CCD_MAX_VISITS)
                {
                    cut = true;
                    break;
                }
                ccdTestShape(w, f, j);
            }
        }
    }
    for (uint s = 0; s < w->statCount; s++)
        ccdTestShape(w, f, w->stat[s]);
    if (bullet)
    {
        for (uint s = 0; s < w->overCount; s++)
            ccdTestShape(w, f, w->over[s]);
    }
    if (cut)
    {
        int old;
        InterlockedAdd(w->ccdStat[1], 1, old);
    }

    if (f.best < 1.0f)
    {
        w->pos[b] = v2add(v2mul(f.sw.c1, 1.0f - f.best), v2mul(f.sw.c2, f.best));
        w->ang[b] = w->initAng[b] * (1.0f - f.best) + w->ang[b] * f.best;
        int old;
        InterlockedAdd(w->ccdStat[0], 1, old);
    }
}
