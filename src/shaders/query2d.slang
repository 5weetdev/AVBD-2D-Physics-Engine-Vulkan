// Batched scene queries on the device (rays and AABB overlaps), resolved against the final poses
// of the last step. The step's broadphase grid is swept and therefore not exact, so a batch builds
// its own cell table over the same shape lists (grid list, oversize list, big statics) and the same
// cell size: shapes that fit a cell (at most 2 x 2 cells) are chained per cell, the rest sit in a
// short brute-force list. Rays walk the cells with a 2D DDA; boxes scan their cell range and report
// each shape once, in the one cell they share. The narrow tests are term-for-term ports of
// the host query code (b2RayCast*, GJK, conservative advancement), so the host and the
// device agree to rounding. Sensors and removed bodies are invisible.

module query2d;

import maths2d;
import types2d;
import collide2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const float TOUCH_TOL = 1e-4f;
static const int MAX_DDA_CELLS = 2048;
static const int MAX_BOX_CELLS = 1024;

// ---------------------------------------------------------------------------------------------
// World-space shapes
// ---------------------------------------------------------------------------------------------
struct WShape
{
    int kind; // 0 circle, 1 polygon (box included), 2 capsule, 3 segment, 4 one-sided chain segment
    vec2 v[8];
    vec2 n[8];
    int count;
    float r;
};

struct Proxy
{
    vec2 pts[8];
    int count;
    float radius;
};

vec2 normalizedOr(vec2 a, vec2 fallback)
{
    float l = v2len(a);
    return l > 1e-12f ? v2mul(a, 1.0f / l) : fallback;
}

bool makeWorldShape(World2 *w, uint s, out WShape ws)
{
    ws.kind = 0;
    ws.count = 0;
    ws.r = 0.0f;
    for (int i = 0; i < 8; i++)
    {
        ws.v[i] = vec2(0.0f, 0.0f);
        ws.n[i] = vec2(0.0f, 0.0f);
    }
    uint b = (uint)w->sBody[s];
    float ba = w->ang[b];
    vec2 o = v2add(w->pos[b], v2rot(ba, w->sOff[s]));
    float a = ba + w->sAng[s];
    int vo = w->sVR[s].x;
    int vc = w->sVR[s].y;
    int t = (int)w->sType[s];
    if (t == SHAPE_BOX)
    {
        vec2 h = w->sHalf[s];
        vec2 lv[4] = { vec2(-h.x, -h.y), vec2(h.x, -h.y), vec2(h.x, h.y), vec2(-h.x, h.y) };
        vec2 ln[4] = { vec2(0.0f, -1.0f), vec2(1.0f, 0.0f), vec2(0.0f, 1.0f), vec2(-1.0f, 0.0f) };
        ws.kind = 1;
        ws.count = 4;
        for (int i = 0; i < 4; i++)
        {
            ws.v[i] = v2add(o, v2rot(a, lv[i]));
            ws.n[i] = v2rot(a, ln[i]);
        }
        return true;
    }
    if (t == SHAPE_CIRCLE)
    {
        ws.kind = 0;
        ws.count = 1;
        if (vc == 0)
        {
            ws.v[0] = o;
            ws.r = w->sHalf[s].x;
        }
        else
        {
            ws.v[0] = v2add(o, v2rot(a, w->verts[vo]));
            ws.r = w->sRad[s];
        }
        return true;
    }
    if (t == SHAPE_POLYGON || t == SHAPE_CAPSULE)
    {
        if (vc < 2 || vc > 8)
            return false;
        ws.count = vc;
        ws.r = w->sRad[s];
        for (int i = 0; i < vc; i++)
        {
            ws.v[i] = v2add(o, v2rot(a, w->verts[vo + i]));
            ws.n[i] = v2rot(a, w->norms[vo + i]);
        }
        ws.kind = (t == SHAPE_CAPSULE || vc == 2) ? 2 : 1;
        return true;
    }
    if (t == SHAPE_SEGMENT)
    {
        if (vc < 2)
            return false;
        ws.kind = 3;
        ws.count = 2;
        ws.v[0] = v2add(o, v2rot(a, w->verts[vo]));
        ws.v[1] = v2add(o, v2rot(a, w->verts[vo + 1]));
        return true;
    }
    if (t == SHAPE_CHAIN)
    {
        if (vc < 4)
            return false;
        ws.kind = 4;
        ws.count = 2;
        ws.v[0] = v2add(o, v2rot(a, w->verts[vo + 1]));
        ws.v[1] = v2add(o, v2rot(a, w->verts[vo + 2]));
        return true;
    }
    return false;
}

Proxy proxyOf(WShape ws)
{
    Proxy p;
    p.count = ws.count;
    p.radius = ws.r;
    for (int i = 0; i < 8; i++)
        p.pts[i] = i < ws.count ? ws.v[i] : vec2(0.0f, 0.0f);
    return p;
}

void boundsOf(Proxy p, out vec2 lo, out vec2 hi)
{
    lo = p.pts[0];
    hi = p.pts[0];
    for (int i = 1; i < p.count; i++)
    {
        lo = vec2(min(lo.x, p.pts[i].x), min(lo.y, p.pts[i].y));
        hi = vec2(max(hi.x, p.pts[i].x), max(hi.y, p.pts[i].y));
    }
    lo = vec2(lo.x - p.radius, lo.y - p.radius);
    hi = vec2(hi.x + p.radius, hi.y + p.radius);
}

// ---------------------------------------------------------------------------------------------
// GJK distance and conservative advancement (ports of the host's coreDistance / castPair)
// ---------------------------------------------------------------------------------------------
int supportOf(Proxy p, vec2 d)
{
    int best = 0;
    float bv = v2dot(p.pts[0], d);
    for (int i = 1; i < p.count; i++)
    {
        float v = v2dot(p.pts[i], d);
        if (v > bv)
        {
            bv = v;
            best = i;
        }
    }
    return best;
}

struct Sv
{
    vec2 wA;
    vec2 wB;
    vec2 w;
    float a;
    int iA;
    int iB;
};

Sv makeSv(Proxy A, Proxy B, int ia, int ib)
{
    Sv s;
    s.iA = ia;
    s.iB = ib;
    s.wA = A.pts[ia];
    s.wB = B.pts[ib];
    s.w = v2sub(s.wA, s.wB);
    s.a = 0.0f;
    return s;
}

void solveSegment(inout Sv s0, inout Sv s1, inout int n)
{
    vec2 e = v2sub(s1.w, s0.w);
    float ee = v2dot(e, e);
    float t = ee > 0.0f ? -v2dot(s0.w, e) / ee : 0.0f;
    if (t <= 0.0f)
    {
        n = 1;
        s0.a = 1.0f;
    }
    else if (t >= 1.0f)
    {
        s0 = s1;
        n = 1;
        s0.a = 1.0f;
    }
    else
    {
        s0.a = 1.0f - t;
        s1.a = t;
        n = 2;
    }
}

float segDistSq(Sv a, Sv b)
{
    vec2 e = v2sub(b.w, a.w);
    float ee = v2dot(e, e);
    float t = ee > 0.0f ? -v2dot(a.w, e) / ee : 0.0f;
    t = min(max(t, 0.0f), 1.0f);
    return v2lenSq(v2add(a.w, v2mul(e, t)));
}

bool solveTriangle(inout Sv s0, inout Sv s1, inout Sv s2, inout int n)
{
    vec2 p0 = s0.w;
    vec2 p1 = s1.w;
    vec2 p2 = s2.w;
    float c0 = v2cross(v2sub(p1, p0), v2neg(p0));
    float c1 = v2cross(v2sub(p2, p1), v2neg(p1));
    float c2 = v2cross(v2sub(p0, p2), v2neg(p2));
    if ((c0 >= 0.0f && c1 >= 0.0f && c2 >= 0.0f) || (c0 <= 0.0f && c1 <= 0.0f && c2 <= 0.0f))
    {
        float area = v2cross(v2sub(p1, p0), v2sub(p2, p0));
        if (abs(area) > 1e-20f)
        {
            s0.a = v2cross(p1, p2) / area;
            s1.a = v2cross(p2, p0) / area;
            s2.a = v2cross(p0, p1) / area;
        }
        else
        {
            s0.a = 1.0f;
            s1.a = 0.0f;
            s2.a = 0.0f;
        }
        return true;
    }
    float d01 = segDistSq(s0, s1);
    float d12 = segDistSq(s1, s2);
    float d20 = segDistSq(s2, s0);
    if (d01 <= d12 && d01 <= d20)
    {
        n = 2;
    }
    else if (d12 <= d20)
    {
        s0 = s1;
        s1 = s2;
        n = 2;
    }
    else
    {
        s1 = s0;
        s0 = s2;
        n = 2;
    }
    solveSegment(s0, s1, n);
    return false;
}

struct CoreDistance
{
    vec2 pA;
    vec2 pB;
    float dist;
};

CoreDistance coreDistance(Proxy A, Proxy B)
{
    Sv s0 = makeSv(A, B, 0, 0);
    Sv s1 = s0;
    Sv s2 = s0;
    s0.a = 1.0f;
    int n = 1;
    bool overlap = false;
    for (int iter = 0; iter < 30; iter++)
    {
        if (n == 2)
            solveSegment(s0, s1, n);
        else if (n == 3)
        {
            if (solveTriangle(s0, s1, s2, n))
            {
                overlap = true;
                break;
            }
        }
        vec2 c = vec2(0.0f, 0.0f);
        c = v2add(c, v2mul(s0.w, s0.a));
        if (n > 1)
            c = v2add(c, v2mul(s1.w, s1.a));
        if (n > 2)
            c = v2add(c, v2mul(s2.w, s2.a));
        if (v2lenSq(c) < 1e-14f)
        {
            overlap = true;
            break;
        }
        vec2 d = v2neg(c);
        int ia = supportOf(A, d);
        int ib = supportOf(B, v2neg(d));
        bool dup = (s0.iA == ia && s0.iB == ib);
        if (n > 1)
            dup = dup || (s1.iA == ia && s1.iB == ib);
        if (n > 2)
            dup = dup || (s2.iA == ia && s2.iB == ib);
        if (dup)
            break;
        if (n == 1)
            s1 = makeSv(A, B, ia, ib);
        else
            s2 = makeSv(A, B, ia, ib);
        n++;
    }
    CoreDistance r;
    r.pA = vec2(0.0f, 0.0f);
    r.pB = vec2(0.0f, 0.0f);
    r.pA = v2add(r.pA, v2mul(s0.wA, s0.a));
    r.pB = v2add(r.pB, v2mul(s0.wB, s0.a));
    if (n > 1)
    {
        r.pA = v2add(r.pA, v2mul(s1.wA, s1.a));
        r.pB = v2add(r.pB, v2mul(s1.wB, s1.a));
    }
    if (n > 2)
    {
        r.pA = v2add(r.pA, v2mul(s2.wA, s2.a));
        r.pB = v2add(r.pB, v2mul(s2.wB, s2.a));
    }
    r.dist = overlap ? 0.0f : v2len(v2sub(r.pA, r.pB));
    return r;
}

struct CastHit
{
    float fraction;
    vec2 point;
    vec2 normal;
};

bool castPair(Proxy target, Proxy caster, vec2 d, float maxFraction, bool skipInitial, out CastHit hit)
{
    hit.fraction = 0.0f;
    hit.point = vec2(0.0f, 0.0f);
    hit.normal = vec2(0.0f, 0.0f);
    float total = target.radius + caster.radius;
    Proxy moved = caster;
    float t = 0.0f;
    for (int iter = 0; iter < 40; iter++)
    {
        for (int i = 0; i < caster.count; i++)
            moved.pts[i] = v2add(caster.pts[i], v2mul(d, t));
        CoreDistance cd = coreDistance(target, moved);
        vec2 sep = v2sub(cd.pB, cd.pA);
        if (cd.dist - total <= TOUCH_TOL)
        {
            if (iter == 0 && skipInitial && cd.dist - total < 0.0f)
                return false;
            vec2 u = cd.dist > 1e-9f ? v2mul(sep, 1.0f / cd.dist) : v2neg(normalizedOr(d, vec2(1.0f, 0.0f)));
            hit.fraction = t;
            hit.normal = u;
            hit.point = v2add(cd.pA, v2mul(u, target.radius));
            return true;
        }
        vec2 u = v2mul(sep, 1.0f / cd.dist);
        float closing = -v2dot(u, d);
        if (closing <= 1e-9f)
            return false;
        t += (cd.dist - total) / closing;
        if (t > maxFraction)
            return false;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Ray tests (b2RayCastCircle / Polygon / Segment)
// ---------------------------------------------------------------------------------------------
bool rayCircle(WShape ws, vec2 o, vec2 d, float maxF, out CastHit hit)
{
    hit.fraction = 0.0f;
    hit.point = vec2(0.0f, 0.0f);
    hit.normal = vec2(0.0f, 0.0f);
    float len = v2len(d);
    if (len == 0.0f)
        return false;
    vec2 dir = v2mul(d, 1.0f / len);
    vec2 s = v2sub(o, ws.v[0]);
    float t = -v2dot(s, dir);
    vec2 c = v2add(s, v2mul(dir, t));
    float cc = v2dot(c, c);
    float rr = ws.r * ws.r;
    if (cc > rr)
        return false;
    float h = sqrt(rr - cc);
    float fr = t - h;
    if (fr < 0.0f || maxF * len < fr)
        return false;
    vec2 hp = v2add(s, v2mul(dir, fr));
    hit.fraction = fr / len;
    hit.normal = normalizedOr(hp, vec2(1.0f, 0.0f));
    hit.point = v2add(ws.v[0], v2mul(hit.normal, ws.r));
    return true;
}

bool rayPolygon(WShape ws, vec2 o, vec2 d, float maxF, out CastHit hit)
{
    hit.fraction = 0.0f;
    hit.point = vec2(0.0f, 0.0f);
    hit.normal = vec2(0.0f, 0.0f);
    float lower = 0.0f;
    float upper = maxF;
    int index = -1;
    for (int i = 0; i < ws.count; i++)
    {
        float num = v2dot(ws.n[i], v2sub(ws.v[i], o));
        float den = v2dot(ws.n[i], d);
        if (den == 0.0f)
        {
            if (num < 0.0f)
                return false;
        }
        else if (den < 0.0f && num < lower * den)
        {
            lower = num / den;
            index = i;
        }
        else if (den > 0.0f && num < upper * den)
        {
            upper = num / den;
        }
        if (upper < lower)
            return false;
    }
    if (index < 0)
        return false;
    hit.fraction = lower;
    hit.normal = ws.n[index];
    hit.point = v2add(o, v2mul(d, lower));
    return true;
}

bool raySegment(WShape ws, vec2 o, vec2 d, float maxF, bool oneSided, out CastHit hit)
{
    hit.fraction = 0.0f;
    hit.point = vec2(0.0f, 0.0f);
    hit.normal = vec2(0.0f, 0.0f);
    vec2 v1 = ws.v[0];
    vec2 v2 = ws.v[1];
    if (oneSided && v2cross(v2sub(o, v1), v2sub(v2, v1)) < 0.0f)
        return false;
    vec2 e = v2sub(v2, v1);
    float len = v2len(e);
    if (len == 0.0f)
        return false;
    vec2 eu = v2mul(e, 1.0f / len);
    vec2 normal = vec2(eu.y, -eu.x);
    float num = v2dot(normal, v2sub(v1, o));
    float den = v2dot(normal, d);
    if (den == 0.0f)
        return false;
    float t = num / den;
    if (t < 0.0f || maxF < t)
        return false;
    vec2 p = v2add(o, v2mul(d, t));
    float s = v2dot(v2sub(p, v1), eu);
    if (s < 0.0f || len < s)
        return false;
    if (num > 0.0f)
        normal = v2neg(normal);
    hit.fraction = t;
    hit.point = p;
    hit.normal = normal;
    return true;
}

bool rayShape(WShape ws, vec2 o, vec2 d, float maxF, out CastHit hit)
{
    if (ws.kind == 0)
        return rayCircle(ws, o, d, maxF, hit);
    if (ws.kind == 1 && ws.r == 0.0f)
        return rayPolygon(ws, o, d, maxF, hit);
    if (ws.kind == 3)
        return raySegment(ws, o, d, maxF, false, hit);
    if (ws.kind == 4)
        return raySegment(ws, o, d, maxF, true, hit);
    Proxy pt;
    for (int i = 0; i < 8; i++)
        pt.pts[i] = o;
    pt.count = 1;
    pt.radius = 0.0f;
    return castPair(proxyOf(ws), pt, d, maxF, true, hit);
}

// Slab test of o + t d, t in [0, maxF], against the box.
bool rayVsBox(vec2 o, vec2 d, float maxF, vec2 lo, vec2 hi)
{
    float t0 = 0.0f;
    float t1 = maxF;
    float oo[2] = { o.x, o.y };
    float dd[2] = { d.x, d.y };
    float l[2] = { lo.x, lo.y };
    float h[2] = { hi.x, hi.y };
    for (int i = 0; i < 2; i++)
    {
        if (abs(dd[i]) < 1e-20f)
        {
            if (oo[i] < l[i] || oo[i] > h[i])
                return false;
        }
        else
        {
            float a = (l[i] - oo[i]) / dd[i];
            float b = (h[i] - oo[i]) / dd[i];
            if (a > b)
            {
                float tmp = a;
                a = b;
                b = tmp;
            }
            t0 = max(t0, a);
            t1 = min(t1, b);
            if (t0 > t1)
                return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// The query-local grid
// ---------------------------------------------------------------------------------------------
uint cellSlotQ(int cx, int cy, uint mask)
{
    uint h = ((uint)cx * 73856093u) ^ ((uint)cy * 19349663u);
    h ^= h >> 15;
    return h & mask;
}

int cellOf(float x, float cell) { return (int)floor(x / cell); }

bool boxEmpty(float4 b) { return b.x > b.z || b.y > b.w; }

bool shapeVisible(World2 *w, uint s)
{
    return w->sBody[s] >= 0 && w->sleepTimer[w->sBody[s]] >= 0 &&
           (((uint)w->sFlags[s]) & SHAPE_FLAG_SENSOR) == 0u;
}

bool filterPasses(QueryBatch2 *q, uint s)
{
    World2 *w = q->w;
    return (w->sCat[s] & q->mask) != 0 && (q->cat & w->sMask[s]) != 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csQueryClear2D(uint3 tid: SV_DispatchThreadID)
{
    QueryBatch2 *q = (QueryBatch2 *)pc.argsAddr;
    uint i = tid.x;
    if (i > q->headMask)
        return;
    q->qHead[i] = -1;
    if (i == 0u)
        q->qCount[0] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csQueryInsert2D(uint3 tid: SV_DispatchThreadID)
{
    QueryBatch2 *q = (QueryBatch2 *)pc.argsAddr;
    World2 *w = q->w;
    uint t = tid.x;
    if (t >= q->insertCount)
        return;
    bool fromGrid = t < w->gridCount;
    bool fromOver = !fromGrid && t < w->gridCount + w->overCount;
    int i;
    if (fromGrid)
        i = w->grid[t];
    else if (fromOver)
        i = w->over[t - w->gridCount];
    else
        i = w->stat[t - w->gridCount - w->overCount];
    q->qBox[i] = float4(1.0f, 1.0f, -1.0f, -1.0f);
    if (!shapeVisible(w, (uint)i))
        return;
    WShape ws;
    if (!makeWorldShape(w, (uint)i, ws))
        return;
    vec2 lo, hi;
    boundsOf(proxyOf(ws), lo, hi);
    if (!(finitef(lo.x) && finitef(lo.y) && finitef(hi.x) && finitef(hi.y)))
        return;
    q->qBox[i] = float4(lo.x, lo.y, hi.x, hi.y);
    float cell = w->cellSize;
    int mx = cellOf(lo.x, cell);
    int my = cellOf(lo.y, cell);
    int nx = cellOf(hi.x, cell) - mx;
    int ny = cellOf(hi.y, cell) - my;
    if (!fromGrid || nx > 1 || ny > 1)
    {
        int idx;
        InterlockedAdd(q->qCount[0], 1, idx);
        q->qBig[idx] = i;
        return;
    }
    for (int ky = 0; ky <= ny; ky++)
    {
        for (int kx = 0; kx <= nx; kx++)
        {
            int node = i * 4 + ky * 2 + kx;
            uint slot = cellSlotQ(mx + kx, my + ky, q->headMask);
            int prev;
            InterlockedExchange(q->qHead[slot], node, prev);
            q->qNext[node] = prev;
        }
    }
}

// Tests shape s against the ray; updates the best hit. Returns true when the ray is finished
// (a hit at fraction 0).
bool rayCandidate(QueryBatch2 *q, uint s, vec2 o, vec2 d, inout float maxF, inout bool found,
                  inout int bestShape, inout CastHit best)
{
    World2 *w = q->w;
    float4 bx = q->qBox[s];
    if (boxEmpty(bx) || !filterPasses(q, s))
        return false;
    if (!rayVsBox(o, d, maxF, vec2(bx.x, bx.y), vec2(bx.z, bx.w)))
        return false;
    WShape ws;
    CastHit h;
    if (!makeWorldShape(w, s, ws) || !rayShape(ws, o, d, maxF, h))
        return false;
    best = h;
    bestShape = (int)s;
    found = true;
    if (h.fraction == 0.0f)
        return true;
    if (h.fraction > 0.0f && h.fraction < maxF)
        maxF = h.fraction;
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csQueryRays2D(uint3 tid: SV_DispatchThreadID)
{
    QueryBatch2 *q = (QueryBatch2 *)pc.argsAddr;
    World2 *w = q->w;
    uint t = tid.x;
    if (t >= q->count)
        return;
    float4 r = q->rays[t];
    vec2 o = vec2(r.x, r.y);
    vec2 d = vec2(r.z, r.w);
    float maxF = 1.0f;
    bool found = false;
    bool done = false;
    int bestShape = -1;
    CastHit best;
    best.fraction = 1.0f;
    best.point = vec2(0.0f, 0.0f);
    best.normal = vec2(0.0f, 0.0f);

    int bigN = q->qCount[0];
    for (int k = 0; k < bigN && !done; k++)
        done = rayCandidate(q, (uint)q->qBig[k], o, d, maxF, found, bestShape, best);

    float cell = w->cellSize;
    int cx = cellOf(o.x, cell);
    int cy = cellOf(o.y, cell);
    int stepX = d.x > 0.0f ? 1 : (d.x < 0.0f ? -1 : 0);
    int stepY = d.y > 0.0f ? 1 : (d.y < 0.0f ? -1 : 0);
    float inf = 3.0e38f;
    float tMaxX = inf;
    float tMaxY = inf;
    float tDeltaX = inf;
    float tDeltaY = inf;
    if (stepX != 0)
    {
        tMaxX = (((float)(stepX > 0 ? cx + 1 : cx)) * cell - o.x) / d.x;
        tDeltaX = cell / abs(d.x);
    }
    if (stepY != 0)
    {
        tMaxY = (((float)(stepY > 0 ? cy + 1 : cy)) * cell - o.y) / d.y;
        tDeltaY = cell / abs(d.y);
    }
    bool walked = false;
    int cells = 0;
    while (!done)
    {
        if (cells++ >= MAX_DDA_CELLS)
            break;
        uint slot = cellSlotQ(cx, cy, q->headMask);
        for (int n = q->qHead[slot]; n >= 0 && !done; n = q->qNext[n])
        {
            uint s = (uint)(n >> 2);
            int k = n & 3;
            float4 bx = q->qBox[s];
            if (cellOf(bx.x, cell) + (k & 1) != cx || cellOf(bx.y, cell) + (k >> 1) != cy)
                continue;
            done = rayCandidate(q, s, o, d, maxF, found, bestShape, best);
        }
        float tn = min(tMaxX, tMaxY);
        if (done || tn > maxF || tn >= inf)
        {
            walked = true;
            break;
        }
        if (tMaxX < tMaxY)
        {
            cx += stepX;
            tMaxX += tDeltaX;
        }
        else
        {
            cy += stepY;
            tMaxY += tDeltaY;
        }
    }
    if (!walked && !done)
    {
        // A ray longer than the walk allows: test every grid shape directly.
        for (uint g = 0; g < w->gridCount && !done; g++)
            done = rayCandidate(q, (uint)w->grid[g], o, d, maxF, found, bestShape, best);
    }
    float4 o0 = float4(asfloat(found ? bestShape : -1), best.fraction, best.point.x, best.point.y);
    float4 o1 = float4(best.normal.x, best.normal.y, 0.0f, 0.0f);
    q->outHit[t * 2u] = o0;
    q->outHit[t * 2u + 1u] = o1;
}

bool boxesOverlap(float4 a, float4 b)
{
    return a.x <= b.z && b.x <= a.z && a.y <= b.w && b.y <= a.w;
}

void emitId(QueryBatch2 *q, uint t, uint s, inout int found)
{
    if ((uint)found < q->maxPer)
        q->outIds[t * q->maxPer + (uint)found] = (int)s;
    found++;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csQueryAabbs2D(uint3 tid: SV_DispatchThreadID)
{
    QueryBatch2 *q = (QueryBatch2 *)pc.argsAddr;
    World2 *w = q->w;
    uint t = tid.x;
    if (t >= q->count)
        return;
    float4 qb = q->aabbs[t];
    int found = 0;
    int bigN = q->qCount[0];
    for (int k = 0; k < bigN; k++)
    {
        uint s = (uint)q->qBig[k];
        float4 bx = q->qBox[s];
        if (!boxEmpty(bx) && boxesOverlap(bx, qb) && filterPasses(q, s))
            emitId(q, t, s, found);
    }
    float cell = w->cellSize;
    int x0 = cellOf(qb.x, cell);
    int y0 = cellOf(qb.y, cell);
    int x1 = cellOf(qb.z, cell);
    int y1 = cellOf(qb.w, cell);
    bool sane = finitef(qb.x) && finitef(qb.y) && finitef(qb.z) && finitef(qb.w) && x1 >= x0 && y1 >= y0;
    if (sane && ((int64_t)(x1 - x0 + 1) * (int64_t)(y1 - y0 + 1)) <= (int64_t)MAX_BOX_CELLS)
    {
        for (int cy = y0; cy <= y1; cy++)
        {
            for (int cx = x0; cx <= x1; cx++)
            {
                uint slot = cellSlotQ(cx, cy, q->headMask);
                for (int n = q->qHead[slot]; n >= 0; n = q->qNext[n])
                {
                    uint s = (uint)(n >> 2);
                    int k = n & 3;
                    float4 bx = q->qBox[s];
                    int mx = cellOf(bx.x, cell);
                    int my = cellOf(bx.y, cell);
                    if (mx + (k & 1) != cx || my + (k >> 1) != cy)
                        continue;
                    // Report the shape in the one cell its box shares with the query's range.
                    if (cx != max(mx, x0) || cy != max(my, y0))
                        continue;
                    if (boxesOverlap(bx, qb) && filterPasses(q, s))
                        emitId(q, t, s, found);
                }
            }
        }
    }
    else if (sane)
    {
        for (uint g = 0; g < w->gridCount; g++)
        {
            uint s = (uint)w->grid[g];
            float4 bx = q->qBox[s];
            if (!boxEmpty(bx) && boxesOverlap(bx, qb) && filterPasses(q, s) &&
                (cellOf(bx.z, cell) - cellOf(bx.x, cell)) <= 1 && (cellOf(bx.w, cell) - cellOf(bx.y, cell)) <= 1)
                emitId(q, t, s, found);
        }
    }
    q->outCounts[t] = found;
}
