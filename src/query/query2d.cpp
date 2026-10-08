#include "query2d.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace avbd2d
{

namespace
{

constexpr float kTouchTol = 1e-4f;

Vec2 vmin(Vec2 a, Vec2 b) { return Vec2(std::min(a.x, b.x), std::min(a.y, b.y)); }
Vec2 vmax(Vec2 a, Vec2 b) { return Vec2(std::max(a.x, b.x), std::max(a.y, b.y)); }
Vec2 normalized(Vec2 a, Vec2 fallback)
{
    const float l = length(a);
    return l > 1e-12f ? a * (1.0f / l) : fallback;
}

// A shape in world space. kind: 0 circle, 1 polygon (box included; may be rounded), 2 capsule,
// 3 segment, 4 one-sided chain segment (p1 -> p2; the solid side is the right of the direction).
struct WShape
{
    int kind = 0;
    Vec2 v[8];
    Vec2 n[8];
    int count = 0;
    float r = 0.0f;
};

bool makeWorldShape(const Scene2D &S, int s, Vec2 bp, float ba, WShape &w)
{
    const size_t k = (size_t)s;
    const Vec2 o = bp + rotate(ba, S.shOff[k]);
    const float a = ba + S.shAng[k];
    const int vo = S.shVOff[k], vc = S.shVCnt[k];
    w = WShape();
    switch (S.shType[k])
    {
    case SHAPE2D_BOX:
    {
        const Vec2 h = S.shHalf[k];
        const Vec2 lv[4] = {Vec2(-h.x, -h.y), Vec2(h.x, -h.y), Vec2(h.x, h.y), Vec2(-h.x, h.y)};
        const Vec2 ln[4] = {Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0)};
        w.kind = 1;
        w.count = 4;
        for (int i = 0; i < 4; i++)
        {
            w.v[i] = o + rotate(a, lv[i]);
            w.n[i] = rotate(a, ln[i]);
        }
        return true;
    }
    case SHAPE2D_CIRCLE:
        w.kind = 0;
        w.count = 1;
        if (vc == 0)
        {
            w.v[0] = o;
            w.r = S.shHalf[k].x;
        }
        else
        {
            w.v[0] = o + rotate(a, S.verts[(size_t)vo]);
            w.r = S.shRad[k];
        }
        return true;
    case SHAPE2D_POLYGON:
    case SHAPE2D_CAPSULE:
        if (vc < 2 || vc > 8)
            return false;
        w.count = vc;
        w.r = S.shRad[k];
        for (int i = 0; i < vc; i++)
        {
            w.v[i] = o + rotate(a, S.verts[(size_t)(vo + i)]);
            w.n[i] = rotate(a, S.norms[(size_t)(vo + i)]);
        }
        w.kind = (S.shType[k] == SHAPE2D_CAPSULE || vc == 2) ? 2 : 1;
        return true;
    case SHAPE2D_SEGMENT:
        if (vc < 2)
            return false;
        w.kind = 3;
        w.count = 2;
        w.v[0] = o + rotate(a, S.verts[(size_t)vo]);
        w.v[1] = o + rotate(a, S.verts[(size_t)vo + 1]);
        return true;
    case SHAPE2D_CHAIN:
        if (vc < 4)
            return false;
        w.kind = 4;
        w.count = 2;
        w.v[0] = o + rotate(a, S.verts[(size_t)vo + 1]);
        w.v[1] = o + rotate(a, S.verts[(size_t)vo + 2]);
        return true;
    default:
        return false;
    }
}

Proxy2D proxyOf(const WShape &w)
{
    Proxy2D p;
    p.count = w.count;
    p.radius = w.r;
    for (int i = 0; i < w.count; i++)
        p.pts[i] = w.v[i];
    return p;
}

void boundsOf(const Proxy2D &p, Vec2 &lo, Vec2 &hi)
{
    lo = hi = p.pts[0];
    for (int i = 1; i < p.count; i++)
    {
        lo = vmin(lo, p.pts[i]);
        hi = vmax(hi, p.pts[i]);
    }
    lo = lo - Vec2(p.radius, p.radius);
    hi = hi + Vec2(p.radius, p.radius);
}

int support(const Proxy2D &p, Vec2 d)
{
    int best = 0;
    float bv = dot(p.pts[0], d);
    for (int i = 1; i < p.count; i++)
    {
        const float v = dot(p.pts[i], d);
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
    Vec2 wA, wB, w;
    float a = 0.0f;
    int iA = 0, iB = 0;
};

Sv makeSv(const Proxy2D &A, const Proxy2D &B, int ia, int ib)
{
    Sv s;
    s.iA = ia;
    s.iB = ib;
    s.wA = A.pts[ia];
    s.wB = B.pts[ib];
    s.w = s.wA - s.wB;
    return s;
}

// Reduces a segment simplex to the feature closest to the origin and sets its weights.
void solveSegment(Sv *s, int &n)
{
    const Vec2 e = s[1].w - s[0].w;
    const float ee = dot(e, e);
    const float t = ee > 0.0f ? -dot(s[0].w, e) / ee : 0.0f;
    if (t <= 0.0f)
    {
        n = 1;
        s[0].a = 1.0f;
    }
    else if (t >= 1.0f)
    {
        s[0] = s[1];
        n = 1;
        s[0].a = 1.0f;
    }
    else
    {
        s[0].a = 1.0f - t;
        s[1].a = t;
        n = 2;
    }
}

float segDistSq(const Sv &a, const Sv &b)
{
    const Vec2 e = b.w - a.w;
    const float ee = dot(e, e);
    float t = ee > 0.0f ? -dot(a.w, e) / ee : 0.0f;
    t = std::min(std::max(t, 0.0f), 1.0f);
    return lengthSq(a.w + e * t);
}

// Returns true if the origin is inside the triangle (overlap); otherwise reduces to the
// closest edge or vertex.
bool solveTriangle(Sv *s, int &n)
{
    const Vec2 p0 = s[0].w, p1 = s[1].w, p2 = s[2].w;
    const float c0 = cross(p1 - p0, -p0), c1 = cross(p2 - p1, -p1), c2 = cross(p0 - p2, -p2);
    if ((c0 >= 0.0f && c1 >= 0.0f && c2 >= 0.0f) || (c0 <= 0.0f && c1 <= 0.0f && c2 <= 0.0f))
    {
        const float area = cross(p1 - p0, p2 - p0);
        if (std::fabs(area) > 1e-20f)
        {
            s[0].a = cross(p1, p2) / area;
            s[1].a = cross(p2, p0) / area;
            s[2].a = cross(p0, p1) / area;
        }
        else
        {
            s[0].a = 1.0f;
            s[1].a = s[2].a = 0.0f;
        }
        return true;
    }
    const float d01 = segDistSq(s[0], s[1]), d12 = segDistSq(s[1], s[2]), d20 = segDistSq(s[2], s[0]);
    if (d01 <= d12 && d01 <= d20)
    {
        n = 2;
    }
    else if (d12 <= d20)
    {
        s[0] = s[1];
        s[1] = s[2];
        n = 2;
    }
    else
    {
        s[1] = s[0];
        s[0] = s[2];
        n = 2;
    }
    solveSegment(s, n);
    return false;
}

struct CoreDistance
{
    Vec2 pA, pB;
    float dist = 0.0f;
};

CoreDistance coreDistance(const Proxy2D &A, const Proxy2D &B)
{
    Sv s[3];
    int n = 1;
    s[0] = makeSv(A, B, 0, 0);
    s[0].a = 1.0f;
    bool overlap = false;
    for (int iter = 0; iter < 30; iter++)
    {
        if (n == 2)
            solveSegment(s, n);
        else if (n == 3)
        {
            if (solveTriangle(s, n))
            {
                overlap = true;
                break;
            }
        }
        Vec2 c;
        for (int i = 0; i < n; i++)
            c = c + s[i].w * s[i].a;
        if (lengthSq(c) < 1e-14f)
        {
            overlap = true;
            break;
        }
        const Vec2 d = -c;
        const int ia = support(A, d), ib = support(B, -d);
        bool dup = false;
        for (int i = 0; i < n; i++)
            dup = dup || (s[i].iA == ia && s[i].iB == ib);
        if (dup)
            break;
        s[n] = makeSv(A, B, ia, ib);
        s[n].a = 0.0f;
        n++;
    }
    CoreDistance r;
    for (int i = 0; i < n; i++)
    {
        r.pA = r.pA + s[i].wA * s[i].a;
        r.pB = r.pB + s[i].wB * s[i].a;
    }
    r.dist = overlap ? 0.0f : length(r.pA - r.pB);
    return r;
}

struct CastHit
{
    float fraction = 0.0f;
    Vec2 point, normal;
};

// Conservative advancement of `caster` along d against `target` (b2ShapeCast's contract). With
// skipInitial, a caster that starts overlapping reports no hit (a ray from inside a shape).
bool castPair(const Proxy2D &target, const Proxy2D &caster, Vec2 d, float maxFraction, bool skipInitial, CastHit &out)
{
    const float total = target.radius + caster.radius;
    Proxy2D moved = caster;
    float t = 0.0f;
    for (int iter = 0; iter < 40; iter++)
    {
        for (int i = 0; i < caster.count; i++)
            moved.pts[i] = caster.pts[i] + d * t;
        const CoreDistance cd = coreDistance(target, moved);
        const Vec2 sep = cd.pB - cd.pA;
        if (cd.dist - total <= kTouchTol)
        {
            if (iter == 0 && skipInitial && cd.dist - total < 0.0f)
                return false;
            const Vec2 u = cd.dist > 1e-9f ? sep * (1.0f / cd.dist) : -normalized(d, Vec2(1, 0));
            out.fraction = t;
            out.normal = u;
            out.point = cd.pA + u * target.radius;
            return true;
        }
        const Vec2 u = sep * (1.0f / cd.dist);
        const float closing = -dot(u, d);
        if (closing <= 1e-9f)
            return false;
        t += (cd.dist - total) / closing;
        if (t > maxFraction)
            return false;
    }
    return false;
}

bool rayCircle(const WShape &w, Vec2 o, Vec2 d, float maxF, CastHit &out)
{
    const float len = length(d);
    if (len == 0.0f)
        return false;
    const Vec2 dir = d * (1.0f / len);
    const Vec2 s = o - w.v[0];
    const float t = -dot(s, dir);
    const Vec2 c = s + dir * t;
    const float cc = dot(c, c), rr = w.r * w.r;
    if (cc > rr)
        return false;
    const float h = std::sqrt(rr - cc);
    const float fr = t - h;
    if (fr < 0.0f || maxF * len < fr)
        return false;
    const Vec2 hp = s + dir * fr;
    out.fraction = fr / len;
    out.normal = normalized(hp, Vec2(1, 0));
    out.point = w.v[0] + out.normal * w.r;
    return true;
}

bool rayPolygon(const WShape &w, Vec2 o, Vec2 d, float maxF, CastHit &out)
{
    float lower = 0.0f, upper = maxF;
    int index = -1;
    for (int i = 0; i < w.count; i++)
    {
        const float num = dot(w.n[i], w.v[i] - o);
        const float den = dot(w.n[i], d);
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
    out.fraction = lower;
    out.normal = w.n[index];
    out.point = o + d * lower;
    return true;
}

bool raySegment(const WShape &w, Vec2 o, Vec2 d, float maxF, bool oneSided, CastHit &out)
{
    const Vec2 v1 = w.v[0], v2 = w.v[1];
    if (oneSided && cross(o - v1, v2 - v1) < 0.0f)
        return false;
    const Vec2 e = v2 - v1;
    const float len = length(e);
    if (len == 0.0f)
        return false;
    const Vec2 eu = e * (1.0f / len);
    Vec2 normal(eu.y, -eu.x);
    const float num = dot(normal, v1 - o);
    const float den = dot(normal, d);
    if (den == 0.0f)
        return false;
    const float t = num / den;
    if (t < 0.0f || maxF < t)
        return false;
    const Vec2 p = o + d * t;
    const float s = dot(p - v1, eu);
    if (s < 0.0f || len < s)
        return false;
    if (num > 0.0f)
        normal = -normal;
    out.fraction = t;
    out.point = p;
    out.normal = normal;
    return true;
}

bool rayShape(const WShape &w, Vec2 o, Vec2 d, float maxF, CastHit &out)
{
    switch (w.kind)
    {
    case 0:
        return rayCircle(w, o, d, maxF, out);
    case 1:
        if (w.r == 0.0f)
            return rayPolygon(w, o, d, maxF, out);
        break;
    case 3:
        return raySegment(w, o, d, maxF, false, out);
    case 4:
        return raySegment(w, o, d, maxF, true, out);
    default:
        break;
    }
    Proxy2D pt;
    pt.count = 1;
    pt.pts[0] = o;
    return castPair(proxyOf(w), pt, d, maxF, true, out);
}

// Slab test of the segment o + t d, t in [0, maxF], against the box [lo - pad, hi + pad].
bool rayVsBox(Vec2 o, Vec2 d, float maxF, Vec2 lo, Vec2 hi, Vec2 pad)
{
    float t0 = 0.0f, t1 = maxF;
    const float oo[2] = {o.x, o.y}, dd[2] = {d.x, d.y};
    const float l[2] = {lo.x - pad.x, lo.y - pad.y}, h[2] = {hi.x + pad.x, hi.y + pad.y};
    for (int i = 0; i < 2; i++)
    {
        if (std::fabs(dd[i]) < 1e-20f)
        {
            if (oo[i] < l[i] || oo[i] > h[i])
                return false;
        }
        else
        {
            float a = (l[i] - oo[i]) / dd[i], b = (h[i] - oo[i]) / dd[i];
            if (a > b)
                std::swap(a, b);
            t0 = std::max(t0, a);
            t1 = std::min(t1, b);
            if (t0 > t1)
                return false;
        }
    }
    return true;
}

bool boxesOverlap(Vec2 alo, Vec2 ahi, Vec2 blo, Vec2 bhi)
{
    return alo.x <= bhi.x && blo.x <= ahi.x && alo.y <= bhi.y && blo.y <= ahi.y;
}

bool isSensor(const Scene2D &S, int s) { return (S.shFlags[(size_t)s] & SHAPE_FLAG_SENSOR) != 0; }

} // namespace

Proxy2D makeCircleProxy(Vec2 center, float radius)
{
    Proxy2D p;
    p.count = 1;
    p.pts[0] = center;
    p.radius = radius;
    return p;
}

Proxy2D makeCapsuleProxy(Vec2 a, Vec2 b, float radius)
{
    Proxy2D p;
    p.count = 2;
    p.pts[0] = a;
    p.pts[1] = b;
    p.radius = radius;
    return p;
}

Proxy2D makeBoxProxy(Vec2 center, float angle, Vec2 h, float radius)
{
    Proxy2D p;
    p.count = 4;
    p.radius = radius;
    const Vec2 lv[4] = {Vec2(-h.x, -h.y), Vec2(h.x, -h.y), Vec2(h.x, h.y), Vec2(-h.x, h.y)};
    for (int i = 0; i < 4; i++)
        p.pts[i] = center + rotate(angle, lv[i]);
    return p;
}

Proxy2D makePolygonProxy(const Vec2 *pts, int count, float radius)
{
    Proxy2D p;
    p.count = std::min(count, 8);
    p.radius = radius;
    for (int i = 0; i < p.count; i++)
        p.pts[i] = pts[i];
    return p;
}

DistanceResult2D proxyDistance(const Proxy2D &a, const Proxy2D &b)
{
    const CoreDistance cd = coreDistance(a, b);
    DistanceResult2D r;
    const float total = a.radius + b.radius;
    if (cd.dist > total)
    {
        const Vec2 u = (cd.pB - cd.pA) * (1.0f / cd.dist);
        r.pointA = cd.pA + u * a.radius;
        r.pointB = cd.pB - u * b.radius;
        r.distance = cd.dist - total;
    }
    else
    {
        r.pointA = r.pointB = (cd.pA + cd.pB) * 0.5f;
        r.distance = 0.0f;
    }
    return r;
}

int Query2D::build(std::vector<int> &ids, int begin, int end)
{
    Node node;
    if (end - begin == 1)
    {
        node.shape = ids[(size_t)begin];
        node.lo = m_lo[(size_t)node.shape];
        node.hi = m_hi[(size_t)node.shape];
        m_nodes.push_back(node);
        return (int)m_nodes.size() - 1;
    }
    Vec2 clo(FLT_MAX, FLT_MAX), chi(-FLT_MAX, -FLT_MAX);
    for (int i = begin; i < end; i++)
    {
        const size_t s = (size_t)ids[(size_t)i];
        const Vec2 c = (m_lo[s] + m_hi[s]) * 0.5f;
        clo = vmin(clo, c);
        chi = vmax(chi, c);
    }
    const bool axisX = (chi.x - clo.x) >= (chi.y - clo.y);
    const int mid = (begin + end) / 2;
    std::nth_element(ids.begin() + begin, ids.begin() + mid, ids.begin() + end, [&](int a, int b) {
        const Vec2 ca = (m_lo[(size_t)a] + m_hi[(size_t)a]) * 0.5f, cb = (m_lo[(size_t)b] + m_hi[(size_t)b]) * 0.5f;
        return axisX ? ca.x < cb.x : ca.y < cb.y;
    });
    const int l = build(ids, begin, mid);
    const int r = build(ids, mid, end);
    node.left = l;
    node.right = r;
    node.lo = vmin(m_nodes[(size_t)l].lo, m_nodes[(size_t)r].lo);
    node.hi = vmax(m_nodes[(size_t)l].hi, m_nodes[(size_t)r].hi);
    m_nodes.push_back(node);
    return (int)m_nodes.size() - 1;
}

void Query2D::rebuild(const QueryView2D &v)
{
    const Scene2D &S = *v.scene;
    const int total = S.shapeTotal();
    m_nodes.clear();
    m_lo.assign((size_t)total, Vec2());
    m_hi.assign((size_t)total, Vec2());
    std::vector<int> ids;
    for (int s = 0; s < total; s++)
    {
        WShape w;
        const int b = S.shBody[(size_t)s];
        if (b < 0) // a destroyed body's shape slot
            continue;
        if (isSensor(S, s) || !makeWorldShape(S, s, v.pos[b], v.ang[b], w))
            continue;
        boundsOf(proxyOf(w), m_lo[(size_t)s], m_hi[(size_t)s]);
        ids.push_back(s);
    }
    m_root = ids.empty() ? -1 : build(ids, 0, (int)ids.size());
    m_builtShapes = total;
    m_dirty = false;
}

void Query2D::castRay(const QueryView2D &v, Vec2 origin, Vec2 translation, const CastFn &fn)
{
    if (m_dirty || m_builtShapes != v.scene->shapeTotal())
        rebuild(v);
    if (m_root < 0)
        return;
    float maxF = 1.0f;
    std::vector<int> stack{m_root};
    while (!stack.empty())
    {
        const Node &nd = m_nodes[(size_t)stack.back()];
        stack.pop_back();
        if (!rayVsBox(origin, translation, maxF, nd.lo, nd.hi, Vec2()))
            continue;
        if (nd.shape < 0)
        {
            stack.push_back(nd.left);
            stack.push_back(nd.right);
            continue;
        }
        const int s = nd.shape;
        const int b = v.scene->shBody[(size_t)s];
        WShape w;
        CastHit h;
        if (!makeWorldShape(*v.scene, s, v.pos[b], v.ang[b], w) || !rayShape(w, origin, translation, maxF, h))
            continue;
        RayHit2D hit;
        hit.shape = s;
        hit.body = b;
        hit.point = h.point;
        hit.normal = h.normal;
        hit.fraction = h.fraction;
        const float r = fn(hit);
        if (r == 0.0f)
            return;
        if (r > 0.0f && r < maxF)
            maxF = r;
    }
}

bool Query2D::castRayClosest(const QueryView2D &v, Vec2 origin, Vec2 translation, RayHit2D &out)
{
    bool found = false;
    castRay(v, origin, translation, [&](const RayHit2D &h) {
        out = h;
        found = true;
        return h.fraction;
    });
    return found;
}

void Query2D::castShape(const QueryView2D &v, const Proxy2D &proxy, Vec2 translation, const CastFn &fn)
{
    if (m_dirty || m_builtShapes != v.scene->shapeTotal())
        rebuild(v);
    if (m_root < 0)
        return;
    Vec2 plo, phi;
    boundsOf(proxy, plo, phi);
    const Vec2 center = (plo + phi) * 0.5f, pad = (phi - plo) * 0.5f;
    float maxF = 1.0f;
    std::vector<int> stack{m_root};
    while (!stack.empty())
    {
        const Node &nd = m_nodes[(size_t)stack.back()];
        stack.pop_back();
        if (!rayVsBox(center, translation, maxF, nd.lo, nd.hi, pad))
            continue;
        if (nd.shape < 0)
        {
            stack.push_back(nd.left);
            stack.push_back(nd.right);
            continue;
        }
        const int s = nd.shape;
        const int b = v.scene->shBody[(size_t)s];
        WShape w;
        CastHit h;
        if (!makeWorldShape(*v.scene, s, v.pos[b], v.ang[b], w) ||
            !castPair(proxyOf(w), proxy, translation, maxF, false, h))
            continue;
        RayHit2D hit;
        hit.shape = s;
        hit.body = b;
        hit.point = h.point;
        hit.normal = h.normal;
        hit.fraction = h.fraction;
        const float r = fn(hit);
        if (r == 0.0f)
            return;
        if (r > 0.0f && r < maxF)
            maxF = r;
    }
}

bool Query2D::castShapeClosest(const QueryView2D &v, const Proxy2D &proxy, Vec2 translation, RayHit2D &out)
{
    bool found = false;
    castShape(v, proxy, translation, [&](const RayHit2D &h) {
        out = h;
        found = true;
        return h.fraction;
    });
    return found;
}

void Query2D::overlapAABB(const QueryView2D &v, Vec2 lo, Vec2 hi, const ShapeFn &fn)
{
    if (m_dirty || m_builtShapes != v.scene->shapeTotal())
        rebuild(v);
    if (m_root < 0)
        return;
    std::vector<int> stack{m_root};
    while (!stack.empty())
    {
        const Node &nd = m_nodes[(size_t)stack.back()];
        stack.pop_back();
        if (!boxesOverlap(nd.lo, nd.hi, lo, hi))
            continue;
        if (nd.shape < 0)
        {
            stack.push_back(nd.left);
            stack.push_back(nd.right);
            continue;
        }
        if (!fn(nd.shape))
            return;
    }
}

void Query2D::overlapShape(const QueryView2D &v, const Proxy2D &proxy, const ShapeFn &fn)
{
    Vec2 lo, hi;
    boundsOf(proxy, lo, hi);
    overlapAABB(v, lo, hi, [&](int s) {
        const int b = v.scene->shBody[(size_t)s];
        WShape w;
        if (!makeWorldShape(*v.scene, s, v.pos[b], v.ang[b], w))
            return true;
        if (proxyDistance(proxyOf(w), proxy).distance <= kTouchTol)
            return fn(s);
        return true;
    });
}

void Query2D::explosion(const QueryView2D &v, Vec2 pos, float radius, float falloff, float impulsePerLength,
                        std::vector<ExplosionImpulse2D> &out)
{
    const Scene2D &S = *v.scene;
    const Proxy2D point = makeCircleProxy(pos, 0.0f);
    for (int s = 0; s < S.shapeTotal(); s++)
    {
        const int b = S.shBody[(size_t)s];
        if (b < 0 || S.mass[(size_t)b] <= 0.0f || isSensor(S, s))
            continue;
        WShape w;
        if (!makeWorldShape(S, s, v.pos[b], v.ang[b], w))
            continue;
        const Proxy2D proxy = proxyOf(w);
        const DistanceResult2D dr = proxyDistance(proxy, point);
        if (dr.distance > radius + falloff)
            continue;
        Vec2 closest = dr.pointA;
        if (dr.distance == 0.0f)
        {
            closest = Vec2();
            for (int i = 0; i < w.count; i++)
                closest = closest + w.v[i];
            closest = closest * (1.0f / (float)w.count);
        }
        const Vec2 dir = normalized(closest - pos, Vec2(1, 0));
        const Vec2 line = perp(dir);
        float perimeter = 0.0f;
        if (w.kind == 0)
            perimeter = 2.0f * w.r;
        else if (w.kind == 2)
            perimeter = std::fabs(dot(w.v[1] - w.v[0], line)) + 2.0f * w.r;
        else if (w.kind == 1)
        {
            float lo = dot(w.v[0], line), hi = lo;
            for (int i = 1; i < w.count; i++)
            {
                const float d = dot(w.v[i], line);
                lo = std::min(lo, d);
                hi = std::max(hi, d);
            }
            perimeter = (hi - lo) + 2.0f * w.r;
        }
        else
            perimeter = std::fabs(dot(w.v[1] - w.v[0], line));
        float scale = 1.0f;
        if (dr.distance > radius && falloff > 0.0f)
            scale = std::min(std::max((radius + falloff - dr.distance) / falloff, 0.0f), 1.0f);
        const Vec2 impulse = dir * (impulsePerLength * perimeter * scale);
        const float invM = 1.0f / S.mass[(size_t)b];
        const float invI = S.moment[(size_t)b] > 0.0f ? 1.0f / S.moment[(size_t)b] : 0.0f;
        const float dw = invI * cross(closest - v.pos[b], impulse);
        const Vec2 dv = impulse * invM;
        bool merged = false;
        for (ExplosionImpulse2D &e : out)
            if (e.body == b)
            {
                e.dv = e.dv + dv;
                e.dw += dw;
                merged = true;
                break;
            }
        if (!merged)
            out.push_back({b, dv, dw});
    }
}

} // namespace avbd2d
