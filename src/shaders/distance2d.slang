// GJK distance and conservative-advancement time of impact for two convex proxies, a term for
// term port of Box2D v3's b2ShapeDistance and b2TimeOfImpact (distance.c). A proxy is up to
// PROXY_MAX points plus a rounding radius, in the owning body's frame (the solver's body
// position is the centre of mass, so a sweep has no local centre). Used by ccd2d.slang.

module distance2d;

import maths2d;

public static const int PROXY_MAX = 8;
public static const float TOI_SLOP = 0.005f; // Box2D's B2_LINEAR_SLOP

public struct Proxy
{
    public vec2 p[PROXY_MAX];
    public int count;
    public float radius;
};

// A rotation is stored as (cos, sin).
public struct Xf
{
    public vec2 p;
    public vec2 q;
};

public struct Sweep
{
    public vec2 c1;
    public vec2 c2;
    public vec2 q1;
    public vec2 q2;
};

public vec2 rotQ(vec2 q, vec2 v) { return vec2(q.x * v.x - q.y * v.y, q.y * v.x + q.x * v.y); }
public vec2 invRotQ(vec2 q, vec2 v) { return vec2(q.x * v.x + q.y * v.y, -q.y * v.x + q.x * v.y); }
public vec2 xfPoint(Xf x, vec2 v) { return v2add(x.p, rotQ(x.q, v)); }

public vec2 normalized2(vec2 v)
{
    float l = v2len(v);
    return l < 1.1920929e-7f ? vec2(0.0f, 0.0f) : v2mul(v, 1.0f / l);
}

public Xf sweepXf(Sweep s, float t)
{
    Xf x;
    x.p = v2add(v2mul(s.c1, 1.0f - t), v2mul(s.c2, t));
    vec2 q = v2add(v2mul(s.q1, 1.0f - t), v2mul(s.q2, t));
    float l = v2len(q);
    x.q = l > 0.0f ? v2mul(q, 1.0f / l) : vec2(1.0f, 0.0f);
    return x;
}

// Transform of B in the frame of A.
Xf invMulXf(Xf a, Xf b)
{
    Xf r;
    r.q = vec2(a.q.x * b.q.x + a.q.y * b.q.y, a.q.x * b.q.y - a.q.y * b.q.x);
    r.p = invRotQ(a.q, v2sub(b.p, a.p));
    return r;
}

vec2 crossSV(float s, vec2 v) { return vec2(-s * v.y, s * v.x); }
vec2 crossVS(vec2 v, float s) { return vec2(s * v.y, -s * v.x); }

int findSupport(Proxy pr, vec2 d)
{
    int best = 0;
    float bestV = v2dot(pr.p[0], d);
    for (int i = 1; i < pr.count; i++)
    {
        float v = v2dot(pr.p[i], d);
        if (v > bestV)
        {
            best = i;
            bestV = v;
        }
    }
    return best;
}

struct SimplexVertex
{
    vec2 wA;
    vec2 wB;
    vec2 w;
    float a;
    int iA;
    int iB;
};

struct Simplex
{
    SimplexVertex v[3];
    int count;
};

public struct SimplexCache
{
    public int count;
    public int iA[3];
    public int iB[3];
};

public struct DistanceOut
{
    public vec2 normal;
    public vec2 pointA;
    public vec2 pointB;
    public float distance;
};

Simplex makeSimplex(SimplexCache cache, Proxy pa, Proxy pb)
{
    Simplex s;
    s.count = cache.count;
    for (int i = 0; i < 3; i++)
    {
        s.v[i].wA = vec2(0.0f, 0.0f);
        s.v[i].wB = vec2(0.0f, 0.0f);
        s.v[i].w = vec2(0.0f, 0.0f);
        s.v[i].a = 0.0f;
        s.v[i].iA = 0;
        s.v[i].iB = 0;
    }
    for (int i = 0; i < s.count; i++)
    {
        s.v[i].iA = cache.iA[i];
        s.v[i].iB = cache.iB[i];
        s.v[i].wA = pa.p[s.v[i].iA];
        s.v[i].wB = pb.p[s.v[i].iB];
        s.v[i].w = v2sub(s.v[i].wA, s.v[i].wB);
        s.v[i].a = -1.0f;
    }
    if (s.count == 0)
    {
        s.v[0].iA = 0;
        s.v[0].iB = 0;
        s.v[0].wA = pa.p[0];
        s.v[0].wB = pb.p[0];
        s.v[0].w = v2sub(s.v[0].wA, s.v[0].wB);
        s.v[0].a = 1.0f;
        s.count = 1;
    }
    return s;
}

SimplexCache makeCache(Simplex s)
{
    SimplexCache c;
    c.count = s.count;
    for (int i = 0; i < 3; i++)
    {
        c.iA[i] = 0;
        c.iB[i] = 0;
    }
    for (int i = 0; i < s.count; i++)
    {
        c.iA[i] = s.v[i].iA;
        c.iB[i] = s.v[i].iB;
    }
    return c;
}

void witnessPoints(Simplex s, out vec2 a, out vec2 b)
{
    if (s.count == 1)
    {
        a = s.v[0].wA;
        b = s.v[0].wB;
    }
    else if (s.count == 2)
    {
        a = v2add(v2mul(s.v[0].wA, s.v[0].a), v2mul(s.v[1].wA, s.v[1].a));
        b = v2add(v2mul(s.v[0].wB, s.v[0].a), v2mul(s.v[1].wB, s.v[1].a));
    }
    else
    {
        a = v2add(v2add(v2mul(s.v[0].wA, s.v[0].a), v2mul(s.v[1].wA, s.v[1].a)), v2mul(s.v[2].wA, s.v[2].a));
        b = a;
    }
}

// Both solvers return a vector that points towards the origin.
vec2 solveSimplex2(inout Simplex s)
{
    vec2 w1 = s.v[0].w;
    vec2 w2 = s.v[1].w;
    vec2 e12 = v2sub(w2, w1);

    float d12_2 = -v2dot(w1, e12);
    if (d12_2 <= 0.0f)
    {
        s.v[0].a = 1.0f;
        s.count = 1;
        return v2neg(w1);
    }
    float d12_1 = v2dot(w2, e12);
    if (d12_1 <= 0.0f)
    {
        s.v[1].a = 1.0f;
        s.count = 1;
        s.v[0] = s.v[1];
        return v2neg(w2);
    }
    float inv = 1.0f / (d12_1 + d12_2);
    s.v[0].a = d12_1 * inv;
    s.v[1].a = d12_2 * inv;
    s.count = 2;
    return crossSV(v2cross(v2add(w1, w2), e12), e12);
}

vec2 solveSimplex3(inout Simplex s)
{
    vec2 w1 = s.v[0].w;
    vec2 w2 = s.v[1].w;
    vec2 w3 = s.v[2].w;

    vec2 e12 = v2sub(w2, w1);
    float d12_1 = v2dot(w2, e12);
    float d12_2 = -v2dot(w1, e12);

    vec2 e13 = v2sub(w3, w1);
    float d13_1 = v2dot(w3, e13);
    float d13_2 = -v2dot(w1, e13);

    vec2 e23 = v2sub(w3, w2);
    float d23_1 = v2dot(w3, e23);
    float d23_2 = -v2dot(w2, e23);

    float n123 = v2cross(e12, e13);
    float d123_1 = n123 * v2cross(w2, w3);
    float d123_2 = n123 * v2cross(w3, w1);
    float d123_3 = n123 * v2cross(w1, w2);

    if (d12_2 <= 0.0f && d13_2 <= 0.0f)
    {
        s.v[0].a = 1.0f;
        s.count = 1;
        return v2neg(w1);
    }
    if (d12_1 > 0.0f && d12_2 > 0.0f && d123_3 <= 0.0f)
    {
        float inv = 1.0f / (d12_1 + d12_2);
        s.v[0].a = d12_1 * inv;
        s.v[1].a = d12_2 * inv;
        s.count = 2;
        return crossSV(v2cross(v2add(w1, w2), e12), e12);
    }
    if (d13_1 > 0.0f && d13_2 > 0.0f && d123_2 <= 0.0f)
    {
        float inv = 1.0f / (d13_1 + d13_2);
        s.v[0].a = d13_1 * inv;
        s.v[2].a = d13_2 * inv;
        s.count = 2;
        s.v[1] = s.v[2];
        return crossSV(v2cross(v2add(w1, w3), e13), e13);
    }
    if (d12_1 <= 0.0f && d23_2 <= 0.0f)
    {
        s.v[1].a = 1.0f;
        s.count = 1;
        s.v[0] = s.v[1];
        return v2neg(w2);
    }
    if (d13_1 <= 0.0f && d23_1 <= 0.0f)
    {
        s.v[2].a = 1.0f;
        s.count = 1;
        s.v[0] = s.v[2];
        return v2neg(w3);
    }
    if (d23_1 > 0.0f && d23_2 > 0.0f && d123_1 <= 0.0f)
    {
        float inv = 1.0f / (d23_1 + d23_2);
        s.v[1].a = d23_1 * inv;
        s.v[2].a = d23_2 * inv;
        s.count = 2;
        s.v[0] = s.v[2];
        return crossSV(v2cross(v2add(w2, w3), e23), e23);
    }
    float inv = 1.0f / (d123_1 + d123_2 + d123_3);
    s.v[0].a = d123_1 * inv;
    s.v[1].a = d123_2 * inv;
    s.v[2].a = d123_3 * inv;
    s.count = 3;
    return vec2(0.0f, 0.0f);
}

// GJK distance between proxy A and proxy B placed by `xf` (B in the frame of A), radii ignored.
// distance == 0 means the cores overlap.
DistanceOut shapeDistance(Proxy pa, Proxy pb, Xf xf, inout SimplexCache cache)
{
    DistanceOut o;
    o.normal = vec2(0.0f, 0.0f);
    o.pointA = vec2(0.0f, 0.0f);
    o.pointB = vec2(0.0f, 0.0f);
    o.distance = 0.0f;

    Proxy lb = pb;
    for (int i = 0; i < lb.count; i++)
        lb.p[i] = xfPoint(xf, pb.p[i]);

    Simplex simplex = makeSimplex(cache, pa, lb);
    vec2 nonUnit = vec2(0.0f, 0.0f);
    int saveA[3] = {0, 0, 0};
    int saveB[3] = {0, 0, 0};

    int iteration = 0;
    while (iteration < 20)
    {
        int saveCount = simplex.count;
        for (int i = 0; i < saveCount; i++)
        {
            saveA[i] = simplex.v[i].iA;
            saveB[i] = simplex.v[i].iB;
        }

        vec2 d = vec2(0.0f, 0.0f);
        if (simplex.count == 1)
            d = v2neg(simplex.v[0].w);
        else if (simplex.count == 2)
            d = solveSimplex2(simplex);
        else
            d = solveSimplex3(simplex);

        if (simplex.count == 3 || v2dot(d, d) < 1.1754944e-35f)
        {
            cache = makeCache(simplex);
            vec2 a, b;
            witnessPoints(simplex, a, b);
            o.distance = 0.0f;
            o.pointA = a;
            o.pointB = b;
            return o;
        }

        nonUnit = d;

        int n = simplex.count;
        simplex.v[n].iA = findSupport(pa, d);
        simplex.v[n].wA = pa.p[simplex.v[n].iA];
        simplex.v[n].iB = findSupport(lb, v2neg(d));
        simplex.v[n].wB = lb.p[simplex.v[n].iB];
        simplex.v[n].w = v2sub(simplex.v[n].wA, simplex.v[n].wB);
        iteration++;

        bool duplicate = false;
        for (int i = 0; i < saveCount; i++)
        {
            if (simplex.v[n].iA == saveA[i] && simplex.v[n].iB == saveB[i])
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            break;
        simplex.count += 1;
    }

    vec2 normal = normalized2(nonUnit);
    vec2 a, b;
    witnessPoints(simplex, a, b);
    o.normal = normal;
    o.distance = v2len(v2sub(a, b));
    o.pointA = a;
    o.pointB = b;
    cache = makeCache(simplex);
    return o;
}

// ---------------------------------------------------------------------------
// Time of impact
// ---------------------------------------------------------------------------
static const int SEP_POINTS = 0;
static const int SEP_FACE_A = 1;
static const int SEP_FACE_B = 2;

struct SepFn
{
    Sweep sA;
    Sweep sB;
    vec2 localPoint;
    vec2 axis;
    int type;
};

SepFn makeSepFn(SimplexCache cache, Proxy pa, Sweep sA, Proxy pb, Sweep sB, float t1)
{
    SepFn f;
    f.sA = sA;
    f.sB = sB;
    f.localPoint = vec2(0.0f, 0.0f);
    f.axis = vec2(0.0f, 0.0f);
    f.type = SEP_POINTS;
    Xf xfA = sweepXf(sA, t1);
    Xf xfB = sweepXf(sB, t1);

    if (cache.count == 1)
    {
        vec2 pointA = xfPoint(xfA, pa.p[cache.iA[0]]);
        vec2 pointB = xfPoint(xfB, pb.p[cache.iB[0]]);
        f.axis = normalized2(v2sub(pointB, pointA));
        return f;
    }
    if (cache.iA[0] == cache.iA[1])
    {
        f.type = SEP_FACE_B;
        vec2 b1 = pb.p[cache.iB[0]];
        vec2 b2 = pb.p[cache.iB[1]];
        f.axis = normalized2(crossVS(v2sub(b2, b1), 1.0f));
        vec2 normal = rotQ(xfB.q, f.axis);
        f.localPoint = vec2(0.5f * (b1.x + b2.x), 0.5f * (b1.y + b2.y));
        vec2 pointB = xfPoint(xfB, f.localPoint);
        vec2 pointA = xfPoint(xfA, pa.p[cache.iA[0]]);
        if (v2dot(v2sub(pointA, pointB), normal) < 0.0f)
            f.axis = v2neg(f.axis);
        return f;
    }
    f.type = SEP_FACE_A;
    vec2 a1 = pa.p[cache.iA[0]];
    vec2 a2 = pa.p[cache.iA[1]];
    f.axis = normalized2(crossVS(v2sub(a2, a1), 1.0f));
    vec2 normal = rotQ(xfA.q, f.axis);
    f.localPoint = vec2(0.5f * (a1.x + a2.x), 0.5f * (a1.y + a2.y));
    vec2 pointA = xfPoint(xfA, f.localPoint);
    vec2 pointB = xfPoint(xfB, pb.p[cache.iB[0]]);
    if (v2dot(v2sub(pointB, pointA), normal) < 0.0f)
        f.axis = v2neg(f.axis);
    return f;
}

float findMinSeparation(SepFn f, Proxy pa, Proxy pb, float t, out int iA, out int iB)
{
    Xf xfA = sweepXf(f.sA, t);
    Xf xfB = sweepXf(f.sB, t);
    iA = -1;
    iB = -1;
    if (f.type == SEP_POINTS)
    {
        vec2 axisA = invRotQ(xfA.q, f.axis);
        vec2 axisB = invRotQ(xfB.q, v2neg(f.axis));
        iA = findSupport(pa, axisA);
        iB = findSupport(pb, axisB);
        vec2 pointA = xfPoint(xfA, pa.p[iA]);
        vec2 pointB = xfPoint(xfB, pb.p[iB]);
        return v2dot(v2sub(pointB, pointA), f.axis);
    }
    if (f.type == SEP_FACE_A)
    {
        vec2 normal = rotQ(xfA.q, f.axis);
        vec2 pointA = xfPoint(xfA, f.localPoint);
        vec2 axisB = invRotQ(xfB.q, v2neg(normal));
        iB = findSupport(pb, axisB);
        vec2 pointB = xfPoint(xfB, pb.p[iB]);
        return v2dot(v2sub(pointB, pointA), normal);
    }
    vec2 normal = rotQ(xfB.q, f.axis);
    vec2 pointB = xfPoint(xfB, f.localPoint);
    vec2 axisA = invRotQ(xfA.q, v2neg(normal));
    iA = findSupport(pa, axisA);
    vec2 pointA = xfPoint(xfA, pa.p[iA]);
    return v2dot(v2sub(pointA, pointB), normal);
}

float evaluateSeparation(SepFn f, Proxy pa, Proxy pb, int iA, int iB, float t)
{
    Xf xfA = sweepXf(f.sA, t);
    Xf xfB = sweepXf(f.sB, t);
    if (f.type == SEP_POINTS)
    {
        vec2 pointA = xfPoint(xfA, pa.p[iA]);
        vec2 pointB = xfPoint(xfB, pb.p[iB]);
        return v2dot(v2sub(pointB, pointA), f.axis);
    }
    if (f.type == SEP_FACE_A)
    {
        vec2 normal = rotQ(xfA.q, f.axis);
        vec2 pointA = xfPoint(xfA, f.localPoint);
        vec2 pointB = xfPoint(xfB, pb.p[iB]);
        return v2dot(v2sub(pointB, pointA), normal);
    }
    vec2 normal = rotQ(xfB.q, f.axis);
    vec2 pointB = xfPoint(xfB, f.localPoint);
    vec2 pointA = xfPoint(xfA, pa.p[iA]);
    return v2dot(v2sub(pointA, pointB), normal);
}

public static const int TOI_UNKNOWN = 0;
public static const int TOI_FAILED = 1;
public static const int TOI_OVERLAPPED = 2;
public static const int TOI_HIT = 3;
public static const int TOI_SEPARATED = 4;

public struct ToiOut
{
    public int state;
    public vec2 point;
    public vec2 normal; // from A to B
    public float fraction;
};

// The largest t in [0, tMax] at which A and B (cores plus radii) are not closer than the
// target distance, A and B moving along their sweeps.
public ToiOut timeOfImpact(Proxy pa, Proxy pb, Sweep sweepA, Sweep sweepB, float tMax)
{
    ToiOut out_;
    out_.state = TOI_UNKNOWN;
    out_.point = vec2(0.0f, 0.0f);
    out_.normal = vec2(0.0f, 0.0f);
    out_.fraction = tMax;

    float totalRadius = pa.radius + pb.radius;
    float target = max(TOI_SLOP, totalRadius - TOI_SLOP);
    float tolerance = 0.25f * TOI_SLOP;

    float t1 = 0.0f;
    int distanceIterations = 0;
    SimplexCache cache;
    cache.count = 0;
    for (int i = 0; i < 3; i++)
    {
        cache.iA[i] = 0;
        cache.iB[i] = 0;
    }

    for (;;)
    {
        Xf xfA = sweepXf(sweepA, t1);
        Xf xfB = sweepXf(sweepB, t1);
        DistanceOut dout = shapeDistance(pa, pb, invMulXf(xfA, xfB), cache);

        vec2 worldNormal = rotQ(xfA.q, dout.normal);
        vec2 worldPointA = xfPoint(xfA, dout.pointA);
        vec2 worldPointB = xfPoint(xfA, dout.pointB);
        distanceIterations += 1;

        if (dout.distance <= 0.0f)
        {
            out_.state = TOI_OVERLAPPED;
            out_.fraction = 0.0f;
            break;
        }
        if (dout.distance <= target + tolerance)
        {
            out_.state = TOI_HIT;
            vec2 pA = v2add(worldPointA, v2mul(worldNormal, pa.radius));
            vec2 pB = v2sub(worldPointB, v2mul(worldNormal, pb.radius));
            out_.point = v2mul(v2add(pA, pB), 0.5f);
            out_.normal = worldNormal;
            out_.fraction = t1;
            break;
        }

        SepFn fcn = makeSepFn(cache, pa, sweepA, pb, sweepB, t1);

        bool done = false;
        float t2 = tMax;
        int pushBack = 0;
        for (;;)
        {
            int iA, iB;
            float s2 = findMinSeparation(fcn, pa, pb, t2, iA, iB);

            if (s2 > target + tolerance)
            {
                out_.state = TOI_SEPARATED;
                out_.fraction = tMax;
                done = true;
                break;
            }
            if (s2 > target - tolerance)
            {
                t1 = t2;
                break;
            }

            float s1 = evaluateSeparation(fcn, pa, pb, iA, iB, t1);
            if (s1 < target - tolerance)
            {
                out_.state = TOI_FAILED;
                out_.fraction = t1;
                done = true;
                break;
            }
            if (s1 <= target + tolerance)
            {
                out_.state = TOI_HIT;
                vec2 pA = v2add(worldPointA, v2mul(worldNormal, pa.radius));
                vec2 pB = v2sub(worldPointB, v2mul(worldNormal, pb.radius));
                out_.point = v2mul(v2add(pA, pB), 0.5f);
                out_.normal = worldNormal;
                out_.fraction = t1;
                done = true;
                break;
            }

            int rootIter = 0;
            float a1 = t1, a2 = t2;
            for (;;)
            {
                float t;
                if ((rootIter & 1) != 0)
                    t = a1 + (target - s1) * (a2 - a1) / (s2 - s1);
                else
                    t = 0.5f * (a1 + a2);
                rootIter += 1;

                float s = evaluateSeparation(fcn, pa, pb, iA, iB, t);
                if (abs(s - target) < tolerance)
                {
                    t2 = t;
                    break;
                }
                if (s > target)
                {
                    a1 = t;
                    s1 = s;
                }
                else
                {
                    a2 = t;
                    s2 = s;
                }
                if (rootIter == 50)
                    break;
            }

            pushBack += 1;
            if (pushBack == PROXY_MAX)
                break;
        }

        if (done)
            break;

        if (distanceIterations == 20)
        {
            out_.state = TOI_FAILED;
            vec2 pA = v2add(worldPointA, v2mul(worldNormal, pa.radius));
            vec2 pB = v2sub(worldPointB, v2mul(worldNormal, pb.radius));
            out_.point = v2mul(v2add(pA, pB), 0.5f);
            out_.normal = worldNormal;
            out_.fraction = t1;
            break;
        }
    }
    return out_;
}
