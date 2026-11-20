// Box2D v3 shape manifolds: rounded convex polygons, capsules, segments, offset circles and chain
// segments (manifold.c), ported function for function.

module manifold2d;

import maths2d;
import collide2d_box;

// ===========================================================================
// Box2D v3 shapes (manifold.c): rounded convex polygons, capsules, segments, offset circles and
// chain segments, ported function for function. Box2D's speculative distance is the caller's
// margin. Shapes are given in world orientation relative to one origin (A's x^t position, for
// round-off, as Box2D shifts to polygon A's first vertex). Box2D's local manifold has the normal
// from A to B and one midpoint per contact with its separation; collideShapes splits each
// midpoint back into the two surface points it came from.
// ===========================================================================

public static const float LINEAR_SLOP = 0.005f;
static const float FLT_EPS = 1.1920929e-7f;
static const float FLT_BIG = 3.4e38f;

public static const int KIND_ROUND = 0; // a point with a radius: circle, offset circle
public static const int KIND_HULL = 1;  // 2..MAX_POLY vertices with a radius: box, polygon, capsule, segment
public static const int KIND_CHAIN = 2; // v[0..3] = ghost1, p1, p2, ghost2; one-sided, no radius

public struct Shape2
{
    public vec2 v[MAX_POLY];
    public vec2 n[MAX_POLY]; // outward edge normals; n[i] belongs to the edge v[i] -> v[i+1]
    public int count;
    public int kind;
    public float radius;
    public vec2 centroid;
};

// Box2D's b2LocalManifold, at most two points.
struct LM
{
    vec2 normal; // from A to B
    int count;
    vec2 p0;
    float s0;
    int id0;
    vec2 p1;
    float s1;
    int id1;
};

LM lmEmpty()
{
    LM m;
    m.normal = vec2(0.0f, 1.0f);
    m.count = 0;
    m.p0 = vec2(0.0f, 0.0f);
    m.s0 = 0.0f;
    m.id0 = 0;
    m.p1 = vec2(0.0f, 0.0f);
    m.s1 = 0.0f;
    m.id1 = 0;
    return m;
}

void lmPush(inout LM m, vec2 p, float s, int id)
{
    if (m.count == 0)
    {
        m.p0 = p;
        m.s0 = s;
        m.id0 = id;
    }
    else
    {
        m.p1 = p;
        m.s1 = s;
        m.id1 = id;
    }
    m.count++;
}

int makeId(int a, int b) { return ((a & 255) << 8) | (b & 255); }
vec2 rightPerp(vec2 a) { return vec2(a.y, -a.x); }
vec2 lerp2(vec2 a, vec2 b, float t) { return vec2(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)); }
vec2 mulAdd(vec2 a, float s, vec2 b) { return vec2(a.x + s * b.x, a.y + s * b.y); }

// b2GetLengthAndNormalize, except that a zero vector gives +y instead of zero: a normal must stay
// a unit vector here (a degenerate direction is still a usable contact axis).
vec2 lengthAndNormalize(vec2 v, out float len)
{
    len = v2len(v);
    if (len < FLT_EPS)
        return vec2(0.0f, 1.0f);
    return v2mul(v, 1.0f / len);
}

vec2 normalize2(vec2 v)
{
    float len;
    return lengthAndNormalize(v, len);
}

LM lmCircles(vec2 pA, float rA, vec2 pB, float rB, float spec)
{
    LM m = lmEmpty();
    float distance;
    vec2 normal = lengthAndNormalize(v2sub(pB, pA), distance);
    float separation = distance - rA - rB;
    if (separation > spec)
        return m;
    vec2 cA = mulAdd(pA, rA, normal);
    vec2 cB = mulAdd(pB, -rB, normal);
    m.normal = normal;
    lmPush(m, lerp2(cA, cB, 0.5f), separation, 0);
    return m;
}

LM lmCapsuleCircle(vec2 p1, vec2 p2, float rA, vec2 pB, float rB, float spec)
{
    LM m = lmEmpty();
    vec2 e = v2sub(p2, p1);
    vec2 pA;
    float s1 = v2dot(v2sub(pB, p1), e);
    float s2 = v2dot(v2sub(p2, pB), e);
    if (s1 < 0.0f)
        pA = p1;
    else if (s2 < 0.0f)
        pA = p2;
    else
        pA = mulAdd(p1, s1 / v2dot(e, e), e);
    float distance;
    vec2 normal = lengthAndNormalize(v2sub(pB, pA), distance);
    float separation = distance - rA - rB;
    if (separation > spec)
        return m;
    vec2 cA = mulAdd(pA, rA, normal);
    vec2 cB = mulAdd(pB, -rB, normal);
    m.normal = normal;
    lmPush(m, lerp2(cA, cB, 0.5f), separation, 0);
    return m;
}

LM lmPolygonCircle(Shape2 A, vec2 center, float rB, float spec)
{
    LM m = lmEmpty();
    float rA = A.radius;
    float radius = rA + rB;
    int normalIndex = 0;
    float separation = -FLT_BIG;
    for (int i = 0; i < A.count; i++)
    {
        float s = v2dot(A.n[i], v2sub(center, A.v[i]));
        if (s > separation)
        {
            separation = s;
            normalIndex = i;
        }
    }
    if (separation > radius + spec)
        return m;
    int i2 = normalIndex + 1 < A.count ? normalIndex + 1 : 0;
    vec2 v1 = A.v[normalIndex];
    vec2 v2 = A.v[i2];
    float u1 = v2dot(v2sub(center, v1), v2sub(v2, v1));
    float u2 = v2dot(v2sub(center, v2), v2sub(v1, v2));
    if ((u1 < 0.0f || u2 < 0.0f) && separation > FLT_EPS)
    {
        // Centre closest to a vertex and safely outside.
        vec2 vv = u1 < 0.0f ? v1 : v2;
        vec2 normal = normalize2(v2sub(center, vv));
        separation = v2dot(v2sub(center, vv), normal);
        if (separation > radius + spec)
            return m;
        vec2 cA = mulAdd(vv, rA, normal);
        vec2 cB = mulAdd(center, -rB, normal);
        m.normal = normal;
        lmPush(m, lerp2(cA, cB, 0.5f), v2dot(v2sub(cB, cA), normal), 0);
        return m;
    }
    // Centre between v1 and v2, maybe inside.
    vec2 normal = A.n[normalIndex];
    m.normal = normal;
    vec2 cA = mulAdd(center, rA - v2dot(v2sub(center, v1), normal), normal);
    vec2 cB = mulAdd(center, -rB, normal);
    lmPush(m, lerp2(cA, cB, 0.5f), separation - radius, 0);
    return m;
}

// b2CollideCapsules (Ericson 5.1.9, plus clipping for two points). Segment A is p1 -> q1.
LM lmCapsules(vec2 p1, vec2 q1, float rA, vec2 p2, vec2 q2, float rB, float spec)
{
    vec2 origin = p1;
    p1 = v2sub(p1, origin);
    q1 = v2sub(q1, origin);
    p2 = v2sub(p2, origin);
    q2 = v2sub(q2, origin);

    vec2 d1 = v2sub(q1, p1);
    vec2 d2 = v2sub(q2, p2);
    const float epsSqr = FLT_EPS * FLT_EPS;
    float dd1 = max(v2dot(d1, d1), epsSqr);
    float dd2 = max(v2dot(d2, d2), epsSqr);
    vec2 r = v2sub(p1, p2);
    float rd1 = v2dot(r, d1);
    float rd2 = v2dot(r, d2);
    float d12 = v2dot(d1, d2);
    float denom = dd1 * dd2 - d12 * d12;
    float f1 = 0.0f;
    if (denom != 0.0f)
        f1 = clampf((d12 * rd2 - rd1 * dd2) / denom, 0.0f, 1.0f);
    float f2 = (d12 * f1 + rd2) / dd2;
    if (f2 < 0.0f)
    {
        f2 = 0.0f;
        f1 = clampf(-rd1 / dd1, 0.0f, 1.0f);
    }
    else if (f2 > 1.0f)
    {
        f2 = 1.0f;
        f1 = clampf((d12 - rd1) / dd1, 0.0f, 1.0f);
    }
    vec2 closest1 = mulAdd(p1, f1, d1);
    vec2 closest2 = mulAdd(p2, f2, d2);
    float distanceSquared = v2lenSq(v2sub(closest1, closest2));

    LM m = lmEmpty();
    float radius = rA + rB;
    float maxDistance = radius + spec;
    if (distanceSquared > maxDistance * maxDistance)
        return m;
    float distance = sqrt(distanceSquared);

    float length1, length2;
    vec2 u1 = lengthAndNormalize(d1, length1);
    vec2 u2 = lengthAndNormalize(d2, length2);

    float fp2 = v2dot(v2sub(p2, p1), u1);
    float fq2 = v2dot(v2sub(q2, p1), u1);
    bool outsideA = (fp2 <= 0.0f && fq2 <= 0.0f) || (fp2 >= length1 && fq2 >= length1);
    float fp1 = v2dot(v2sub(p1, p2), u2);
    float fq1 = v2dot(v2sub(q1, p2), u2);
    bool outsideB = (fp1 <= 0.0f && fq1 <= 0.0f) || (fp1 >= length2 && fq1 >= length2);

    if (!outsideA && !outsideB)
    {
        vec2 normalA = v2perp(u1);
        float separationA;
        {
            float ss1 = v2dot(v2sub(p2, p1), normalA);
            float ss2 = v2dot(v2sub(q2, p1), normalA);
            float s1p = min(ss1, ss2);
            float s1n = min(-ss1, -ss2);
            if (s1p > s1n)
                separationA = s1p;
            else
            {
                separationA = s1n;
                normalA = v2neg(normalA);
            }
        }
        vec2 normalB = v2perp(u2);
        float separationB;
        {
            float ss1 = v2dot(v2sub(p1, p2), normalB);
            float ss2 = v2dot(v2sub(q1, p2), normalB);
            float s1p = min(ss1, ss2);
            float s1n = min(-ss1, -ss2);
            if (s1p > s1n)
                separationB = s1p;
            else
            {
                separationB = s1n;
                normalB = v2neg(normalB);
            }
        }

        if (separationA + 0.1f * LINEAR_SLOP >= separationB)
        {
            vec2 cp = p2;
            vec2 cq = q2;
            if (fp2 < 0.0f && fq2 > 0.0f)
                cp = lerp2(p2, q2, (0.0f - fp2) / (fq2 - fp2));
            else if (fq2 < 0.0f && fp2 > 0.0f)
                cq = lerp2(q2, p2, (0.0f - fq2) / (fp2 - fq2));
            if (fp2 > length1 && fq2 < length1)
                cp = lerp2(p2, q2, (fp2 - length1) / (fp2 - fq2));
            else if (fq2 > length1 && fp2 < length1)
                cq = lerp2(q2, p2, (fq2 - length1) / (fq2 - fp2));
            float sp = v2dot(v2sub(cp, p1), normalA);
            float sq = v2dot(v2sub(cq, p1), normalA);
            if (sp <= distance + LINEAR_SLOP || sq <= distance + LINEAR_SLOP)
            {
                m.normal = normalA;
                lmPush(m, mulAdd(cp, 0.5f * (rA - rB - sp), normalA), sp - radius, makeId(0, 0));
                lmPush(m, mulAdd(cq, 0.5f * (rA - rB - sq), normalA), sq - radius, makeId(0, 1));
            }
        }
        else
        {
            vec2 cp = p1;
            vec2 cq = q1;
            if (fp1 < 0.0f && fq1 > 0.0f)
                cp = lerp2(p1, q1, (0.0f - fp1) / (fq1 - fp1));
            else if (fq1 < 0.0f && fp1 > 0.0f)
                cq = lerp2(q1, p1, (0.0f - fq1) / (fp1 - fq1));
            if (fp1 > length2 && fq1 < length2)
                cp = lerp2(p1, q1, (fp1 - length2) / (fp1 - fq1));
            else if (fq1 > length2 && fp1 < length2)
                cq = lerp2(q1, p1, (fq1 - length2) / (fq1 - fp1));
            float sp = v2dot(v2sub(cp, p2), normalB);
            float sq = v2dot(v2sub(cq, p2), normalB);
            if (sp <= distance + LINEAR_SLOP || sq <= distance + LINEAR_SLOP)
            {
                m.normal = v2neg(normalB);
                lmPush(m, mulAdd(cp, 0.5f * (rB - rA - sp), normalB), sp - radius, makeId(0, 0));
                lmPush(m, mulAdd(cq, 0.5f * (rB - rA - sq), normalB), sq - radius, makeId(1, 0));
            }
        }
    }

    if (m.count == 0)
    {
        vec2 normal = v2sub(closest2, closest1);
        if (v2dot(normal, normal) > epsSqr)
            normal = normalize2(normal);
        else
            normal = v2perp(u1);
        vec2 c1 = mulAdd(closest1, rA, normal);
        vec2 c2 = mulAdd(closest2, -rB, normal);
        m.normal = normal;
        lmPush(m, lerp2(c1, c2, 0.5f), distance - radius, makeId(f1 == 0.0f ? 0 : 1, f2 == 0.0f ? 0 : 1));
    }
    m.p0 = v2add(m.p0, origin);
    m.p1 = v2add(m.p1, origin);
    return m;
}

struct SegDist
{
    vec2 closest1;
    vec2 closest2;
    float fraction1;
    float fraction2;
    float distanceSquared;
};

// b2SegmentDistance.
SegDist segmentDistance(vec2 p1, vec2 q1, vec2 p2, vec2 q2)
{
    SegDist res;
    vec2 d1 = v2sub(q1, p1);
    vec2 d2 = v2sub(q2, p2);
    vec2 r = v2sub(p1, p2);
    float dd1 = v2dot(d1, d1);
    float dd2 = v2dot(d2, d2);
    float rd1 = v2dot(r, d1);
    float rd2 = v2dot(r, d2);
    const float epsSqr = FLT_EPS * FLT_EPS;
    res.fraction1 = 0.0f;
    res.fraction2 = 0.0f;
    if (dd1 < epsSqr || dd2 < epsSqr)
    {
        if (dd1 >= epsSqr)
            res.fraction1 = clampf(-rd1 / dd1, 0.0f, 1.0f);
        else if (dd2 >= epsSqr)
            res.fraction2 = clampf(rd2 / dd2, 0.0f, 1.0f);
    }
    else
    {
        float d12 = v2dot(d1, d2);
        float denominator = dd1 * dd2 - d12 * d12;
        float f1 = 0.0f;
        if (denominator != 0.0f)
            f1 = clampf((d12 * rd2 - rd1 * dd2) / denominator, 0.0f, 1.0f);
        float f2 = (d12 * f1 + rd2) / dd2;
        if (f2 < 0.0f)
        {
            f2 = 0.0f;
            f1 = clampf(-rd1 / dd1, 0.0f, 1.0f);
        }
        else if (f2 > 1.0f)
        {
            f2 = 1.0f;
            f1 = clampf((d12 - rd1) / dd1, 0.0f, 1.0f);
        }
        res.fraction1 = f1;
        res.fraction2 = f2;
    }
    res.closest1 = mulAdd(p1, res.fraction1, d1);
    res.closest2 = mulAdd(p2, res.fraction2, d2);
    res.distanceSquared = v2lenSq(v2sub(res.closest1, res.closest2));
    return res;
}

// b2ClipPolygons: reference edge on A (or on B when flip), incident edge on the other.
LM clipPolygons(Shape2 A, Shape2 B, int edgeA, int edgeB, bool flip)
{
    LM m = lmEmpty();
    int i11, i12, i21, i22;
    vec2 normal, v11, v12, v21, v22;
    float r1, r2;
    if (flip)
    {
        i11 = edgeB;
        i12 = edgeB + 1 < B.count ? edgeB + 1 : 0;
        i21 = edgeA;
        i22 = edgeA + 1 < A.count ? edgeA + 1 : 0;
        normal = B.n[i11];
        v11 = B.v[i11];
        v12 = B.v[i12];
        v21 = A.v[i21];
        v22 = A.v[i22];
        r1 = B.radius;
        r2 = A.radius;
    }
    else
    {
        i11 = edgeA;
        i12 = edgeA + 1 < A.count ? edgeA + 1 : 0;
        i21 = edgeB;
        i22 = edgeB + 1 < B.count ? edgeB + 1 : 0;
        normal = A.n[i11];
        v11 = A.v[i11];
        v12 = A.v[i12];
        v21 = B.v[i21];
        v22 = B.v[i22];
        r1 = A.radius;
        r2 = B.radius;
    }
    vec2 tangent = v2perp(normal);
    float lower1 = 0.0f;
    float upper1 = v2dot(v2sub(v12, v11), tangent);
    float upper2 = v2dot(v2sub(v21, v11), tangent);
    float lower2 = v2dot(v2sub(v22, v11), tangent);
    if (upper2 < lower1 || upper1 < lower2)
        return m;
    vec2 vLower = v22;
    if (lower2 < lower1 && upper2 - lower2 > FLT_EPS)
        vLower = lerp2(v22, v21, (lower1 - lower2) / (upper2 - lower2));
    vec2 vUpper = v21;
    if (upper2 > upper1 && upper2 - lower2 > FLT_EPS)
        vUpper = lerp2(v22, v21, (upper1 - lower2) / (upper2 - lower2));
    float separationLower = v2dot(v2sub(vLower, v11), normal);
    float separationUpper = v2dot(v2sub(vUpper, v11), normal);
    vLower = mulAdd(vLower, 0.5f * (r1 - r2 - separationLower), normal);
    vUpper = mulAdd(vUpper, 0.5f * (r1 - r2 - separationUpper), normal);
    float radius = r1 + r2;
    if (!flip)
    {
        m.normal = normal;
        lmPush(m, vLower, separationLower - radius, makeId(i11, i22));
        lmPush(m, vUpper, separationUpper - radius, makeId(i12, i21));
    }
    else
    {
        m.normal = v2neg(normal);
        lmPush(m, vUpper, separationUpper - radius, makeId(i21, i12));
        lmPush(m, vLower, separationLower - radius, makeId(i22, i11));
    }
    return m;
}

float findMaxSeparation(Shape2 P1, Shape2 P2, out int edgeIndex)
{
    int bestIndex = 0;
    float maxSeparation = -FLT_BIG;
    for (int i = 0; i < P1.count; i++)
    {
        float si = FLT_BIG;
        for (int j = 0; j < P2.count; j++)
            si = min(si, v2dot(P1.n[i], v2sub(P2.v[j], P1.v[i])));
        if (si > maxSeparation)
        {
            maxSeparation = si;
            bestIndex = i;
        }
    }
    edgeIndex = bestIndex;
    return maxSeparation;
}

int incidentEdgeOf(Shape2 P, vec2 searchDirection)
{
    int edge = 0;
    float minDot = FLT_BIG;
    for (int i = 0; i < P.count; i++)
    {
        float d = v2dot(searchDirection, P.n[i]);
        if (d < minDot)
        {
            minDot = d;
            edge = i;
        }
    }
    return edge;
}

// b2CollidePolygons (its "#if 1" branch): SAT, then clip, falling back to one vertex-vertex point
// when the clipped points are much further apart than the closest vertices.
LM lmPolygons(Shape2 A, Shape2 B, float spec)
{
    vec2 origin = A.v[0];
    for (int i = 0; i < A.count; i++)
        A.v[i] = v2sub(A.v[i], origin);
    for (int i = 0; i < B.count; i++)
        B.v[i] = v2sub(B.v[i], origin);
    float radius = A.radius + B.radius;

    LM m = lmEmpty();
    int edgeA = 0;
    float separationA = findMaxSeparation(A, B, edgeA);
    if (separationA > spec + radius)
        return m;
    int edgeB = 0;
    float separationB = findMaxSeparation(B, A, edgeB);
    if (separationB > spec + radius)
        return m;

    bool flip;
    if (separationA >= separationB)
    {
        flip = false;
        edgeB = incidentEdgeOf(B, A.n[edgeA]);
    }
    else
    {
        flip = true;
        edgeA = incidentEdgeOf(A, B.n[edgeB]);
    }

    if (separationA > 0.1f * LINEAR_SLOP || separationB > 0.1f * LINEAR_SLOP)
    {
        int i11 = edgeA;
        int i12 = edgeA + 1 < A.count ? edgeA + 1 : 0;
        int i21 = edgeB;
        int i22 = edgeB + 1 < B.count ? edgeB + 1 : 0;
        vec2 v11 = A.v[i11];
        vec2 v12 = A.v[i12];
        vec2 v21 = B.v[i21];
        vec2 v22 = B.v[i22];
        SegDist result = segmentDistance(v11, v12, v21, v22);
        float distance = sqrt(result.distanceSquared);
        float separation = distance - radius;
        if (separation > spec)
            return m;
        m = clipPolygons(A, B, edgeA, edgeB, flip);
        float minSeparation = FLT_BIG;
        if (m.count > 0)
            minSeparation = min(minSeparation, m.s0);
        if (m.count > 1)
            minSeparation = min(minSeparation, m.s1);
        if (separation + 0.1f * LINEAR_SLOP < minSeparation &&
            (result.fraction1 == 0.0f || result.fraction1 == 1.0f) &&
            (result.fraction2 == 0.0f || result.fraction2 == 1.0f) && distance > 0.0f)
        {
            int ia = result.fraction1 == 0.0f ? i11 : i12;
            int ib = result.fraction2 == 0.0f ? i21 : i22;
            vec2 va = result.fraction1 == 0.0f ? v11 : v12;
            vec2 vb = result.fraction2 == 0.0f ? v21 : v22;
            vec2 normal = v2mul(v2sub(vb, va), 1.0f / distance);
            vec2 c1 = mulAdd(va, A.radius, normal);
            vec2 c2 = mulAdd(vb, -B.radius, normal);
            m = lmEmpty();
            m.normal = normal;
            lmPush(m, lerp2(c1, c2, 0.5f), distance - radius, makeId(ia, ib));
        }
    }
    else
        m = clipPolygons(A, B, edgeA, edgeB, flip);

    m.p0 = v2add(m.p0, origin);
    m.p1 = v2add(m.p1, origin);
    return m;
}

LM lmChainCircle(vec2 g1, vec2 p1, vec2 p2, vec2 g2, vec2 pB, float rB, float spec)
{
    LM m = lmEmpty();
    vec2 e = v2sub(p2, p1);
    float offset = v2dot(rightPerp(e), v2sub(pB, p1));
    if (offset < 0.0f)
        return m; // one-sided
    float u = v2dot(e, v2sub(p2, pB));
    float v = v2dot(e, v2sub(pB, p1));
    vec2 pA;
    if (v <= 0.0f)
    {
        // Behind p1: the previous segment owns it.
        if (v2dot(v2sub(p1, g1), v2sub(pB, p1)) <= 0.0f)
            return m;
        pA = p1;
    }
    else if (u <= 0.0f)
    {
        if (v2dot(v2sub(g2, p2), v2sub(pB, p2)) > 0.0f)
            return m;
        pA = p2;
    }
    else
    {
        float ee = v2dot(e, e);
        pA = ee > 0.0f ? v2mul(vec2(u * p1.x + v * p2.x, u * p1.y + v * p2.y), 1.0f / ee) : p1;
    }
    float distance;
    vec2 normal = lengthAndNormalize(v2sub(pB, pA), distance);
    float separation = distance - rB;
    if (separation > spec)
        return m;
    vec2 cB = mulAdd(pB, -rB, normal);
    m.normal = normal;
    lmPush(m, lerp2(pA, cB, 0.5f), separation, 0);
    return m;
}

// b2ClipSegments.
LM clipSegments(vec2 a1, vec2 a2, vec2 b1, vec2 b2, vec2 normal, float ra, float rb, int id1, int id2)
{
    LM m = lmEmpty();
    vec2 tangent = v2perp(normal);
    float lower1 = 0.0f;
    float upper1 = v2dot(v2sub(a2, a1), tangent);
    float upper2 = v2dot(v2sub(b1, a1), tangent);
    float lower2 = v2dot(v2sub(b2, a1), tangent);
    if (upper2 < lower1 || upper1 < lower2)
        return m;
    vec2 vLower = b2;
    if (lower2 < lower1 && upper2 - lower2 > FLT_EPS)
        vLower = lerp2(b2, b1, (lower1 - lower2) / (upper2 - lower2));
    vec2 vUpper = b1;
    if (upper2 > upper1 && upper2 - lower2 > FLT_EPS)
        vUpper = lerp2(b2, b1, (upper1 - lower2) / (upper2 - lower2));
    float separationLower = v2dot(v2sub(vLower, a1), normal);
    float separationUpper = v2dot(v2sub(vUpper, a1), normal);
    vLower = mulAdd(vLower, 0.5f * (ra - rb - separationLower), normal);
    vUpper = mulAdd(vUpper, 0.5f * (ra - rb - separationUpper), normal);
    float radius = ra + rb;
    m.normal = normal;
    lmPush(m, vLower, separationLower - radius, id1);
    lmPush(m, vUpper, separationUpper - radius, id2);
    return m;
}

static const int NORMAL_SKIP = 0;
static const int NORMAL_ADMIT = 1;
static const int NORMAL_SNAP = 2;

struct ChainParams
{
    vec2 edge1;
    vec2 normal0;
    vec2 normal2;
    bool convex1;
    bool convex2;
};

// b2ClassifyNormal: the Gauss map test that keeps a body from catching on a chain's joints.
int classifyNormal(ChainParams params, vec2 normal)
{
    const float sinTol = 0.01f;
    if (v2dot(normal, params.edge1) <= 0.0f)
    {
        if (params.convex1)
            return v2cross(normal, params.normal0) > sinTol ? NORMAL_SKIP : NORMAL_ADMIT;
        return NORMAL_SNAP;
    }
    if (params.convex2)
        return v2cross(params.normal2, normal) > sinTol ? NORMAL_SKIP : NORMAL_ADMIT;
    return NORMAL_SNAP;
}

// The closest points between segment p1-p2 and a convex hull (radius ignored): what
// b2ShapeDistance returns, with its simplex cache, found by brute force over the hull's edges
// instead of GJK (at most MAX_POLY edges, and a thread has no cache to carry between steps).
// cacheCount 1 = vertex-vertex; 2 with ia1 == ia2 = a segment vertex against hull edge
// (ib1, ib2); 2 with ia1 != ia2 = the segment interior against hull vertex ib1 == ib2.
struct ChainDist
{
    float distance;
    vec2 pointA;
    vec2 pointB;
    int cacheCount;
    int ia1;
    int ia2;
    int ib1;
    int ib2;
};

bool insideHull(Shape2 B, vec2 p)
{
    for (int i = 0; i < B.count; i++)
        if (v2dot(B.n[i], v2sub(p, B.v[i])) > 0.0f)
            return false;
    return true;
}

ChainDist chainDistance(vec2 p1, vec2 p2, Shape2 B)
{
    ChainDist d;
    d.pointA = p1;
    d.pointB = B.v[0];
    d.cacheCount = 1;
    d.ia1 = 0;
    d.ia2 = 0;
    d.ib1 = 0;
    d.ib2 = 0;
    // A capsule's two "edges" are the same segment both ways; one pass covers it.
    int edges = B.count == 2 ? 1 : B.count;
    float best = FLT_BIG;
    SegDist bestR = segmentDistance(p1, p2, B.v[0], B.v[1]);
    int bestEdge = 0;
    for (int i = 0; i < edges; i++)
    {
        int j = i + 1 < B.count ? i + 1 : 0;
        SegDist r = segmentDistance(p1, p2, B.v[i], B.v[j]);
        if (r.distanceSquared < best)
        {
            best = r.distanceSquared;
            bestR = r;
            bestEdge = i;
        }
    }
    if (B.count > 2 && (insideHull(B, p1) || insideHull(B, p2)))
        best = 0.0f; // the segment reaches inside: overlapping, as GJK reports it
    d.distance = sqrt(best);
    if (best == 0.0f)
        return d;
    d.pointA = bestR.closest1;
    d.pointB = bestR.closest2;
    int j = bestEdge + 1 < B.count ? bestEdge + 1 : 0;
    bool endA = bestR.fraction1 == 0.0f || bestR.fraction1 == 1.0f;
    bool endB = bestR.fraction2 == 0.0f || bestR.fraction2 == 1.0f;
    int vb = bestR.fraction2 == 0.0f ? bestEdge : j;
    int va = bestR.fraction1 == 0.0f ? 0 : 1;
    if (endA && endB)
    {
        d.cacheCount = 1;
        d.ia1 = va;
        d.ia2 = va;
        d.ib1 = vb;
        d.ib2 = vb;
    }
    else if (endA)
    {
        d.cacheCount = 2;
        d.ia1 = va;
        d.ia2 = va;
        d.ib1 = bestEdge;
        d.ib2 = j;
    }
    else
    {
        // The segment interior against a hull vertex (two parallel edges settle on a vertex
        // in GJK too).
        d.cacheCount = 2;
        d.ia1 = 0;
        d.ia2 = 1;
        d.ib1 = endB ? vb : bestEdge;
        d.ib2 = d.ib1;
    }
    return d;
}

// b2CollideChainSegmentAndPolygon. A capsule comes in as its two-vertex hull.
LM lmChainPolygon(vec2 g1, vec2 p1, vec2 p2, vec2 g2, Shape2 B, float spec)
{
    LM m = lmEmpty();
    float radiusB = B.radius;
    vec2 centroidB = B.centroid;
    vec2 edge1 = normalize2(v2sub(p2, p1));
    ChainParams sp;
    sp.edge1 = edge1;
    const float convexTol = 0.01f;
    vec2 edge0 = normalize2(v2sub(p1, g1));
    sp.normal0 = rightPerp(edge0);
    sp.convex1 = v2cross(edge0, edge1) >= convexTol;
    vec2 edge2 = normalize2(v2sub(g2, p2));
    sp.normal2 = rightPerp(edge2);
    sp.convex2 = v2cross(edge1, edge2) >= convexTol;

    vec2 normal1 = rightPerp(edge1);
    bool behind1 = v2dot(normal1, v2sub(centroidB, p1)) < 0.0f;
    bool behind0 = true;
    bool behind2 = true;
    if (sp.convex1)
        behind0 = v2dot(sp.normal0, v2sub(centroidB, p1)) < 0.0f;
    if (sp.convex2)
        behind2 = v2dot(sp.normal2, v2sub(centroidB, p2)) < 0.0f;
    if (behind1 && behind0 && behind2)
        return m;

    int count = B.count;
    ChainDist output = chainDistance(p1, p2, B);
    if (output.distance > radiusB + spec)
        return m;

    vec2 n0 = sp.convex1 ? sp.normal0 : normal1;
    vec2 n2 = sp.convex2 ? sp.normal2 : normal1;
    int incidentIndex = -1;
    int incidentNormal = -1;

    if (!behind1 && output.distance > 0.1f * LINEAR_SLOP)
    {
        if (output.cacheCount == 1)
        {
            vec2 pA = output.pointA;
            vec2 pB = output.pointB;
            vec2 normal = normalize2(v2sub(pB, pA));
            int type = classifyNormal(sp, normal);
            if (type == NORMAL_SKIP)
                return m;
            if (type == NORMAL_ADMIT)
            {
                // Box2D anchors this point on the segment; the midpoint form puts it halfway.
                float s = output.distance - radiusB;
                m.normal = normal;
                lmPush(m, mulAdd(pA, 0.5f * s, normal), s, makeId(output.ia1, output.ib1));
                return m;
            }
            incidentIndex = output.ib1;
        }
        else
        {
            int ia1 = output.ia1;
            int ia2 = output.ia2;
            int ib1 = output.ib1;
            int ib2 = output.ib2;
            if (ia1 == ia2)
            {
                vec2 normalB = v2sub(output.pointA, output.pointB);
                float dot1 = v2dot(normalB, B.n[ib1]);
                float dot2 = v2dot(normalB, B.n[ib2]);
                int ib = dot1 > dot2 ? ib1 : ib2;
                normalB = B.n[ib];
                int type = classifyNormal(sp, v2neg(normalB));
                if (type == NORMAL_SKIP)
                    return m;
                if (type == NORMAL_ADMIT)
                {
                    ib1 = ib;
                    ib2 = ib < count - 1 ? ib + 1 : 0;
                    vec2 b1 = B.v[ib1];
                    vec2 b2 = B.v[ib2];
                    dot1 = v2dot(normalB, v2sub(p1, b1));
                    dot2 = v2dot(normalB, v2sub(p2, b1));
                    if (dot1 < dot2)
                    {
                        if (v2dot(n0, normalB) < v2dot(normal1, normalB))
                            return m; // the neighbour is incident
                    }
                    else
                    {
                        if (v2dot(n2, normalB) < v2dot(normal1, normalB))
                            return m;
                    }
                    m = clipSegments(b1, b2, p1, p2, normalB, radiusB, 0.0f, makeId(ib1, 1), makeId(ib2, 0));
                    m.normal = v2neg(normalB);
                    return m;
                }
                incidentNormal = ib;
            }
            else
            {
                float dot1 = v2dot(normal1, v2sub(B.v[ib1], p1));
                float dot2 = v2dot(normal1, v2sub(B.v[ib2], p2));
                incidentIndex = dot1 < dot2 ? ib1 : ib2;
            }
        }
    }
    else
    {
        // SAT edge normal.
        float edgeSeparation = FLT_BIG;
        for (int i = 0; i < count; i++)
        {
            float s = v2dot(normal1, v2sub(B.v[i], p1));
            if (s < edgeSeparation)
            {
                edgeSeparation = s;
                incidentIndex = i;
            }
        }
        if (sp.convex1)
        {
            float s0 = FLT_BIG;
            for (int i = 0; i < count; i++)
                s0 = min(s0, v2dot(sp.normal0, v2sub(B.v[i], p1)));
            if (s0 > edgeSeparation)
            {
                edgeSeparation = s0;
                incidentIndex = -1;
            }
        }
        if (sp.convex2)
        {
            float s2 = FLT_BIG;
            for (int i = 0; i < count; i++)
                s2 = min(s2, v2dot(sp.normal2, v2sub(B.v[i], p2)));
            if (s2 > edgeSeparation)
            {
                edgeSeparation = s2;
                incidentIndex = -1;
            }
        }
        // SAT polygon normals.
        float polygonSeparation = -FLT_BIG;
        int referenceIndex = -1;
        for (int i = 0; i < count; i++)
        {
            vec2 n = B.n[i];
            if (classifyNormal(sp, v2neg(n)) != NORMAL_ADMIT)
                continue;
            vec2 p = B.v[i];
            float s = min(v2dot(n, v2sub(p2, p)), v2dot(n, v2sub(p1, p)));
            if (s > polygonSeparation)
            {
                polygonSeparation = s;
                referenceIndex = i;
            }
        }
        if (polygonSeparation > edgeSeparation)
        {
            int ia1 = referenceIndex;
            int ia2 = ia1 < count - 1 ? ia1 + 1 : 0;
            vec2 a1 = B.v[ia1];
            vec2 a2 = B.v[ia2];
            vec2 n = B.n[ia1];
            float dot1 = v2dot(n, v2sub(p1, a1));
            float dot2 = v2dot(n, v2sub(p2, a1));
            if (dot1 < dot2)
            {
                if (v2dot(n0, n) < v2dot(normal1, n))
                    return m;
            }
            else
            {
                if (v2dot(n2, n) < v2dot(normal1, n))
                    return m;
            }
            m = clipSegments(a1, a2, p1, p2, n, radiusB, 0.0f, makeId(ia1, 1), makeId(ia2, 0));
            m.normal = v2neg(n);
            return m;
        }
        if (incidentIndex == -1)
            return m; // the neighbouring segment is the separating axis
    }

    // Segment normal: the incident hull edge is the one at the incident vertex most
    // anti-parallel to the segment normal.
    int ib1, ib2;
    if (incidentNormal != -1)
    {
        ib1 = incidentNormal;
        ib2 = ib1 < count - 1 ? ib1 + 1 : 0;
    }
    else
    {
        int i2 = incidentIndex;
        int i1 = i2 > 0 ? i2 - 1 : count - 1;
        float d1 = v2dot(normal1, B.n[i1]);
        float d2 = v2dot(normal1, B.n[i2]);
        if (d1 < d2)
        {
            ib1 = i1;
            ib2 = i2;
        }
        else
        {
            ib1 = i2;
            ib2 = i2 < count - 1 ? i2 + 1 : 0;
        }
    }
    return clipSegments(p1, p2, B.v[ib1], B.v[ib2], normal1, 0.0f, radiusB, makeId(0, ib2), makeId(1, ib1));
}

// Shape rank for the dispatch, as Box2D registers its pair functions: the higher rank is A.
int shapeRank(Shape2 s)
{
    if (s.kind == KIND_CHAIN)
        return 3;
    if (s.kind == KIND_ROUND)
        return 0;
    return s.count == 2 ? 1 : 2;
}

LM collideOrdered(Shape2 A, Shape2 B, float spec)
{
    if (A.kind == KIND_CHAIN)
    {
        if (B.kind == KIND_ROUND)
            return lmChainCircle(A.v[0], A.v[1], A.v[2], A.v[3], B.v[0], B.radius, spec);
        if (B.kind == KIND_HULL)
            return lmChainPolygon(A.v[0], A.v[1], A.v[2], A.v[3], B, spec);
        return lmEmpty(); // chain against chain: both static, never paired
    }
    if (A.kind == KIND_ROUND)
        return lmCircles(A.v[0], A.radius, B.v[0], B.radius, spec);
    if (B.kind == KIND_ROUND)
    {
        if (A.count == 2)
            return lmCapsuleCircle(A.v[0], A.v[1], A.radius, B.v[0], B.radius, spec);
        return lmPolygonCircle(A, B.v[0], B.radius, spec);
    }
    if (A.count == 2 && B.count == 2)
        return lmCapsules(A.v[0], A.v[1], A.radius, B.v[0], B.v[1], B.radius, spec);
    return lmPolygons(A, B, spec);
}

// Collides two shapes given relative to one origin (both in world orientation). Returns the
// contact count (0..2); `normal` points from B to A, and each contact carries the two surface
// points (relative to the origin). Points separated by more than `margin` are dropped (the legacy
// rule). Keys carry FEATURE_SHAPES so they never alias a legacy box-box feature key.
public static const int FEATURE_SHAPES = 0x20000000;

public int collideShapes(Shape2 a, Shape2 b, float margin, out vec2 normal, out vec2 xA0, out vec2 xB0, out int k0,
                         out vec2 xA1, out vec2 xB1, out int k1)
{
    bool swap = shapeRank(b) > shapeRank(a);
    LM m = swap ? collideOrdered(b, a, margin) : collideOrdered(a, b, margin);
    xA0 = vec2(0.0f, 0.0f);
    xB0 = vec2(0.0f, 0.0f);
    xA1 = vec2(0.0f, 0.0f);
    xB1 = vec2(0.0f, 0.0f);
    k0 = 0;
    k1 = 0;
    // m.normal points from m's A to m's B; ours from B to A.
    vec2 nAB = swap ? v2neg(m.normal) : m.normal;
    normal = v2neg(nAB);
    int cnt = 0;
    for (int i = 0; i < m.count; i++)
    {
        vec2 p = i == 0 ? m.p0 : m.p1;
        float s = i == 0 ? m.s0 : m.s1;
        int id = (i == 0 ? m.id0 : m.id1) | FEATURE_SHAPES;
        if (s > margin)
            continue;
        // Box2D's midpoint: the surfaces lie half the separation either side along its normal.
        vec2 sa = mulAdd(p, -0.5f * s, m.normal);
        vec2 sb = mulAdd(p, 0.5f * s, m.normal);
        vec2 pa = swap ? sb : sa;
        vec2 pb = swap ? sa : sb;
        if (cnt == 0)
        {
            xA0 = pa;
            xB0 = pb;
            k0 = id;
        }
        else
        {
            xA1 = pa;
            xB1 = pb;
            k1 = id;
        }
        cnt++;
    }
    return cnt;
}
