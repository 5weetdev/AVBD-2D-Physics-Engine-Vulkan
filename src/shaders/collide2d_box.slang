// 2D shape collision: box-box (SAT + reference/incident clipping, at most two contacts),
// box-circle and circle-circle. Pure functions of two poses; no buffers.
//
// Adding a shape is one function here and one enum value. The convention matches the 3D
// collider (collide.slang): contacts carry one anchor on each body's surface, in each body's
// local frame, and the constraint normal points from B to A, so that the contact error
// C = n . (xA - xB) + COLLISION_MARGIN2 is negative when the shapes overlap.
//
// Box-box is Box2D-lite's clipper (feature ids from the clipped edges give the stable
// warm-start keys), extended with a margin reach: the caller's COLLISION_MARGIN2 plus a
// speculative distance (narrowphase2d.slang).


module collide2d_box;

import maths2d;

public static const float COLLISION_MARGIN2 = 0.01f;
public static const int FEATURE_ROUND = 0x40000000;
public static const int SHAPE_BOX = 0;
public static const int SHAPE_CIRCLE = 1;  // an offset circle has its centre as one vertex in the pool
public static const int SHAPE_POLYGON = 2; // rounded convex polygon, at most MAX_POLY vertices
public static const int SHAPE_CAPSULE = 3;
public static const int SHAPE_SEGMENT = 4; // static, two-sided
public static const int SHAPE_CHAIN = 5;   // static, one-sided: ghost1, p1, p2, ghost2
public static const int MAX_POLY = 8;

// How a body's contact anchor moves with it (narrowphase2d.slang stores it, solver2d.slang
// anchorMotion moves it). ANCHOR_MATERIAL: a point of the body (box corners, sharp polygons,
// segments, chains). ANCHOR_DISC: the legacy centred circle's rolling surface point.
// ANCHOR_CORE: a rounded shape (capsule, rounded polygon, offset circle): the anchor is the
// material point of the core under the contact, and the surface sits `radius` from it along the
// contact normal, so the core turns with the body while the rounded surface rolls under the
// contact.
public static const int ANCHOR_MATERIAL = 0;
public static const int ANCHOR_DISC = 1;
public static const int ANCHOR_CORE = 2;

public int anchorModeOf(int shape, int vCnt, float radius)
{
    if (shape == SHAPE_CIRCLE && vCnt == 0)
        return ANCHOR_DISC;
    return radius > 0.0f ? ANCHOR_CORE : ANCHOR_MATERIAL;
}

// True for the pairs the original box/circle collider handles (collide2D); every other pair goes
// through the Box2D port (collideShapes).
public bool legacyShape(int shape, int vCnt)
{
    return shape == SHAPE_BOX || (shape == SHAPE_CIRCLE && vCnt == 0);
}

public struct Contact2
{
    public vec2 rA; // local anchors
    public vec2 rB;
    public int key; // feature id, stable while the same features touch
};

struct Man2
{
    vec2 nAB; // from A to B
    int count;
    vec2 xA0;
    vec2 xB0;
    int k0;
    vec2 xA1;
    vec2 xB1;
    int k1;
};

struct ClipV
{
    vec2 v;
    int inE1;
    int outE1;
    int inE2;
    int outE2;
};

int packFeature(ClipV c)
{
    return (c.inE1 & 255) | ((c.outE1 & 255) << 8) | ((c.inE2 & 255) << 16) | ((c.outE2 & 255) << 24);
}

ClipV flipFeature(ClipV c)
{
    ClipV r;
    r.v = c.v;
    r.inE1 = c.inE2;
    r.inE2 = c.inE1;
    r.outE1 = c.outE2;
    r.outE2 = c.outE1;
    return r;
}

int clipSegment(ClipV in0, ClipV in1, vec2 normal, float offset, int clipEdge, out ClipV o0, out ClipV o1)
{
    o0 = in0;
    o1 = in1;
    int numOut = 0;
    float d0 = v2dot(normal, in0.v) - offset;
    float d1 = v2dot(normal, in1.v) - offset;

    if (d0 <= 0.0f)
    {
        o0 = in0;
        numOut = 1;
    }
    if (d1 <= 0.0f)
    {
        if (numOut == 0)
            o0 = in1;
        else
            o1 = in1;
        numOut++;
    }
    if (d0 * d1 < 0.0f)
    {
        float interp = d0 / (d0 - d1);
        ClipV n;
        n.v = v2add(in0.v, v2mul(v2sub(in1.v, in0.v), interp));
        if (d0 > 0.0f)
        {
            n.inE1 = clipEdge;
            n.outE1 = in0.outE1;
            n.inE2 = 0;
            n.outE2 = in0.outE2;
        }
        else
        {
            n.inE1 = in1.inE1;
            n.outE1 = clipEdge;
            n.inE2 = in1.inE2;
            n.outE2 = 0;
        }
        if (numOut == 0)
            o0 = n;
        else if (numOut == 1)
            o1 = n;
        numOut++;
    }
    return numOut;
}

void incidentEdge(vec2 h, vec2 pos, float angle, vec2 normal, out ClipV c0, out ClipV c1)
{
    float c = cos(angle);
    float s = sin(angle);
    // The reference normal in the incident box's frame, flipped.
    vec2 n = vec2(-(c * normal.x + s * normal.y), -(-s * normal.x + c * normal.y));

    vec2 a0, a1;
    int in0, out0, in1, out1;
    if (abs(n.x) > abs(n.y))
    {
        if (n.x >= 0.0f)
        {
            a0 = vec2(h.x, -h.y); in0 = 3; out0 = 4;
            a1 = vec2(h.x, h.y); in1 = 4; out1 = 1;
        }
        else
        {
            a0 = vec2(-h.x, h.y); in0 = 1; out0 = 2;
            a1 = vec2(-h.x, -h.y); in1 = 2; out1 = 3;
        }
    }
    else
    {
        if (n.y >= 0.0f)
        {
            a0 = vec2(h.x, h.y); in0 = 4; out0 = 1;
            a1 = vec2(-h.x, h.y); in1 = 1; out1 = 2;
        }
        else
        {
            a0 = vec2(-h.x, -h.y); in0 = 2; out0 = 3;
            a1 = vec2(h.x, -h.y); in1 = 3; out1 = 4;
        }
    }
    c0.v = v2add(pos, v2rot(angle, a0));
    c0.inE1 = 0; c0.outE1 = 0; c0.inE2 = in0; c0.outE2 = out0;
    c1.v = v2add(pos, v2rot(angle, a1));
    c1.inE1 = 0; c1.outE1 = 0; c1.inE2 = in1; c1.outE2 = out1;
}

void collideBoxBox(vec2 posA, float angA, vec2 hA, vec2 posB, float angB, vec2 hB, float margin, inout Man2 m)
{
    m.count = 0;
    float cA = cos(angA), sA = sin(angA);
    float cB = cos(angB), sB = sin(angB);
    vec2 colA1 = vec2(cA, sA), colA2 = vec2(-sA, cA);
    vec2 colB1 = vec2(cB, sB), colB2 = vec2(-sB, cB);

    vec2 dp = v2sub(posB, posA);
    vec2 dA = vec2(v2dot(colA1, dp), v2dot(colA2, dp));
    vec2 dB = vec2(v2dot(colB1, dp), v2dot(colB2, dp));

    // |RotA^T RotB| is [|c| |s|; |s| |c|] for the relative angle, and it is symmetric.
    float dth = angB - angA;
    float ac = abs(cos(dth)), as_ = abs(sin(dth));

    vec2 faceA = vec2(abs(dA.x) - hA.x - (ac * hB.x + as_ * hB.y),
                      abs(dA.y) - hA.y - (as_ * hB.x + ac * hB.y));
    if (faceA.x > margin || faceA.y > margin)
        return;
    vec2 faceB = vec2(abs(dB.x) - (ac * hA.x + as_ * hA.y) - hB.x,
                      abs(dB.y) - (as_ * hA.x + ac * hA.y) - hB.y);
    if (faceB.x > margin || faceB.y > margin)
        return;

    // Best axis, preferring A's faces on near ties.
    int axis = 0;
    float separation = faceA.x;
    vec2 normal = colA1;
    if (dA.x <= 0.0f)
        normal = v2neg(colA1);
    const float relativeTol = 0.95f;
    const float absoluteTol = 0.01f;
    if (faceA.y > relativeTol * separation + absoluteTol * hA.y)
    {
        axis = 1;
        separation = faceA.y;
        normal = colA2;
        if (dA.y <= 0.0f)
            normal = v2neg(colA2);
    }
    if (faceB.x > relativeTol * separation + absoluteTol * hB.x)
    {
        axis = 2;
        separation = faceB.x;
        normal = colB1;
        if (dB.x <= 0.0f)
            normal = v2neg(colB1);
    }
    if (faceB.y > relativeTol * separation + absoluteTol * hB.y)
    {
        axis = 3;
        separation = faceB.y;
        normal = colB2;
        if (dB.y <= 0.0f)
            normal = v2neg(colB2);
    }

    vec2 frontNormal = normal;
    vec2 sideNormal = colA2;
    float front = 0.0f, negSide = 0.0f, posSide = 0.0f;
    int negEdge = 0, posEdge = 0;
    ClipV inc0, inc1;
    if (axis == 0)
    {
        frontNormal = normal;
        front = v2dot(posA, frontNormal) + hA.x;
        sideNormal = colA2;
        float side = v2dot(posA, sideNormal);
        negSide = -side + hA.y;
        posSide = side + hA.y;
        negEdge = 3;
        posEdge = 1;
        incidentEdge(hB, posB, angB, frontNormal, inc0, inc1);
    }
    else if (axis == 1)
    {
        frontNormal = normal;
        front = v2dot(posA, frontNormal) + hA.y;
        sideNormal = colA1;
        float side = v2dot(posA, sideNormal);
        negSide = -side + hA.x;
        posSide = side + hA.x;
        negEdge = 2;
        posEdge = 4;
        incidentEdge(hB, posB, angB, frontNormal, inc0, inc1);
    }
    else if (axis == 2)
    {
        frontNormal = v2neg(normal);
        front = v2dot(posB, frontNormal) + hB.x;
        sideNormal = colB2;
        float side = v2dot(posB, sideNormal);
        negSide = -side + hB.y;
        posSide = side + hB.y;
        negEdge = 3;
        posEdge = 1;
        incidentEdge(hA, posA, angA, frontNormal, inc0, inc1);
    }
    else
    {
        frontNormal = v2neg(normal);
        front = v2dot(posB, frontNormal) + hB.y;
        sideNormal = colB1;
        float side = v2dot(posB, sideNormal);
        negSide = -side + hB.x;
        posSide = side + hB.x;
        negEdge = 2;
        posEdge = 4;
        incidentEdge(hA, posA, angA, frontNormal, inc0, inc1);
    }

    // Clip the incident edge against the reference face's two side planes.
    ClipV p0, p1, q0, q1;
    int np = clipSegment(inc0, inc1, v2neg(sideNormal), negSide, negEdge, p0, p1);
    if (np < 2)
        return;
    np = clipSegment(p0, p1, sideNormal, posSide, posEdge, q0, q1);
    if (np < 2)
        return;

    m.nAB = normal;
    for (int i = 0; i < 2; i++)
    {
        ClipV q = q0;
        if (i == 1)
            q = q1;
        float sep = v2dot(frontNormal, q.v) - front;
        if (sep > margin)
            continue;
        // Slide the incident point onto the reference face.
        vec2 onRef = v2sub(q.v, v2mul(frontNormal, sep));
        ClipV fp = q;
        if (axis >= 2)
            fp = flipFeature(q);
        vec2 xA = q.v;
        vec2 xB = onRef;
        if (axis < 2)
        {
            xA = onRef;
            xB = q.v;
        }
        if (m.count == 0)
        {
            m.xA0 = xA;
            m.xB0 = xB;
            m.k0 = packFeature(fp);
        }
        else
        {
            m.xA1 = xA;
            m.xB1 = xB;
            m.k1 = packFeature(fp);
        }
        m.count++;
    }
}

void collideCircles(vec2 posA, float rA, vec2 posB, float rB, float margin, inout Man2 m)
{
    m.count = 0;
    vec2 d = v2sub(posB, posA);
    float dist = v2len(d);
    if (dist > rA + rB + COLLISION_MARGIN2)
        return;
    vec2 n = vec2(0.0f, 1.0f);
    if (dist > 1.0e-6f)
        n = v2mul(d, 1.0f / dist);
    m.nAB = n;
    m.xA0 = v2add(posA, v2mul(n, rA));
    m.xB0 = v2sub(posB, v2mul(n, rB));
    m.k0 = FEATURE_ROUND;
    m.count = 1;
}

// A is the box, B the circle.
void collideBoxCircle(vec2 posA, float angA, vec2 hA, vec2 posB, float rB, float margin, inout Man2 m)
{
    m.count = 0;
    vec2 local = v2rot(-angA, v2sub(posB, posA));
    vec2 closest = vec2(clampf(local.x, -hA.x, hA.x), clampf(local.y, -hA.y, hA.y));
    bool inside = (closest.x == local.x) && (closest.y == local.y);

    vec2 nLocal;
    if (inside)
    {
        // Centre inside the box: leave through the nearest face.
        float dx = hA.x - abs(local.x);
        float dy = hA.y - abs(local.y);
        if (dx < dy)
        {
            float sx = local.x >= 0.0f ? 1.0f : -1.0f;
            nLocal = vec2(sx, 0.0f);
            closest = vec2(sx * hA.x, local.y);
        }
        else
        {
            float sy = local.y >= 0.0f ? 1.0f : -1.0f;
            nLocal = vec2(0.0f, sy);
            closest = vec2(local.x, sy * hA.y);
        }
    }
    else
    {
        vec2 delta = v2sub(local, closest);
        float dist = v2len(delta);
        if (dist > rB + COLLISION_MARGIN2)
            return;
        nLocal = vec2(0.0f, 1.0f);
        if (dist > 1.0e-6f)
            nLocal = v2mul(delta, 1.0f / dist);
    }

    vec2 n = v2rot(angA, nLocal);
    m.nAB = n;
    m.xA0 = v2add(posA, v2rot(angA, closest));
    m.xB0 = v2sub(posB, v2mul(n, rB));
    m.k0 = FEATURE_ROUND;
    m.count = 1;
}

// Collides two shapes. Returns the contact count (0..2). `normal` points from B to A. Points
// separated by up to `margin` are kept (COLLISION_MARGIN2 plus any speculative reach).
public int collide2D(vec2 posA, float angA, vec2 halfA, int shapeA,
                     vec2 posB, float angB, vec2 halfB, int shapeB, float margin,
                     out vec2 normal, out Contact2 c0, out Contact2 c1)
{
    Man2 m;
    m.count = 0;
    m.nAB = vec2(0.0f, 1.0f);
    m.xA0 = vec2(0.0f, 0.0f);
    m.xB0 = vec2(0.0f, 0.0f);
    m.k0 = 0;
    m.xA1 = vec2(0.0f, 0.0f);
    m.xB1 = vec2(0.0f, 0.0f);
    m.k1 = 0;

    if (shapeA == SHAPE_BOX && shapeB == SHAPE_BOX)
    {
        collideBoxBox(posA, angA, halfA, posB, angB, halfB, margin, m);
    }
    else if (shapeA == SHAPE_CIRCLE && shapeB == SHAPE_CIRCLE)
    {
        collideCircles(posA, halfA.x, posB, halfB.x, margin, m);
    }
    else if (shapeA == SHAPE_BOX)
    {
        collideBoxCircle(posA, angA, halfA, posB, halfB.x, margin, m);
    }
    else
    {
        // Circle first: collide with the box as A and swap the roles back.
        collideBoxCircle(posB, angB, halfB, posA, halfA.x, margin, m);
        m.nAB = v2neg(m.nAB);
        vec2 t0 = m.xA0;
        m.xA0 = m.xB0;
        m.xB0 = t0;
        vec2 t1 = m.xA1;
        m.xA1 = m.xB1;
        m.xB1 = t1;
    }

    normal = v2neg(m.nAB);
    c0.rA = v2rot(-angA, v2sub(m.xA0, posA));
    c0.rB = v2rot(-angB, v2sub(m.xB0, posB));
    c0.key = m.k0;
    c1.rA = v2rot(-angA, v2sub(m.xA1, posA));
    c1.rB = v2rot(-angB, v2sub(m.xB1, posB));
    c1.key = m.k1;
    return m.count;
}
