#pragma once

// Authoring for the 2D solver: Scene2D is a host-only record of bodies and joints that
// VkWorld2D::build() uploads. It knows no solver and no Vulkan. Sizes are full extents,
// density 0 makes a body static, gravity points along -y.
//
// Joints need anchors that coincide
// in world space at authoring time (C0 ~ 0), or penalty ramping amplifies the error. The
// *At helpers take a world point and derive both local anchors from the current poses,
// which is the easy way to satisfy it.

#include "joint2d.h"
#include "maths2d.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace avbd2d
{

constexpr float kStiffInf = std::numeric_limits<float>::infinity();

// Box2D-like shape and body definitions. Every default reproduces what the plain helpers do.
struct ShapeDef2D
{
    float density = 1.0f;
    float friction = 0.5f;
    float restitution = 0.0f;
    float tangentSpeed = 0.0f;
    float rollingResistance = 0.0f;
    uint64_t categoryBits = 1;
    uint64_t maskBits = ~uint64_t(0);
    int groupIndex = 0;
    int userMaterialId = 0;
};

struct BodyDef2D
{
    Vec2 position;
    float angle = 0.0f;
    Vec2 linearVelocity;
    float angularVelocity = 0.0f;
    Body2DType type = BODY2D_DYNAMIC;
    float gravityScale = 1.0f;
    float linearDamping = 0.0f;
    float angularDamping = 0.0f;
    bool lockLinearX = false;
    bool lockLinearY = false;
    bool fixedRotation = false;
    uint32_t color = 0;
};

struct Scene2D
{
    SolverParams2D params;

    // Bodies (host SoA, index = body id). A body's position is its centre of mass; `half` is
    // the half extents of a box about it that bounds every shape (what the joint arm and the
    // demo's culling read; the broadphase reads the shapes). localCenter is where the centre of
    // mass sits in the frame the body was authored in (zero for a one-shape body).
    std::vector<Vec2> pos, vel, half, localCenter;
    std::vector<float> ang, angVel, mass, moment;
    // Body type and per-body dynamics. A kinematic or static body has mass 0 here and on the
    // device; type tells them apart. Locks drop that degree of freedom from the solve.
    std::vector<uint8_t> bodyType, lockX, lockY, lockRot;
    std::vector<float> gravityScale, linearDamping, angularDamping;
    std::vector<uint32_t> color; // 0 = let the renderer pick
    std::vector<int> shapeFirst, shapeCount; // the body's shapes are [shapeFirst, shapeFirst + shapeCount)
    // Continuous collision (SolverParams2D::enableContinuous): a bullet is also swept against the
    // non-bullet dynamic bodies, not just the static geometry.
    std::vector<uint8_t> isBullet;

    // Shape table (host SoA, index = shape id; a body's shapes are contiguous and in body order,
    // so with one shape per body the shape id is the body id). A shape sits at shOff / shAng in
    // the body's frame (about its centre of mass). `shRad` is the rounding radius (a circle's
    // radius, a capsule's, a rounded polygon's; 0 for boxes, segments and chains). A shape's
    // vertices are verts[shVOff, shVOff + shVCnt), shape-local, with the matching outward edge
    // normals in norms (polygon: n CCW vertices about the centroid; capsule/segment: the two
    // core ends; chain: ghost1, p1, p2, ghost2; offset or compound circle: its centre). Boxes and
    // centred circles have none. `shHalf` is the local AABB half extents about the shape origin,
    // which is all the broadphase reads.
    std::vector<uint8_t> shType;
    std::vector<Vec2> shOff, shHalf;
    std::vector<float> shAng, shRad, shFric;
    std::vector<int> shVOff, shVCnt, shBody;
    // Per-shape collision filter (Box2D semantics: two shapes with the same nonzero group collide
    // when it is positive and never when it is negative; otherwise they collide when each one's
    // category is in the other's mask) and material. tangentSpeed is a conveyor belt's surface
    // speed along the surface's clockwise tangent (rightward on a top face); rolling resistance
    // is a dimensionless coefficient on the round shapes' rolling torque; restitution and the user
    // material id are stored and unused by the solver.
    std::vector<uint64_t> shCat, shMask;
    std::vector<int> shGroup, shUserId;
    std::vector<float> shTangentSpeed, shRolling, shRestitution;
    std::vector<Vec2> verts, norms;
    // Per-shape event flags (SHAPE_FLAG_*). Default: contact events and sensor events on.
    std::vector<uint8_t> shFlags;

    // Joints. A joint with jointA < 0 is anchored to the world: jointRA is then a world point.
    std::vector<int> jointA, jointB;
    std::vector<Vec2> jointRA, jointRB;
    // jointFrameA: angle of the frame-A x axis in body A (world when anchored); jointRest: reference
    // angle (relative angle at rest); jointAxis: three JointAxis2 per joint (x, y, angle).
    std::vector<float> jointFrameA, jointRest, jointArm, jointFrac;
    std::vector<JointAxis2> jointAxis;

    // Pairs that must not collide, as (lo << 32 | hi) body keys; finalize() sorts them.
    std::vector<uint64_t> ignore;

    // Camera hint for the demo.
    Vec2 cameraCenter{0.0f, 5.0f};
    float cameraHeight = 20.0f; // visible world height in metres

    int bodyCount() const { return (int)pos.size(); }
    int shapeTotal() const { return (int)shBody.size(); }
    int jointCount() const { return (int)jointA.size(); }

    // Event flags of a body's shapes (all of them). A sensor shape detects overlaps and never
    // collides; see docs/SOLVER_2D_ROADMAP.md (events).
    void setShapeFlag(int body, uint8_t flag, bool on)
    {
        const size_t b = (size_t)body;
        for (int s = shapeFirst[b]; s < shapeFirst[b] + shapeCount[b]; s++)
        {
            uint8_t &f = shFlags[(size_t)s];
            f = on ? (uint8_t)(f | flag) : (uint8_t)(f & ~flag);
        }
    }
    void setSensor(int body, bool on = true) { setShapeFlag(body, SHAPE_FLAG_SENSOR, on); }
    void setHitEvents(int body, bool on = true) { setShapeFlag(body, SHAPE_FLAG_HIT_EVENTS, on); }
    void setContactEvents(int body, bool on) { setShapeFlag(body, SHAPE_FLAG_CONTACT_EVENTS, on); }
    void setSensorEvents(int body, bool on) { setShapeFlag(body, SHAPE_FLAG_SENSOR_EVENTS, on); }

    void clear()
    {
        *this = Scene2D();
    }

    int addBox(Vec2 p, float angle, Vec2 size, float density, float mu = 0.5f, Vec2 v = Vec2(), float w = 0.0f,
               uint32_t rgba = 0)
    {
        const float m = density > 0.0f ? density * size.x * size.y : 0.0f;
        return addBody(p, angle, Vec2(size.x * 0.5f, size.y * 0.5f), SHAPE2D_BOX, m,
                       m * (size.x * size.x + size.y * size.y) / 12.0f, mu, v, w, rgba);
    }

    int addCircle(Vec2 p, float radius, float density, float mu = 0.5f, Vec2 v = Vec2(), float w = 0.0f,
                  uint32_t rgba = 0)
    {
        const float m = density > 0.0f ? density * kPi * radius * radius : 0.0f;
        const int id = addBody(p, 0.0f, Vec2(radius, radius), SHAPE2D_CIRCLE, m, 0.5f * m * radius * radius, mu, v, w,
                               rgba);
        shRad[(size_t)shapeFirst[(size_t)id]] = radius;
        return id;
    }

    // A circle whose centre sits at `center` in the frame of a body at (p, angle), so the body
    // origin is not its centre (a wheel hub, a ball on a stick). Inertia is about the origin.
    int addOffsetCircle(Vec2 p, float angle, Vec2 center, float r, float density, float mu = 0.5f, Vec2 v = Vec2(),
                        float w = 0.0f, uint32_t rgba = 0)
    {
        const float m = density > 0.0f ? density * kPi * r * r : 0.0f;
        const float reach = length(center) + r;
        const int id = addBody(p, angle, Vec2(reach, reach), SHAPE2D_CIRCLE, m, m * (0.5f * r * r + lengthSq(center)),
                               mu, v, w, rgba);
        const Vec2 pts[1] = {center};
        const Vec2 nrm[1] = {Vec2()};
        setShape(id, pts, nrm, 1, r);
        return id;
    }

    // A capsule between two world points (its core segment) with rounding radius r. The body
    // sits at the midpoint with its local x axis along p1 -> p2.
    int addCapsule(Vec2 p1, Vec2 p2, float r, float density, float mu = 0.5f, Vec2 v = Vec2(), float w = 0.0f,
                   uint32_t rgba = 0)
    {
        const Vec2 d = p2 - p1;
        const float L = length(d);
        const float h = 0.5f * L;
        float m = 0.0f, inertia = 0.0f;
        if (density > 0.0f)
        {
            // b2ComputeCapsuleMass.
            const float circleMass = density * kPi * r * r;
            const float boxMass = density * 2.0f * r * L;
            const float lc = 4.0f * r / (3.0f * kPi);
            m = circleMass + boxMass;
            inertia = circleMass * (0.5f * r * r + h * h + 2.0f * h * lc) + boxMass * (4.0f * r * r + L * L) / 12.0f;
        }
        const float angle = L > 0.0f ? std::atan2(d.y, d.x) : 0.0f;
        const int id = addBody((p1 + p2) * 0.5f, angle, Vec2(h + r, r), SHAPE2D_CAPSULE, m, inertia, mu, v, w, rgba);
        const Vec2 pts[2] = {Vec2(-h, 0.0f), Vec2(h, 0.0f)};
        const Vec2 nrm[2] = {Vec2(0.0f, -1.0f), Vec2(0.0f, 1.0f)}; // b2MakeCapsule: rightPerp(axis), its negation
        setShape(id, pts, nrm, 2, r);
        return id;
    }

    // A static two-sided line segment between two world points.
    int addSegment(Vec2 p1, Vec2 p2, float mu = 0.6f, uint32_t rgba = 0)
    {
        const Vec2 d = p2 - p1;
        const float h = 0.5f * length(d);
        const float angle = std::atan2(d.y, d.x);
        const int id =
            addBody((p1 + p2) * 0.5f, angle, Vec2(h, 0.0f), SHAPE2D_SEGMENT, 0.0f, 0.0f, mu, Vec2(), 0.0f, rgba);
        const Vec2 pts[2] = {Vec2(-h, 0.0f), Vec2(h, 0.0f)};
        const Vec2 nrm[2] = {Vec2(0.0f, -1.0f), Vec2(0.0f, 1.0f)};
        setShape(id, pts, nrm, 2, 0.0f);
        return id;
    }

    // A static chain through world points (b2CreateChain): one body per segment, each a one-sided
    // segment that collides only on the right of p_i -> p_i+1 (Box2D's rule: a CCW loop is solid
    // inside, so terrain with bodies on top runs right to left), with ghost vertices so a body
    // sliding over the joint between two segments does not catch on it. An open chain's end
    // ghosts continue its end segments straight on. Returns the first segment's body.
    int addChain(const Vec2 *points, int count, bool loop, float mu = 0.6f, uint32_t rgba = 0)
    {
        if (count < (loop ? 3 : 2))
            return -1;
        const int n = loop ? count : count - 1;
        auto P = [&](int i) {
            if (loop)
                return points[((i % count) + count) % count];
            if (i < 0)
                return points[0] * 2.0f - points[1];
            if (i >= count)
                return points[count - 1] * 2.0f - points[count - 2];
            return points[i];
        };
        int first = -1;
        for (int i = 0; i < n; i++)
        {
            const Vec2 g1 = P(i - 1), p1 = P(i), p2 = P(i + 1), g2 = P(i + 2);
            const Vec2 c = (p1 + p2) * 0.5f;
            const Vec2 h(std::fabs(p2.x - p1.x) * 0.5f, std::fabs(p2.y - p1.y) * 0.5f);
            const int id = addBody(c, 0.0f, h, SHAPE2D_CHAIN, 0.0f, 0.0f, mu, Vec2(), 0.0f, rgba);
            const Vec2 e = p2 - p1;
            const float len = length(e);
            const Vec2 nr = len > 0.0f ? Vec2(e.y / len, -e.x / len) : Vec2(0.0f, 1.0f); // right of p1 -> p2
            const Vec2 pts[4] = {g1 - c, p1 - c, p2 - c, g2 - c};
            const Vec2 nrm[4] = {nr, nr, nr, nr};
            setShape(id, pts, nrm, 4, 0.0f);
            if (first < 0)
                first = id;
        }
        return first;
    }

    // A convex polygon: the hull of `count` points (at most kMaxPolygonVertices) given in the
    // frame of a body at (p, angle), rounded by r. The body is placed at the hull's centroid,
    // so its position is p only when the points are centred. Returns -1 for a degenerate hull.
    int addPolygon(Vec2 p, float angle, const Vec2 *points, int count, float r, float density, float mu = 0.5f,
                   Vec2 v = Vec2(), float w = 0.0f, uint32_t rgba = 0)
    {
        PolyProps P;
        if (!polyProps(points, count, r, density, P))
            return -1;
        const int id = addBody(p + rotate(angle, P.centroid), angle, P.half, SHAPE2D_POLYGON, P.mass, P.inertia, mu, v,
                               w, rgba);
        setShape(id, P.v, P.nr, P.n, r);
        return id;
    }

    // A box as a polygon (b2MakeRoundedBox): `size` is the core's full extent, rounding adds r
    // on every side.
    int addRoundedBox(Vec2 p, float angle, Vec2 size, float r, float density, float mu = 0.5f, Vec2 v = Vec2(),
                      float w = 0.0f, uint32_t rgba = 0)
    {
        const float hx = 0.5f * size.x, hy = 0.5f * size.y;
        const Vec2 pts[4] = {Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy)};
        return addPolygon(p, angle, pts, 4, r, density, mu, v, w, rgba);
    }

    // Convex hull of 3..kMaxPolygonVertices points, CCW. Points closer than half a linear slop
    // are welded and vertices within two slops of the line through their neighbours dropped
    // (b2ComputeHull's tolerances; the algorithm is a monotone chain, not Box2D's quickhull).
    // Returns the vertex count, 0 for a degenerate input.
    static int computeHull(const Vec2 *points, int count, Vec2 *out)
    {
        if (count < 3 || count > kMaxPolygonVertices)
            return 0;
        Vec2 ps[kMaxPolygonVertices];
        int n = 0;
        const float tolSq = 0.25f * kLinearSlop * kLinearSlop;
        for (int i = 0; i < count; i++)
        {
            bool unique = true;
            for (int j = 0; j < n && unique; j++)
                unique = lengthSq(points[i] - ps[j]) >= tolSq;
            if (unique)
                ps[n++] = points[i];
        }
        if (n < 3)
            return 0;
        std::sort(ps, ps + n, [](Vec2 a, Vec2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
        Vec2 h[2 * kMaxPolygonVertices + 1];
        int k = 0;
        auto turn = [](Vec2 o, Vec2 a, Vec2 b) { return cross(a - o, b - o); };
        for (int i = 0; i < n; i++)
        {
            while (k >= 2 && turn(h[k - 2], h[k - 1], ps[i]) <= 0.0f)
                k--;
            h[k++] = ps[i];
        }
        for (int i = n - 2, lo = k + 1; i >= 0; i--)
        {
            while (k >= lo && turn(h[k - 2], h[k - 1], ps[i]) <= 0.0f)
                k--;
            h[k++] = ps[i];
        }
        k--; // the last point repeats the first
        bool dropped = true;
        while (dropped && k >= 3)
        {
            dropped = false;
            for (int i = 0; i < k && !dropped; i++)
            {
                const Vec2 a = h[(i + k - 1) % k], b = h[i], c = h[(i + 1) % k];
                const float len = length(c - a);
                if (len > 0.0f && std::fabs(cross(b - a, c - a)) / len < 2.0f * kLinearSlop)
                {
                    for (int j = i; j < k - 1; j++)
                        h[j] = h[j + 1];
                    k--;
                    dropped = true;
                }
            }
        }
        if (k < 3)
            return 0;
        for (int i = 0; i < k; i++)
            out[i] = h[i];
        return k;
    }

    // Revolute (pin) joint through a world point. a < 0 pins b to the world.
    int addRevoluteAt(int a, int b, Vec2 worldPoint, bool collide = false, float breakForce = kStiffInf)
    {
        return addJointAt(a, b, worldPoint, kStiffInf, 0.0f, collide, breakForce);
    }

    // Weld joint through a world point: pin plus a locked relative angle.
    // kAng may be finite (a stiff angular spring) or kStiffInf (locked).
    // breakForce: the joint is cut when its linear multiplier exceeds it (N).
    int addWeldAt(int a, int b, Vec2 worldPoint, bool collide = false, float breakForce = kStiffInf,
                  float kAng = kStiffInf)
    {
        return addJointAt(a, b, worldPoint, kStiffInf, kAng, collide, breakForce);
    }

    // Raw form: local anchors given directly (rA is a world point when a < 0). kLin infinite locks
    // x and y (finite: a ramped spring); kAng 0 leaves the angle free, infinite locks it, finite is
    // a ramped spring (in solver units, per arm^2). Kept for the C ABI's revolute/weld descs.
    int addJoint(int a, int b, Vec2 rA, Vec2 rB, float kLin, float kAng, bool collide = false,
                 float breakForce = kStiffInf)
    {
        JointAxis2 ax[3];
        for (int k = 0; k < 2; k++)
        {
            ax[k].pos = std::isinf(kLin) ? JOINT_AXIS_LOCK : JOINT_AXIS_SPRING;
            ax[k].stiff = kLin;
        }
        if (kAng != 0.0f)
        {
            ax[2].pos = std::isinf(kAng) ? JOINT_AXIS_LOCK : JOINT_AXIS_SPRING;
            ax[2].stiff = kAng;
        }
        const float angA = a >= 0 ? ang[(size_t)a] : 0.0f;
        return pushJoint(a, b, rA, rB, 0.0f, ang[(size_t)b] - angA, ax, collide, breakForce);
    }

    // ----- Box2D-style joint definitions -------------------------------------------------------
    enum class JointType2D
    {
        Revolute,
        Weld,
        Prismatic,
        Wheel,
        Motor,
        Distance,
        Filter, // no rows: only turns the pair's collision off
    };

    // Mirrors the b2*JointDef fields the solver uses. Anchors are local to each body (world points
    // when bodyA < 0); localAxisA is in body A's frame (world when bodyA < 0).
    struct JointDef2D
    {
        JointType2D type = JointType2D::Revolute;
        int bodyA = -1, bodyB = -1;
        Vec2 localAnchorA, localAnchorB;
        Vec2 localAxisA{1.0f, 0.0f};
        float referenceAngle = 0.0f; // relative angle at rest (angB - angA)
        bool enableSpring = false;
        float hertz = 0.0f;
        float dampingRatio = 0.0f; // accepted, ignored by the solver
        bool enableLimit = false;
        float lower = 0.0f, upper = 0.0f; // angle (revolute), translation (prismatic, wheel)
        bool enableMotor = false;
        float motorSpeed = 0.0f;
        float maxMotorForce = 0.0f, maxMotorTorque = 0.0f;
        // Weld: linear / angular stiffness as a frequency; 0 = rigid.
        float linearHertz = 0.0f, linearDampingRatio = 0.0f;
        float angularHertz = 0.0f, angularDampingRatio = 0.0f;
        // Motor joint: target offset of B's frame in A's frame.
        Vec2 linearOffset;
        float angularOffset = 0.0f;
        // Distance joint.
        float length = 1.0f, minLength = 0.0f, maxLength = 1.0e6f;
        bool collideConnected = false;
        float breakForce = kStiffInf;
    };

    // A def through one world point, with the local anchors, axis and reference angle derived from
    // the bodies' current poses. Fill in the type-specific fields, then pass it to addJoint.
    JointDef2D makeJointDef(JointType2D type, int a, int b, Vec2 worldAnchor, Vec2 worldAxis = Vec2(1.0f, 0.0f)) const
    {
        JointDef2D d;
        d.type = type;
        d.bodyA = a;
        d.bodyB = b;
        d.localAnchorA = a >= 0 ? rotate(-ang[(size_t)a], worldAnchor - pos[(size_t)a]) : worldAnchor;
        d.localAnchorB = rotate(-ang[(size_t)b], worldAnchor - pos[(size_t)b]);
        d.localAxisA = a >= 0 ? rotate(-ang[(size_t)a], worldAxis) : worldAxis;
        d.referenceAngle = ang[(size_t)b] - (a >= 0 ? ang[(size_t)a] : 0.0f);
        return d;
    }

    int addJoint(const JointDef2D &d)
    {
        JointAxis2 ax[3];
        float frame = std::atan2(d.localAxisA.y, d.localAxisA.x);
        auto lockAxis = [&](int k) { ax[k].pos = JOINT_AXIS_LOCK; };
        auto limitMotorSpring = [&](int k, bool spring, bool limit, bool motor, float maxF)
        {
            if (spring)
            {
                ax[k].pos = JOINT_AXIS_SPRING;
                ax[k].hertz = d.hertz;
                ax[k].stiff = kStiffInf;
                ax[k].damp = d.dampingRatio;
            }
            if (limit)
            {
                ax[k].limit = 1;
                ax[k].lo = d.lower;
                ax[k].hi = d.upper;
            }
            if (motor)
            {
                ax[k].motor = JOINT_MOTOR_VELOCITY;
                ax[k].mtarget = d.motorSpeed;
                ax[k].maxF = maxF;
            }
        };
        auto weldAxis = [&](int k, float hz, float zeta)
        {
            if (hz > 0.0f)
            {
                ax[k].pos = JOINT_AXIS_SPRING;
                ax[k].hertz = hz;
                ax[k].stiff = kStiffInf;
                ax[k].damp = zeta;
            }
            else
                ax[k].pos = JOINT_AXIS_LOCK;
        };
        switch (d.type)
        {
        case JointType2D::Revolute:
            lockAxis(0);
            lockAxis(1);
            limitMotorSpring(2, d.enableSpring, d.enableLimit, d.enableMotor, d.maxMotorTorque);
            break;
        case JointType2D::Prismatic:
            limitMotorSpring(0, d.enableSpring, d.enableLimit, d.enableMotor, d.maxMotorForce);
            lockAxis(1);
            lockAxis(2);
            break;
        case JointType2D::Wheel: // suspension spring and limit along the axis, motor spins the wheel
            limitMotorSpring(0, d.enableSpring, d.enableLimit, false, 0.0f);
            lockAxis(1);
            if (d.enableMotor)
            {
                ax[2].motor = JOINT_MOTOR_VELOCITY;
                ax[2].mtarget = d.motorSpeed;
                ax[2].maxF = d.maxMotorTorque;
            }
            break;
        case JointType2D::Weld:
            weldAxis(0, d.linearHertz, d.linearDampingRatio);
            weldAxis(1, d.linearHertz, d.linearDampingRatio);
            weldAxis(2, d.angularHertz, d.angularDampingRatio);
            break;
        case JointType2D::Motor:
            for (int k = 0; k < 3; k++)
                ax[k].motor = JOINT_MOTOR_TARGET;
            ax[0].mtarget = d.linearOffset.x;
            ax[1].mtarget = d.linearOffset.y;
            ax[2].mtarget = d.angularOffset;
            ax[0].maxF = ax[1].maxF = d.maxMotorForce;
            ax[2].maxF = d.maxMotorTorque;
            break;
        case JointType2D::Distance:
            ax[0].flags = JOINT_FLAG_DISTANCE;
            ax[0].rest = d.length;
            if (d.enableSpring)
            {
                ax[0].pos = JOINT_AXIS_SPRING;
                ax[0].hertz = d.hertz;
                ax[0].stiff = kStiffInf;
                ax[0].damp = d.dampingRatio;
            }
            else
                ax[0].pos = JOINT_AXIS_LOCK;
            if (d.enableLimit)
            {
                ax[0].limit = 1;
                ax[0].lo = d.minLength;
                ax[0].hi = d.maxLength;
            }
            if (d.enableMotor)
            {
                ax[0].motor = JOINT_MOTOR_VELOCITY;
                ax[0].mtarget = d.motorSpeed;
                ax[0].maxF = d.maxMotorForce;
            }
            frame = 0.0f;
            break;
        case JointType2D::Filter:
            if (d.bodyA >= 0 && d.bodyB >= 0)
                ignoreCollision(d.bodyA, d.bodyB);
            break;
        }
        // Box2D accepts dampingRatio on every spring; this solver ignores it (TODO: add a damping row).
        return pushJoint(d.bodyA, d.bodyB, d.localAnchorA, d.localAnchorB, frame, d.referenceAngle, ax,
                         d.collideConnected, d.breakForce);
    }

    // Convenience wrappers: one world point (and axis), the type-specific fields passed in.
    int addPrismaticAt(int a, int b, Vec2 worldPoint, Vec2 worldAxis, bool enableLimit = false, float lower = 0.0f,
                       float upper = 0.0f, bool enableMotor = false, float motorSpeed = 0.0f, float maxForce = 0.0f)
    {
        JointDef2D d = makeJointDef(JointType2D::Prismatic, a, b, worldPoint, worldAxis);
        d.enableLimit = enableLimit;
        d.lower = lower;
        d.upper = upper;
        d.enableMotor = enableMotor;
        d.motorSpeed = motorSpeed;
        d.maxMotorForce = maxForce;
        return addJoint(d);
    }
    int addWheelAt(int a, int b, Vec2 worldPoint, Vec2 worldAxis, float hertz = 4.0f, float dampingRatio = 0.7f,
                   bool enableMotor = false, float motorSpeed = 0.0f, float maxTorque = 0.0f)
    {
        JointDef2D d = makeJointDef(JointType2D::Wheel, a, b, worldPoint, worldAxis);
        d.enableSpring = hertz > 0.0f;
        d.hertz = hertz;
        d.dampingRatio = dampingRatio;
        d.enableMotor = enableMotor;
        d.motorSpeed = motorSpeed;
        d.maxMotorTorque = maxTorque;
        return addJoint(d);
    }
    int addMotorJoint(int a, int b, Vec2 linearOffset, float angularOffset, float maxForce, float maxTorque)
    {
        JointDef2D d = makeJointDef(JointType2D::Motor, a, b, pos[(size_t)b]);
        if (a >= 0)
            d.localAnchorA = Vec2();
        d.localAnchorB = Vec2();
        d.localAxisA = Vec2(1.0f, 0.0f); // Box2D: the offset is in body A's own frame
        d.referenceAngle = 0.0f;
        d.linearOffset = linearOffset;
        d.angularOffset = angularOffset;
        d.maxMotorForce = maxForce;
        d.maxMotorTorque = maxTorque;
        return addJoint(d);
    }
    // Distance joint between two world points (one per body). length < 0 keeps the current distance.
    int addDistanceJoint(int a, int b, Vec2 worldA, Vec2 worldB, float restLength = -1.0f, bool spring = false,
                         float hertz = 0.0f, float dampingRatio = 0.0f, float minLength = 0.0f,
                         float maxLength = 1.0e6f)
    {
        JointDef2D d = makeJointDef(JointType2D::Distance, a, b, worldB);
        d.localAnchorA = a >= 0 ? rotate(-ang[(size_t)a], worldA - pos[(size_t)a]) : worldA;
        d.length = restLength >= 0.0f ? restLength : length(worldB - worldA);
        d.enableSpring = spring;
        d.hertz = hertz;
        d.dampingRatio = dampingRatio;
        d.enableLimit = minLength > 0.0f || maxLength < 1.0e5f;
        d.minLength = minLength;
        d.maxLength = maxLength;
        return addJoint(d);
    }
    int addFilterJoint(int a, int b)
    {
        return addJoint(makeJointDef(JointType2D::Filter, a, b, pos[(size_t)b]));
    }

    // Applies a body definition's type, dynamics and locks to body `id` (the pose and velocity are
    // taken as they are). A static or kinematic body loses its mass; a static one its velocity.
    void setBodyDef(int id, const BodyDef2D &d)
    {
        const size_t b = (size_t)id;
        bodyType[b] = (uint8_t)d.type;
        gravityScale[b] = d.gravityScale;
        linearDamping[b] = d.linearDamping;
        angularDamping[b] = d.angularDamping;
        lockX[b] = d.lockLinearX ? 1 : 0;
        lockY[b] = d.lockLinearY ? 1 : 0;
        lockRot[b] = d.fixedRotation ? 1 : 0;
        if (d.type != BODY2D_DYNAMIC)
        {
            mass[b] = 0.0f;
            moment[b] = 0.0f;
        }
        if (d.type == BODY2D_STATIC)
        {
            vel[b] = Vec2();
            angVel[b] = 0.0f;
        }
    }

    // Turns body `id` into a kinematic body moving at (v, w); it keeps its shapes.
    void setKinematic(int id, Vec2 v, float w = 0.0f)
    {
        BodyDef2D d;
        d.type = BODY2D_KINEMATIC;
        setBodyDef(id, d);
        vel[(size_t)id] = v;
        angVel[(size_t)id] = w;
    }

    // Sets the filter and material of every shape of body `id` (density is ignored: mass is set).
    void setShapeDef(int id, const ShapeDef2D &d)
    {
        const size_t b = (size_t)id;
        for (int s = shapeFirst[b]; s < shapeFirst[b] + shapeCount[b]; s++)
            applyShapeDef((size_t)s, d);
    }

    // ----- Compound bodies: several shapes on one body -----------------------------------------
    // beginBody(p, angle) opens a body whose authoring frame sits at p, angle; the add*Shape calls
    // place shapes in that frame (a local position and angle each, its own density and friction);
    // endBody() computes the mass, the centre of mass and the moment about it from the sum of the
    // shapes (parallel-axis theorem), makes the centre of mass the body position, stores every
    // shape relative to it and returns the body id (-1 when no shape was added). localCenter[id]
    // is then the centre of mass in the authoring frame. A compound circle is a core point plus a
    // radius (its anchors are material, as an offset circle's are), never the legacy disc.
    void beginBody(Vec2 p, float angle)
    {
        pending.clear();
        pendingBody = BodyDef2D();
        pendingP = p;
        pendingAngle = angle;
    }

    void addBoxShape(Vec2 localPos, float localAngle, Vec2 size, float density, float mu = 0.5f)
    {
        Pending q = pendingShape(SHAPE2D_BOX, localPos, localAngle, Vec2(size.x * 0.5f, size.y * 0.5f), 0.0f, mu);
        q.mass = density > 0.0f ? density * size.x * size.y : 0.0f;
        q.inertia = q.mass * (size.x * size.x + size.y * size.y) / 12.0f;
        pending.push_back(q);
    }

    void addCircleShape(Vec2 localPos, float r, float density, float mu = 0.5f)
    {
        Pending q = pendingShape(SHAPE2D_CIRCLE, localPos, 0.0f, Vec2(r, r), r, mu);
        q.mass = density > 0.0f ? density * kPi * r * r : 0.0f;
        q.inertia = 0.5f * q.mass * r * r;
        q.n = 1; // the core point at the shape origin
        q.v[0] = Vec2();
        q.nr[0] = Vec2();
        pending.push_back(q);
    }

    // The convex hull of `points`, given in the frame of a shape at (localPos, localAngle),
    // rounded by r. The shape origin moves to the hull's centroid. False for a degenerate hull.
    bool addPolygonShape(Vec2 localPos, float localAngle, const Vec2 *points, int count, float r, float density,
                         float mu = 0.5f)
    {
        PolyProps P;
        if (!polyProps(points, count, r, density, P))
            return false;
        Pending q = pendingShape(SHAPE2D_POLYGON, localPos + rotate(localAngle, P.centroid), localAngle, P.half, r, mu);
        q.mass = P.mass;
        q.inertia = P.inertia;
        q.n = P.n;
        for (int i = 0; i < P.n; i++)
        {
            q.v[i] = P.v[i];
            q.nr[i] = P.nr[i];
        }
        pending.push_back(q);
        return true;
    }

    // A capsule between two points of the body frame, rounded by r.
    void addCapsuleShape(Vec2 p1, Vec2 p2, float r, float density, float mu = 0.5f)
    {
        const Vec2 d = p2 - p1;
        const float L = length(d);
        const float h = 0.5f * L;
        Pending q = pendingShape(SHAPE2D_CAPSULE, (p1 + p2) * 0.5f, L > 0.0f ? std::atan2(d.y, d.x) : 0.0f,
                                 Vec2(h + r, r), r, mu);
        if (density > 0.0f)
        {
            const float circleMass = density * kPi * r * r;
            const float boxMass = density * 2.0f * r * L;
            const float lc = 4.0f * r / (3.0f * kPi);
            q.mass = circleMass + boxMass;
            q.inertia = circleMass * (0.5f * r * r + h * h + 2.0f * h * lc) + boxMass * (4.0f * r * r + L * L) / 12.0f;
        }
        q.n = 2;
        q.v[0] = Vec2(-h, 0.0f);
        q.v[1] = Vec2(h, 0.0f);
        q.nr[0] = Vec2(0.0f, -1.0f);
        q.nr[1] = Vec2(0.0f, 1.0f);
        pending.push_back(q);
    }

    // A two-sided segment between two points of the body frame (massless).
    void addSegmentShape(Vec2 p1, Vec2 p2, const ShapeDef2D &d)
    {
        const Vec2 e = p2 - p1;
        const float h = 0.5f * length(e);
        Pending q = pendingShape(SHAPE2D_SEGMENT, (p1 + p2) * 0.5f, std::atan2(e.y, e.x), Vec2(h, 0.0f), 0.0f,
                                 d.friction);
        q.n = 2;
        q.v[0] = Vec2(-h, 0.0f);
        q.v[1] = Vec2(h, 0.0f);
        q.nr[0] = Vec2(0.0f, -1.0f);
        q.nr[1] = Vec2(0.0f, 1.0f);
        q.def = d;
        pending.push_back(q);
    }

    // A one-sided chain segment p1 -> p2 (solid on its right) with its two ghost vertices
    // (massless), points in the body frame.
    void addChainSegmentShape(Vec2 g1, Vec2 p1, Vec2 p2, Vec2 g2, const ShapeDef2D &d)
    {
        const Vec2 c = (p1 + p2) * 0.5f;
        const Vec2 e = p2 - p1;
        const float len = length(e);
        const Vec2 nr = len > 0.0f ? Vec2(e.y / len, -e.x / len) : Vec2(0.0f, 1.0f);
        Pending q = pendingShape(SHAPE2D_CHAIN, c, 0.0f, Vec2(std::fabs(e.x) * 0.5f, std::fabs(e.y) * 0.5f), 0.0f,
                                 d.friction);
        q.n = 4;
        q.v[0] = g1 - c;
        q.v[1] = p1 - c;
        q.v[2] = p2 - c;
        q.v[3] = g2 - c;
        for (int i = 0; i < 4; i++)
            q.nr[i] = nr;
        q.def = d;
        pending.push_back(q);
    }

    void beginBody(const BodyDef2D &d)
    {
        beginBody(d.position, d.angle);
        pendingBody = d;
    }

    void addBoxShape(Vec2 localPos, float localAngle, Vec2 size, const ShapeDef2D &d)
    {
        addBoxShape(localPos, localAngle, size, d.density, d.friction);
        pending.back().def = d;
    }

    void addCircleShape(Vec2 localPos, float r, const ShapeDef2D &d)
    {
        addCircleShape(localPos, r, d.density, d.friction);
        pending.back().def = d;
    }

    bool addPolygonShape(Vec2 localPos, float localAngle, const Vec2 *points, int count, float r, const ShapeDef2D &d)
    {
        if (!addPolygonShape(localPos, localAngle, points, count, r, d.density, d.friction))
            return false;
        pending.back().def = d;
        return true;
    }

    void addCapsuleShape(Vec2 p1, Vec2 p2, float r, const ShapeDef2D &d)
    {
        addCapsuleShape(p1, p2, r, d.density, d.friction);
        pending.back().def = d;
    }

    // Closes the body opened by beginBody(BodyDef2D) (or by the two-argument beginBody, with
    // default body properties).
    int endBody()
    {
        const BodyDef2D d = pendingBody;
        const int id = endBody(d.linearVelocity, d.angularVelocity, d.color);
        if (id >= 0)
        {
            // A dynamic body whose shapes have no density: Box2D has no default mass, so this solver
            // gives it mass 1 and the inertia of a disc of the body's reach (0.5 * reach^2).
            if (d.type == BODY2D_DYNAMIC && mass[(size_t)id] <= 0.0f)
            {
                mass[(size_t)id] = 1.0f;
                moment[(size_t)id] = 0.5f * half[(size_t)id].x * half[(size_t)id].x;
                vel[(size_t)id] = d.linearVelocity;
                angVel[(size_t)id] = d.angularVelocity;
            }
            const Vec2 v = vel[(size_t)id];
            const float w = angVel[(size_t)id];
            setBodyDef(id, d);
            if (d.type == BODY2D_KINEMATIC)
            {
                vel[(size_t)id] = d.linearVelocity;
                angVel[(size_t)id] = d.angularVelocity;
            }
            else if (d.type == BODY2D_DYNAMIC)
            {
                vel[(size_t)id] = v;
                angVel[(size_t)id] = w;
            }
        }
        pendingBody = BodyDef2D();
        return id;
    }

    int endBody(Vec2 v, float w = 0.0f, uint32_t rgba = 0)
    {
        if (pending.empty())
            return -1;
        float M = 0.0f;
        Vec2 c;
        for (const Pending &q : pending)
        {
            M += q.mass;
            c = c + q.off * q.mass;
        }
        c = M > 0.0f ? c * (1.0f / M) : Vec2();
        float I = 0.0f, reach = 0.0f;
        for (const Pending &q : pending)
        {
            I += q.inertia + q.mass * lengthSq(q.off - c);
            const float own = q.type == SHAPE2D_CIRCLE ? q.rad : length(q.half);
            reach = std::max(reach, length(q.off - c) + own);
        }
        const int id = bodyCount();
        pushBody(pendingP + rotate(pendingAngle, c), pendingAngle, Vec2(reach, reach), M, I, v, w, rgba);
        localCenter[(size_t)id] = c;
        shapeCount[(size_t)id] = (int)pending.size();
        for (const Pending &q : pending)
        {
            const int vo = (int)verts.size();
            verts.insert(verts.end(), q.v, q.v + q.n);
            norms.insert(norms.end(), q.nr, q.nr + q.n);
            pushShape(id, q.type, q.off - c, q.angle, q.half, q.rad, q.mu, vo, q.n);
            applyShapeDef((size_t)shapeTotal() - 1, q.def);
        }
        pending.clear();
        return id;
    }

    // A body with no shape (a pivot, a joint anchor, an empty parent). A dynamic one gets the default
    // mass: 1, with the inertia of a disc of reach kShapelessReach.
    static constexpr float kShapelessReach = 0.5f;
    int addShapelessBody(const BodyDef2D &d)
    {
        const bool dyn = d.type == BODY2D_DYNAMIC;
        const int id = bodyCount();
        pushBody(d.position, d.angle, Vec2(kShapelessReach, kShapelessReach), dyn ? 1.0f : 0.0f,
                 dyn ? 0.5f * kShapelessReach * kShapelessReach : 0.0f, d.linearVelocity, d.angularVelocity, d.color);
        setBodyDef(id, d);
        if (d.type == BODY2D_KINEMATIC)
        {
            vel[(size_t)id] = d.linearVelocity;
            angVel[(size_t)id] = d.angularVelocity;
        }
        return id;
    }

    // Joint record j takes the value of record `from` (live slot reuse: the last joint moves into a
    // freed slot), and popJoint() drops the last record.
    void moveJoint(int from, int to)
    {
        const size_t f = (size_t)from, t = (size_t)to;
        jointA[t] = jointA[f];
        jointB[t] = jointB[f];
        jointRA[t] = jointRA[f];
        jointRB[t] = jointRB[f];
        jointFrameA[t] = jointFrameA[f];
        jointRest[t] = jointRest[f];
        jointArm[t] = jointArm[f];
        jointFrac[t] = jointFrac[f];
        for (size_t k = 0; k < 3; k++)
            jointAxis[3 * t + k] = jointAxis[3 * f + k];
    }
    void popJoint()
    {
        jointA.pop_back();
        jointB.pop_back();
        jointRA.pop_back();
        jointRB.pop_back();
        jointFrameA.pop_back();
        jointRest.pop_back();
        jointArm.pop_back();
        jointFrac.pop_back();
        jointAxis.resize(jointAxis.size() - 3);
    }
    // Marks shapes [first, first + n) dead (no owner); the slots wait for reuse.
    void releaseShapes(int first, int n)
    {
        for (int s = first; s < first + n; s++)
        {
            shBody[(size_t)s] = -1;
            shVCnt[(size_t)s] = 0;
        }
    }
    // Drops every no-collide pair that involves body b. True when any was removed.
    bool eraseIgnore(int b)
    {
        const size_t n = ignore.size();
        ignore.erase(std::remove_if(ignore.begin(), ignore.end(),
                                    [b](uint64_t k) { return (int)(k >> 32) == b || (int)(k & 0xffffffffu) == b; }),
                     ignore.end());
        return ignore.size() != n;
    }

    // Grows every shape array to n entries (new ones are dead: shBody -1).
    void resizeShapes(int n)
    {
        const size_t c = (size_t)n;
        shType.resize(c, (uint8_t)SHAPE2D_BOX);
        shOff.resize(c);
        shHalf.resize(c);
        shAng.resize(c, 0.0f);
        shRad.resize(c, 0.0f);
        shFric.resize(c, 0.5f);
        shVOff.resize(c, 0);
        shVCnt.resize(c, 0);
        shBody.resize(c, -1);
        shCat.resize(c, 1);
        shMask.resize(c, ~uint64_t(0));
        shGroup.resize(c, 0);
        shUserId.resize(c, 0);
        shTangentSpeed.resize(c, 0.0f);
        shRolling.resize(c, 0.0f);
        shRestitution.resize(c, 0.0f);
        shFlags.resize(c, (uint8_t)SHAPE_FLAG_DEFAULT);
    }
    int vertTotal(int k) const
    {
        int t = 0;
        for (int s = shapeFirst[(size_t)k]; s < shapeFirst[(size_t)k] + shapeCount[(size_t)k]; s++)
            t += shVCnt[(size_t)s];
        return t;
    }

    // Copies body k of `src` into body slot `slot` (-1 appends a new body), its shapes into the shape
    // slots [shapeOff, shapeOff + shapeCount) and their vertices into the pool from vertOff (both
    // grown when they run past the end). Returns the body index.
    int installBody(const Scene2D &src, int k, int slot, int shapeOff, int vertOff)
    {
        const size_t q = (size_t)k;
        int id = slot;
        if (id < 0)
        {
            id = bodyCount();
            pushBody(src.pos[q], src.ang[q], src.half[q], src.mass[q], src.moment[q], src.vel[q], src.angVel[q],
                     src.color[q]);
        }
        const size_t d = (size_t)id;
        pos[d] = src.pos[q];
        ang[d] = src.ang[q];
        half[d] = src.half[q];
        mass[d] = src.mass[q];
        moment[d] = src.moment[q];
        vel[d] = src.vel[q];
        angVel[d] = src.angVel[q];
        color[d] = src.color[q];
        localCenter[d] = src.localCenter[q];
        bodyType[d] = src.bodyType[q];
        gravityScale[d] = src.gravityScale[q];
        linearDamping[d] = src.linearDamping[q];
        angularDamping[d] = src.angularDamping[q];
        lockX[d] = src.lockX[q];
        lockY[d] = src.lockY[q];
        lockRot[d] = src.lockRot[q];
        isBullet[d] = src.isBullet[q];
        const int n = src.shapeCount[q];
        shapeFirst[d] = shapeOff;
        shapeCount[d] = n;
        if (shapeOff + n > shapeTotal())
            resizeShapes(shapeOff + n);
        const int vt = src.vertTotal(k);
        if (vertOff + vt > (int)verts.size())
        {
            verts.resize((size_t)(vertOff + vt));
            norms.resize((size_t)(vertOff + vt));
        }
        int vo = vertOff;
        for (int i = 0; i < n; i++)
        {
            const size_t s = (size_t)(shapeOff + i), p = (size_t)(src.shapeFirst[q] + i);
            const int cnt = src.shVCnt[p];
            shType[s] = src.shType[p];
            shOff[s] = src.shOff[p];
            shHalf[s] = src.shHalf[p];
            shAng[s] = src.shAng[p];
            shRad[s] = src.shRad[p];
            shFric[s] = src.shFric[p];
            shVOff[s] = cnt > 0 ? vo : 0;
            shVCnt[s] = cnt;
            shBody[s] = id;
            shCat[s] = src.shCat[p];
            shMask[s] = src.shMask[p];
            shGroup[s] = src.shGroup[p];
            shUserId[s] = src.shUserId[p];
            shTangentSpeed[s] = src.shTangentSpeed[p];
            shRolling[s] = src.shRolling[p];
            shRestitution[s] = src.shRestitution[p];
            shFlags[s] = src.shFlags[p];
            for (int c = 0; c < cnt; c++)
            {
                verts[(size_t)(vo + c)] = src.verts[(size_t)(src.shVOff[p] + c)];
                norms[(size_t)(vo + c)] = src.norms[(size_t)(src.shVOff[p] + c)];
            }
            vo += cnt;
        }
        return id;
    }

    // Copies body i of `src` (with its shapes) to the end of this scene, moved by `offset`.
    int appendBody(const Scene2D &src, int i, Vec2 offset)
    {
        const size_t k = (size_t)i;
        const int id = bodyCount();
        pushBody(src.pos[k] + offset, src.ang[k], src.half[k], src.mass[k], src.moment[k], src.vel[k], src.angVel[k],
                 src.color[k]);
        localCenter[(size_t)id] = src.localCenter[k];
        shapeCount[(size_t)id] = src.shapeCount[k];
        bodyType[(size_t)id] = src.bodyType[k];
        gravityScale[(size_t)id] = src.gravityScale[k];
        linearDamping[(size_t)id] = src.linearDamping[k];
        angularDamping[(size_t)id] = src.angularDamping[k];
        lockX[(size_t)id] = src.lockX[k];
        lockY[(size_t)id] = src.lockY[k];
        lockRot[(size_t)id] = src.lockRot[k];
        vel[(size_t)id] = src.vel[k];
        angVel[(size_t)id] = src.angVel[k];
        isBullet[(size_t)id] = src.isBullet[k];
        for (int s = src.shapeFirst[k]; s < src.shapeFirst[k] + src.shapeCount[k]; s++)
        {
            const size_t q = (size_t)s;
            const int o = src.shVOff[q], n = src.shVCnt[q];
            const int vo = (int)verts.size();
            verts.insert(verts.end(), src.verts.begin() + o, src.verts.begin() + o + n);
            norms.insert(norms.end(), src.norms.begin() + o, src.norms.begin() + o + n);
            pushShape(id, src.shType[q], src.shOff[q], src.shAng[q], src.shHalf[q], src.shRad[q], src.shFric[q], vo, n);
            const size_t d = (size_t)shapeTotal() - 1;
            shCat[d] = src.shCat[q];
            shMask[d] = src.shMask[q];
            shGroup[d] = src.shGroup[q];
            shUserId[d] = src.shUserId[q];
            shTangentSpeed[d] = src.shTangentSpeed[q];
            shRolling[d] = src.shRolling[q];
            shRestitution[d] = src.shRestitution[q];
            shFlags.back() = src.shFlags[q];
        }
        return id;
    }

    // True when the contact anchor of shape s is a core point with the surface `radius` beyond it
    // along the normal (collide2d.slang ANCHOR_CORE): capsules, rounded polygons, offset circles.
    bool coreAnchored(int s) const
    {
        const size_t k = (size_t)s;
        return !(shType[k] == SHAPE2D_CIRCLE && shVCnt[k] == 0) && shRad[k] > 0.0f;
    }

    // Whether the body-local point p (relative to the body position) lies inside any shape of
    // body i (the mouse pick). Segments and chains have no inside.
    bool contains(int i, Vec2 p) const
    {
        const size_t b = (size_t)i;
        for (int s = shapeFirst[b]; s < shapeFirst[b] + shapeCount[b]; s++)
            if (shapeContains(s, rotate(-shAng[(size_t)s], p - shOff[(size_t)s])))
                return true;
        return false;
    }

    // Whether the shape-local point p lies inside shape s.
    bool shapeContains(int s, Vec2 p) const
    {
        const size_t k = (size_t)s;
        const Vec2 h = shHalf[k];
        const float r = shRad[k];
        const Vec2 *v = verts.data() + shVOff[k];
        const int n = shVCnt[k];
        switch (shType[k])
        {
        case SHAPE2D_BOX:
            return std::fabs(p.x) <= h.x && std::fabs(p.y) <= h.y;
        case SHAPE2D_CIRCLE:
            return n == 0 ? lengthSq(p) <= h.x * h.x : lengthSq(p - v[0]) <= r * r;
        case SHAPE2D_CAPSULE:
        case SHAPE2D_POLYGON:
        {
            if (n < 2)
                return false;
            const Vec2 *nr = norms.data() + shVOff[k];
            bool inside = n > 2;
            for (int j = 0; j < n && inside; j++)
                inside = dot(nr[j], p - v[j]) <= 0.0f;
            if (inside)
                return true;
            for (int j = 0; j < n; j++)
            {
                const Vec2 a = v[j], e = v[(j + 1) % n] - a;
                const float ee = lengthSq(e);
                const float t = ee > 0.0f ? std::min(std::max(dot(p - a, e) / ee, 0.0f), 1.0f) : 0.0f;
                if (lengthSq(p - (a + e * t)) <= r * r)
                    return true;
            }
            return false;
        }
        default:
            return false;
        }
    }

    void ignoreCollision(int a, int b)
    {
        if (a == b || a < 0 || b < 0)
            return;
        const uint32_t lo = (uint32_t)std::min(a, b), hi = (uint32_t)std::max(a, b);
        ignore.push_back(((uint64_t)lo << 32) | hi);
    }

    void setBullet(int id, bool bullet = true) { isBullet[(size_t)id] = bullet ? 1 : 0; }

    // Sorts and dedupes the ignore list. VkWorld2D::build() calls it.
    void finalize()
    {
        std::sort(ignore.begin(), ignore.end());
        ignore.erase(std::unique(ignore.begin(), ignore.end()), ignore.end());
    }

private:
    // A shape being collected between beginBody and endBody, in the authoring frame.
    struct Pending
    {
        uint8_t type = SHAPE2D_BOX;
        Vec2 off, half;
        float angle = 0.0f, rad = 0.0f, mu = 0.5f, mass = 0.0f, inertia = 0.0f; // inertia about the shape origin
        int n = 0;
        Vec2 v[kMaxPolygonVertices], nr[kMaxPolygonVertices];
        ShapeDef2D def;
    };
    std::vector<Pending> pending;
    BodyDef2D pendingBody;
    Vec2 pendingP;
    float pendingAngle = 0.0f;

    static Pending pendingShape(uint8_t type, Vec2 off, float angle, Vec2 half, float rad, float mu)
    {
        Pending q;
        q.type = type;
        q.off = off;
        q.angle = angle;
        q.half = half;
        q.rad = rad;
        q.mu = mu;
        q.def.friction = mu;
        return q;
    }

    // A convex polygon's hull, outward normals and mass properties (b2ComputePolygonMass); the
    // hull is recentred on the centroid, and `half` bounds it about there.
    struct PolyProps
    {
        Vec2 v[kMaxPolygonVertices], nr[kMaxPolygonVertices];
        int n = 0;
        Vec2 centroid, half;
        float mass = 0.0f, inertia = 0.0f;
    };

    static bool polyProps(const Vec2 *points, int count, float r, float density, PolyProps &P)
    {
        Vec2 hull[kMaxPolygonVertices];
        const int n = computeHull(points, count, hull);
        if (n < 3)
            return false;
        Vec2 nrm[kMaxPolygonVertices];
        for (int i = 0; i < n; i++)
        {
            const Vec2 e = hull[(i + 1) % n] - hull[i];
            const float len = length(e);
            nrm[i] = Vec2(e.y / len, -e.x / len);
        }
        // b2ComputePolygonMass: a rounded polygon's vertices are pushed out along the bisectors.
        Vec2 mv[kMaxPolygonVertices];
        for (int i = 0; i < n; i++)
        {
            mv[i] = hull[i];
            if (r > 0.0f)
            {
                const Vec2 mid = nrm[(i + n - 1) % n] + nrm[i];
                mv[i] = hull[i] + mid * (1.412f * r / length(mid));
            }
        }
        Vec2 center;
        float area = 0.0f, inertia = 0.0f;
        const Vec2 o = mv[0];
        for (int i = 1; i < n - 1; i++)
        {
            const Vec2 e1 = mv[i] - o, e2 = mv[i + 1] - o;
            const float D = cross(e1, e2);
            const float a = 0.5f * D;
            area += a;
            center = center + (e1 + e2) * (a / 3.0f);
            const float intx2 = e1.x * e1.x + e2.x * e1.x + e2.x * e2.x;
            const float inty2 = e1.y * e1.y + e2.y * e1.y + e2.y * e2.y;
            inertia += (0.25f / 3.0f * D) * (intx2 + inty2);
        }
        center = center * (1.0f / area);
        float m = 0.0f, I = 0.0f;
        if (density > 0.0f)
        {
            m = density * area;
            I = density * inertia - m * dot(center, center);
        }
        const Vec2 c = o + center; // the centroid, in the authoring frame
        Vec2 h;
        for (int i = 0; i < n; i++)
        {
            hull[i] = hull[i] - c;
            h = Vec2(std::max(h.x, std::fabs(hull[i].x) + r), std::max(h.y, std::fabs(hull[i].y) + r));
        }
        P.n = n;
        for (int i = 0; i < n; i++)
        {
            P.v[i] = hull[i];
            P.nr[i] = nrm[i];
        }
        P.centroid = c;
        P.half = h;
        P.mass = m;
        P.inertia = I;
        return true;
    }

    // One body record, with no shape yet.
    void pushBody(Vec2 p, float angle, Vec2 h, float m, float inertia, Vec2 v, float w, uint32_t rgba)
    {
        pos.push_back(p);
        ang.push_back(angle);
        half.push_back(h);
        mass.push_back(m);
        moment.push_back(inertia);
        vel.push_back(m > 0.0f ? v : Vec2());
        angVel.push_back(m > 0.0f ? w : 0.0f);
        color.push_back(rgba);
        localCenter.push_back(Vec2());
        shapeFirst.push_back(shapeTotal());
        shapeCount.push_back(0);
        bodyType.push_back(m > 0.0f ? BODY2D_DYNAMIC : BODY2D_STATIC);
        lockX.push_back(0);
        lockY.push_back(0);
        lockRot.push_back(0);
        gravityScale.push_back(1.0f);
        linearDamping.push_back(0.0f);
        angularDamping.push_back(0.0f);
        isBullet.push_back(0);
    }

    void applyShapeDef(size_t s, const ShapeDef2D &d)
    {
        shFric[s] = d.friction;
        shCat[s] = d.categoryBits;
        shMask[s] = d.maskBits;
        shGroup[s] = d.groupIndex;
        shUserId[s] = d.userMaterialId;
        shTangentSpeed[s] = d.tangentSpeed;
        shRolling[s] = d.rollingResistance;
        shRestitution[s] = d.restitution;
    }

    void pushShape(int body, uint8_t type, Vec2 off, float angle, Vec2 h, float rad, float mu, int vOff, int vCnt)
    {
        shType.push_back(type);
        shOff.push_back(off);
        shAng.push_back(angle);
        shHalf.push_back(h);
        shRad.push_back(rad);
        shFric.push_back(mu);
        shVOff.push_back(vOff);
        shVCnt.push_back(vCnt);
        shBody.push_back(body);
        shCat.push_back(1);
        shMask.push_back(~uint64_t(0));
        shGroup.push_back(0);
        shUserId.push_back(0);
        shTangentSpeed.push_back(0.0f);
        shRolling.push_back(0.0f);
        shRestitution.push_back(0.0f);
        shFlags.push_back(SHAPE_FLAG_DEFAULT);
    }

    // A one-shape body at its authoring origin: the shape sits at the centre with no rotation.
    int addBody(Vec2 p, float angle, Vec2 h, uint8_t sh, float m, float inertia, float mu, Vec2 v, float w,
                uint32_t rgba)
    {
        const int id = bodyCount();
        pushBody(p, angle, h, m, inertia, v, w, rgba);
        shapeCount[(size_t)id] = 1;
        pushShape(id, sh, Vec2(), 0.0f, h, 0.0f, mu, (int)verts.size(), 0);
        return id;
    }

    // Gives the one shape of body `id` its vertices and rounding radius.
    void setShape(int id, const Vec2 *pts, const Vec2 *nrm, int n, float r)
    {
        const size_t s = (size_t)shapeFirst[(size_t)id];
        shRad[s] = r;
        shVOff[s] = (int)verts.size();
        shVCnt[s] = n;
        verts.insert(verts.end(), pts, pts + n);
        norms.insert(norms.end(), nrm, nrm + n);
    }

    int addJointAt(int a, int b, Vec2 worldPoint, float kLin, float kAng, bool collide, float breakForce)
    {
        const Vec2 rA = a >= 0 ? rotate(-ang[a], worldPoint - pos[a]) : worldPoint;
        const Vec2 rB = rotate(-ang[b], worldPoint - pos[b]);
        return addJoint(a, b, rA, rB, kLin, kAng, collide, breakForce);
    }

    int pushJoint(int a, int b, Vec2 rA, Vec2 rB, float frameA, float refAngle, const JointAxis2 *ax, bool collide,
                  float breakForce)
    {
        const int id = jointCount();
        jointA.push_back(a);
        jointB.push_back(b);
        jointRA.push_back(rA);
        jointRB.push_back(rB);
        jointFrameA.push_back(frameA);
        jointRest.push_back(refAngle);
        jointFrac.push_back(breakForce);
        for (int k = 0; k < 3; k++)
            jointAxis.push_back(ax[k]);
        // Squared length of the summed full widths.
        const Vec2 sizeA = a >= 0 ? half[(size_t)a] * 2.0f : Vec2();
        const Vec2 sizeB = half[(size_t)b] * 2.0f;
        jointArm.push_back(lengthSq(sizeA + sizeB));
        if (!collide && a >= 0)
            ignoreCollision(a, b);
        return id;
    }
};

} // namespace avbd2d
