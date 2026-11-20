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
    std::vector<uint32_t> color;             // 0 = let the renderer pick
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

    void clear() { *this = Scene2D(); }

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
        const int id =
            addBody(p, 0.0f, Vec2(radius, radius), SHAPE2D_CIRCLE, m, 0.5f * m * radius * radius, mu, v, w, rgba);
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
                   uint32_t rgba = 0);

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
    int addChain(const Vec2 *points, int count, bool loop, float mu = 0.6f, uint32_t rgba = 0);

    // A convex polygon: the hull of `count` points (at most kMaxPolygonVertices) given in the
    // frame of a body at (p, angle), rounded by r. The body is placed at the hull's centroid,
    // so its position is p only when the points are centred. Returns -1 for a degenerate hull.
    int addPolygon(Vec2 p, float angle, const Vec2 *points, int count, float r, float density, float mu = 0.5f,
                   Vec2 v = Vec2(), float w = 0.0f, uint32_t rgba = 0)
    {
        PolyProps P;
        if (!polyProps(points, count, r, density, P))
            return -1;
        const int id =
            addBody(p + rotate(angle, P.centroid), angle, P.half, SHAPE2D_POLYGON, P.mass, P.inertia, mu, v, w, rgba);
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
    static int computeHull(const Vec2 *points, int count, Vec2 *out);

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
        bool enableLimit = false;
        float lower = 0.0f, upper = 0.0f; // angle (revolute), translation (prismatic, wheel)
        bool enableMotor = false;
        float motorSpeed = 0.0f;
        // What the motor drives: 0/velocity = at motorSpeed, target = hold motorTarget (a Box2D
        // motor joint). 0 with enableMotor set means velocity, the historical meaning.
        uint32_t motorMode = 0;
        float motorTarget = 0.0f;
        float maxMotorForce = 0.0f, maxMotorTorque = 0.0f;
        // Weld: linear / angular stiffness as a frequency; 0 = rigid.
        float linearHertz = 0.0f;
        float angularHertz = 0.0f;
        // Motor joint: target offset of B's frame in A's frame.
        Vec2 linearOffset;
        float angularOffset = 0.0f;
        // Distance joint.
        float length = 1.0f, minLength = 0.0f, maxLength = 1.0e6f;
        bool collideConnected = false;
        float breakForce = kStiffInf;
        // Multiplier of the hard-row penalty cap (0/1 = stock, up to 16). Baked into the axes at creation
        // so a joint created with a stiff human lock carries it without a runtime patch.
        float lockScale = 0.0f;
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

    int addJoint(const JointDef2D &d);

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
                         float hertz = 0.0f, float minLength = 0.0f, float maxLength = 1.0e6f)
    {
        JointDef2D d = makeJointDef(JointType2D::Distance, a, b, worldB);
        d.localAnchorA = a >= 0 ? rotate(-ang[(size_t)a], worldA - pos[(size_t)a]) : worldA;
        d.length = restLength >= 0.0f ? restLength : length(worldB - worldA);
        d.enableSpring = spring;
        d.hertz = hertz;
        d.enableLimit = minLength > 0.0f || maxLength < 1.0e5f;
        d.minLength = minLength;
        d.maxLength = maxLength;
        return addJoint(d);
    }

    // Applies a body definition's type, dynamics and locks to body `id` (the pose and velocity are
    // taken as they are). A static or kinematic body loses its mass; a static one its velocity.
    void setBodyDef(int id, const BodyDef2D &d);

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
    void addCapsuleShape(Vec2 p1, Vec2 p2, float r, float density, float mu = 0.5f);

    // A two-sided segment between two points of the body frame (massless).
    void addSegmentShape(Vec2 p1, Vec2 p2, const ShapeDef2D &d)
    {
        const Vec2 e = p2 - p1;
        const float h = 0.5f * length(e);
        Pending q =
            pendingShape(SHAPE2D_SEGMENT, (p1 + p2) * 0.5f, std::atan2(e.y, e.x), Vec2(h, 0.0f), 0.0f, d.friction);
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
        Pending q =
            pendingShape(SHAPE2D_CHAIN, c, 0.0f, Vec2(std::fabs(e.x) * 0.5f, std::fabs(e.y) * 0.5f), 0.0f, d.friction);
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
    int endBody();

    int endBody(Vec2 v, float w = 0.0f, uint32_t rgba = 0);

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
    void resizeShapes(int n);
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
    int installBody(const Scene2D &src, int k, int slot, int shapeOff, int vertOff);

    // Copies body i of `src` (with its shapes) to the end of this scene, moved by `offset`.
    int appendBody(const Scene2D &src, int i, Vec2 offset);

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
    bool shapeContains(int s, Vec2 p) const;

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

    static bool polyProps(const Vec2 *points, int count, float r, float density, PolyProps &P);

    // One body record, with no shape yet.
    void pushBody(Vec2 p, float angle, Vec2 h, float m, float inertia, Vec2 v, float w, uint32_t rgba);

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
