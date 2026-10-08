// The C ABI of the 2D solver (avbd2d.h, ABI 3) on VkWorld2D + Scene2D.
//
// Shape of the layer: the host keeps slot tables (with free lists and generations) of bodies,
// shapes and joints. A created object is a record in its table and is "pending" until the next
// step_begin, which replays every pending body (with its shapes) and joint into the live world's
// authored Scene2D in one go and commits them (VkWorld2D::commitAuthored). Setters on a live
// object go to VkWorld2D's mutation queue; on a pending object they edit its record. Getters read
// VkWorld2D's host mirrors (the last completed step plus queued pose/velocity sets).
//
// Every entry point validates its handle or id first, so a null, never-created or destroyed world
// and a stale id are error codes and not crashes.

#include "avbd2d.h"

#include "query2d.h"
#include "scene2d.h"
#include "vk_util.h"
#include "vk_world2d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace avbd2d;

namespace
{
using JointDef = Scene2D::JointDef2D;
using JointType = Scene2D::JointType2D;

struct BodyRec
{
    bool alive = false;
    uint16_t gen = 1;
    Avbd2dBodyDef def{};
    bool flushed = false;
    bool reshape = false; // flushed, and its shape set changed since: replaced at the next step
    int solver = -1;      // solver body index once flushed
    std::vector<int> shapes; // shape slots, in the order they were added
    std::vector<int> joints; // joint slots
    // Mass properties, lazily computed (pending bodies replay into a scratch scene).
    bool propsValid = false;
    float mass = 0.0f, inertia = 0.0f;
    Vec2 lc;
};

enum ShapeKind
{
    SK_POLYGON,
    SK_CIRCLE,
    SK_CAPSULE,
    SK_SEGMENT,
    SK_CHAIN
};

struct ShapeRec
{
    bool alive = false;
    uint16_t gen = 1;
    int body = -1; // body slot
    int kind = SK_POLYGON;
    Avbd2dPolygon poly{};
    Avbd2dCircle circle{};
    Avbd2dCapsule capsule{};
    Avbd2dSegment seg{};
    Vec2 chain[4]; // ghost1, p1, p2, ghost2
    Avbd2dShapeDef def{};
    int solver = -1; // solver shape index once flushed
};

struct JointRec
{
    bool alive = false;
    uint16_t gen = 1;
    JointDef d; // anchors relative to the body origins here; converted at flush
    int bodyA = -1, bodyB = -1; // body slots
    void *userData = nullptr;
    bool flushed = false;
    int solver = -1; // authored joint index; -1 = not live (or skipped at flush)
};

template <class T> struct Table
{
    std::vector<T> items;
    std::vector<int> freeList;
    int acquire()
    {
        int i;
        if (!freeList.empty())
        {
            i = freeList.back();
            freeList.pop_back();
        }
        else
        {
            items.emplace_back();
            i = (int)items.size() - 1;
        }
        items[(size_t)i].alive = true;
        return i;
    }
    T *find(int32_t index1, uint16_t gen)
    {
        if (index1 < 1 || index1 > (int32_t)items.size())
            return nullptr;
        T &t = items[(size_t)index1 - 1];
        return (t.alive && t.gen == gen) ? &t : nullptr;
    }
    void release(int i)
    {
        uint16_t g = (uint16_t)(items[(size_t)i].gen + 1);
        if (g == 0)
            g = 1;
        items[(size_t)i] = T();
        items[(size_t)i].gen = g;
        freeList.push_back(i);
    }
    int live() const { return (int)items.size() - (int)freeList.size(); }
};
} // namespace

struct Avbd2dWorld
{
    std::unique_ptr<VkWorld2D> world;
    uint16_t id = 0;
    bool lost = false; // a device error or a failed step: every call but destroy returns DEVICE_LOST
    Avbd2dWorldDef def{};
    StepResult2D last;
    bool inFlight = false;
    Avbd2dResult deferred = AVBD2D_OK; // result of a step settled by a query, for the next step_end

    Table<BodyRec> bodies;
    Table<ShapeRec> shapes;
    Table<JointRec> joints;
    std::vector<int> pendingBodies, pendingJoints;
    std::vector<int> postBodies;                         // disable / sleep to queue once their flush is built
    std::vector<int> replaceBodies;                     // body slots whose shape set changed
    std::vector<int> removeBodies, removeJoints;         // solver indices destroyed since the last step
    std::vector<int> solverBody, solverShape, solverJoint; // solver index -> slot, or -1
    // Solver shape ids that no longer map to a live shape (destroyed, or replaced by a shape edit) keep
    // the id they had until the events of the next step are delivered, so END events still name them.
    struct Tomb
    {
        Avbd2dShapeId id;
        uint64_t epoch;
    };
    std::unordered_map<int, Tomb> tomb;
    uint64_t stepEpoch = 0; // steps begun

    std::vector<Avbd2dBodyMoveEvent> bodyEvents;
    std::vector<Avbd2dContactBeginTouchEvent> beginEvents;
    std::vector<Avbd2dContactEndTouchEvent> endEvents;
    std::vector<Avbd2dContactHitEvent> hitEvents;
    std::vector<Avbd2dSensorBeginTouchEvent> sensorBegin;
    std::vector<Avbd2dSensorEndTouchEvent> sensorEnd;
    std::vector<Avbd2dJointEvent> jointEvents;
};

namespace
{
std::mutex g_mutex;
uint16_t g_nextWorldId = 0;

std::unordered_set<Avbd2dWorld *> &registry()
{
    static std::unordered_set<Avbd2dWorld *> r;
    return r;
}

std::unordered_map<uint16_t, Avbd2dWorld *> &byId()
{
    static std::unordered_map<uint16_t, Avbd2dWorld *> r;
    return r;
}

Avbd2dWorld *lookup(Avbd2dWorld *w)
{
    std::lock_guard<std::mutex> lk(g_mutex);
    return registry().count(w) ? w : nullptr;
}

Avbd2dWorld *worldFromId(uint16_t id)
{
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = byId().find(id);
    return it == byId().end() ? nullptr : it->second;
}

bool isOutOfMemory(const avbdvk::VkError &e)
{
    const VkResult r = e.result();
    return r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY;
}

// Runs `f` (which touches the device or allocates) and turns what it throws into a result code.
// Memory exhaustion is AVBD2D_ERR_OUT_OF_MEMORY, anything else from Vulkan is AVBD2D_ERR_DEVICE_LOST.
// Either kills the world (later calls return DEVICE_LOST) unless `recoverable`.
template <class F> Avbd2dResult guarded(Avbd2dWorld *w, bool recoverable, F &&f)
{
    try
    {
        return f();
    }
    catch (const std::bad_alloc &)
    {
        if (!recoverable)
            w->lost = true;
        return AVBD2D_ERR_OUT_OF_MEMORY;
    }
    catch (const avbdvk::VkError &e)
    {
        const bool oom = isOutOfMemory(e);
        if (!(oom && recoverable))
            w->lost = true;
        return oom ? AVBD2D_ERR_OUT_OF_MEMORY : AVBD2D_ERR_DEVICE_LOST;
    }
    catch (...)
    {
        w->lost = true;
        return AVBD2D_ERR_DEVICE_LOST;
    }
}

bool finite1(float a) { return std::isfinite(a); }
bool finite2(Avbd2dVec2 v) { return std::isfinite(v.x) && std::isfinite(v.y); }
Vec2 V(Avbd2dVec2 v) { return Vec2(v.x, v.y); }
Avbd2dVec2 A(Vec2 v) { return Avbd2dVec2{v.x, v.y}; }
float angleOf(Avbd2dRot q) { return std::atan2(q.s, q.c); }
Avbd2dRot rotOf(float a) { return Avbd2dRot{std::cos(a), std::sin(a)}; }

// ----- Parameters ------------------------------------------------------------------------------
SolverParams2D toParams(const Avbd2dWorldDef &d)
{
    SolverParams2D s;
    s.dt = d.timeStep;
    s.gravity = V(d.gravity);
    s.iterations = d.iterations;
    s.alpha = d.alpha;
    s.betaLin = d.betaLin;
    s.betaAng = d.betaAng;
    s.gamma = d.gamma;
    s.autoKStart = d.autoKStart != 0;
    s.dualDamping = d.dualDamping;
    s.dualDampVel = d.dualDampVel;
    s.killFallenEnabled = d.killFallen != 0;
    s.killY = d.killY;
    s.grabStrength = d.grabStrength;
    s.grabMaxSpeed = d.grabMaxSpeed;
    s.sleepEnabled = d.enableSleep != 0;
    s.sleepFrames = d.sleepFrames;
    s.sleepLinVel = d.sleepLinVel;
    s.sleepAngVel = d.sleepAngVel;
    s.sleepDisp = d.sleepDisp;
    s.sleepFracture = d.sleepFracture;
    s.maxSpeed = d.maxSpeed;
    s.contactCap = d.contactCap;
    s.frictionMix = d.frictionMix;
    s.enableContinuous = d.enableContinuous != 0;
    return s;
}

void fromParams(const SolverParams2D &s, Avbd2dWorldDef &d)
{
    d.timeStep = s.dt;
    d.gravity = A(s.gravity);
    d.iterations = s.iterations;
    d.alpha = s.alpha;
    d.betaLin = s.betaLin;
    d.betaAng = s.betaAng;
    d.gamma = s.gamma;
    d.autoKStart = s.autoKStart ? 1 : 0;
    d.dualDamping = s.dualDamping;
    d.dualDampVel = s.dualDampVel;
    d.killFallen = s.killFallenEnabled ? 1 : 0;
    d.killY = s.killY;
    d.grabStrength = s.grabStrength;
    d.grabMaxSpeed = s.grabMaxSpeed;
    d.enableSleep = s.sleepEnabled ? 1 : 0;
    d.sleepFrames = s.sleepFrames;
    d.sleepLinVel = s.sleepLinVel;
    d.sleepAngVel = s.sleepAngVel;
    d.sleepDisp = s.sleepDisp;
    d.sleepFracture = s.sleepFracture;
    d.maxSpeed = s.maxSpeed;
    d.contactCap = s.contactCap;
    d.frictionMix = s.frictionMix;
    d.enableContinuous = s.enableContinuous ? 1 : 0;
}

bool validWorldDef(const Avbd2dWorldDef *d)
{
    return d && finite2(d->gravity) && d->timeStep > 0.0f && std::isfinite(d->timeStep) && d->iterations >= 1 &&
           d->sleepFrames >= 0 && d->contactCap >= 0;
}

// ----- Building bodies in a Scene2D ------------------------------------------------------------
uint8_t flagsOf(const Avbd2dShapeDef &d)
{
    uint8_t f = 0;
    if (d.isSensor)
        f |= SHAPE_FLAG_SENSOR;
    if (d.enableSensorEvents)
        f |= SHAPE_FLAG_SENSOR_EVENTS;
    if (d.enableContactEvents)
        f |= SHAPE_FLAG_CONTACT_EVENTS;
    if (d.enableHitEvents)
        f |= SHAPE_FLAG_HIT_EVENTS;
    return f;
}

ShapeDef2D toShapeDef(const Avbd2dShapeDef &d)
{
    ShapeDef2D s;
    s.density = d.density;
    s.friction = d.friction;
    s.restitution = d.restitution;
    s.tangentSpeed = d.tangentSpeed;
    s.rollingResistance = d.rollingResistance;
    s.categoryBits = d.filter.categoryBits;
    s.maskBits = d.filter.maskBits;
    s.groupIndex = d.filter.groupIndex;
    s.userMaterialId = d.userMaterialId;
    return s;
}

// A 4-vertex, unrounded, right-angled CCW polygon is a box (the solver's fast shape).
bool asRect(const Avbd2dPolygon &p, Vec2 &center, float &angle, Vec2 &size)
{
    if (p.count != 4 || p.radius > 0.0f)
        return false;
    Vec2 v[4];
    for (int i = 0; i < 4; i++)
        v[i] = V(p.vertices[i]);
    const Vec2 e0 = v[1] - v[0], e1 = v[2] - v[1], e2 = v[3] - v[2], e3 = v[0] - v[3];
    const float l0 = length(e0), l1 = length(e1);
    if (!(l0 > 0.0f) || !(l1 > 0.0f) || cross(e0, e1) <= 0.0f)
        return false;
    const float tol = 1.0e-4f * (l0 + l1);
    if (std::fabs(dot(e0, e1)) > 1.0e-4f * l0 * l1 || length(e2 + e0) > tol || length(e3 + e1) > tol)
        return false;
    center = (v[0] + v[2]) * 0.5f;
    angle = std::atan2(e0.y, e0.x);
    size = Vec2(l0, l1);
    return true;
}

void addShapeToScene(Scene2D &sc, const ShapeRec &s)
{
    const ShapeDef2D d = toShapeDef(s.def);
    switch (s.kind)
    {
    case SK_POLYGON:
    {
        Vec2 c, size;
        float a = 0.0f;
        if (asRect(s.poly, c, a, size))
            sc.addBoxShape(c, a, size, d);
        else
        {
            Vec2 pts[AVBD2D_MAX_POLYGON_VERTICES];
            for (int i = 0; i < s.poly.count; i++)
                pts[i] = V(s.poly.vertices[i]);
            sc.addPolygonShape(Vec2(), 0.0f, pts, s.poly.count, s.poly.radius, d);
        }
        break;
    }
    case SK_CIRCLE:
        sc.addCircleShape(V(s.circle.center), s.circle.radius, d);
        break;
    case SK_CAPSULE:
        sc.addCapsuleShape(V(s.capsule.center1), V(s.capsule.center2), s.capsule.radius, d);
        break;
    case SK_SEGMENT:
        sc.addSegmentShape(V(s.seg.point1), V(s.seg.point2), d);
        break;
    case SK_CHAIN:
        sc.addChainSegmentShape(s.chain[0], s.chain[1], s.chain[2], s.chain[3], d);
        break;
    }
}

BodyDef2D toBodyDef(const Avbd2dBodyDef &d)
{
    BodyDef2D b;
    b.position = V(d.position);
    b.angle = angleOf(d.rotation);
    b.linearVelocity = V(d.linearVelocity);
    b.angularVelocity = d.angularVelocity;
    b.type = d.type == AVBD2D_STATIC_BODY ? BODY2D_STATIC : d.type == AVBD2D_KINEMATIC_BODY ? BODY2D_KINEMATIC
                                                                                            : BODY2D_DYNAMIC;
    b.gravityScale = d.gravityScale;
    b.linearDamping = d.linearDamping;
    b.angularDamping = d.angularDamping;
    b.lockLinearX = d.lockLinearX != 0;
    b.lockLinearY = d.lockLinearY != 0;
    b.fixedRotation = d.fixedRotation != 0;
    return b;
}

// Appends body `b` with its shapes to `sc`; returns the solver body id or -1.
// A body with no shape becomes a shapeless body (a real body with default mass when dynamic).
int buildBody(Scene2D &sc, const BodyRec &b, const Table<ShapeRec> &shapes, const BodyDef2D &bd)
{
    int id;
    if (b.shapes.empty())
        id = sc.addShapelessBody(bd);
    else
    {
        sc.beginBody(bd);
        for (int s : b.shapes)
            addShapeToScene(sc, shapes.items[(size_t)s]);
        id = sc.endBody();
        if (id >= 0)
        {
            const int first = sc.shapeFirst[(size_t)id];
            for (size_t k = 0; k < b.shapes.size(); k++)
                sc.shFlags[(size_t)first + k] = flagsOf(shapes.items[(size_t)b.shapes[k]].def);
        }
    }
    if (id >= 0 && b.def.isBullet)
        sc.setBullet(id, true);
    return id;
}

void ensureProps(Avbd2dWorld *w, BodyRec &b)
{
    if (b.propsValid)
        return;
    b.mass = b.inertia = 0.0f;
    b.lc = Vec2();
    if (b.flushed && !b.reshape)
    {
        if (b.solver >= 0)
        {
            const Scene2D &sc = w->world->authoring();
            b.mass = sc.mass[(size_t)b.solver];
            b.inertia = sc.moment[(size_t)b.solver];
            b.lc = sc.localCenter[(size_t)b.solver];
            b.propsValid = true;
        }
        return;
    }
    Scene2D tmp;
    const int id = buildBody(tmp, b, w->shapes, toBodyDef(b.def));
    if (id >= 0)
    {
        b.mass = tmp.mass[(size_t)id];
        b.inertia = tmp.moment[(size_t)id];
        b.lc = tmp.localCenter[(size_t)id];
    }
}

// ----- Pose helpers ----------------------------------------------------------------------------
struct Pose
{
    Vec2 com;
    float angle = 0.0f;
    Vec2 lc;
    Vec2 origin() const { return com - rotate(angle, lc); }
};

bool live(const BodyRec &b) { return b.flushed && b.solver >= 0; }

Pose poseOf(Avbd2dWorld *w, BodyRec &b)
{
    ensureProps(w, b);
    Pose p;
    p.lc = b.lc;
    if (live(b))
    {
        p.lc = w->world->authoring().localCenter[(size_t)b.solver]; // the world's, until a shape edit is applied
        p.com = w->world->positions()[(size_t)b.solver];
        p.angle = w->world->angles()[(size_t)b.solver];
    }
    else
    {
        p.angle = angleOf(b.def.rotation);
        p.com = V(b.def.position) + rotate(p.angle, b.lc);
    }
    return p;
}

Avbd2dBodyId bodyIdOf(Avbd2dWorld *w, int slot)
{
    return Avbd2dBodyId{slot + 1, w->id, w->bodies.items[(size_t)slot].gen};
}

Avbd2dShapeId shapeIdOf(Avbd2dWorld *w, int slot)
{
    return Avbd2dShapeId{slot + 1, w->id, w->shapes.items[(size_t)slot].gen};
}

Avbd2dJointId jointIdOf(Avbd2dWorld *w, int slot)
{
    return Avbd2dJointId{slot + 1, w->id, w->joints.items[(size_t)slot].gen};
}

// ----- Flush: pending bodies and joints become part of the live world ---------------------------
int motorAxis(JointType t) { return (t == JointType::Revolute || t == JointType::Wheel) ? 2 : 0; }
int limitAxis(JointType t) { return t == JointType::Revolute ? 2 : 0; }
float maxForceOf(const JointDef &d)
{
    return (d.type == JointType::Revolute || d.type == JointType::Wheel) ? d.maxMotorTorque : d.maxMotorForce;
}

void flushJoint(Avbd2dWorld *w, int slot)
{
    JointRec &j = w->joints.items[(size_t)slot];
    BodyRec &A = w->bodies.items[(size_t)j.bodyA];
    BodyRec &B = w->bodies.items[(size_t)j.bodyB];
    j.flushed = true;
    if (!live(A) || !live(B))
        return;
    VkWorld2D &vw = *w->world;
    const Scene2D &sc = vw.authoring();
    const Vec2 lcA = sc.localCenter[(size_t)A.solver], lcB = sc.localCenter[(size_t)B.solver];
    JointDef d = j.d;
    d.bodyA = A.solver;
    d.bodyB = B.solver;
    d.localAnchorA = d.localAnchorA - lcA;
    d.localAnchorB = d.localAnchorB - lcB;
    if (d.type == JointType::Motor)
    {
        // The offset is measured between the body origins (anchors are in the centre-of-mass frame).
        d.localAnchorA = Vec2() - lcA;
        d.localAnchorB = Vec2() - lcB;
        d.localAxisA = Vec2(1.0f, 0.0f);
        d.referenceAngle = 0.0f;
    }
    j.solver = vw.liveAddJoint(d);
    if (j.solver >= 0)
    {
        if ((int)w->solverJoint.size() <= j.solver)
            w->solverJoint.resize((size_t)j.solver + 1, -1);
        w->solverJoint[(size_t)j.solver] = slot;
    }
}

// Remembers the id a solver shape had, for the events of the next step.
void tombstone(Avbd2dWorld *w, int solverShape, Avbd2dShapeId id)
{
    if (solverShape >= 0)
        w->tomb[solverShape] = Avbd2dWorld::Tomb{id, w->stepEpoch};
}

// The edits are sized before anything changes, so a budget failure leaves the world untouched.
void flushReserve(Avbd2dWorld *w)
{
    if (w->pendingBodies.empty() && w->pendingJoints.empty() && w->replaceBodies.empty())
        return; // removals allocate nothing; a step without creates touches no memory
    VkWorld2D::LiveNeeds n;
    n.bodies = (int)w->pendingBodies.size();
    n.joints = (int)w->pendingJoints.size();
    for (int slot : w->pendingBodies)
    {
        const int ns = (int)w->bodies.items[(size_t)slot].shapes.size();
        n.shapes += ns;
        n.verts += 8 * ns;
    }
    for (int slot : w->replaceBodies)
    {
        const int ns = (int)w->bodies.items[(size_t)slot].shapes.size();
        n.shapes += ns;
        n.verts += 8 * ns;
    }
    w->world->liveReserve(n);
}

void flush(Avbd2dWorld *w)
{
    VkWorld2D &vw = *w->world;
    const bool edits = !(w->pendingBodies.empty() && w->pendingJoints.empty() && w->replaceBodies.empty() &&
                         w->removeBodies.empty() && w->removeJoints.empty());
    if (!edits && !vw.uploadsPending())
        return;
    if (edits)
    {
        for (int j : w->removeJoints)
            vw.liveRemoveJoint(j);
        w->removeJoints.clear();
        for (int b : w->removeBodies)
            vw.liveRemoveBody(b);
        w->removeBodies.clear();

        const Scene2D &sc = vw.authoring();
        std::vector<int> posts;
        // Shape edits on live bodies: the body keeps its slot and pose, its shapes get new ids.
        for (int slot : w->replaceBodies)
        {
            BodyRec &b = w->bodies.items[(size_t)slot];
            if (!b.alive || !b.reshape || b.solver < 0)
                continue;
            const int id = b.solver;
            const Vec2 lcOld = sc.localCenter[(size_t)id];
            const float ang = vw.angles()[(size_t)id];
            BodyDef2D bd = toBodyDef(b.def);
            bd.angle = ang;
            bd.position = vw.positions()[(size_t)id] - rotate(ang, lcOld);
            bd.linearVelocity = vw.velocities()[(size_t)id];
            bd.angularVelocity = vw.angularVelocities()[(size_t)id];
            for (int s : b.shapes)
            {
                ShapeRec &r = w->shapes.items[(size_t)s];
                if (r.solver >= 0 && r.solver < (int)w->solverShape.size())
                {
                    tombstone(w, r.solver, shapeIdOf(w, s));
                    w->solverShape[(size_t)r.solver] = -1;
                }
                r.solver = -1;
            }
            Scene2D tmp;
            const int ti = buildBody(tmp, b, w->shapes, bd);
            vw.liveReplaceBody(id, tmp, ti);
            vw.queueSetPose(id, sc.pos[(size_t)id], sc.ang[(size_t)id]);
            const int first = sc.shapeFirst[(size_t)id];
            if ((int)w->solverShape.size() < first + (int)b.shapes.size())
                w->solverShape.resize((size_t)first + b.shapes.size(), -1);
            for (size_t k = 0; k < b.shapes.size(); k++)
            {
                ShapeRec &r = w->shapes.items[(size_t)b.shapes[k]];
                r.solver = first + (int)k;
                w->solverShape[(size_t)r.solver] = b.shapes[k];
            }
            b.reshape = false;
            b.propsValid = false;
            if (!b.def.isEnabled)
                posts.push_back(slot);
            else if (!b.def.isAwake)
                posts.push_back(slot);
        }
        w->replaceBodies.clear();

        for (int slot : w->pendingBodies)
        {
            BodyRec &b = w->bodies.items[(size_t)slot];
            b.propsValid = false;
            Scene2D tmp;
            const int ti = buildBody(tmp, b, w->shapes, toBodyDef(b.def));
            const int id = vw.liveAddBody(tmp, ti);
            b.flushed = true;
            b.solver = id;
            if ((int)w->solverBody.size() <= id)
                w->solverBody.resize((size_t)id + 1, -1);
            w->solverBody[(size_t)id] = slot;
            const int first = sc.shapeFirst[(size_t)id];
            if ((int)w->solverShape.size() < first + (int)b.shapes.size())
                w->solverShape.resize((size_t)first + b.shapes.size(), -1);
            for (size_t k = 0; k < b.shapes.size(); k++)
            {
                ShapeRec &s = w->shapes.items[(size_t)b.shapes[k]];
                s.solver = first + (int)k;
                w->solverShape[(size_t)s.solver] = b.shapes[k];
            }
            if (!b.def.isAwake || !b.def.isEnabled)
                posts.push_back(slot);
        }
        w->pendingBodies.clear();
        for (int slot : w->pendingJoints)
            flushJoint(w, slot);
        w->pendingJoints.clear();
        w->postBodies = posts;
    }
    vw.liveFinish();
    vw.setHitEventThreshold(w->def.hitEventThreshold);
    for (int slot : w->postBodies)
    {
        const BodyRec &b = w->bodies.items[(size_t)slot];
        if (!b.alive || !live(b))
            continue;
        if (!b.def.isEnabled)
            vw.queueDisable(b.solver);
        else
            vw.queueSleep(b.solver);
    }
    w->postBodies.clear();
}

// ----- Stepping --------------------------------------------------------------------------------
void rebuildEvents(Avbd2dWorld *w)
{
    VkWorld2D &vw = *w->world;
    const Scene2D &sc = vw.authoring();
    w->bodyEvents.clear();
    for (const VkWorld2D::MovedBody2D &m : vw.movedBodies())
    {
        if (m.body < 0 || m.body >= (int)w->solverBody.size() || w->solverBody[(size_t)m.body] < 0)
            continue;
        const int slot = w->solverBody[(size_t)m.body];
        const BodyRec &b = w->bodies.items[(size_t)slot];
        Avbd2dBodyMoveEvent e;
        const Vec2 o = m.pos - rotate(m.angle, sc.localCenter[(size_t)m.body]);
        e.transform = Avbd2dTransform{A(o), rotOf(m.angle)};
        e.bodyId = bodyIdOf(w, slot);
        e.userData = b.def.userData;
        e.fellAsleep = m.fellAsleep ? 1 : 0;
        w->bodyEvents.push_back(e);
    }
    // The id of a solver shape: its live shape, else the id it had when it was destroyed or replaced.
    auto shapeSlot = [&](int s) -> int {
        return (s >= 0 && s < (int)w->solverShape.size()) ? w->solverShape[(size_t)s] : -1;
    };
    auto shapeId = [&](int s, Avbd2dShapeId &out) -> bool {
        const int slot = shapeSlot(s);
        if (slot >= 0)
        {
            out = shapeIdOf(w, slot);
            return true;
        }
        const auto it = w->tomb.find(s);
        if (it == w->tomb.end())
            return false;
        out = it->second.id;
        return true;
    };
    const Events2D &ev = vw.events();
    w->beginEvents.clear();
    w->endEvents.clear();
    w->hitEvents.clear();
    w->sensorBegin.clear();
    w->sensorEnd.clear();
    w->jointEvents.clear();
    Avbd2dShapeId ia, ib;
    for (const ContactEvent2D &e : ev.contactBegin)
        if (shapeSlot(e.shapeA) >= 0 && shapeSlot(e.shapeB) >= 0 && shapeId(e.shapeA, ia) && shapeId(e.shapeB, ib))
            w->beginEvents.push_back({ia, ib});
    for (const ContactEvent2D &e : ev.contactEnd)
        if (shapeId(e.shapeA, ia) && shapeId(e.shapeB, ib))
            w->endEvents.push_back({ia, ib});
    for (const HitEvent2D &e : ev.hit)
        if (shapeSlot(e.shapeA) >= 0 && shapeSlot(e.shapeB) >= 0 && shapeId(e.shapeA, ia) && shapeId(e.shapeB, ib))
            w->hitEvents.push_back({ia, ib, A(e.point), A(e.normal), e.approachSpeed});
    for (const SensorEvent2D &e : ev.sensorBegin)
        if (shapeSlot(e.sensorShape) >= 0 && shapeSlot(e.visitorShape) >= 0 && shapeId(e.sensorShape, ia) &&
            shapeId(e.visitorShape, ib))
            w->sensorBegin.push_back({ia, ib});
    for (const SensorEvent2D &e : ev.sensorEnd)
        if (shapeId(e.sensorShape, ia) && shapeId(e.visitorShape, ib))
            w->sensorEnd.push_back({ia, ib});
    for (auto it = w->tomb.begin(); it != w->tomb.end();)
        it = it->second.epoch < w->stepEpoch ? w->tomb.erase(it) : std::next(it);
    for (const JointBreakEvent2D &e : ev.jointBreak)
    {
        if (e.joint < 0 || e.joint >= (int)w->solverJoint.size() || w->solverJoint[(size_t)e.joint] < 0)
            continue;
        const int slot = w->solverJoint[(size_t)e.joint];
        w->jointEvents.push_back({jointIdOf(w, slot), w->joints.items[(size_t)slot].userData});
    }
}

Avbd2dResult stepEndImpl(Avbd2dWorld *w)
{
    if (!w->inFlight)
        return AVBD2D_OK;
    w->inFlight = false;
    return guarded(w, false, [&]() -> Avbd2dResult {
        w->last = w->world->stepEnd();
        rebuildEvents(w);
        return w->last.ok ? AVBD2D_OK : AVBD2D_ERR_CAPACITY;
    });
}

// Ends a step in flight (queries, grab and explosions read settled state); its result is handed to
// the next avbd2d_step_end.
void settle(Avbd2dWorld *w)
{
    if (!w->inFlight)
        return;
    const Avbd2dResult r = stepEndImpl(w);
    if (r != AVBD2D_OK)
        w->deferred = r;
}

Avbd2dResult stepBeginImpl(Avbd2dWorld *w)
{
    settle(w);
    // Sizing the device for the pending edits is the only part that can run out of memory
    // recoverably: it throws before anything is edited, so the world and its pending edits stay as
    // they were and the next step retries.
    const Avbd2dResult rr = guarded(w, true, [&]() -> Avbd2dResult {
        flushReserve(w);
        flush(w);
        return AVBD2D_OK;
    });
    if (rr != AVBD2D_OK)
        return rr;
    return guarded(w, false, [&]() -> Avbd2dResult {
        if (w->world->bodyCount() == 0)
        {
            w->bodyEvents.clear();
            return AVBD2D_OK;
        }
        w->stepEpoch++;
        w->world->setHitEventThreshold(w->def.hitEventThreshold);
        w->world->stepBegin();
        w->inFlight = true;
        return AVBD2D_OK;
    });
}

Avbd2dResult createImpl(const Avbd2dWorldDef *defIn, const Avbd2dVulkanDevice *ext, Avbd2dWorld **out_world)
{
    *out_world = nullptr;
    const Avbd2dWorldDef def = defIn ? *defIn : avbd2d_default_world_def();
    if (!validWorldDef(&def))
        return AVBD2D_ERR_INVALID_ARG;
    std::unique_ptr<Avbd2dWorld> w;
    try
    {
        w.reset(new Avbd2dWorld());
        w->def = def;
        w->world.reset(new VkWorld2D());
        avbdvk::Device &dev = w->world->device();
        dev.setPreferDedicatedCompute(true);
        if (ext)
        {
            w->world->configureDevice();
            dev.adopt(reinterpret_cast<VkInstance>(ext->instance),
                      reinterpret_cast<VkPhysicalDevice>(ext->physical_device),
                      reinterpret_cast<VkDevice>(ext->device), ext->queue_family_index, ext->queue_index,
                      ext->api_version);
        }
        // Test hook: the Nth step submission of this world reports VK_ERROR_DEVICE_LOST.
        if (const char *f = std::getenv("AVBD2D_DEBUG_FAIL_SUBMIT"))
            dev.setDebugFailSubmit((uint64_t)std::strtoull(f, nullptr, 10));
        if (def.memoryBudget)
            dev.setMemoryBudget(def.memoryBudget);
        w->world->init();
        w->world->params() = toParams(def);
        w->world->setHitEventThreshold(def.hitEventThreshold);
        w->world->setFullReadback(false);
        w->world->setJointForceReadback(true);
    }
    catch (const std::bad_alloc &)
    {
        return AVBD2D_ERR_OUT_OF_MEMORY;
    }
    catch (const avbdvk::DeviceError &e)
    {
        fprintf(stderr, "[avbd2d] no usable Vulkan device: %s\n", e.what());
        return AVBD2D_ERR_DEVICE;
    }
    catch (const avbdvk::VkError &e)
    {
        return isOutOfMemory(e) ? AVBD2D_ERR_OUT_OF_MEMORY : AVBD2D_ERR_DEVICE;
    }
    catch (...)
    {
        return AVBD2D_ERR_DEVICE;
    }
    Avbd2dWorld *raw = w.release();
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        // World indices are never reused until the 16-bit counter wraps, so a stale id of a
        // destroyed world cannot alias a new one.
        for (int tries = 0; tries < 65536 && byId().count(g_nextWorldId); tries++)
            g_nextWorldId++;
        raw->id = g_nextWorldId++;
        registry().insert(raw);
        byId()[raw->id] = raw;
    }
    *out_world = raw;
    return AVBD2D_OK;
}

// A packed polygon from a convex hull (CCW), with normals and the area centroid.
Avbd2dPolygon fillPolygon(const Vec2 *pts, int n, float radius)
{
    Avbd2dPolygon p{};
    p.count = n;
    p.radius = radius;
    Vec2 c;
    float area = 0.0f;
    for (int i = 0; i < n; i++)
    {
        p.vertices[i] = A(pts[i]);
        const Vec2 e = pts[(i + 1) % n] - pts[i];
        const float len = length(e);
        p.normals[i] = len > 0.0f ? Avbd2dVec2{e.y / len, -e.x / len} : Avbd2dVec2{0.0f, 1.0f};
    }
    for (int i = 1; i < n - 1; i++)
    {
        const Vec2 e1 = pts[i] - pts[0], e2 = pts[i + 1] - pts[0];
        const float a = 0.5f * cross(e1, e2);
        area += a;
        c = c + (e1 + e2) * (a / 3.0f);
    }
    p.centroid = area > 0.0f ? A(pts[0] + c * (1.0f / area)) : A(pts[0]);
    return p;
}

bool validShapeDef(const Avbd2dShapeDef *d)
{
    return d && std::isfinite(d->density) && d->density >= 0.0f && std::isfinite(d->friction) && d->friction >= 0.0f &&
           std::isfinite(d->restitution) && std::isfinite(d->rollingResistance) && std::isfinite(d->tangentSpeed);
}

bool passes(const ShapeRec &s, const Avbd2dQueryFilter &f)
{
    return (s.def.filter.categoryBits & f.maskBits) != 0 && (f.categoryBits & s.def.filter.maskBits) != 0;
}

// The shape slot behind a solver shape, or null when it was destroyed or its body is disabled.
const ShapeRec *queryShape(Avbd2dWorld *w, int solverShape, int *slotOut)
{
    if (solverShape < 0 || solverShape >= (int)w->solverShape.size() || w->solverShape[(size_t)solverShape] < 0)
        return nullptr;
    const int slot = w->solverShape[(size_t)solverShape];
    const ShapeRec &s = w->shapes.items[(size_t)slot];
    if (!s.alive || s.body < 0 || !w->bodies.items[(size_t)s.body].def.isEnabled)
        return nullptr;
    *slotOut = slot;
    return &s;
}

Proxy2D toProxy(const Avbd2dShapeProxy &p)
{
    Proxy2D q;
    q.count = p.count;
    q.radius = p.radius;
    for (int i = 0; i < p.count && i < 8; i++)
        q.pts[i] = V(p.points[i]);
    return q;
}

bool validProxy(const Avbd2dShapeProxy *p)
{
    if (!p || p->count < 1 || p->count > AVBD2D_MAX_POLYGON_VERTICES || !(p->radius >= 0.0f))
        return false;
    for (int i = 0; i < p->count; i++)
        if (!finite2(p->points[i]))
            return false;
    return true;
}
} // namespace

#define AVBD2D_GET_WORLD(handle)                  \
    Avbd2dWorld *w = lookup(handle);              \
    if (!w)                                       \
        return AVBD2D_ERR_NULL_HANDLE;            \
    if (w->lost)                                  \
        return AVBD2D_ERR_DEVICE_LOST

#define ID_WORLD(id)                              \
    Avbd2dWorld *w = worldFromId((id).world0);    \
    if (!w)                                       \
        return AVBD2D_ERR_INVALID_ARG;            \
    if (w->lost)                                  \
        return AVBD2D_ERR_DEVICE_LOST

#define BODY_CTX(id)                                              \
    ID_WORLD(id);                                                 \
    BodyRec *b = w->bodies.find((id).index1, (id).generation);    \
    if (!b)                                                       \
        return AVBD2D_ERR_INVALID_ARG;                            \
    const int slot = (id).index1 - 1;                             \
    (void)slot

#define SHAPE_CTX(id)                                             \
    ID_WORLD(id);                                                 \
    ShapeRec *s = w->shapes.find((id).index1, (id).generation);   \
    if (!s)                                                       \
        return AVBD2D_ERR_INVALID_ARG;                            \
    const int slot = (id).index1 - 1;                             \
    (void)slot

#define JOINT_CTX(id)                                             \
    ID_WORLD(id);                                                 \
    JointRec *j = w->joints.find((id).index1, (id).generation);   \
    if (!j)                                                       \
        return AVBD2D_ERR_INVALID_ARG;                            \
    const int slot = (id).index1 - 1;                             \
    (void)slot

#define NEED(p)                                                   \
    if (!(p))                                                     \
    return AVBD2D_ERR_INVALID_ARG

extern "C"
{

AVBD2D_API uint32_t avbd2d_abi_version(void)
{
    return ((uint32_t)AVBD2D_ABI_VERSION_MAJOR << 16) | (uint32_t)AVBD2D_ABI_VERSION_MINOR;
}

AVBD2D_API const char *avbd2d_result_string(Avbd2dResult r)
{
    switch (r)
    {
    case AVBD2D_OK: return "ok";
    case AVBD2D_ERR_NULL_HANDLE: return "null, unknown or destroyed world handle";
    case AVBD2D_ERR_INVALID_ARG: return "invalid argument (or a stale id)";
    case AVBD2D_ERR_UNSUPPORTED: return "unsupported";
    case AVBD2D_ERR_NOT_COMMITTED: return "unused";
    case AVBD2D_ERR_DEVICE: return "no usable Vulkan device";
    case AVBD2D_ERR_CAPACITY: return "a hard capacity ceiling or the memory budget clamped the step";
    case AVBD2D_ERR_DEVICE_LOST: return "the Vulkan device failed or was lost; destroy the world and create a new one";
    case AVBD2D_ERR_OUT_OF_MEMORY: return "out of memory (host, device or the memory budget)";
    }
    return "unknown result";
}

AVBD2D_API Avbd2dRot avbd2d_make_rot(float radians) { return rotOf(radians); }
AVBD2D_API float avbd2d_rot_get_angle(Avbd2dRot q) { return angleOf(q); }

AVBD2D_API int32_t avbd2d_body_is_valid(Avbd2dBodyId id)
{
    Avbd2dWorld *w = worldFromId(id.world0);
    return w && w->bodies.find(id.index1, id.generation) ? 1 : 0;
}
AVBD2D_API int32_t avbd2d_shape_is_valid(Avbd2dShapeId id)
{
    Avbd2dWorld *w = worldFromId(id.world0);
    return w && w->shapes.find(id.index1, id.generation) ? 1 : 0;
}
AVBD2D_API int32_t avbd2d_joint_is_valid(Avbd2dJointId id)
{
    Avbd2dWorld *w = worldFromId(id.world0);
    return w && w->joints.find(id.index1, id.generation) ? 1 : 0;
}
AVBD2D_API int32_t avbd2d_chain_is_valid(Avbd2dChainId id)
{
    Avbd2dWorld *w = worldFromId(id.world0);
    if (!w || id.index1 < 1 || id.index1 > (int32_t)w->shapes.items.size())
        return 0;
    // A chain id names its first segment shape.
    const ShapeRec &s = w->shapes.items[(size_t)id.index1 - 1];
    return s.alive && s.gen == id.generation && s.kind == SK_CHAIN ? 1 : 0;
}

// ----- Defaults --------------------------------------------------------------------------------
AVBD2D_API Avbd2dWorldDef avbd2d_default_world_def(void)
{
    Avbd2dWorldDef d{};
    fromParams(SolverParams2D(), d);
    d.hitEventThreshold = 1.0f;
    return d;
}

AVBD2D_API Avbd2dBodyDef avbd2d_default_body_def(void)
{
    Avbd2dBodyDef d{};
    d.type = AVBD2D_STATIC_BODY;
    d.rotation = Avbd2dRot{1.0f, 0.0f};
    d.gravityScale = 1.0f;
    d.enableSleep = 1;
    d.isAwake = 1;
    d.isEnabled = 1;
    return d;
}

AVBD2D_API Avbd2dShapeDef avbd2d_default_shape_def(void)
{
    Avbd2dShapeDef d{};
    d.friction = 0.6f;
    d.density = 1.0f;
    d.filter.categoryBits = 1;
    d.filter.maskBits = ~uint64_t(0);
    d.enableSensorEvents = 1;
    d.enableContactEvents = 1;
    return d;
}

AVBD2D_API Avbd2dChainDef avbd2d_default_chain_def(void)
{
    Avbd2dChainDef d{};
    d.friction = 0.6f;
    d.filter.categoryBits = 1;
    d.filter.maskBits = ~uint64_t(0);
    return d;
}

AVBD2D_API Avbd2dRevoluteJointDef avbd2d_default_revolute_joint_def(void)
{
    Avbd2dRevoluteJointDef d{};
    d.breakForce = 3.4e38f;
    return d;
}
AVBD2D_API Avbd2dWeldJointDef avbd2d_default_weld_joint_def(void)
{
    Avbd2dWeldJointDef d{};
    d.breakForce = 3.4e38f;
    return d;
}
AVBD2D_API Avbd2dPrismaticJointDef avbd2d_default_prismatic_joint_def(void)
{
    Avbd2dPrismaticJointDef d{};
    d.localAxisA = Avbd2dVec2{1.0f, 0.0f};
    d.breakForce = 3.4e38f;
    return d;
}
AVBD2D_API Avbd2dWheelJointDef avbd2d_default_wheel_joint_def(void)
{
    Avbd2dWheelJointDef d{};
    d.localAxisA = Avbd2dVec2{1.0f, 0.0f};
    d.enableSpring = 1;
    d.hertz = 1.0f;
    d.dampingRatio = 0.7f;
    d.breakForce = 3.4e38f;
    return d;
}
AVBD2D_API Avbd2dDistanceJointDef avbd2d_default_distance_joint_def(void)
{
    Avbd2dDistanceJointDef d{};
    d.length = 1.0f;
    d.maxLength = 1.0e5f;
    d.breakForce = 3.4e38f;
    return d;
}
AVBD2D_API Avbd2dMotorJointDef avbd2d_default_motor_joint_def(void)
{
    Avbd2dMotorJointDef d{};
    d.maxForce = 1.0f;
    d.maxTorque = 1.0f;
    d.breakForce = 3.4e38f;
    return d;
}
AVBD2D_API Avbd2dFilterJointDef avbd2d_default_filter_joint_def(void)
{
    Avbd2dFilterJointDef d{};
    return d;
}
AVBD2D_API Avbd2dExplosionDef avbd2d_default_explosion_def(void)
{
    Avbd2dExplosionDef d{};
    d.maskBits = ~uint64_t(0);
    return d;
}
AVBD2D_API Avbd2dQueryFilter avbd2d_default_query_filter(void)
{
    Avbd2dQueryFilter f{};
    f.categoryBits = 1;
    f.maskBits = ~uint64_t(0);
    return f;
}

// ----- World lifetime --------------------------------------------------------------------------
AVBD2D_API Avbd2dResult avbd2d_create(Avbd2dWorld **out_world)
{
    NEED(out_world);
    return createImpl(nullptr, nullptr, out_world);
}

AVBD2D_API Avbd2dResult avbd2d_create_world(const Avbd2dWorldDef *def, Avbd2dWorld **out_world)
{
    NEED(out_world);
    *out_world = nullptr;
    NEED(def);
    return createImpl(def, nullptr, out_world);
}

static Avbd2dResult validateDevice(const Avbd2dVulkanDevice *device)
{
    if (!device || device->struct_size < sizeof(Avbd2dVulkanDevice) || !device->instance || !device->physical_device ||
        !device->device)
        return AVBD2D_ERR_INVALID_ARG;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_create_with_vulkan(const Avbd2dVulkanDevice *device, Avbd2dWorld **out_world)
{
    NEED(out_world);
    *out_world = nullptr;
    if (validateDevice(device) != AVBD2D_OK)
        return AVBD2D_ERR_INVALID_ARG;
    return createImpl(nullptr, device, out_world);
}

AVBD2D_API Avbd2dResult avbd2d_create_world_with_vulkan(const Avbd2dWorldDef *def, const Avbd2dVulkanDevice *device,
                                                        Avbd2dWorld **out_world)
{
    NEED(out_world);
    *out_world = nullptr;
    if (!def || validateDevice(device) != AVBD2D_OK)
        return AVBD2D_ERR_INVALID_ARG;
    return createImpl(def, device, out_world);
}

AVBD2D_API Avbd2dResult avbd2d_destroy(Avbd2dWorld *world)
{
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (!registry().erase(world))
            return AVBD2D_ERR_NULL_HANDLE;
        byId().erase(world->id);
    }
    try
    {
        delete world; // ~VkWorld2D tears the device down; a lost device must not take the host with it
    }
    catch (...)
    {
    }
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_set_memory_budget(Avbd2dWorld *world, uint64_t bytes)
{
    AVBD2D_GET_WORLD(world);
    w->world->device().setMemoryBudget(bytes);
    w->def.memoryBudget = bytes;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_set_world_def(Avbd2dWorld *world, const Avbd2dWorldDef *def)
{
    AVBD2D_GET_WORLD(world);
    if (!validWorldDef(def))
        return AVBD2D_ERR_INVALID_ARG;
    const uint64_t budget = w->def.memoryBudget;
    w->def = *def;
    w->def.memoryBudget = budget;
    w->world->params() = toParams(*def);
    w->world->setHitEventThreshold(def->hitEventThreshold);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_get_world_def(Avbd2dWorld *world, Avbd2dWorldDef *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    *out = w->def;
    fromParams(w->world->params(), *out);
    out->hitEventThreshold = w->def.hitEventThreshold;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_world_set_gravity(Avbd2dWorld *world, Avbd2dVec2 gravity)
{
    AVBD2D_GET_WORLD(world);
    if (!finite2(gravity))
        return AVBD2D_ERR_INVALID_ARG;
    w->world->params().gravity = V(gravity);
    w->def.gravity = gravity;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_world_get_gravity(Avbd2dWorld *world, Avbd2dVec2 *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    *out = A(w->world->params().gravity);
    return AVBD2D_OK;
}

// ----- Stepping --------------------------------------------------------------------------------
AVBD2D_API Avbd2dResult avbd2d_step_begin(Avbd2dWorld *world)
{
    AVBD2D_GET_WORLD(world);
    return stepBeginImpl(w);
}

AVBD2D_API Avbd2dResult avbd2d_step_end(Avbd2dWorld *world)
{
    AVBD2D_GET_WORLD(world);
    if (!w->inFlight)
    {
        const Avbd2dResult r = w->deferred;
        w->deferred = AVBD2D_OK;
        return r;
    }
    return stepEndImpl(w);
}

AVBD2D_API Avbd2dResult avbd2d_step(Avbd2dWorld *world)
{
    AVBD2D_GET_WORLD(world);
    Avbd2dResult r = stepBeginImpl(w);
    if (r != AVBD2D_OK)
        return r;
    w->deferred = AVBD2D_OK;
    return stepEndImpl(w);
}

AVBD2D_API Avbd2dResult avbd2d_wake_all(Avbd2dWorld *world)
{
    AVBD2D_GET_WORLD(world);
    w->world->wakeAll();
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_get_stats(Avbd2dWorld *world, Avbd2dStats *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    *out = Avbd2dStats{};
    out->bodies = w->bodies.live();
    out->joints = w->joints.live();
    out->pairs = w->last.pairs;
    out->contacts = w->last.contacts;
    out->colors = w->last.colors;
    out->capacity_reruns = w->last.capacityReruns;
    out->submits = w->last.submits;
    out->ok = w->last.ok ? 1 : 0;
    out->step_ms = w->last.stepMs;
    if (w->world->bodyCount() > 0)
    {
        const WorldStats2D ws = w->world->stats();
        out->awake = ws.awake;
        out->asleep = ws.asleep;
        out->removed = ws.removed;
        out->broken_joints = ws.brokenJoints;
        out->gpu_ms = w->world->timing().valid ? w->world->timing().gpuMs : 0.0f;
    }
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_get_capabilities(Avbd2dWorld *world, Avbd2dCapabilities *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    if (out->struct_size < sizeof(Avbd2dCapabilities))
        return AVBD2D_ERR_INVALID_ARG;
    Avbd2dCapabilities c{};
    c.struct_size = sizeof(Avbd2dCapabilities);
    c.restitution = 0;
    c.joint_damping_ratio = 0;
    c.determinism = 0;
    c.continuous = 1;
    c.sensors = 1;
    c.contact_events = 1;
    c.hit_events = 1;
    c.joint_events = 1;
    c.move_events = 1;
    c.joint_forces = 1;
    c.explosions = 1;
    c.queries = 1;
    c.chains = 1;
    c.live_body_create_destroy = 1;
    c.live_shape_add = 1;
    c.per_body_sleep_disable = 0;
    c.set_body_type = 0;
    c.kinematic_bodies = 1;
    c.grab = 1;
    c.max_polygon_vertices = AVBD2D_MAX_POLYGON_VERTICES;
    *out = c;
    return AVBD2D_OK;
}

// ----- Bodies ----------------------------------------------------------------------------------
AVBD2D_API Avbd2dResult avbd2d_create_body(Avbd2dWorld *world, const Avbd2dBodyDef *def, Avbd2dBodyId *out_id)
{
    AVBD2D_GET_WORLD(world);
    NEED(def);
    NEED(out_id);
    if (!finite2(def->position) || !finite1(def->rotation.c) || !finite1(def->rotation.s) ||
        !finite2(def->linearVelocity) || !finite1(def->angularVelocity) || def->type < AVBD2D_STATIC_BODY ||
        def->type > AVBD2D_DYNAMIC_BODY)
        return AVBD2D_ERR_INVALID_ARG;
    return guarded(w, true, [&]() -> Avbd2dResult {
        const int slot = w->bodies.acquire();
        BodyRec &b = w->bodies.items[(size_t)slot];
        b.def = *def;
        w->pendingBodies.push_back(slot);
        *out_id = bodyIdOf(w, slot);
        return AVBD2D_OK;
    });
}

static void destroyJointSlot(Avbd2dWorld *w, int slot)
{
    JointRec &j = w->joints.items[(size_t)slot];
    if (j.flushed && j.solver >= 0)
    {
        // Applied at the next step: the slot goes to the next joint created.
        w->removeJoints.push_back(j.solver);
        w->solverJoint[(size_t)j.solver] = -1;
    }
    else
        w->pendingJoints.erase(std::remove(w->pendingJoints.begin(), w->pendingJoints.end(), slot),
                               w->pendingJoints.end());
    for (int bs : {j.bodyA, j.bodyB})
    {
        if (bs < 0)
            continue;
        std::vector<int> &v = w->bodies.items[(size_t)bs].joints;
        v.erase(std::remove(v.begin(), v.end(), slot), v.end());
    }
    w->joints.release(slot);
}

AVBD2D_API Avbd2dResult avbd2d_destroy_body(Avbd2dBodyId id)
{
    BODY_CTX(id);
    return guarded(w, false, [&]() -> Avbd2dResult {
        const std::vector<int> js = b->joints;
        for (int js1 : js)
            destroyJointSlot(w, js1);
        for (int s : b->shapes)
        {
            const int solver = w->shapes.items[(size_t)s].solver;
            if (solver >= 0 && solver < (int)w->solverShape.size())
            {
                tombstone(w, solver, shapeIdOf(w, s));
                w->solverShape[(size_t)solver] = -1;
            }
            w->shapes.release(s);
        }
        if (!b->flushed)
            w->pendingBodies.erase(std::remove(w->pendingBodies.begin(), w->pendingBodies.end(), slot),
                                   w->pendingBodies.end());
        else if (b->solver >= 0)
        {
            w->removeBodies.push_back(b->solver);
            w->solverBody[(size_t)b->solver] = -1;
            w->replaceBodies.erase(std::remove(w->replaceBodies.begin(), w->replaceBodies.end(), slot),
                                   w->replaceBodies.end());
        }
        w->bodies.release(slot);
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_body_get_position(Avbd2dBodyId id, Avbd2dVec2 *out)
{
    BODY_CTX(id);
    NEED(out);
    *out = A(poseOf(w, *b).origin());
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_rotation(Avbd2dBodyId id, Avbd2dRot *out)
{
    BODY_CTX(id);
    NEED(out);
    *out = live(*b) ? rotOf(poseOf(w, *b).angle) : b->def.rotation;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_angle(Avbd2dBodyId id, float *out)
{
    BODY_CTX(id);
    NEED(out);
    *out = poseOf(w, *b).angle;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_transform(Avbd2dBodyId id, Avbd2dTransform *out)
{
    BODY_CTX(id);
    NEED(out);
    const Pose p = poseOf(w, *b);
    out->p = A(p.origin());
    out->q = live(*b) ? rotOf(p.angle) : b->def.rotation;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_world_center(Avbd2dBodyId id, Avbd2dVec2 *out)
{
    BODY_CTX(id);
    NEED(out);
    *out = A(poseOf(w, *b).com);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_local_center(Avbd2dBodyId id, Avbd2dVec2 *out)
{
    BODY_CTX(id);
    NEED(out);
    ensureProps(w, *b);
    *out = A(b->lc);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_linear_velocity(Avbd2dBodyId id, Avbd2dVec2 *out)
{
    BODY_CTX(id);
    NEED(out);
    *out = live(*b) ? A(w->world->velocities()[(size_t)b->solver]) : b->def.linearVelocity;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_angular_velocity(Avbd2dBodyId id, float *out)
{
    BODY_CTX(id);
    NEED(out);
    *out = live(*b) ? w->world->angularVelocities()[(size_t)b->solver] : b->def.angularVelocity;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_mass(Avbd2dBodyId id, float *out)
{
    BODY_CTX(id);
    NEED(out);
    ensureProps(w, *b);
    *out = b->mass;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_rotational_inertia(Avbd2dBodyId id, float *out)
{
    BODY_CTX(id);
    NEED(out);
    ensureProps(w, *b);
    *out = b->inertia;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_type(Avbd2dBodyId id, int32_t *out)
{
    BODY_CTX(id);
    NEED(out);
    *out = b->def.type;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_is_awake(Avbd2dBodyId id, int32_t *out)
{
    BODY_CTX(id);
    NEED(out);
    if (live(*b))
        *out = w->world->awakeFlags()[(size_t)b->solver] ? 1 : 0;
    else
        *out = (b->def.isAwake && b->def.isEnabled && b->def.type != AVBD2D_STATIC_BODY) ? 1 : 0;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_is_enabled(Avbd2dBodyId id, int32_t *out)
{
    BODY_CTX(id);
    NEED(out);
    *out = b->def.isEnabled ? 1 : 0;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_get_user_data(Avbd2dBodyId id, void **out)
{
    BODY_CTX(id);
    NEED(out);
    *out = b->def.userData;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_set_user_data(Avbd2dBodyId id, void *userData)
{
    BODY_CTX(id);
    b->def.userData = userData;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_set_transform(Avbd2dBodyId id, Avbd2dVec2 position, Avbd2dRot rotation)
{
    BODY_CTX(id);
    if (!finite2(position) || !finite1(rotation.c) || !finite1(rotation.s))
        return AVBD2D_ERR_INVALID_ARG;
    ensureProps(w, *b);
    b->def.position = position;
    b->def.rotation = rotation;
    if (live(*b))
    {
        const float a = angleOf(rotation);
        const Vec2 lc = w->world->authoring().localCenter[(size_t)b->solver]; // the world's, until a shape edit applies
        w->world->queueSetPose(b->solver, V(position) + rotate(a, lc), a);
    }
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_set_linear_velocity(Avbd2dBodyId id, Avbd2dVec2 v)
{
    BODY_CTX(id);
    if (!finite2(v))
        return AVBD2D_ERR_INVALID_ARG;
    if (live(*b))
        w->world->queueSetVelocity(b->solver, V(v), w->world->angularVelocities()[(size_t)b->solver]);
    else
        b->def.linearVelocity = v;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_set_angular_velocity(Avbd2dBodyId id, float av)
{
    BODY_CTX(id);
    if (!finite1(av))
        return AVBD2D_ERR_INVALID_ARG;
    if (live(*b))
        w->world->queueSetVelocity(b->solver, w->world->velocities()[(size_t)b->solver], av);
    else
        b->def.angularVelocity = av;
    return AVBD2D_OK;
}

// An impulse on a body: queued on a live one, folded into the velocity of a pending one.
static void impulseOn(Avbd2dWorld *w, BodyRec &b, Vec2 J, const Vec2 *point, float angular, int wake)
{
    if (live(b))
    {
        if (J.x != 0.0f || J.y != 0.0f)
            w->world->queueImpulse(b.solver, J, point);
        if (angular != 0.0f)
            w->world->queueAngularImpulse(b.solver, angular);
        if (wake)
            w->world->queueWake(b.solver);
        return;
    }
    ensureProps(w, b);
    if (!(b.mass > 0.0f) || b.def.type != AVBD2D_DYNAMIC_BODY)
        return;
    b.def.linearVelocity.x += J.x / b.mass;
    b.def.linearVelocity.y += J.y / b.mass;
    if (b.inertia > 0.0f)
    {
        float dw = angular;
        if (point)
        {
            const Pose p = poseOf(w, b);
            dw += cross(*point - p.com, J);
        }
        b.def.angularVelocity += dw / b.inertia;
    }
}

AVBD2D_API Avbd2dResult avbd2d_body_apply_force(Avbd2dBodyId id, Avbd2dVec2 force, Avbd2dVec2 worldPoint, int32_t wake)
{
    BODY_CTX(id);
    if (!finite2(force) || !finite2(worldPoint))
        return AVBD2D_ERR_INVALID_ARG;
    const float dt = w->world->params().dt;
    const Vec2 p = V(worldPoint);
    impulseOn(w, *b, V(force) * dt, &p, 0.0f, wake);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_apply_force_to_center(Avbd2dBodyId id, Avbd2dVec2 force, int32_t wake)
{
    BODY_CTX(id);
    if (!finite2(force))
        return AVBD2D_ERR_INVALID_ARG;
    impulseOn(w, *b, V(force) * w->world->params().dt, nullptr, 0.0f, wake);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_apply_torque(Avbd2dBodyId id, float torque, int32_t wake)
{
    BODY_CTX(id);
    if (!finite1(torque))
        return AVBD2D_ERR_INVALID_ARG;
    impulseOn(w, *b, Vec2(), nullptr, torque * w->world->params().dt, wake);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_apply_linear_impulse(Avbd2dBodyId id, Avbd2dVec2 impulse, Avbd2dVec2 worldPoint,
                                                         int32_t wake)
{
    BODY_CTX(id);
    if (!finite2(impulse) || !finite2(worldPoint))
        return AVBD2D_ERR_INVALID_ARG;
    const Vec2 p = V(worldPoint);
    impulseOn(w, *b, V(impulse), &p, 0.0f, wake);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_apply_linear_impulse_to_center(Avbd2dBodyId id, Avbd2dVec2 impulse, int32_t wake)
{
    BODY_CTX(id);
    if (!finite2(impulse))
        return AVBD2D_ERR_INVALID_ARG;
    impulseOn(w, *b, V(impulse), nullptr, 0.0f, wake);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_apply_angular_impulse(Avbd2dBodyId id, float impulse, int32_t wake)
{
    BODY_CTX(id);
    if (!finite1(impulse))
        return AVBD2D_ERR_INVALID_ARG;
    impulseOn(w, *b, Vec2(), nullptr, impulse, wake);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_set_awake(Avbd2dBodyId id, int32_t awake)
{
    BODY_CTX(id);
    b->def.isAwake = awake ? 1 : 0;
    if (live(*b))
    {
        if (awake)
            w->world->queueWake(b->solver);
        else
            w->world->queueSleep(b->solver);
        w->world->setHostAwake(b->solver, awake != 0);
    }
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_enable(Avbd2dBodyId id)
{
    BODY_CTX(id);
    b->def.isEnabled = 1;
    if (live(*b))
        w->world->queueEnable(b->solver);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_body_disable(Avbd2dBodyId id)
{
    BODY_CTX(id);
    b->def.isEnabled = 0;
    if (live(*b))
        w->world->queueDisable(b->solver);
    return AVBD2D_OK;
}

// ----- Shape construction ----------------------------------------------------------------------
AVBD2D_API Avbd2dPolygon avbd2d_make_box(float hx, float hy)
{
    const Vec2 pts[4] = {Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy)};
    Avbd2dPolygon p = fillPolygon(pts, 4, 0.0f);
    p.centroid = Avbd2dVec2{0.0f, 0.0f};
    return p;
}

AVBD2D_API Avbd2dPolygon avbd2d_make_offset_box(float hx, float hy, Avbd2dVec2 center, Avbd2dRot rotation)
{
    const float a = angleOf(rotation);
    const Vec2 pts[4] = {Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy)};
    Vec2 q[4];
    for (int i = 0; i < 4; i++)
        q[i] = V(center) + rotate(a, pts[i]);
    Avbd2dPolygon p = fillPolygon(q, 4, 0.0f);
    p.centroid = center;
    return p;
}

AVBD2D_API Avbd2dPolygon avbd2d_make_rounded_box(float hx, float hy, float radius)
{
    Avbd2dPolygon p = avbd2d_make_box(hx, hy);
    p.radius = radius;
    return p;
}

AVBD2D_API Avbd2dResult avbd2d_make_polygon(const Avbd2dVec2 *points, int32_t count, float radius, Avbd2dPolygon *out)
{
    NEED(points);
    NEED(out);
    if (count < 3 || count > AVBD2D_MAX_POLYGON_VERTICES || !(radius >= 0.0f))
        return AVBD2D_ERR_INVALID_ARG;
    Vec2 in[AVBD2D_MAX_POLYGON_VERTICES], hull[AVBD2D_MAX_POLYGON_VERTICES];
    for (int i = 0; i < count; i++)
    {
        if (!finite2(points[i]))
            return AVBD2D_ERR_INVALID_ARG;
        in[i] = V(points[i]);
    }
    const int n = Scene2D::computeHull(in, count, hull);
    if (n < 3)
        return AVBD2D_ERR_INVALID_ARG;
    *out = fillPolygon(hull, n, radius);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_make_proxy(const Avbd2dVec2 *points, int32_t count, float radius, Avbd2dShapeProxy *out)
{
    NEED(points);
    NEED(out);
    if (count < 1 || count > AVBD2D_MAX_POLYGON_VERTICES || !(radius >= 0.0f))
        return AVBD2D_ERR_INVALID_ARG;
    *out = Avbd2dShapeProxy{};
    for (int i = 0; i < count; i++)
        out->points[i] = points[i];
    out->count = count;
    out->radius = radius;
    return AVBD2D_OK;
}

// Releases a shape record; a live shape keeps its solver id for the next step's events.
static void retireShape(Avbd2dWorld *w, int slot)
{
    const int solver = w->shapes.items[(size_t)slot].solver;
    if (solver >= 0 && solver < (int)w->solverShape.size())
    {
        tombstone(w, solver, shapeIdOf(w, slot));
        w->solverShape[(size_t)solver] = -1;
    }
    w->shapes.release(slot);
}

// A shape edit on a live body: its shape range is replaced at the next step.
static void markReshape(Avbd2dWorld *w, int bodySlot)
{
    BodyRec &b = w->bodies.items[(size_t)bodySlot];
    b.propsValid = false;
    if (b.flushed && b.solver >= 0 && !b.reshape)
    {
        b.reshape = true;
        w->replaceBodies.push_back(bodySlot);
    }
}

// Adds a shape record to a body (a live body gets it at the next step).
static Avbd2dResult addShapeRec(Avbd2dBodyId bid, const Avbd2dShapeDef *def, const ShapeRec &proto,
                                Avbd2dShapeId *out_id)
{
    BODY_CTX(bid);
    NEED(out_id);
    if (!validShapeDef(def))
        return AVBD2D_ERR_INVALID_ARG;
    return guarded(w, true, [&]() -> Avbd2dResult {
        const int s = w->shapes.acquire();
        ShapeRec &r = w->shapes.items[(size_t)s];
        const uint16_t gen = r.gen;
        r = proto;
        r.alive = true;
        r.gen = gen;
        r.body = slot;
        r.def = *def;
        b->shapes.push_back(s);
        markReshape(w, slot);
        *out_id = shapeIdOf(w, s);
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_create_polygon_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                    const Avbd2dPolygon *polygon, Avbd2dShapeId *out_id)
{
    NEED(polygon);
    if (polygon->count < 3 || polygon->count > AVBD2D_MAX_POLYGON_VERTICES || !(polygon->radius >= 0.0f))
        return AVBD2D_ERR_INVALID_ARG;
    Vec2 in[AVBD2D_MAX_POLYGON_VERTICES], hull[AVBD2D_MAX_POLYGON_VERTICES];
    for (int i = 0; i < polygon->count; i++)
    {
        if (!finite2(polygon->vertices[i]))
            return AVBD2D_ERR_INVALID_ARG;
        in[i] = V(polygon->vertices[i]);
    }
    if (Scene2D::computeHull(in, polygon->count, hull) < 3)
        return AVBD2D_ERR_INVALID_ARG;
    ShapeRec r;
    r.kind = SK_POLYGON;
    r.poly = *polygon;
    return addShapeRec(body, def, r, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_circle_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                   const Avbd2dCircle *circle, Avbd2dShapeId *out_id)
{
    NEED(circle);
    if (!(circle->radius > 0.0f) || !finite2(circle->center))
        return AVBD2D_ERR_INVALID_ARG;
    ShapeRec r;
    r.kind = SK_CIRCLE;
    r.circle = *circle;
    return addShapeRec(body, def, r, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_capsule_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                    const Avbd2dCapsule *capsule, Avbd2dShapeId *out_id)
{
    NEED(capsule);
    if (!(capsule->radius > 0.0f) || !finite2(capsule->center1) || !finite2(capsule->center2))
        return AVBD2D_ERR_INVALID_ARG;
    ShapeRec r;
    r.kind = SK_CAPSULE;
    r.capsule = *capsule;
    return addShapeRec(body, def, r, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_segment_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                    const Avbd2dSegment *segment, Avbd2dShapeId *out_id)
{
    NEED(segment);
    if (!finite2(segment->point1) || !finite2(segment->point2))
        return AVBD2D_ERR_INVALID_ARG;
    ShapeRec r;
    r.kind = SK_SEGMENT;
    r.seg = *segment;
    return addShapeRec(body, def, r, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_chain(Avbd2dBodyId body, const Avbd2dChainDef *def, Avbd2dChainId *out_id)
{
    BODY_CTX(body);
    NEED(def);
    NEED(out_id);
    if (!def->points || def->count < (def->isLoop ? 3 : 2))
        return AVBD2D_ERR_INVALID_ARG;
    for (int i = 0; i < def->count; i++)
        if (!finite2(def->points[i]))
            return AVBD2D_ERR_INVALID_ARG;
    const int count = def->count;
    const int n = def->isLoop ? count : count - 1;
    const Avbd2dVec2 *pts = def->points;
    auto P = [&](int i) -> Vec2 {
        if (def->isLoop)
            return V(pts[((i % count) + count) % count]);
        if (i < 0)
            return V(pts[0]) * 2.0f - V(pts[1]);
        if (i >= count)
            return V(pts[count - 1]) * 2.0f - V(pts[count - 2]);
        return V(pts[i]);
    };
    return guarded(w, true, [&]() -> Avbd2dResult {
        Avbd2dShapeDef sd = avbd2d_default_shape_def();
        sd.friction = def->friction;
        sd.restitution = def->restitution;
        sd.filter = def->filter;
        sd.userData = def->userData;
        sd.density = 0.0f;
        int first = -1;
        for (int i = 0; i < n; i++)
        {
            const int s = w->shapes.acquire();
            ShapeRec &r = w->shapes.items[(size_t)s];
            r.body = slot;
            r.kind = SK_CHAIN;
            r.chain[0] = P(i - 1);
            r.chain[1] = P(i);
            r.chain[2] = P(i + 1);
            r.chain[3] = P(i + 2);
            r.def = sd;
            b->shapes.push_back(s);
            if (first < 0)
                first = s;
        }
        markReshape(w, slot);
        *out_id = Avbd2dChainId{first + 1, w->id, w->shapes.items[(size_t)first].gen};
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_destroy_shape(Avbd2dShapeId id)
{
    SHAPE_CTX(id);
    const int bodySlot = s->body;
    BodyRec &b = w->bodies.items[(size_t)bodySlot];
    b.shapes.erase(std::remove(b.shapes.begin(), b.shapes.end(), slot), b.shapes.end());
    retireShape(w, slot);
    markReshape(w, bodySlot);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_destroy_chain(Avbd2dChainId id)
{
    ID_WORLD(id);
    if (!avbd2d_chain_is_valid(id))
        return AVBD2D_ERR_INVALID_ARG;
    // A chain is its run of consecutive chain-segment shapes on one body, starting at `id`.
    const int first = id.index1 - 1;
    const int bodySlot = w->shapes.items[(size_t)first].body;
    BodyRec &b = w->bodies.items[(size_t)bodySlot];
    std::vector<int> run;
    size_t at = 0;
    while (at < b.shapes.size() && b.shapes[at] != first)
        at++;
    // The segments of one chain were created together, so they are adjacent in the body's list.
    for (size_t k = at; k < b.shapes.size() && w->shapes.items[(size_t)b.shapes[k]].kind == SK_CHAIN; k++)
        run.push_back(b.shapes[k]);
    for (int s : run)
    {
        b.shapes.erase(std::remove(b.shapes.begin(), b.shapes.end(), s), b.shapes.end());
        retireShape(w, s);
    }
    markReshape(w, bodySlot);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_shape_get_body(Avbd2dShapeId id, Avbd2dBodyId *out)
{
    SHAPE_CTX(id);
    NEED(out);
    *out = bodyIdOf(w, s->body);
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_shape_get_user_data(Avbd2dShapeId id, void **out)
{
    SHAPE_CTX(id);
    NEED(out);
    *out = s->def.userData;
    return AVBD2D_OK;
}

// ----- Joints ----------------------------------------------------------------------------------
static Avbd2dResult makeJoint(Avbd2dWorld *world, Avbd2dBodyId ida, Avbd2dBodyId idb, JointDef d, float breakForce,
                              void *userData, Avbd2dJointId *out_id)
{
    AVBD2D_GET_WORLD(world);
    NEED(out_id);
    if (ida.world0 != w->id || idb.world0 != w->id)
        return AVBD2D_ERR_INVALID_ARG;
    if (!w->bodies.find(ida.index1, ida.generation) || !w->bodies.find(idb.index1, idb.generation) ||
        ida.index1 == idb.index1)
        return AVBD2D_ERR_INVALID_ARG;
    if (!(breakForce > 0.0f))
        return AVBD2D_ERR_INVALID_ARG;
    d.breakForce = breakForce >= 1.0e38f ? kStiffInf : breakForce;
    return guarded(w, true, [&]() -> Avbd2dResult {
        const int slot = w->joints.acquire();
        JointRec &j = w->joints.items[(size_t)slot];
        j.d = d;
        j.bodyA = ida.index1 - 1;
        j.bodyB = idb.index1 - 1;
        j.userData = userData;
        w->bodies.items[(size_t)j.bodyA].joints.push_back(slot);
        w->bodies.items[(size_t)j.bodyB].joints.push_back(slot);
        w->pendingJoints.push_back(slot);
        *out_id = jointIdOf(w, slot);
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_create_revolute_joint(Avbd2dWorld *world, const Avbd2dRevoluteJointDef *def,
                                                     Avbd2dJointId *out_id)
{
    NEED(def);
    if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !finite1(def->referenceAngle))
        return AVBD2D_ERR_INVALID_ARG;
    JointDef d;
    d.type = JointType::Revolute;
    d.localAnchorA = V(def->localAnchorA);
    d.localAnchorB = V(def->localAnchorB);
    d.referenceAngle = def->referenceAngle;
    d.enableSpring = def->enableSpring != 0;
    d.hertz = def->hertz;
    d.dampingRatio = def->dampingRatio;
    d.enableLimit = def->enableLimit != 0;
    d.lower = def->lowerAngle;
    d.upper = def->upperAngle;
    d.enableMotor = def->enableMotor != 0;
    d.motorSpeed = def->motorSpeed;
    d.maxMotorTorque = def->maxMotorTorque;
    d.collideConnected = def->collideConnected != 0;
    return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_weld_joint(Avbd2dWorld *world, const Avbd2dWeldJointDef *def,
                                                 Avbd2dJointId *out_id)
{
    NEED(def);
    if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !finite1(def->referenceAngle))
        return AVBD2D_ERR_INVALID_ARG;
    JointDef d;
    d.type = JointType::Weld;
    d.localAnchorA = V(def->localAnchorA);
    d.localAnchorB = V(def->localAnchorB);
    d.referenceAngle = def->referenceAngle;
    d.linearHertz = def->linearHertz;
    d.angularHertz = def->angularHertz;
    d.linearDampingRatio = def->linearDampingRatio;
    d.angularDampingRatio = def->angularDampingRatio;
    d.collideConnected = def->collideConnected != 0;
    return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_prismatic_joint(Avbd2dWorld *world, const Avbd2dPrismaticJointDef *def,
                                                      Avbd2dJointId *out_id)
{
    NEED(def);
    if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !finite2(def->localAxisA) ||
        (def->localAxisA.x == 0.0f && def->localAxisA.y == 0.0f) || !finite1(def->referenceAngle))
        return AVBD2D_ERR_INVALID_ARG;
    JointDef d;
    d.type = JointType::Prismatic;
    d.localAnchorA = V(def->localAnchorA);
    d.localAnchorB = V(def->localAnchorB);
    d.localAxisA = V(def->localAxisA);
    d.referenceAngle = def->referenceAngle;
    d.enableSpring = def->enableSpring != 0;
    d.hertz = def->hertz;
    d.dampingRatio = def->dampingRatio;
    d.enableLimit = def->enableLimit != 0;
    d.lower = def->lowerTranslation;
    d.upper = def->upperTranslation;
    d.enableMotor = def->enableMotor != 0;
    d.motorSpeed = def->motorSpeed;
    d.maxMotorForce = def->maxMotorForce;
    d.collideConnected = def->collideConnected != 0;
    return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_wheel_joint(Avbd2dWorld *world, const Avbd2dWheelJointDef *def,
                                                  Avbd2dJointId *out_id)
{
    NEED(def);
    if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !finite2(def->localAxisA) ||
        (def->localAxisA.x == 0.0f && def->localAxisA.y == 0.0f))
        return AVBD2D_ERR_INVALID_ARG;
    JointDef d;
    d.type = JointType::Wheel;
    d.localAnchorA = V(def->localAnchorA);
    d.localAnchorB = V(def->localAnchorB);
    d.localAxisA = V(def->localAxisA);
    d.enableSpring = def->enableSpring != 0;
    d.hertz = def->hertz;
    d.dampingRatio = def->dampingRatio;
    d.enableLimit = def->enableLimit != 0;
    d.lower = def->lowerTranslation;
    d.upper = def->upperTranslation;
    d.enableMotor = def->enableMotor != 0;
    d.motorSpeed = def->motorSpeed;
    d.maxMotorTorque = def->maxMotorTorque;
    d.collideConnected = def->collideConnected != 0;
    return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_distance_joint(Avbd2dWorld *world, const Avbd2dDistanceJointDef *def,
                                                     Avbd2dJointId *out_id)
{
    NEED(def);
    if (!finite2(def->localAnchorA) || !finite2(def->localAnchorB) || !(def->length >= 0.0f))
        return AVBD2D_ERR_INVALID_ARG;
    JointDef d;
    d.type = JointType::Distance;
    d.localAnchorA = V(def->localAnchorA);
    d.localAnchorB = V(def->localAnchorB);
    d.length = def->length;
    d.enableSpring = def->enableSpring != 0;
    d.hertz = def->hertz;
    d.dampingRatio = def->dampingRatio;
    d.enableLimit = def->enableLimit != 0;
    d.minLength = def->minLength;
    d.maxLength = def->maxLength;
    d.enableMotor = def->enableMotor != 0;
    d.motorSpeed = def->motorSpeed;
    d.maxMotorForce = def->maxMotorForce;
    d.collideConnected = def->collideConnected != 0;
    return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_motor_joint(Avbd2dWorld *world, const Avbd2dMotorJointDef *def,
                                                  Avbd2dJointId *out_id)
{
    NEED(def);
    if (!finite2(def->linearOffset) || !finite1(def->angularOffset))
        return AVBD2D_ERR_INVALID_ARG;
    JointDef d;
    d.type = JointType::Motor;
    d.linearOffset = V(def->linearOffset);
    d.angularOffset = def->angularOffset;
    d.maxMotorForce = def->maxForce;
    d.maxMotorTorque = def->maxTorque;
    d.collideConnected = def->collideConnected != 0;
    return makeJoint(world, def->bodyIdA, def->bodyIdB, d, def->breakForce, def->userData, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_create_filter_joint(Avbd2dWorld *world, const Avbd2dFilterJointDef *def,
                                                   Avbd2dJointId *out_id)
{
    NEED(def);
    JointDef d;
    d.type = JointType::Filter;
    return makeJoint(world, def->bodyIdA, def->bodyIdB, d, 3.4e38f, def->userData, out_id);
}

AVBD2D_API Avbd2dResult avbd2d_destroy_joint(Avbd2dJointId id)
{
    JOINT_CTX(id);
    return guarded(w, false, [&]() -> Avbd2dResult {
        destroyJointSlot(w, slot);
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_joint_get_user_data(Avbd2dJointId id, void **out)
{
    JOINT_CTX(id);
    NEED(out);
    *out = j->userData;
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_joint_get_constraint_force(Avbd2dJointId id, Avbd2dVec2 *out)
{
    JOINT_CTX(id);
    NEED(out);
    float f[3] = {};
    if (j->flushed && j->solver >= 0)
        w->world->jointForce(j->solver, f);
    *out = Avbd2dVec2{f[0], f[1]};
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_joint_get_constraint_torque(Avbd2dJointId id, float *out)
{
    JOINT_CTX(id);
    NEED(out);
    float f[3] = {};
    if (j->flushed && j->solver >= 0)
        w->world->jointForce(j->solver, f);
    *out = f[2];
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_joint_wake_bodies(Avbd2dJointId id)
{
    JOINT_CTX(id);
    if (j->flushed && j->solver >= 0)
        w->world->queueJointWake(j->solver);
    return AVBD2D_OK;
}

static void syncMotor(Avbd2dWorld *w, const JointRec &j)
{
    if (j.flushed && j.solver >= 0)
        w->world->queueJointMotor(j.solver, motorAxis(j.d.type), j.d.enableMotor ? 1u : 0u, j.d.motorSpeed,
                                  maxForceOf(j.d));
}

static void syncLimit(Avbd2dWorld *w, const JointRec &j)
{
    if (j.flushed && j.solver >= 0)
        w->world->queueJointLimit(j.solver, limitAxis(j.d.type), j.d.enableLimit, j.d.lower, j.d.upper);
}

#define JOINT_SETTER(name, TYPE, ARGDECL, BODY, SYNC)                          \
    AVBD2D_API Avbd2dResult name(Avbd2dJointId id ARGDECL)                      \
    {                                                                          \
        JOINT_CTX(id);                                                         \
        if (j->d.type != JointType::TYPE)                                      \
            return AVBD2D_ERR_INVALID_ARG;                                     \
        BODY;                                                                  \
        SYNC(w, *j);                                                           \
        return AVBD2D_OK;                                                      \
    }

#define COMMA ,
JOINT_SETTER(avbd2d_revolute_joint_enable_limit, Revolute, COMMA int32_t flag, j->d.enableLimit = flag != 0, syncLimit)
JOINT_SETTER(avbd2d_revolute_joint_set_limits, Revolute, COMMA float lower COMMA float upper,
             (j->d.lower = lower, j->d.upper = upper), syncLimit)
JOINT_SETTER(avbd2d_revolute_joint_enable_motor, Revolute, COMMA int32_t flag, j->d.enableMotor = flag != 0, syncMotor)
JOINT_SETTER(avbd2d_revolute_joint_set_motor_speed, Revolute, COMMA float speed, j->d.motorSpeed = speed, syncMotor)
JOINT_SETTER(avbd2d_revolute_joint_set_max_motor_torque, Revolute, COMMA float torque, j->d.maxMotorTorque = torque,
             syncMotor)

JOINT_SETTER(avbd2d_prismatic_joint_enable_limit, Prismatic, COMMA int32_t flag, j->d.enableLimit = flag != 0,
             syncLimit)
JOINT_SETTER(avbd2d_prismatic_joint_set_limits, Prismatic, COMMA float lower COMMA float upper,
             (j->d.lower = lower, j->d.upper = upper), syncLimit)
JOINT_SETTER(avbd2d_prismatic_joint_enable_motor, Prismatic, COMMA int32_t flag, j->d.enableMotor = flag != 0,
             syncMotor)
JOINT_SETTER(avbd2d_prismatic_joint_set_motor_speed, Prismatic, COMMA float speed, j->d.motorSpeed = speed, syncMotor)
JOINT_SETTER(avbd2d_prismatic_joint_set_max_motor_force, Prismatic, COMMA float force, j->d.maxMotorForce = force,
             syncMotor)

JOINT_SETTER(avbd2d_wheel_joint_enable_limit, Wheel, COMMA int32_t flag, j->d.enableLimit = flag != 0, syncLimit)
JOINT_SETTER(avbd2d_wheel_joint_set_limits, Wheel, COMMA float lower COMMA float upper,
             (j->d.lower = lower, j->d.upper = upper), syncLimit)
JOINT_SETTER(avbd2d_wheel_joint_enable_motor, Wheel, COMMA int32_t flag, j->d.enableMotor = flag != 0, syncMotor)
JOINT_SETTER(avbd2d_wheel_joint_set_motor_speed, Wheel, COMMA float speed, j->d.motorSpeed = speed, syncMotor)
JOINT_SETTER(avbd2d_wheel_joint_set_max_motor_torque, Wheel, COMMA float torque, j->d.maxMotorTorque = torque,
             syncMotor)

JOINT_SETTER(avbd2d_distance_joint_enable_limit, Distance, COMMA int32_t flag, j->d.enableLimit = flag != 0, syncLimit)
JOINT_SETTER(avbd2d_distance_joint_set_length_range, Distance, COMMA float minLength COMMA float maxLength,
             (j->d.minLength = minLength, j->d.maxLength = maxLength), syncLimit)
JOINT_SETTER(avbd2d_distance_joint_enable_motor, Distance, COMMA int32_t flag, j->d.enableMotor = flag != 0,
             syncMotor)
JOINT_SETTER(avbd2d_distance_joint_set_motor_speed, Distance, COMMA float speed, j->d.motorSpeed = speed, syncMotor)
JOINT_SETTER(avbd2d_distance_joint_set_max_motor_force, Distance, COMMA float force, j->d.maxMotorForce = force,
             syncMotor)

// ----- Events ----------------------------------------------------------------------------------
AVBD2D_API Avbd2dResult avbd2d_world_get_body_events(Avbd2dWorld *world, Avbd2dBodyEvents *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    out->moveEvents = w->bodyEvents.empty() ? nullptr : w->bodyEvents.data();
    out->moveCount = (int32_t)w->bodyEvents.size();
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_world_get_contact_events(Avbd2dWorld *world, Avbd2dContactEvents *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    out->beginEvents = w->beginEvents.empty() ? nullptr : w->beginEvents.data();
    out->endEvents = w->endEvents.empty() ? nullptr : w->endEvents.data();
    out->hitEvents = w->hitEvents.empty() ? nullptr : w->hitEvents.data();
    out->beginCount = (int32_t)w->beginEvents.size();
    out->endCount = (int32_t)w->endEvents.size();
    out->hitCount = (int32_t)w->hitEvents.size();
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_world_get_sensor_events(Avbd2dWorld *world, Avbd2dSensorEvents *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    out->beginEvents = w->sensorBegin.empty() ? nullptr : w->sensorBegin.data();
    out->endEvents = w->sensorEnd.empty() ? nullptr : w->sensorEnd.data();
    out->beginCount = (int32_t)w->sensorBegin.size();
    out->endCount = (int32_t)w->sensorEnd.size();
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_world_get_joint_events(Avbd2dWorld *world, Avbd2dJointEvents *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    out->jointEvents = w->jointEvents.empty() ? nullptr : w->jointEvents.data();
    out->count = (int32_t)w->jointEvents.size();
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_world_get_transforms(Avbd2dWorld *world, Avbd2dBodyTransform *out, int32_t capacity,
                                                    int32_t *out_count)
{
    AVBD2D_GET_WORLD(world);
    if (!out && capacity > 0)
        return AVBD2D_ERR_INVALID_ARG;
    int32_t n = 0;
    for (size_t i = 0; i < w->bodies.items.size(); i++)
    {
        BodyRec &b = w->bodies.items[i];
        if (!b.alive)
            continue;
        if (n < capacity)
        {
            const Pose p = poseOf(w, b);
            out[n].bodyId = bodyIdOf(w, (int)i);
            out[n].transform.p = A(p.origin());
            out[n].transform.q = live(b) ? rotOf(p.angle) : b.def.rotation;
        }
        n++;
    }
    if (out_count)
        *out_count = n;
    return AVBD2D_OK;
}

// ----- Explosions and queries ------------------------------------------------------------------
AVBD2D_API Avbd2dResult avbd2d_world_explode(Avbd2dWorld *world, const Avbd2dExplosionDef *def)
{
    AVBD2D_GET_WORLD(world);
    NEED(def);
    if (!finite2(def->position) || !(def->radius >= 0.0f) || !(def->falloff >= 0.0f) ||
        !finite1(def->impulsePerLength))
        return AVBD2D_ERR_INVALID_ARG;
    settle(w);
    return guarded(w, false, [&]() -> Avbd2dResult {
        w->world->explode(V(def->position), def->radius, def->falloff, def->impulsePerLength);
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_world_cast_ray_closest(Avbd2dWorld *world, Avbd2dVec2 origin, Avbd2dVec2 translation,
                                                      Avbd2dQueryFilter filter, Avbd2dRayResult *out)
{
    AVBD2D_GET_WORLD(world);
    NEED(out);
    if (!finite2(origin) || !finite2(translation))
        return AVBD2D_ERR_INVALID_ARG;
    *out = Avbd2dRayResult{};
    return guarded(w, false, [&]() -> Avbd2dResult {
        w->world->castRay(V(origin), V(translation), [&](const RayHit2D &h) -> float {
            int slot;
            const ShapeRec *s = queryShape(w, h.shape, &slot);
            if (!s || !passes(*s, filter))
                return -1.0f;
            out->hit = 1;
            out->shapeId = shapeIdOf(w, slot);
            out->point = A(h.point);
            out->normal = A(h.normal);
            out->fraction = h.fraction;
            return h.fraction;
        });
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_world_cast_ray(Avbd2dWorld *world, Avbd2dVec2 origin, Avbd2dVec2 translation,
                                              Avbd2dQueryFilter filter, Avbd2dCastResultFcn fcn, void *context)
{
    AVBD2D_GET_WORLD(world);
    NEED(fcn);
    if (!finite2(origin) || !finite2(translation))
        return AVBD2D_ERR_INVALID_ARG;
    return guarded(w, false, [&]() -> Avbd2dResult {
        w->world->castRay(V(origin), V(translation), [&](const RayHit2D &h) -> float {
            int slot;
            const ShapeRec *s = queryShape(w, h.shape, &slot);
            if (!s || !passes(*s, filter))
                return -1.0f;
            return fcn(shapeIdOf(w, slot), A(h.point), A(h.normal), h.fraction, context);
        });
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_world_cast_rays_batch(Avbd2dWorld *world, const Avbd2dRayQuery *rays, int32_t count,
                                                     Avbd2dQueryFilter filter, Avbd2dRayResult *out)
{
    AVBD2D_GET_WORLD(world);
    if (count < 0)
        return AVBD2D_ERR_INVALID_ARG;
    if (count == 0)
        return AVBD2D_OK;
    NEED(rays);
    NEED(out);
    for (int32_t i = 0; i < count; i++)
        if (!finite2(rays[i].origin) || !finite2(rays[i].translation))
            return AVBD2D_ERR_INVALID_ARG;
    return guarded(w, false, [&]() -> Avbd2dResult {
        std::vector<VkWorld2D::RayQuery2D> q((size_t)count);
        std::vector<VkWorld2D::RayBatchHit2D> hits((size_t)count);
        for (int32_t i = 0; i < count; i++)
        {
            q[(size_t)i].origin = V(rays[i].origin);
            q[(size_t)i].translation = V(rays[i].translation);
        }
        w->world->castRaysBatch(q.data(), count, filter.categoryBits, filter.maskBits, hits.data());
        for (int32_t i = 0; i < count; i++)
        {
            out[i] = Avbd2dRayResult{};
            const VkWorld2D::RayBatchHit2D &h = hits[(size_t)i];
            int slot;
            const ShapeRec *s = h.shape >= 0 ? queryShape(w, h.shape, &slot) : nullptr;
            if (h.shape >= 0 && (!s || !passes(*s, filter)))
            {
                // A shape the ABI hides (destroyed before its body flushed): the answer for this
                // ray is the closest visible one, which only the host query can walk past.
                w->world->castRay(q[(size_t)i].origin, q[(size_t)i].translation, [&](const RayHit2D &r) -> float {
                    int sl;
                    const ShapeRec *sr = queryShape(w, r.shape, &sl);
                    if (!sr || !passes(*sr, filter))
                        return -1.0f;
                    out[i].hit = 1;
                    out[i].shapeId = shapeIdOf(w, sl);
                    out[i].point = A(r.point);
                    out[i].normal = A(r.normal);
                    out[i].fraction = r.fraction;
                    return r.fraction;
                });
                continue;
            }
            if (h.shape < 0)
                continue;
            out[i].hit = 1;
            out[i].shapeId = shapeIdOf(w, slot);
            out[i].point = A(h.point);
            out[i].normal = A(h.normal);
            out[i].fraction = h.fraction;
        }
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_world_overlap_aabbs_batch(Avbd2dWorld *world, const Avbd2dAABB *aabbs, int32_t count,
                                                         Avbd2dQueryFilter filter, int32_t maxPerQuery,
                                                         Avbd2dShapeId *outShapeIds, int32_t *outCounts)
{
    AVBD2D_GET_WORLD(world);
    if (count < 0 || maxPerQuery < 0)
        return AVBD2D_ERR_INVALID_ARG;
    if (count == 0)
        return AVBD2D_OK;
    NEED(aabbs);
    NEED(outCounts);
    if (maxPerQuery > 0)
        NEED(outShapeIds);
    for (int32_t i = 0; i < count; i++)
        if (!finite2(aabbs[i].lowerBound) || !finite2(aabbs[i].upperBound))
            return AVBD2D_ERR_INVALID_ARG;
    return guarded(w, false, [&]() -> Avbd2dResult {
        const int per = std::max(maxPerQuery, 1);
        std::vector<Vec2> lo((size_t)count), hi((size_t)count);
        for (int32_t i = 0; i < count; i++)
        {
            lo[(size_t)i] = V(aabbs[i].lowerBound);
            hi[(size_t)i] = V(aabbs[i].upperBound);
        }
        std::vector<int> ids((size_t)count * (size_t)per), counts((size_t)count);
        w->world->overlapAabbsBatch(lo.data(), hi.data(), count, filter.categoryBits, filter.maskBits, per, ids.data(),
                                    counts.data());
        for (int32_t i = 0; i < count; i++)
        {
            int n = 0;
            bool dropped = false;
            const int got = std::min(counts[(size_t)i], per);
            for (int k = 0; k < got; k++)
            {
                int slot;
                const ShapeRec *s = queryShape(w, ids[(size_t)i * (size_t)per + (size_t)k], &slot);
                if (!s || !passes(*s, filter))
                {
                    dropped = true;
                    continue;
                }
                if (n < maxPerQuery)
                    outShapeIds[(size_t)i * (size_t)maxPerQuery + (size_t)n] = shapeIdOf(w, slot);
                n++;
            }
            if (dropped && counts[(size_t)i] > per)
            {
                // The device list was cut and some of it is hidden by the ABI: count this box on the host.
                n = 0;
                w->world->overlapAABB(lo[(size_t)i], hi[(size_t)i], [&](int shape) -> bool {
                    int slot;
                    const ShapeRec *s = queryShape(w, shape, &slot);
                    if (!s || !passes(*s, filter))
                        return true;
                    if (n < maxPerQuery)
                        outShapeIds[(size_t)i * (size_t)maxPerQuery + (size_t)n] = shapeIdOf(w, slot);
                    n++;
                    return true;
                });
            }
            else if (counts[(size_t)i] > per)
                n += counts[(size_t)i] - per;
            outCounts[i] = n;
        }
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_world_overlap_aabb(Avbd2dWorld *world, Avbd2dAABB aabb, Avbd2dQueryFilter filter,
                                                  Avbd2dOverlapResultFcn fcn, void *context)
{
    AVBD2D_GET_WORLD(world);
    NEED(fcn);
    if (!finite2(aabb.lowerBound) || !finite2(aabb.upperBound))
        return AVBD2D_ERR_INVALID_ARG;
    return guarded(w, false, [&]() -> Avbd2dResult {
        w->world->overlapAABB(V(aabb.lowerBound), V(aabb.upperBound), [&](int shape) -> bool {
            int slot;
            const ShapeRec *s = queryShape(w, shape, &slot);
            if (!s || !passes(*s, filter))
                return true;
            return fcn(shapeIdOf(w, slot), context) != 0;
        });
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_world_overlap_shape(Avbd2dWorld *world, const Avbd2dShapeProxy *proxy,
                                                   Avbd2dQueryFilter filter, Avbd2dOverlapResultFcn fcn,
                                                   void *context)
{
    AVBD2D_GET_WORLD(world);
    NEED(fcn);
    if (!validProxy(proxy))
        return AVBD2D_ERR_INVALID_ARG;
    const Proxy2D p = toProxy(*proxy);
    return guarded(w, false, [&]() -> Avbd2dResult {
        w->world->overlapShape(p, [&](int shape) -> bool {
            int slot;
            const ShapeRec *s = queryShape(w, shape, &slot);
            if (!s || !passes(*s, filter))
                return true;
            return fcn(shapeIdOf(w, slot), context) != 0;
        });
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_world_cast_shape(Avbd2dWorld *world, const Avbd2dShapeProxy *proxy,
                                                Avbd2dVec2 translation, Avbd2dQueryFilter filter,
                                                Avbd2dCastResultFcn fcn, void *context)
{
    AVBD2D_GET_WORLD(world);
    NEED(fcn);
    if (!validProxy(proxy) || !finite2(translation))
        return AVBD2D_ERR_INVALID_ARG;
    const Proxy2D p = toProxy(*proxy);
    return guarded(w, false, [&]() -> Avbd2dResult {
        w->world->castShape(p, V(translation), [&](const RayHit2D &h) -> float {
            int slot;
            const ShapeRec *s = queryShape(w, h.shape, &slot);
            if (!s || !passes(*s, filter))
                return -1.0f;
            return fcn(shapeIdOf(w, slot), A(h.point), A(h.normal), h.fraction, context);
        });
        return AVBD2D_OK;
    });
}

// ----- Mouse grab ------------------------------------------------------------------------------
AVBD2D_API Avbd2dResult avbd2d_grab_begin(Avbd2dWorld *world, Avbd2dVec2 point, Avbd2dBodyId *out_body)
{
    AVBD2D_GET_WORLD(world);
    NEED(out_body);
    if (!finite2(point))
        return AVBD2D_ERR_INVALID_ARG;
    *out_body = Avbd2dBodyId{};
    settle(w);
    return guarded(w, false, [&]() -> Avbd2dResult {
        flush(w);
        if (w->world->bodyCount() == 0)
            return AVBD2D_OK;
        if (w->world->grabBegin(V(point)))
        {
            const int sb = w->world->grabbedBody();
            if (sb >= 0 && sb < (int)w->solverBody.size() && w->solverBody[(size_t)sb] >= 0)
                *out_body = bodyIdOf(w, w->solverBody[(size_t)sb]);
        }
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_grab_move(Avbd2dWorld *world, Avbd2dVec2 point)
{
    AVBD2D_GET_WORLD(world);
    if (!finite2(point))
        return AVBD2D_ERR_INVALID_ARG;
    if (w->world->bodyCount() > 0)
        w->world->grabMove(V(point));
    return AVBD2D_OK;
}

AVBD2D_API Avbd2dResult avbd2d_grab_end(Avbd2dWorld *world)
{
    AVBD2D_GET_WORLD(world);
    settle(w);
    return guarded(w, false, [&]() -> Avbd2dResult {
        if (w->world->bodyCount() > 0)
            w->world->grabEnd();
        return AVBD2D_OK;
    });
}

AVBD2D_API Avbd2dResult avbd2d_register_render_target(Avbd2dWorld *world, void *native_handle)
{
    (void)native_handle;
    AVBD2D_GET_WORLD(world);
    return AVBD2D_ERR_UNSUPPORTED;
}

} // extern "C"
