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


#include "avbd2d_world.h"

namespace avbd2d_capi
{
std::mutex g_mutex;

uint16_t g_nextWorldId = 0;


std::unordered_set<Avbd2dWorld *> &registry()
{
    static std::unordered_set<Avbd2dWorld *> r;
    return r;
}


// World index -> world, read without the lock by every id call. Written only under g_mutex, by
// create (after the world is complete) and destroy (before it is deleted).
std::atomic<Avbd2dWorld *> g_worldTable[65536];


Avbd2dWorld *lookup(Avbd2dWorld *w)
{
    std::lock_guard<std::mutex> lk(g_mutex);
    return registry().count(w) ? w : nullptr;
}


Avbd2dWorld *worldFromId(uint16_t id)
{
    return g_worldTable[id].load(std::memory_order_acquire);
}



bool finite1(float a)
{
    return std::isfinite(a);
}

bool finite2(Avbd2dVec2 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y);
}

Vec2 V(Avbd2dVec2 v)
{
    return Vec2(v.x, v.y);
}

Avbd2dVec2 A(Vec2 v)
{
    return Avbd2dVec2{v.x, v.y};
}

float angleOf(Avbd2dRot q)
{
    return std::atan2(q.s, q.c);
}

Avbd2dRot rotOf(float a)
{
    return Avbd2dRot{std::cos(a), std::sin(a)};
}


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
    b.type = d.type == AVBD2D_STATIC_BODY      ? BODY2D_STATIC
             : d.type == AVBD2D_KINEMATIC_BODY ? BODY2D_KINEMATIC
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


bool live(const BodyRec &b)
{
    return b.flushed && b.solver >= 0;
}


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


// The body's state view entry is rebuilt from its record at the next refresh (create, destroy,
// teleport, and every change the moved list does not report).
void markDirty(Avbd2dWorld *w, int slot)
{
    if ((int)w->viewDirtyMark.size() <= slot)
        w->viewDirtyMark.resize(w->bodies.items.size() > (size_t)slot ? w->bodies.items.size() : (size_t)slot + 1, 0);
    if (w->viewDirtyMark[(size_t)slot])
        return;
    w->viewDirtyMark[(size_t)slot] = 1;
    w->viewDirty.push_back(slot);
}


// ----- Flush: pending bodies and joints become part of the live world ---------------------------
int motorAxis(JointType t)
{
    return (t == JointType::Revolute || t == JointType::Wheel) ? 2 : 0;
}

int limitAxis(JointType t)
{
    return t == JointType::Revolute ? 2 : 0;
}

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
    d.lockScale = j.lockScale;
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
        if (j.lockScale > 1.0f)
            vw.queueJointLockScale(j.solver, j.lockScale);
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
        Scene2D tmp; // every rebuilt and new body of this flush is authored here, then installed
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
    // The id of a solver shape: its live shape, else the id it had when it was destroyed or replaced.
    auto shapeSlot = [&](int s) -> int
    { return (s >= 0 && s < (int)w->solverShape.size()) ? w->solverShape[(size_t)s] : -1; };
    auto shapeId = [&](int s, Avbd2dShapeId &out) -> bool
    {
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


// ----- Body state view -------------------------------------------------------------------------
uint32_t baseFlags(const BodyRec &b)
{
    uint32_t f = AVBD2D_BODY_ALIVE | ((uint32_t)b.def.type << AVBD2D_BODY_TYPE_SHIFT);
    if (b.def.isEnabled)
        f |= AVBD2D_BODY_ENABLED;
    if (!b.flushed)
        f |= AVBD2D_BODY_PENDING;
    if (b.removed)
        f |= AVBD2D_BODY_REMOVED;
    return f;
}


// One slot's entry from its record and the host mirrors.
void writeViewSlot(Avbd2dWorld *w, int slot)
{
    const size_t i = (size_t)slot;
    BodyRec &b = w->bodies.items[i];
    w->viewGen[i] = b.gen;
    w->viewUserData[i] = b.alive ? b.def.userData : nullptr;
    if (!b.alive)
    {
        w->viewXf[i] = Avbd2dTransform{};
        w->viewVel[i] = Avbd2dBodyVelocity{};
        w->viewFlags[i] = 0;
        return;
    }
    const Pose p = poseOf(w, b);
    uint32_t f = baseFlags(b);
    if (live(b))
    {
        const size_t s = (size_t)b.solver;
        w->viewXf[i] = Avbd2dTransform{A(p.origin()), rotOf(p.angle)};
        w->viewVel[i] = Avbd2dBodyVelocity{A(w->world->velocities()[s]), w->world->angularVelocities()[s], 0.0f};
        if (w->world->awakeFlags()[s] && !b.removed)
            f |= AVBD2D_BODY_AWAKE;
    }
    else
    {
        w->viewXf[i] = Avbd2dTransform{A(p.origin()), b.def.rotation};
        w->viewVel[i] = Avbd2dBodyVelocity{b.def.linearVelocity, b.def.angularVelocity, 0.0f};
        if (b.def.isAwake && b.def.isEnabled && b.def.type != AVBD2D_STATIC_BODY)
            f |= AVBD2D_BODY_AWAKE;
    }
    w->viewFlags[i] = f | (w->viewFlags[i] & AVBD2D_BODY_GRABBED);
}


// Brings the view up to date: the moved list of a step that just ended, the grab set, then every
// dirty slot. Called only from the step calls, which is what makes the view stable between them.
void refreshView(Avbd2dWorld *w, bool stepEnded)
{
    const size_t n = w->bodies.items.size();
    if (w->viewFlags.size() != n)
    {
        w->viewXf.resize(n);
        w->viewVel.resize(n);
        w->viewFlags.resize(n, 0);
        w->viewGen.resize(n, 1);
        w->viewUserData.resize(n, nullptr);
        w->viewDirtyMark.resize(n, 0);
    }
    VkWorld2D &vw = *w->world;
    if (stepEnded)
    {
        // The moved list feeds the move events and the view in one pass: one cos/sin per body.
        // A slot that is not dirty already holds its record's flags, generation and user data
        // (writeViewSlot wrote them, and every change to them marks the slot dirty), so the pass
        // reads those from the view and touches a body record only for a dirty or removed body.
        // Each record names a distinct slot, so the chunks run in parallel: event k is written at
        // k, and the removed bodies are gathered per chunk and appended in chunk order, which
        // keeps both lists in moved-list order whatever the thread count.
        w->removedBodies.clear();
        const Scene2D &sc = vw.authoring();
        const std::vector<VkWorld2D::MovedBody2D> &moved = vw.movedBodies();
        const size_t chunk = VkWorld2D::kMovedChunk;
        w->bodyEvents.resize(moved.size());
        w->removedByChunk.resize(TaskPool2D::chunkCount(moved.size(), chunk));
        std::atomic<bool> skipped{false};
        vw.hostPool().run(
            moved.size(), chunk,
            [&](size_t c, size_t begin, size_t end)
            {
                std::vector<Avbd2dBodyId> &removed = w->removedByChunk[c];
                removed.clear();
                for (size_t k = begin; k < end; k++)
                {
                    const VkWorld2D::MovedBody2D &m = moved[k];
                    Avbd2dBodyMoveEvent &e = w->bodyEvents[k];
                    if (m.body < 0 || m.body >= (int)w->solverBody.size() || w->solverBody[(size_t)m.body] < 0)
                    {
                        e.bodyId.index1 = 0; // dropped below
                        skipped.store(true, std::memory_order_relaxed);
                        continue;
                    }
                    const int slot = w->solverBody[(size_t)m.body];
                    const size_t i = (size_t)slot;
                    uint32_t f;
                    uint16_t gen;
                    void *userData;
                    if (w->viewDirtyMark[i])
                    {
                        const BodyRec &b = w->bodies.items[i];
                        f = baseFlags(b);
                        gen = b.gen;
                        userData = b.def.userData;
                    }
                    else
                    {
                        f = w->viewFlags[i] & ~(uint32_t)AVBD2D_BODY_AWAKE;
                        gen = w->viewGen[i];
                        userData = w->viewUserData[i];
                    }
                    if (m.removed && (f & AVBD2D_BODY_ENABLED) && !(f & AVBD2D_BODY_REMOVED))
                    {
                        w->bodies.items[i].removed = true;
                        f |= AVBD2D_BODY_REMOVED;
                        removed.push_back(Avbd2dBodyId{slot + 1, w->id, gen});
                    }
                    const Avbd2dRot q = rotOf(m.angle);
                    const Vec2 lc = sc.localCenter[(size_t)m.body];
                    const Vec2 o = m.pos - Vec2(q.c * lc.x - q.s * lc.y, q.s * lc.x + q.c * lc.y);
                    const Avbd2dTransform xf{A(o), q};
                    w->viewXf[i] = xf;
                    w->viewVel[i] = Avbd2dBodyVelocity{A(m.vel), m.angVel, 0.0f};
                    // GRABBED is kept here and rebuilt from the grab set below.
                    w->viewFlags[i] = f | (m.awake && !(f & AVBD2D_BODY_REMOVED) ? (uint32_t)AVBD2D_BODY_AWAKE : 0u);

                    e.transform = xf;
                    e.bodyId = Avbd2dBodyId{slot + 1, w->id, gen};
                    e.userData = userData;
                    e.fellAsleep = m.fellAsleep ? 1 : 0;
                }
            });
        if (skipped.load(std::memory_order_relaxed))
            w->bodyEvents.erase(std::remove_if(w->bodyEvents.begin(), w->bodyEvents.end(),
                                               [](const Avbd2dBodyMoveEvent &e) { return e.bodyId.index1 == 0; }),
                                w->bodyEvents.end());
        for (const std::vector<Avbd2dBodyId> &removed : w->removedByChunk)
            w->removedBodies.insert(w->removedBodies.end(), removed.begin(), removed.end());
        if (w->debugFlags & AVBD2D_DEBUG_BODY_EXTRAS)
        {
            w->slotColors.assign(n, -1);
            const int *colors = vw.graphColors();
            for (size_t i = 0; i < n; i++)
            {
                const BodyRec &b = w->bodies.items[i];
                if (b.alive && live(b))
                    w->slotColors[i] = colors[(size_t)b.solver];
            }
        }
    }
    for (int slot : w->viewGrabbed)
        if ((size_t)slot < n)
            w->viewFlags[(size_t)slot] &= ~(uint32_t)AVBD2D_BODY_GRABBED;
    w->viewGrabbed.clear();
    for (int sb : vw.grabbedBodies())
    {
        if (sb < 0 || sb >= (int)w->solverBody.size() || w->solverBody[(size_t)sb] < 0)
            continue;
        const int slot = w->solverBody[(size_t)sb];
        w->viewFlags[(size_t)slot] |= AVBD2D_BODY_GRABBED;
        w->viewGrabbed.push_back(slot);
    }
    for (int slot : w->viewDirty)
    {
        if ((size_t)slot >= n)
            continue;
        writeViewSlot(w, slot);
        w->viewDirtyMark[(size_t)slot] = 0;
    }
    w->viewDirty.clear();
    w->viewVersion++;
}


Avbd2dResult stepEndImpl(Avbd2dWorld *w)
{
    if (!w->inFlight)
        return AVBD2D_OK;
    w->inFlight = false;
    return guarded(w, false,
                   [&]() -> Avbd2dResult
                   {
                       w->last = w->world->stepEnd();
                       w->stepsEnded++;
                       rebuildEvents(w);
                       refreshView(w, true);
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
    const Avbd2dResult rr = guarded(w, true,
                                    [&]() -> Avbd2dResult
                                    {
                                        flushReserve(w);
                                        flush(w);
                                        return AVBD2D_OK;
                                    });
    if (rr != AVBD2D_OK)
        return rr;
    return guarded(w, false,
                   [&]() -> Avbd2dResult
                   {
                       refreshView(w, false);
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
                      reinterpret_cast<VkPhysicalDevice>(ext->physical_device), reinterpret_cast<VkDevice>(ext->device),
                      ext->queue_family_index, ext->queue_index, ext->api_version);
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
        return avbdvk::isOutOfMemory(e) ? AVBD2D_ERR_OUT_OF_MEMORY : AVBD2D_ERR_DEVICE;
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
        for (int tries = 0; tries < 65536 && g_worldTable[g_nextWorldId].load(std::memory_order_relaxed); tries++)
            g_nextWorldId++;
        raw->id = g_nextWorldId++;
        registry().insert(raw);
        g_worldTable[raw->id].store(raw, std::memory_order_release);
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

void destroyJointSlot(Avbd2dWorld *w, int slot)
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

// Destroys a body's joints and shapes and releases its slot. With `sweep` false the caller removes
// the slot from pendingBodies and replaceBodies itself (one pass for a whole batch).
void destroyBodySlot(Avbd2dWorld *w, int slot, bool sweep)
{
    BodyRec *b = &w->bodies.items[(size_t)slot];
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
    {
        if (sweep)
            w->pendingBodies.erase(std::remove(w->pendingBodies.begin(), w->pendingBodies.end(), slot),
                                   w->pendingBodies.end());
    }
    else if (b->solver >= 0)
    {
        w->removeBodies.push_back(b->solver);
        w->solverBody[(size_t)b->solver] = -1;
        if (sweep)
            w->replaceBodies.erase(std::remove(w->replaceBodies.begin(), w->replaceBodies.end(), slot),
                                   w->replaceBodies.end());
    }
    w->bodies.release(slot);
    markDirty(w, slot);
}

// ----- Body setters: one helper per change, shared by the single-id and the bulk calls ------------
void setTransformOn(Avbd2dWorld *w, BodyRec &b, int slot, Avbd2dVec2 position, Avbd2dRot rotation)
{
    ensureProps(w, b);
    b.def.position = position;
    b.def.rotation = rotation;
    if (live(b))
    {
        const float a = angleOf(rotation);
        const Vec2 lc =
            w->world->authoring().localCenter[(size_t)b.solver]; // the world's, until a shape edit applies
        w->world->queueSetPose(b.solver, V(position) + rotate(a, lc), a);
    }
    markDirty(w, slot);
}

// Sets the linear and/or the angular velocity (a null pointer keeps that part).
void setVelocityOn(Avbd2dWorld *w, BodyRec &b, int slot, const Avbd2dVec2 *v, const float *av)
{
    if (live(b))
    {
        const size_t s = (size_t)b.solver;
        w->world->queueSetVelocity(b.solver, v ? V(*v) : w->world->velocities()[s],
                                   av ? *av : w->world->angularVelocities()[s]);
    }
    else
    {
        if (v)
            b.def.linearVelocity = *v;
        if (av)
            b.def.angularVelocity = *av;
    }
    markDirty(w, slot);
}

// An impulse on a body: queued on a live one, folded into the velocity of a pending one.
void impulseOn(Avbd2dWorld *w, BodyRec &b, Vec2 J, const Vec2 *point, float angular, int wake)
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

void setAwakeOn(Avbd2dWorld *w, BodyRec &b, int slot, int32_t awake)
{
    b.def.isAwake = awake ? 1 : 0;
    if (live(b))
    {
        if (awake)
            w->world->queueWake(b.solver);
        else
            w->world->queueSleep(b.solver);
        w->world->setHostAwake(b.solver, awake != 0);
    }
    markDirty(w, slot);
}

void setEnabledOn(Avbd2dWorld *w, BodyRec &b, int slot, int32_t enabled)
{
    b.def.isEnabled = enabled ? 1 : 0;
    if (enabled)
        b.removed = false;
    if (live(b))
    {
        if (enabled)
            w->world->queueEnable(b.solver);
        else
            w->world->queueDisable(b.solver);
    }
    markDirty(w, slot);
}

bool finiteRot(Avbd2dRot q)
{
    return finite1(q.c) && finite1(q.s);
}

void syncMotor(Avbd2dWorld *w, const JointRec &j)
{
    if (j.flushed && j.solver >= 0)
        w->world->queueJointMotor(
            j.solver, motorAxis(j.d.type), j.d.enableMotor ? j.d.motorMode : (uint32_t)JOINT_MOTOR_OFF,
            j.d.motorMode == JOINT_MOTOR_TARGET ? j.d.motorTarget : j.d.motorSpeed, maxForceOf(j.d));
}

Avbd2dBodyId bodyIdOfSolver(Avbd2dWorld *w, int solver)
{
    if (solver < 0 || solver >= (int)w->solverBody.size() || w->solverBody[(size_t)solver] < 0)
        return Avbd2dBodyId{};
    return bodyIdOf(w, w->solverBody[(size_t)solver]);
}

} // namespace avbd2d_capi
