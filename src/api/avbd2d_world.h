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


#pragma once

// Internal header of the 2D C ABI implementation (avbd2d_api*.cpp): the host records, the slot tables, the
// world object and the helpers every entry-point file shares.

#include "avbd2d.h"

#include "query2d.h"
#include "slot_table.h"
#include "vk_guard.h"
#include "scene2d.h"
#include "vk_util.h"
#include "vk_world2d.h"

#include <algorithm>
#include <atomic>
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

namespace avbd2d_capi
{
using JointDef = Scene2D::JointDef2D;
using JointType = Scene2D::JointType2D;

struct BodyRec
{
    bool alive = false;
    uint16_t gen = 1;
    Avbd2dBodyDef def{};
    bool flushed = false;
    bool reshape = false;    // flushed, and its shape set changed since: replaced at the next step
    int solver = -1;         // solver body index once flushed
    std::vector<int> shapes; // shape slots, in the order they were added
    std::vector<int> joints; // joint slots
    // Mass properties, lazily computed (pending bodies replay into a scratch scene).
    bool propsValid = false;
    float mass = 0.0f, inertia = 0.0f;
    Vec2 lc;
    bool removed = false; // the guard or the kill plane took it out of the simulation
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
    JointDef d;                 // anchors relative to the body origins here; converted at flush
    int bodyA = -1, bodyB = -1; // body slots
    void *userData = nullptr;
    bool flushed = false;
    int solver = -1;        // authored joint index; -1 = not live (or skipped at flush)
    float lockScale = 1.0f; // multiplier of the hard-row penalty cap (see set_lock_scale)
};

template <class T> using Table = SlotTable<T>;
// Shared by the entry-point files (defined in avbd2d_api_body/joint/debug.cpp).
void destroyJointSlot(Avbd2dWorld *w, int slot);
void destroyBodySlot(Avbd2dWorld *w, int slot, bool sweep);
void setTransformOn(Avbd2dWorld *w, BodyRec &b, int slot, Avbd2dVec2 position, Avbd2dRot rotation);
void setVelocityOn(Avbd2dWorld *w, BodyRec &b, int slot, const Avbd2dVec2 *v, const float *av);
void impulseOn(Avbd2dWorld *w, BodyRec &b, Vec2 J, const Vec2 *point, float angular, int wake);
void setAwakeOn(Avbd2dWorld *w, BodyRec &b, int slot, int32_t awake);
void setEnabledOn(Avbd2dWorld *w, BodyRec &b, int slot, int32_t enabled);
bool finiteRot(Avbd2dRot q);
void syncMotor(Avbd2dWorld *w, const JointRec &j);
Avbd2dBodyId bodyIdOfSolver(Avbd2dWorld *w, int solver);

} // namespace avbd2d_capi

using namespace avbd2d_capi;

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
    std::vector<int> postBodies;                           // disable / sleep to queue once their flush is built
    std::vector<int> replaceBodies;                        // body slots whose shape set changed
    std::vector<int> removeBodies, removeJoints;           // solver indices destroyed since the last step
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

    // The body state view (ABI 3.2), by body slot. Written only inside the step calls (refreshView).
    std::vector<Avbd2dTransform> viewXf;
    std::vector<Avbd2dBodyVelocity> viewVel;
    std::vector<uint32_t> viewFlags;
    std::vector<uint16_t> viewGen;
    std::vector<void *> viewUserData; // internal: lets the moved pass skip the body records
    uint64_t viewVersion = 0;
    std::vector<int> viewDirty;              // slots whose entry is rebuilt from their record at the next refresh
    std::vector<uint8_t> viewDirtyMark;      // by slot: already in viewDirty
    std::vector<int> viewGrabbed;            // slots flagged GRABBED
    std::vector<Avbd2dBodyId> removedBodies; // removed by the guard in the last step
    std::vector<std::vector<Avbd2dBodyId>> removedByChunk; // internal: the moved pass's per-chunk lists

    // Debug read-backs (ABI 3.2).
    uint32_t debugFlags = 0;
    uint64_t stepsEnded = 0;
    std::vector<int32_t> slotColors;
    std::vector<Avbd2dContactDebug> contactDebug;
    uint64_t contactDebugAt = ~0ull; // stepsEnded when contactDebug was built
    std::vector<Avbd2dJointDebug> jointDebug;
    uint64_t jointDebugAt = ~0ull;
};


namespace avbd2d_capi
{
extern std::mutex g_mutex;

extern uint16_t g_nextWorldId;

std::unordered_set<Avbd2dWorld *> &registry();

// World index -> world, read without the lock by every id call. Written only under g_mutex, by
// create (after the world is complete) and destroy (before it is deleted).
extern std::atomic<Avbd2dWorld *> g_worldTable[65536];

Avbd2dWorld *lookup(Avbd2dWorld *w);

Avbd2dWorld *worldFromId(uint16_t id);



// Runs `f` (which touches the device or allocates) and turns what it throws into a result code.
// Memory exhaustion is AVBD2D_ERR_OUT_OF_MEMORY, anything else from Vulkan is AVBD2D_ERR_DEVICE_LOST.
// Either kills the world (later calls return DEVICE_LOST) unless `recoverable`.
template <class F> Avbd2dResult guarded(Avbd2dWorld *w, bool recoverable, F &&f)
{
    return avbdvk::guardedCall<Avbd2dResult>(f, [&](const avbdvk::Classified &c) {
        switch (c.kind)
        {
        case avbdvk::Failure::HostOutOfMemory:
        case avbdvk::Failure::DeviceOutOfMemory:
            if (!recoverable)
                w->lost = true;
            return AVBD2D_ERR_OUT_OF_MEMORY;
        default:
            w->lost = true;
            return AVBD2D_ERR_DEVICE_LOST;
        }
    });
}

bool finite1(float a);

bool finite2(Avbd2dVec2 v);

Vec2 V(Avbd2dVec2 v);

Avbd2dVec2 A(Vec2 v);

float angleOf(Avbd2dRot q);

Avbd2dRot rotOf(float a);

// ----- Parameters ------------------------------------------------------------------------------
SolverParams2D toParams(const Avbd2dWorldDef &d);

void fromParams(const SolverParams2D &s, Avbd2dWorldDef &d);

bool validWorldDef(const Avbd2dWorldDef *d);

// ----- Building bodies in a Scene2D ------------------------------------------------------------
uint8_t flagsOf(const Avbd2dShapeDef &d);

ShapeDef2D toShapeDef(const Avbd2dShapeDef &d);

// A 4-vertex, unrounded, right-angled CCW polygon is a box (the solver's fast shape).
bool asRect(const Avbd2dPolygon &p, Vec2 &center, float &angle, Vec2 &size);

void addShapeToScene(Scene2D &sc, const ShapeRec &s);

BodyDef2D toBodyDef(const Avbd2dBodyDef &d);

// Appends body `b` with its shapes to `sc`; returns the solver body id or -1.
// A body with no shape becomes a shapeless body (a real body with default mass when dynamic).
int buildBody(Scene2D &sc, const BodyRec &b, const Table<ShapeRec> &shapes, const BodyDef2D &bd);

void ensureProps(Avbd2dWorld *w, BodyRec &b);


// ----- Pose helpers ----------------------------------------------------------------------------
struct Pose
{
    Vec2 com;
    float angle = 0.0f;
    Vec2 lc;
    Vec2 origin() const { return com - rotate(angle, lc); }
};

bool live(const BodyRec &b);

Pose poseOf(Avbd2dWorld *w, BodyRec &b);

Avbd2dBodyId bodyIdOf(Avbd2dWorld *w, int slot);

Avbd2dShapeId shapeIdOf(Avbd2dWorld *w, int slot);

Avbd2dJointId jointIdOf(Avbd2dWorld *w, int slot);

// The body's state view entry is rebuilt from its record at the next refresh (create, destroy,
// teleport, and every change the moved list does not report).
void markDirty(Avbd2dWorld *w, int slot);

// ----- Flush: pending bodies and joints become part of the live world ---------------------------
int motorAxis(JointType t);

int limitAxis(JointType t);

float maxForceOf(const JointDef &d);

void flushJoint(Avbd2dWorld *w, int slot);

// Remembers the id a solver shape had, for the events of the next step.
void tombstone(Avbd2dWorld *w, int solverShape, Avbd2dShapeId id);

// The edits are sized before anything changes, so a budget failure leaves the world untouched.
void flushReserve(Avbd2dWorld *w);

void flush(Avbd2dWorld *w);

// ----- Stepping --------------------------------------------------------------------------------
void rebuildEvents(Avbd2dWorld *w);

// ----- Body state view -------------------------------------------------------------------------
uint32_t baseFlags(const BodyRec &b);

// One slot's entry from its record and the host mirrors.
void writeViewSlot(Avbd2dWorld *w, int slot);

// Brings the view up to date: the moved list of a step that just ended, the grab set, then every
// dirty slot. Called only from the step calls, which is what makes the view stable between them.
void refreshView(Avbd2dWorld *w, bool stepEnded);

Avbd2dResult stepEndImpl(Avbd2dWorld *w);

// Ends a step in flight (queries, grab and explosions read settled state); its result is handed to
// the next avbd2d_step_end.
void settle(Avbd2dWorld *w);

Avbd2dResult stepBeginImpl(Avbd2dWorld *w);

Avbd2dResult createImpl(const Avbd2dWorldDef *defIn, const Avbd2dVulkanDevice *ext, Avbd2dWorld **out_world);

// A packed polygon from a convex hull (CCW), with normals and the area centroid.
Avbd2dPolygon fillPolygon(const Vec2 *pts, int n, float radius);

bool validShapeDef(const Avbd2dShapeDef *d);

bool passes(const ShapeRec &s, const Avbd2dQueryFilter &f);

// The shape slot behind a solver shape, or null when it was destroyed or its body is disabled.
const ShapeRec *queryShape(Avbd2dWorld *w, int solverShape, int *slotOut);

Proxy2D toProxy(const Avbd2dShapeProxy &p);

bool validProxy(const Avbd2dShapeProxy *p);

} // namespace avbd2d_capi


#define AVBD2D_GET_WORLD(handle)                                                                                       \
    Avbd2dWorld *w = lookup(handle);                                                                                   \
    if (!w)                                                                                                            \
        return AVBD2D_ERR_NULL_HANDLE;                                                                                 \
    if (w->lost)                                                                                                       \
    return AVBD2D_ERR_DEVICE_LOST

#define ID_WORLD(id)                                                                                                   \
    Avbd2dWorld *w = worldFromId((id).world0);                                                                         \
    if (!w)                                                                                                            \
        return AVBD2D_ERR_INVALID_ARG;                                                                                 \
    if (w->lost)                                                                                                       \
    return AVBD2D_ERR_DEVICE_LOST

#define BODY_CTX(id)                                                                                                   \
    ID_WORLD(id);                                                                                                      \
    BodyRec *b = w->bodies.find((id).index1, (id).generation);                                                         \
    if (!b)                                                                                                            \
        return AVBD2D_ERR_INVALID_ARG;                                                                                 \
    const int slot = (id).index1 - 1;                                                                                  \
    (void)slot

#define SHAPE_CTX(id)                                                                                                  \
    ID_WORLD(id);                                                                                                      \
    ShapeRec *s = w->shapes.find((id).index1, (id).generation);                                                        \
    if (!s)                                                                                                            \
        return AVBD2D_ERR_INVALID_ARG;                                                                                 \
    const int slot = (id).index1 - 1;                                                                                  \
    (void)slot

#define JOINT_CTX(id)                                                                                                  \
    ID_WORLD(id);                                                                                                      \
    JointRec *j = w->joints.find((id).index1, (id).generation);                                                        \
    if (!j)                                                                                                            \
        return AVBD2D_ERR_INVALID_ARG;                                                                                 \
    const int slot = (id).index1 - 1;                                                                                  \
    (void)slot

#define NEED(p)                                                                                                        \
    if (!(p))                                                                                                          \
    return AVBD2D_ERR_INVALID_ARG
