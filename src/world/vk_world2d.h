#pragma once

// VkWorld2D: the 2D Vulkan AVBD solver. It owns every device buffer, the pipelines and the
// step, and runs the AVBD pipeline on 3-DOF bodies (x, y, theta):
//
//   joint init -> integrate -> grid broadphase -> radix sort of the pair keys -> narrowphase
//   (persistent contact cache) -> wake -> CSR constraint graph + Jones-Plassmann colouring
//   (VkConstraints) -> iterations of { per-colour primal sweep, contact
//   dual, joint dual } -> velocity -> guard (kill fallen / exploded) -> sleep.
//
// Joint init runs BEFORE integrate: a hard joint measures its error C0 at x^t, and C0 taken
// at the warm-started pose would cancel the very error the joint has to correct.
//
// The whole step is recorded into one command buffer and submitted once. Count-producing
// phases count `required` on the device and guard their writes; if a capacity or a recorded
// bound (pairs, contacts, Jones-Plassmann rounds, colours) was too small, the pass is
// discarded, the dynamic state it touched is restored from the copy taken at the top of the
// step, the resource grows and the step is recorded again. Contact warm-start state survives
// growth. Capacity ceilings scale with the body count, so a diverged simulation cannot ask the
// driver for gigabytes: StepResult2D::ok is false when a ceiling clamped the result.
//
// step() is stepBegin() + stepEnd(): stepBegin applies queued mutations, records and submits;
// stepEnd waits and updates the host mirrors (positions()/angles()/... and movedBodies()).
// Between the two, getters read the previous mirror and queue*() calls apply at the next step.

#include "maths2d.h"
#include "query2d.h"
#include "scene2d.h"
#include "world2d_args.h"

#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_constraints.h"
#include "vk_device.h"
#include "vk_pipeline.h"
#include "vk_primitives.h"

#include <chrono>
#include <cstdint>
#include <vector>

namespace avbd2d
{

struct StepResult2D
{
    bool ok = true;         // false only when a hard capacity ceiling forced clamped results
    bool truncated = false; // the committed step used clamped pair/contact buffers
    int capacityReruns = 0;
    int jpReruns = 0;
    int colorReruns = 0;
    int submits = 0;        // vkQueueSubmit calls this step made (1 unless it reran)
    int pairs = 0;          // broadphase pairs
    int contacts = 0;       // contact points
    int colors = 0;         // graph colours
    int awake = 0;          // dynamic bodies solved this step
    int removed = 0;        // bodies the guard removed this step
    int brokenJoints = 0;   // joints that fractured this step
    int eventsTruncated = 0; // bit k set: event kind k overflowed its buffer this step (it grows for the next)
    float stepMs = 0.0f;    // host wall clock
    const char *failResource = "";
};

// GPU timestamps of the committed pass, in milliseconds. `valid` is false if the device has
// no timestamp support.
struct StepTiming2D
{
    bool valid = false;
    float gpuMs = 0.0f;       // whole pass
    float integrateMs = 0.0f; // joint init + integrate
    float broadMs = 0.0f;     // grid, pair emission, key sort
    float narrowMs = 0.0f;    // narrowphase + wake
    float buildMs = 0.0f;     // constraint graph + colouring
    float primalMs = 0.0f;    // all primal sweeps
    float dualMs = 0.0f;      // all dual updates
    float tailMs = 0.0f;      // velocity, guard, sleep, read-backs
};

// What continuous collision did in the last committed step (ccd2d.slang).
struct CcdStats2D
{
    int clamped = 0;   // bodies moved back to a time-of-impact pose
    int truncated = 0; // candidate walks that hit the visit or cell-rectangle cap
};

struct WorldStats2D
{
    int bodies = 0;
    int statics = 0;
    int dynamic = 0;        // dynamic bodies still in the simulation
    int awake = 0;
    int asleep = 0;
    int removed = 0;        // dynamic bodies the guard has removed in total
    int joints = 0;
    int brokenJoints = 0;
    int pairCap = 0;
    int contactCap = 0;
    int bodyCap = 0;
    uint64_t deviceBytes = 0;
};

struct SpawnDesc2D
{
    int shape = SHAPE2D_BOX; // Shape2D
    Vec2 pos;
    float angle = 0.0f;
    Vec2 size{1.0f, 1.0f};   // box: full extents; circle: size.x is the radius; capsule: full length and
                             // width along local x (radius size.y / 2); polygon: size.x is the circumradius
    int sides = 6;           // polygon: a regular polygon with 3..8 sides
    float radius = 0.0f;     // polygon: rounding radius
    float density = 1.0f;
    float friction = 0.5f;
    Vec2 vel;
    float angVel = 0.0f;
    uint32_t color = 0;
};

// Adds one SpawnDesc2D body to an authored scene; returns its index. A degenerate polygon
// falls back to a circle of the same circumradius.
int addSpawnDesc(Scene2D &s, const SpawnDesc2D &d);

struct ContactDebug2D
{
    Vec2 p;        // world point
    Vec2 n;        // normal, from B to A
    float force;   // normal multiplier magnitude (N)
    float sep;     // n . (xA - xB) at the stored anchors; negative = overlap (m)
    int bodyA, bodyB;
};

// The events of the last committed step, by shape index (Scene2D shape ids). A kind that
// overflowed holds the first `capacity` records and `required` says how many there were.
struct ContactEvent2D
{
    int shapeA, shapeB;
};
struct SensorEvent2D
{
    int sensorShape, visitorShape;
};
struct HitEvent2D
{
    int shapeA, shapeB;
    Vec2 point, normal;
    float approachSpeed;
};
struct JointBreakEvent2D
{
    int joint; // authored joint index
};
enum EventKind2D
{
    EVENT2D_CONTACT_BEGIN = 0,
    EVENT2D_CONTACT_END,
    EVENT2D_SENSOR_BEGIN,
    EVENT2D_SENSOR_END,
    EVENT2D_HIT,
    EVENT2D_JOINT_BREAK,
    EVENT2D_KINDS
};
struct Events2D
{
    std::vector<ContactEvent2D> contactBegin, contactEnd;
    std::vector<SensorEvent2D> sensorBegin, sensorEnd;
    std::vector<HitEvent2D> hit;
    std::vector<JointBreakEvent2D> jointBreak;
    bool truncated[EVENT2D_KINDS] = {};
    int required[EVENT2D_KINDS] = {};
};

class VkWorld2D
{
public:
    VkWorld2D() = default;
    ~VkWorld2D() { destroy(); }
    VkWorld2D(const VkWorld2D &) = delete;
    VkWorld2D &operator=(const VkWorld2D &) = delete;

    // Creates the Vulkan device and every pipeline. A caller that presents (the demo)
    // configures device() first -- instance extensions and the surface hook -- then calls this.
    // To adopt a host-created device instead, call configureDevice() and device().adopt(...)
    // before init(); init() then skips creating one.
    void init();
    void configureDevice();
    void destroy();
    bool initialised() const { return m_init; }
    avbdvk::Device &device() { return m_dev; }

    // Replaces the world with `scene` (bodies, joints, parameters). Resets every cache. With
    // `keepParams` the solver parameters are those instead of the scene's, so a UI's settings
    // survive a scene change.
    void build(const Scene2D &scene, const SolverParams2D *keepParams = nullptr);

    // One timestep with params(): stepBegin() + stepEnd().
    StepResult2D step();
    // The async halves. stepBegin() applies the queued mutations, records the whole step and
    // submits it once; stepEnd() waits, reruns the pass if a capacity or bound was too small, and
    // commits. Between the two, the getters read the previous step's mirrors and the queue*
    // calls queue for the next step; nothing else may be called.
    void stepBegin();
    StepResult2D stepEnd();
    bool stepInFlight() const { return m_stepActive; }

    SolverParams2D &params() { return m_params; }
    const SolverParams2D &params() const { return m_params; }
    const StepResult2D &lastResult() const { return m_result; }
    const StepTiming2D &timing() const { return m_timing; }
    // Ops recorded by the last pass (dispatches, compute barriers, transfers).
    avbdvk::CommandList::OpCounts lastOps() const { return m_cmd.opCounts(); }
    // The one-workgroup iteration path (csIterate2D) replaces the per-colour dispatches when at most
    // this many dynamic bodies exist; 0 forces the multi-dispatch path. Same arithmetic either way.
    void setSmallWorldLimit(int bodies) { m_smallLimit = bodies; }
    int smallWorldLimit() const { return m_smallLimit; }
    bool lastPassFused() const { return m_lastFused; }
    WorldStats2D stats() const;

    // --- Bodies --------------------------------------------------------------------------
    int bodyCount() const { return m_bodyCount; }
    // Host copies of the poses, refreshed by every step() (and by build()/setBodyState()).
    const Vec2 *positions() const { return m_hostPos.data(); }
    const float *angles() const { return m_hostAng.data(); }
    const Scene2D &scene() const { return m_scene; }
    float cellSize() const { return m_cellSize; }
    // Marks a body as a bullet for continuous collision (also swept against the non-bullet
    // dynamic bodies). Between steps only.
    void setBullet(int body, bool bullet);
    CcdStats2D ccdStats() const { return m_ccdStats; }
    // Overwrites one body's pose and velocities and wakes it. Between steps only.
    void setBodyState(int body, Vec2 pos, float angle, Vec2 vel, float angVel);

    // Adds bodies to the live world and returns the index of the first (or -1). Grows the
    // body buffers when needed, so it can be called repeatedly. Between steps only.
    int spawn(const SpawnDesc2D *descs, int count);
    // Appends every body, joint and no-collide pair of `frag` to the live world, translated by
    // `offset` (a ragdoll authored at the origin, say). Returns the first new body, or -1.
    int spawnScene(const Scene2D &frag, Vec2 offset);
    // Wakes every sleeping body at the start of the next step.
    void wakeAll() { m_wakeAllPending = true; }

    // --- Mutation queue ---------------------------------------------------------------------
    // Every call appends to a host list that is uploaded once and applied on the device at the
    // top of the next step (before the state copy, so a rerun keeps it); there is no per-call
    // upload. Calls on one body merge into one record. Safe between stepBegin and stepEnd.
    // Positions are the centre of mass. set pose / set velocity also update the host mirror.
    void queueSetPose(int body, Vec2 pos, float angle);
    void queueSetVelocity(int body, Vec2 vel, float angVel);
    // A linear impulse at a world point (or at the centre of mass), and an angular impulse.
    void queueImpulse(int body, Vec2 impulse, const Vec2 *worldPoint);
    void queueAngularImpulse(int body, float impulse);
    void queueWake(int body);
    void queueSleep(int body);
    // Disables a body: it leaves the broadphase and the solve and never wakes (until enabled).
    void queueDisable(int body);
    void queueEnable(int body);
    // Joint patches: motor mode (0 off, 1 velocity) on an axis, limit on an axis, fracture, wake
    // both bodies. `joint` is the authored joint index.
    void queueJointMotor(int joint, int axis, uint32_t mode, float speed, float maxForce);
    void queueJointLimit(int joint, int axis, bool on, float lower, float upper);
    void queueJointBreak(int joint);
    void queueJointWake(int joint);

    // The moved-body list of the last step: dynamic bodies that were awake at its start or are
    // awake at its end, and kinematic bodies with a velocity. Written on the device straight into
    // host-visible memory; the host mirrors below are updated from it.
    struct MovedBody2D
    {
        int body;
        bool awake;       // awake at the end of the step
        bool fellAsleep;  // awake at its start, not at its end
        Vec2 pos;         // centre of mass
        float angle;
        Vec2 vel;
        float angVel;
    };
    const std::vector<MovedBody2D> &movedBodies() const { return m_moved2d; }
    // Host mirrors (the last completed step, plus queued set pose / velocity). velocities() is
    // always kept; awakeFlags() too.
    const float *angularVelocities() const { return m_hostAngVel.data(); }
    // When off, a step copies no full pose arrays (only the moved list), so positions() and
    // angles() stay valid through the moved list only. Default on.
    void setFullReadback(bool on) { m_fullReadback = on; }
    // When on, every step also copies the constraint force of every joint to the host.
    void setJointForceReadback(bool on) { m_jointForceReadback = on; }
    // Constraint force on body B (x, y) and torque from the last dual of the last step, or zeros.
    void jointForce(int joint, float out[3]) const;

    // --- Live authoring ---------------------------------------------------------------------
    // The authored scene, for appending bodies, shapes and joints directly (Scene2D::beginBody ...
    // endBody, addJoint). Then commitAuthored() with the counts from before the edit uploads what
    // was appended (building the world on the first call). Between steps only.
    Scene2D &authoring() { return m_scene; }
    int commitAuthored(int firstBody, int jointsBefore, int ignoreBefore);

    // --- Live slot edits (the C ABI's create / destroy path) ------------------------------------
    // Edit the authored scene in place and queue the difference for the NEXT step's own command
    // buffer (a persistent host-visible staging buffer; no submission of their own). A destroyed
    // body, shape or joint leaves a dead slot that the next create takes, so memory is bounded by
    // the peak count. Between steps only. liveReserve() is the one call that allocates device
    // memory: it throws (BudgetExceeded) BEFORE anything is edited, so a caller reserves first and
    // the world stays usable and unchanged when it fails.
    struct LiveNeeds
    {
        int bodies = 0, shapes = 0, verts = 0, joints = 0; // upper bounds of what the batch touches
    };
    void liveReserve(const LiveNeeds &n);
    // Installs body `ti` of `tmp` (authored with Scene2D::beginBody/endBody or addShapelessBody) in
    // a free body slot, or appends. Returns the body index.
    int liveAddBody(const Scene2D &tmp, int ti);
    // Replaces the shapes, mass and centre of mass of a live body with those of `tmp`'s body `ti`,
    // keeping the slot. Pose and velocity come from `tmp`. Joint anchors on the body follow the new
    // centre of mass. The new shapes get new ids (the old ones are retired for a step, so their end
    // events still carry the old ids).
    void liveReplaceBody(int body, const Scene2D &tmp, int ti);
    // Frees a body's slot and shapes and drops its no-collide pairs. Its joints go first.
    void liveRemoveBody(int body);
    // Adds a joint in a free slot (or appends); returns the authored joint index, or -1 for a type
    // that makes no joint (a filter joint).
    int liveAddJoint(const Scene2D::JointDef2D &d);
    void liveRemoveJoint(int joint);
    // Queues everything the live edits changed. Builds the world on the first call.
    void liveFinish();
    // The queued edits have not reached the device yet (the next step uploads them).
    bool uploadsPending() const { return !m_upOps.empty() || m_liveDirty; }
    // The reported host awake flag of a body (the mirror), set immediately.
    void setHostAwake(int body, bool awake);
    // Test hooks: slots waiting for reuse, and the joint slot count.
    int deadBodySlots() const { return (int)m_freeBodies.size(); }
    int deadJointSlots() const { return (int)m_freeJoints.size(); }
    int jointSlotCount() const { return m_jointSlots; }
    bool jointAlive(int joint) const { return joint >= 0 && joint < m_jointCount && !m_jointDead[(size_t)joint]; }

    // --- Debug read-backs -------------------------------------------------------------------
    // When on, every step also copies velocities, sleep flags and graph colours to the host.
    void setExtraReadback(bool on) { m_extras = on; }
    const Vec2 *velocities() const { return m_hostVel.data(); }
    const uint8_t *awakeFlags() const { return m_hostAwake.data(); }
    const int *graphColors() const { return m_hostColor.data(); }
    // The contact points of the last committed step (a synchronous read-back; for overlays).
    void downloadContacts(std::vector<ContactDebug2D> &out) const;
    // What the host mirrors cannot tell: per body, 1 when the guard / kill plane removed it (or it
    // is disabled), and per authored joint, 1 when it fractured or its slot is free. A
    // synchronous read-back for tools (the demo's scene capture); between steps only.
    void downloadLiveFlags(std::vector<uint8_t> &bodyRemoved, std::vector<uint8_t> &jointBroken) const;

    // --- Mouse grab ------------------------------------------------------------------------
    // The grab is a set of world-anchored spring joints, one per held body (up to kGrabSlots).
    // grabMove only records the cursor; each step re-aims every spring at it, no further than
    // params.grabMaxSpeed * dt from its body (SolverParams2D::grabMaxSpeed).
    // pick() returns the dynamic body under `p`, or -1.
    struct GrabOptions
    {
        float radius = 0.0f;        // 0 = just the body under the cursor
        bool followJoints = true;   // a jointed body brings what its joints reach, within the radius
        float falloff = 2.0f;       // exponent of the strength falloff towards the rim
        float minStrength = 0.05f;  // strength at the rim, as a fraction of the centre's
    };
    // The strength weight of a body `dist` from the click, 1 at the cursor and `minStrength` at
    // the rim.
    static float grabWeight(float dist, const GrabOptions &opt);
    static constexpr int kGrabSlots = 256;
    int pick(Vec2 p) const;
    bool grabBegin(Vec2 p);                        // just the body under the cursor
    bool grabBegin(Vec2 p, const GrabOptions &opt);
    void grabMove(Vec2 p);
    void grabEnd();
    bool grabbing() const { return m_grabBody >= 0; }
    int grabbedBody() const { return m_grabBody; }
    int grabCount() const { return (int)m_grabSorted.size(); }
    bool isGrabbed(int body) const;

    // --- Events ---------------------------------------------------------------------------
    // Filled by every step() (cleared first). Contact begin/end follow the manifold: a pair
    // begins when it first has a contact point (speculative ones count, as in Box2D) and ends
    // when it has none. A pair that sleeps keeps its manifold and does not end; a disabled or
    // destroyed body (queueDisable) ends its contacts and sensor overlaps in the next step. Sensors
    // report overlaps with other shapes of non-sensor bodies; a sensor makes no manifold and never
    // wakes a body. A hit fires once per new contact (not on every step it persists), needs
    // SHAPE_FLAG_HIT_EVENTS on either shape and an approach speed above hitEventThreshold.
    const Events2D &events() const { return m_events2d; }
    void setHitEventThreshold(float v) { m_hitThreshold = v; }
    float hitEventThreshold() const { return m_hitThreshold; }
    void setEventsEnabled(bool on) { m_eventsOn = on; }
    // Queues an explosion (Box2D's b2World_Explode) for the top of the next step. Between steps only.
    void explode(Vec2 pos, float radius, float falloff, float impulsePerLength);

    // --- Queries (against the poses of the last finished step) ---------------------------------
    // Legal at any time, also between stepBegin() and stepEnd(): the host query mirror is a stable
    // snapshot of the last finished step (stepEnd refreshes it), so a query in flight answers from
    // that snapshot and never touches the step's buffers.
    bool castRayClosest(Vec2 origin, Vec2 translation, RayHit2D &out);
    void castRay(Vec2 origin, Vec2 translation, const Query2D::CastFn &fn);
    void overlapAABB(Vec2 lo, Vec2 hi, const Query2D::ShapeFn &fn);
    void overlapShape(const Proxy2D &proxy, const Query2D::ShapeFn &fn);
    bool castShapeClosest(const Proxy2D &proxy, Vec2 translation, RayHit2D &out);
    void castShape(const Proxy2D &proxy, Vec2 translation, const Query2D::CastFn &fn);

    // Batched queries. Outside a step they run on the device against the final poses of the last
    // step (a query-local cell grid over the broadphase lists), in one small submission of their own,
    // and the results are read back before the call returns. In flight (or with nothing to
    // query) they are answered from the host mirror instead, so the results are the same snapshot
    // either way. Both return true when the device answered. Shapes are filtered like the ABI's
    // queries: (shape.category & mask) != 0 and (cat & shape.mask) != 0; sensors never match.
    struct RayQuery2D
    {
        Vec2 origin, translation;
    };
    struct RayBatchHit2D
    {
        int shape = -1; // -1 = no hit
        float fraction = 1.0f;
        Vec2 point, normal;
    };
    bool castRaysBatch(const RayQuery2D *rays, int count, uint64_t cat, uint64_t mask, RayBatchHit2D *out);
    // maxPer ids per box go to outIds (count * maxPer ints); outCounts[i] is the number of overlaps
    // found, which may exceed maxPer. Ids are shape ids in no particular order.
    bool overlapAabbsBatch(const Vec2 *lo, const Vec2 *hi, int count, uint64_t cat, uint64_t mask, int maxPer,
                           int *outIds, int *outCounts);

    // --- Test hook -----------------------------------------------------------------------
    // Forces tiny pair/contact capacities so the first steps exercise the rerun path.
    void debugForceTinyCapacity(int pairCap, int contactCap);
    // Forces every event kind's buffer to `cap` records at the next build().
    void debugForceTinyEventCapacity(int cap) { m_forcedEventCap = cap; }

private:
    // One side of the double-buffered manifold store: slot arrays then contact arrays.
    struct ManBufs
    {
        avbdvk::Buffer key, bodyA, bodyB, normal, friction, offset, count, flags;
        avbdvk::Buffer feature, rA, rB, c0, penalty, lambda, stick, roll;
        void init(avbdvk::Device &dev);
        void destroy();
        void size(int pairCap, int contactCap, bool preserve);
        ManSet2Gpu gpu() const;
        uint64_t bytes() const;
    };

    // A per-body buffer, its element size, and whether growth must keep its contents.
    struct BodyBuf
    {
        avbdvk::Buffer *buf;
        size_t elem;
        bool preserve;
    };

    void createPipelines();
    std::vector<avbdvk::Buffer *> buffers();
    std::vector<BodyBuf> bodyBuffers();
    void sizeBodyBuffers(int cap, bool growing);
    void uploadBodyProps(size_t first, size_t count);
    void sizeHostMirrors(int cap);
    void uploadLists();
    void sizeShapeBuffers(int cap, bool growing);
    void uploadShapes(int from, int to = -1);
    void uploadCcdBodies(int from, int to = -1);
    // A buffer upload: immediate (Buffer::upload) normally, queued for the next step's command buffer
    // while m_deferUp is set.
    void up(avbdvk::Buffer &b, const void *src, size_t bytes, size_t off = 0);
    void recordUploads();
    void flushUploadsNow();
    void releaseRetired();
    void uploadBodies(int first, int count);
    void fitBuf(avbdvk::Buffer &b, size_t bytes);
    void uploadVerts(int off, int n);
    void clearBodyPatch(int body, uint32_t keepMask);
    void clearJointPatch(int joint);
    void updateGridSizing();
    int pairCeiling() const;
    int contactCeiling() const { return 2 * pairCeiling(); }
    void setCapacity(int pairCap, int contactCap, bool preserve);
    World2Gpu makeWorld(int colorBound) const;
    void recordPass(int pass, int colorBound, int jpRounds);
    void saveState();
    void restoreState();
    void refreshHostCopies();
    void readTimings(int iterations);
    void putBody(int body, const uint8_t awake);
    bool sleepActive() const { return m_params.sleepEnabled && m_params.sleepFrames > 0; }
    void uploadGrabSlots();
    void recordGrabTargets();
    void sizeJointBuffers(int slotCap, bool preserve);
    void sizeEvents();
    void collectEvents();
    QueryView2D queryView();
    void uploadAuthoredJoints(int from, int to);
    int commitSpawn(int first, int jointsBefore, int ignoreBefore);

    // Mutation queue records (the layouts csApplyBodyPatches2D / csApplyJointPatches2D read).
    struct BodyPatch
    {
        uint32_t body = 0, flags = 0;
        float pose[3] = {};
        float vel[3] = {};
        float dv[3] = {};
        uint32_t pad = 0;
    };
    struct JointPatch
    {
        uint32_t slot = 0, flags = 0, motorAxis = 0, motorMode = 0;
        float speed = 0.0f, maxForce = 0.0f;
        uint32_t limitAxis = 0;
        float lower = 0.0f, upper = 0.0f;
        uint32_t pad[3] = {};
    };
    BodyPatch &bodyPatch(int body);
    JointPatch &jointPatch(int joint);
    void uploadPatches();
    void clearPatches();

    avbdvk::Device m_dev;
    avbdvk::PipelineLayout m_layout;
    avbdvk::Pipeline m_integrate, m_gravityX, m_velocity, m_guard, m_velocityGuard;
    avbdvk::Pipeline m_ccdFlagPl, m_ccdArgsPl, m_ccdPl;
    avbdvk::Pipeline m_gridClear, m_gridInsert, m_findPairs, m_oversizePairs, m_narrowphase;
    avbdvk::Pipeline m_jointInit, m_integrateJoint, m_jointDual, m_colorArgs, m_primal, m_dual, m_dualAll, m_iterate, m_sortSmall, m_buildSmall;
    avbdvk::Pipeline m_sleepPrep, m_sleepBlockPre, m_sleepMark, m_sleepNbrC, m_sleepNbrJ, m_sleepCommit;
    avbdvk::Pipeline m_eventBegin, m_eventEnd, m_eventHit, m_eventJoint, m_applyImpulses;
    avbdvk::Pipeline m_queryClear, m_queryInsert, m_queryRays, m_queryAabbs;
    avbdvk::Pipeline m_applyBodyPatches, m_applyJointPatches, m_compactMoved;
    avbdvk::Pipeline m_touchClear, m_touchAccumC, m_touchAccumJ, m_wakeReqC, m_wakeReqJ, m_wakeLost, m_wakeApply,
        m_wakeAll;
    avbdvk::ConstraintsPipelines m_csPl;
    avbdvk::PrimitivesPipelines m_primPl;
    avbdvk::RadixScratch m_radix;
    avbdvk::VkConstraints m_constraints;
    avbdvk::CommandList m_cmd;
    avbdvk::HostRingBuffer m_ring;
    bool m_init = false;

    SolverParams2D m_params;
    Scene2D m_scene;
    StepResult2D m_result;
    StepTiming2D m_timing;
    int m_smallLimit = 512;
    bool m_lastFused = false;
    bool m_lastSmallBuild = false;
    static constexpr int kSmallBuildMaxBodies = 16384;

    // Body state (device), sized to m_bodyCap.
    avbdvk::Buffer m_pos, m_ang, m_initPos, m_initAng, m_inertPos, m_inertAng, m_vel, m_angVel, m_prevVel;
    avbdvk::Buffer m_moment, m_mass, m_awake, m_sleepTimer, m_blocked, m_bProps;
    avbdvk::Buffer m_dynIdx, m_statIdx;
    // Shape table (sized to m_shapeCap): owning body, type, placement in the body frame, rounding
    // radius, local AABB, vertex range and friction of every shape, and the vertex/normal pools.
    avbdvk::Buffer m_sBody, m_sType, m_sOff, m_sAng, m_sRad, m_sHalf, m_sVR, m_sFric, m_sFlags, m_verts, m_norms;
    // Per-shape filter (category, mask, group) and material (tangent speed, rolling resistance,
    // restitution, user id).
    avbdvk::Buffer m_sCat, m_sMask, m_sGroup, m_sMat;
    // Broadphase lists: grid members (dynamic, then small statics) and oversize dynamics.
    avbdvk::Buffer m_gridList, m_overList;
    // Continuous collision: per-dynamic-body fast flags and the compacted list of flagged bodies,
    // its count, device stat counters (clamped, truncated), indirect args, and per-body
    // (shape range, min/max extent, bullet).
    avbdvk::SelectScratch m_select;
    avbdvk::Buffer m_ccdFlag, m_ccdList, m_ccdCount, m_ccdStat, m_ccdArgs, m_bShape, m_bExt, m_bBullet;
    avbdvk::MappedBuffer m_ccdHost;
    CcdStats2D m_ccdStats;
    // Sleep: the rest window's start pose, candidate flags, contact fingerprints (this step's
    // and the one taken when the body fell asleep) and wake requests.
    avbdvk::Buffer m_restPos, m_restAng, m_cand, m_touchCount, m_touchHash, m_snapCount, m_snapHash, m_wakeReq;
    // Dynamic state at the top of the step, for restoring before a rerun.
    avbdvk::Buffer m_savePos, m_saveAng, m_saveVel, m_saveAngVel, m_savePrevVel, m_saveAwake, m_saveTimer;
    avbdvk::Buffer m_saveRestPos, m_saveRestAng, m_saveCand;
    avbdvk::Buffer m_saveJPen, m_saveJLam, m_saveJBroken;

    // Broadphase.
    avbdvk::Buffer m_gridHead, m_gridNext, m_pairsRaw, m_pairsSorted, m_ignore, m_counters;

    // Manifolds: m_man[m_curIdx] is written this step, the other one is last step's.
    ManBufs m_man[2];
    int m_curIdx = 0;

    // Joints: kGrabSlots grab slots, then the authored joints.
    avbdvk::Buffer m_jA, m_jB, m_jRA, m_jRB, m_jC0, m_jPen, m_jLam, m_jAxis, m_jFrameA, m_jRest, m_jArm, m_jFrac,
        m_jForce, m_jBroken;

    avbdvk::Buffer m_colorIndirect;

    // Events: one device buffer (counters, then a payload region per kind; see events2d.slang),
    // its host copy filled inside the step's submission, and the explosion impulse queue.
    avbdvk::Buffer m_events, m_impulses;
    avbdvk::MappedBuffer m_eventsHost;
    uint32_t m_evOff[EVENT2D_KINDS] = {}, m_evCap[EVENT2D_KINDS] = {};
    Events2D m_events2d;
    float m_hitThreshold = 1.0f;
    bool m_eventsOn = true;
    int m_forcedEventCap = 0;
    size_t m_evBytes = 0;
    std::vector<ExplosionImpulse2D> m_pendingImpulses;
    int m_impulseCap = 0, m_impulseCount = 0;
    Query2D m_query;

    // World2Gpu is staged through host memory and copied to device memory once per pass, so
    // the kernels read their pointer table from VRAM rather than across the bus.
    avbdvk::MappedBuffer m_worldStage;
    avbdvk::MappedBuffer m_grabStage; // the grab springs' targets, copied into jRA by each step
    avbdvk::Buffer m_worldDev;
    // Batched queries: the query-local cell table (device) and the in/out arrays (mapped).
    avbdvk::Buffer m_qBox, m_qHead, m_qNext, m_qBig, m_qCount;
    avbdvk::MappedBuffer m_qRays, m_qAabbs, m_qOutHit, m_qOutIds, m_qOutCounts;
    avbdvk::Buffer m_qOutHitDev, m_qOutIdsDev, m_qOutCountsDev; // the kernels write these
    bool runQueryBatch(bool rays, const float *in, int count, uint64_t cat, uint64_t mask, int maxPer);

    // Host-visible copies filled inside the step's submission.
    avbdvk::MappedBuffer m_statusHost, m_posHost, m_angHost, m_velHost, m_awakeHost, m_colorHost;
    std::vector<Vec2> m_hostPos, m_hostVel;
    std::vector<float> m_hostAng, m_hostAngVel;

    // Mutation queue: merged records (one per body / joint), the slot of each, the device-side
    // copies (host-visible, read by the patch kernels by address) and the counts the step uses.
    std::vector<BodyPatch> m_bodyPatches;
    std::vector<JointPatch> m_jointPatches;
    std::vector<int> m_bodyPatchSlot, m_jointPatchSlot;
    avbdvk::MappedBuffer m_bPatchBuf, m_jPatchBuf;
    size_t m_bPatchCap = 0, m_jPatchCap = 0;
    int m_bPatchCount = 0, m_jPatchCount = 0;
    // The moved-body list the step writes (host-visible), its device counter and host copy.
    avbdvk::MappedBuffer m_movedHost, m_movedCountHost;
    avbdvk::Buffer m_movedCount, m_movedDev; // the list is built device-local, then copied
    std::vector<MovedBody2D> m_moved2d;
    avbdvk::MappedBuffer m_jForceHost;
    bool m_fullReadback = true, m_jointForceReadback = false;

    // The async step.
    bool m_stepActive = false;
    int m_stepColorBound = 0, m_stepJpRounds = 0;
    bool m_stepFirst = true;
    uint64_t m_stepSubmits0 = 0;
    std::chrono::steady_clock::time_point m_stepT0;
    StepResult2D m_stepResult;
    std::vector<uint8_t> m_hostAwake;
    std::vector<int> m_hostColor;

    // Deferred uploads (see up()) and live slot reuse.
    struct UpOp
    {
        avbdvk::Buffer *dst;
        size_t src, bytes, dstOff;
    };
    struct Range
    {
        int off, n;
    };
    // First-fit free list of [off, off + n) ranges of a table; an allocation that finds no hole goes
    // to the end of the table (extending a free tail range when there is one).
    struct RangeAlloc
    {
        std::vector<Range> freeR;
        int alloc(int n, int tableEnd);
        void release(int off, int n);
    };
    struct Retired
    {
        bool verts;
        int off, n;
    };
    std::vector<UpOp> m_upOps;
    std::vector<uint8_t> m_upData;
    avbdvk::MappedBuffer m_upStage;
    bool m_deferUp = false;
    bool m_liveBuilt = false;  // the device buffers exist (build() ran with bodies)
    bool m_liveDirty = false;  // host scene edited, build() still to run
    bool m_dirtyIgnore = false;
    RangeAlloc m_shapeAlloc, m_vertAlloc;
    std::vector<Retired> m_retired; // shape / vertex ranges freed this step window, released after the next commit
    size_t m_retiredMark = 0;       // how many of them the step in flight covers
    std::vector<int> m_freeBodies, m_freeJoints;
    std::vector<uint8_t> m_jointDead;     // per authored joint: a freed slot
    std::vector<int> m_dirtyJoints, m_dirtyBodiesNew;
    std::vector<Range> m_dirtyShapeR, m_dirtyVertR;
    bool m_listsDirty = false;
    bool m_keepLive = false; // build() called by liveFinish: keep the free lists and the retired ranges

    int m_bodyCount = 0, m_bodyCap = 0, m_dynCount = 0, m_statCount = 0;
    int m_shapeCap = 0; // allocated shape slots (also the grid's link array)
    int m_gridCount = 0, m_gridDyn = 0, m_overCount = 0; // broadphase lists of shapes (uploadLists)
    int m_jointCount = 0; // authored
    int m_jointSlots = 0; // grab slots + authored
    int m_jointCap = 0;   // allocated joint slots
    int m_csRes[3] = {0, 0, 0}; // what m_constraints.reserve last sized (bodies, pairs, forces); it reallocates every call
    int m_ignoreCount = 0;
    int m_tableSize = 1024;
    float m_cellSize = 1.0f;
    int m_pairCap = 0, m_contactCap = 0;
    int m_prevPairs = 0;  // live pairs of the manifold set that is now "previous"
    int m_lastColors = 0;
    int m_forcedPairCap = 0, m_forcedContactCap = 0;
    int m_removedTotal = 0, m_brokenTotal = 0, m_awakeNow = -1;
    bool m_wakeAllPending = false;
    bool m_sleepWasOn = true;
    bool m_extras = false;

    // The grab: m_grabBody is the body under the click, the rest hold the group in slot order.
    int m_grabBody = -1;
    GrabOptions m_grabOpt;
    Vec2 m_grabClick;
    std::vector<int> m_grabGroup;     // body of each used slot
    std::vector<Vec2> m_grabOffset;   // target of each slot's anchor, relative to the cursor
    std::vector<Vec2> m_grabLocal;    // each slot's anchor on its body (body frame)
    Vec2 m_grabCursor;                // where grabMove last put the cursor
    std::vector<int> m_grabSorted;    // m_grabGroup sorted, for isGrabbed()
};

} // namespace avbd2d
