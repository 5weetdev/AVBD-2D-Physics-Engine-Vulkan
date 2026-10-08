// The one argument block every 2D kernel reads: World2 holds the device address of every
// buffer a step touches plus the step's scalars, and Args2 is what a dispatch's push constant
// points at (the World2 address and four per-dispatch words).
//
// The host mirrors both structs byte for byte in world/world2d_args.h. The layout
// rule that keeps them in step under -fvk-use-scalar-layout: every pointer first (8 bytes
// each), then 4-byte scalars in an even count, so no padding is ever implied. Change a
// field here and in world2d_args.h in the same edit.

module types2d;

import maths2d;

// One manifold set (this step's, or last step's for the warm start). Slot t is the t-th
// pair of the sorted pair list; contacts of slot t occupy [offset[t], offset[t] + count[t]).
// Bits of the flags word in bProps.w.
public static const uint BODY_KINEMATIC = 1u;
public static const uint BODY_LOCK_X = 2u;
public static const uint BODY_LOCK_Y = 4u;
public static const uint BODY_LOCK_ROT = 8u;

public struct ManSet2
{
    public uint64_t *key;
    public int *bodyA;
    public int *bodyB;
    public vec2 *normal;   // contact normal, pointing from B to A
    public float *friction;
    public int *offset;
    public int *count;
    public int *feature;   // warm start key of each contact
    public vec2 *rA;       // contact anchors in each body's local frame
    public vec2 *rB;
    public vec2 *c0;       // constraint error at the start of the step: (normal, tangent)
    public vec2 *penalty;  // (normal, tangent)
    public vec2 *lambda;
    public uint8_t *stick;
    public float4 *roll;   // rolling-resistance row of the manifold: (lambda, penalty, force-bound coefficient, 0); penalty 0 = no row
    public uint8_t *flags;    // per slot: bit 0 = a sensor pair that overlaps this step
};

// One axis of a joint (core/joint2d.h mirrors this field for field).
public struct JointAxis2
{
    public uint pos;     // 0 free, 1 lock, 2 spring
    public uint motor;   // 0 off, 1 velocity, 2 target
    public uint limit;
    public uint flags;   // bit 0 grab, bit 1 distance
    public float lo;
    public float hi;
    public float rest;
    public float mtarget;
    public float maxF;
    public float hertz;
    public float stiff;
    public float damp;
};

public static const uint SHAPE_FLAG_SENSOR = 1;
public static const uint SHAPE_FLAG_CONTACT_EVENTS = 2;
public static const uint SHAPE_FLAG_HIT_EVENTS = 4;
public static const uint SHAPE_FLAG_SENSOR_EVENTS = 8;

// A velocity change queued for a body (explosions), applied at the top of the next step.
public struct Impulse2
{
    public int body;
    public float dvx;
    public float dvy;
    public float dw;
};

public struct World2
{
    // Bodies.
    public vec2 *pos;
    public float *ang;
    public vec2 *initPos;     // x^t
    public float *initAng;
    public vec2 *inertPos;    // y, the inertial target
    public float *inertAng;
    public vec2 *vel;
    public float *angVel;
    public vec2 *prevVel;
    public float *moment;
    public float *mass;       // 0 = static
    public uint8_t *awake;    // 1 = solved this step; 0 = asleep or removed (acts as a fixed anchor)
    public int *sleepTimer;   // consecutive quiet steps; -1 = removed from the simulation (never wakes)
    public int *blocked;      // scratch: nonzero = this body may not sleep this step
    public vec2 *restPos;     // pose at the start of the body's rest window
    public float *restAng;
    public uint8_t *cand;     // sleep candidate: 0 no, 1 rested through the window, 2 asleep
    public int *touchCount;   // contact + joint fingerprint of each body, this step
    public int *touchHash;
    public int *snapCount;    // the fingerprint taken when the body fell asleep; < 0 = none yet
    public int *snapHash;
    public uint8_t *wakeReq;  // scratch: a sleeper that must wake
    public float4 *bProps;    // per body: (gravity scale, linear damping, angular damping, asfloat(flags: BODY_*))
    public int *dyn;          // indices of the dynamic bodies
    public int *stat;         // indices of the big static SHAPES (broadphase brute force)

    // Broadphase.
    public int *gridHead;
    public int *gridNext;
    public uint64_t *pairsRaw;
    public uint64_t *pairsSorted;
    public uint64_t *ignore;  // sorted (lo << 32 | hi) pairs that must not collide
    public int *counters;     // [0] pairs required, [1] contacts required

    // Manifolds.
    public ManSet2 cur;
    public ManSet2 prev;

    // Constraint graph (built by the shared constraints kernels) and the colour sweep.
    public int *colorBodies;
    public int *colorStart;
    public int *colorEnd;
    public int *gStart;
    public int *gEnd;
    public uint64_t *gKey;
    public int *gIsA;
    public int *colorIndirect;

    // Joints (authored joints, then the grab slot).
    public int *jA;
    public int *jB;
    public vec2 *jRA;
    public vec2 *jRB;
    public float *jC0;        // axis coordinate at the start of the step; index sub * 3 + axis (9 per joint)
    public float *jPen;       // penalty of each row, same indexing
    public float *jLam;       // multiplier of each hard row, same indexing
    public JointAxis2 *jAxis; // three per joint: frame-A x, frame-A y, relative angle
    public float *jFrameA;    // angle of the frame-A x axis in body A (world when anchored)
    public float *jRest;      // reference angle (angB - angA at authoring)
    public float *jArm;
    public float *jFrac;      // break force on the linear lock multipliers; infinite = unbreakable
    public vec3 *jForce;      // constraint force on body B (x, y) and torque, from the last dual
    public uint8_t *jBroken;

    // Shape table (scene2d.h): a body owns a contiguous run of shapes, and the broadphase,
    // the narrowphase and the manifold key work on shapes (a single-shape body's shape index is
    // its body index). sOff / sAng place the shape relative to the body's centre of mass (the
    // solver's body position). sHalf is the shape's local AABB half extents (a legacy circle's
    // radius is sHalf.x), sRad the rounding radius, sVR the (offset, count) of its vertices in
    // the pool; a box, a legacy circle and a plain segment-free body have count 0. Polygon and
    // capsule vertices are shape-local, with outward edge normals; a chain segment's four are
    // ghost1, p1, p2, ghost2.
    public int *sBody;
    public uint8_t *sType;    // Shape2D (maths2d.h): 0 box, 1 circle, 2 polygon, 3 capsule, 4 segment, 5 chain
    public vec2 *sOff;
    public float *sAng;
    public float *sRad;
    public vec2 *sHalf;
    public int2 *sVR;
    public float *sFric;
    public uint8_t *sFlags;   // SHAPE_FLAG_* per shape
    public vec2 *verts;
    public vec2 *norms;
    // Broadphase lists (vk_world2d.cpp uploadLists), all of shapes: `grid` holds gridDyn dynamic
    // shapes, then the statics small enough for the cell; `over` holds the dynamic shapes too big.
    public int *grid;
    public int *over;
    // Per-shape filter and material (Box2D semantics). sMat = (tangent speed, rolling resistance,
    // restitution (stored, unused), asfloat(user material id)).
    public uint64_t *sCat;
    public uint64_t *sMask;
    public int *sGroup;
    public float4 *sMat;
    // Events: ev[0..5] are the per-kind counters, payloads follow at evOff[k] (ints) with evCap[k] records.
    public int *ev;
    public uint8_t *jBrokenPrev; // jBroken as it was at the start of the step
    public Impulse2 *impulses;   // queued explosion impulses, impulseCount of them

    // Continuous collision (ccd2d.slang): per-dynamic-body fast flags, the compacted list of
    // flagged bodies and its count, [0] bodies clamped / [1] candidate walks cut short (device
    // counters, not part of the step status), the indirect dispatch args, and per-body data
    // (shape range, min/max extent, bullet flag).
    public uint8_t *ccdFlag;
    public int *ccdList;
    public int *ccdCount;
    public int *ccdStat;
    public int *ccdArgs;
    public int2 *bShape;
    public vec2 *bExt;
    public uint8_t *bBullet;

    // Host mutations (events2d.slang): BodyPatch2 / JointPatch2 records (12 / 12 words) applied at the
    // top of the step, the moved-body list the step builds device-local and copies to the host (8
    // words per record, see csCompactMoved2D) and its device counter.
    public uint *bPatch;
    public uint *jPatch;
    public uint *moved;
    public int *movedCount;
    public uint8_t *saveAwake; // awake flags at the start of the step (after patches and impulses)
    public vec2 *savePos;      // pose at the start of the step (x^t before integrate), see csIntegrateJointInit2D
    public float *saveAng;

    // Scalars (an even count of 4-byte words).
    public uint bodyCount;
    public uint dynCount;
    public uint statCount;   // big static shapes in `stat`
    public uint tableMask;
    public uint ignoreCount;
    public uint pairCap;
    public uint contactCap;
    public uint prevCount;
    public uint jointCount;
    public float dt;
    public float gravityX;
    public float alpha;
    public float betaLin;
    public float betaAng;
    public float gamma;
    public float dualDamping;
    public float dualDampVel;
    public float cellSize;
    public float killY;
    public uint autoK;
    public uint killOn;
    public int colorBound;
    public uint sleepOn;
    public uint sleepFrames;
    public float sleepLin;    // resting below twice this speed; also the speed at which a mover wakes a sleeper
    public float sleepAng;    // the same for angular speed (rad/s)
    public float maxSpeed;    // a body faster than this is treated as exploded and removed
    public float sleepDisp;   // metres of travel allowed over the rest window
    public float sleepFrac;   // joint load fraction (of its break force) that keeps its bodies awake
    public uint contactCapPts; // max points per manifold; 0 = no cap
    public uint gridCount;    // shapes in the grid list
    public uint gridDyn;      // of which the first gridDyn are dynamic
    public uint overCount;    // oversize dynamic shapes
    public uint frictionMix;  // 0 / 1 geometric sqrt(a b) (the default), 2 min, 3 max, 4 average
    public uint evOff[6];
    public uint evCap[6];
    public float hitThreshold;
    public uint evOn;
    public uint impulseCount;
    public uint ccdOn;        // continuous collision enabled
    public float ccdSafety;   // a body is fast when it moves more than this times its min extent
    public float gravityY;
    public uint bPatchCount;
    public uint jPatchCount;
};

public struct Args2
{
    public World2 *w;
    public uint a;
    public uint b;
    public uint c;
    public uint d;
};

// Argument block of the batched query kernels (query2d.slang). The push constant points straight
// at one of these (not at an Args2). Same layout rule as World2: pointers first, then an even
// number of 4-byte scalars. Mirrored by QueryBatch2Gpu in world2d_args.h.
public struct QueryBatch2
{
    public World2 *w;
    public float4 *qBox;      // per shape: world AABB (lo.xy, hi.xy); lo > hi = not queryable
    public int *qHead;        // query-local cell table, headMask + 1 entries
    public int *qNext;        // 4 chain links per shape: node = shape * 4 + corner
    public int *qBig;         // shapes too big for a cell (and every oversize / big static one)
    public int *qCount;       // [0] number of qBig entries
    public float4 *rays;      // (origin.xy, translation.xy)
    public float4 *aabbs;     // (lo.xy, hi.xy)
    public float4 *outHit;    // two per ray: (asfloat(shape or -1), fraction, point.xy), (normal.xy, 0, 0)
    public int *outIds;       // maxPer per box
    public int *outCounts;    // per box: overlaps found (may exceed maxPer)
    public uint64_t cat;      // the query filter's categoryBits
    public uint64_t mask;     // and maskBits
    public uint headMask;
    public uint count;        // rays or boxes in this dispatch
    public uint maxPer;
    public uint insertCount;  // entries of the grid, oversize and big-static lists
    public uint pad0;
    public uint pad1;
};
