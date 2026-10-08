/*
 * avbd2d.h -- the C ABI of the 2D AVBD solver, shaped like Box2D v3.
 *
 * A standalone 2D physics world on the GPU. Names, structs and the call order follow Box2D v3, so
 * b2CreateBody(world, &def) ports mechanically to avbd2d_create_body(world, &def): a world is
 * created from a world def, bodies carry shapes made from defs (b2ShapeDef -> Avbd2dShapeDef),
 * joints come one def and one create call per type, and the world is advanced with avbd2d_step.
 * Gravity is a 2-vector (default (0, -10)).
 *
 * Conventions
 *  - World calls take the world handle first; body, shape, joint and chain calls take their id
 *    (as Box2D does) and find the world inside it. Every call returns Avbd2dResult. A null, never
 *    created or destroyed world handle yields AVBD2D_ERR_NULL_HANDLE; an id that is not valid
 *    (never issued, or destroyed: slots are reused with a new generation) yields
 *    AVBD2D_ERR_INVALID_ARG. Neither crashes.
 *  - Every struct has a default filler (avbd2d_default_*); fill it, then override fields.
 *  - Calls on one world must not overlap; different worlds are independent.
 *  - Positions in the API are body ORIGINS (b2Body_GetPosition); the solver integrates the centre
 *    of mass, and avbd2d_body_get_world_center returns that.
 *  - Creating and destroying are live: nothing is rebuilt. Creations are queued and flushed once
 *    at the next step. Every setter is queued too and applied at the top of the next step; getters
 *    read the last completed step plus what was queued since for pose and velocity. Forces and
 *    torques act for the next step only; impulses change velocity before it.
 *  - avbd2d_step = avbd2d_step_begin + avbd2d_step_end. step_begin submits the whole step to the
 *    GPU and returns; between the two, getters, setters, creates and destroys are allowed (they
 *    read the previous state and queue for the next step). Queries, explosions and grab settle the
 *    step in flight first.
 *  - Joints are authored in body-local anchors (relative to each body's origin) and must connect
 *    two bodies (use a static body for the ground).
 *  - Shapes can be added to and destroyed on a live body: the body's whole shape set is replaced at
 *    the next step (mass and centre of mass are recomputed, pose and velocity are kept, the
 *    body's shapes get new solver ids, so its unchanged shapes may report an END and a BEGIN
 *    event). A body with no shape is a real body (a dynamic one has mass 1 and inertia 0.5*0.25).
 *    A dynamic body whose shapes have density 0 gets that same default mass.
 *
 * Anything outside this surface (render-target interop, ...) returns AVBD2D_ERR_UNSUPPORTED.
 */
#ifndef AVBD2D_H
#define AVBD2D_H

#include <stdint.h>

#define AVBD2D_ABI_VERSION_MAJOR 3
#define AVBD2D_ABI_VERSION_MINOR 0

#if defined(_WIN32)
#if defined(AVBD2D_BUILD_SHARED)
#define AVBD2D_API __declspec(dllexport)
#elif defined(AVBD2D_USE_SHARED)
#define AVBD2D_API __declspec(dllimport)
#else
#define AVBD2D_API
#endif
#else
#define AVBD2D_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Avbd2dWorld Avbd2dWorld;

typedef enum Avbd2dResult
{
    AVBD2D_OK = 0,
    AVBD2D_ERR_NULL_HANDLE = -1,   /* null, never created, or destroyed world handle */
    AVBD2D_ERR_INVALID_ARG = -2,   /* null pointer argument, bad value, an invalid or stale id */
    AVBD2D_ERR_UNSUPPORTED = -3,   /* not part of the 2D solver */
    AVBD2D_ERR_NOT_COMMITTED = -4, /* unused since ABI 3 (there is no commit); kept for the numbering */
    AVBD2D_ERR_DEVICE = -5,        /* no usable Vulkan device */
    AVBD2D_ERR_CAPACITY = -6,      /* a hard capacity ceiling (or the memory budget) clamped the last step */
    AVBD2D_ERR_DEVICE_LOST = -7,   /* the Vulkan device failed or was lost; the world is dead, only
                                    * avbd2d_destroy works on it. Create a new world to continue. */
    AVBD2D_ERR_OUT_OF_MEMORY = -8  /* host or device memory ran out, or the memory budget forbade a
                                    * required allocation */
} Avbd2dResult;

/* ----- Math types (b2Vec2, b2Rot, b2Transform, b2AABB) ------------------------------------- */
typedef struct Avbd2dVec2
{
    float x, y;
} Avbd2dVec2;

typedef struct Avbd2dRot
{
    float c, s; /* cosine and sine of the angle */
} Avbd2dRot;

typedef struct Avbd2dTransform
{
    Avbd2dVec2 p;
    Avbd2dRot q;
} Avbd2dTransform;

typedef struct Avbd2dAABB
{
    Avbd2dVec2 lowerBound, upperBound;
} Avbd2dAABB;

AVBD2D_API Avbd2dRot avbd2d_make_rot(float radians);
AVBD2D_API float avbd2d_rot_get_angle(Avbd2dRot q);

/* ----- Ids ---------------------------------------------------------------------------------- */
/* index1 is the slot + 1 (0 = null id), generation counts the slot's reuses, world0 is the
 * world's index. An id is valid until its object is destroyed. */
typedef struct Avbd2dBodyId
{
    int32_t index1;
    uint16_t world0;
    uint16_t generation;
} Avbd2dBodyId;

typedef struct Avbd2dShapeId
{
    int32_t index1;
    uint16_t world0;
    uint16_t generation;
} Avbd2dShapeId;

typedef struct Avbd2dJointId
{
    int32_t index1;
    uint16_t world0;
    uint16_t generation;
} Avbd2dJointId;

typedef struct Avbd2dChainId
{
    int32_t index1;
    uint16_t world0;
    uint16_t generation;
} Avbd2dChainId;

AVBD2D_API int32_t avbd2d_body_is_valid(Avbd2dBodyId id);
AVBD2D_API int32_t avbd2d_shape_is_valid(Avbd2dShapeId id);
AVBD2D_API int32_t avbd2d_joint_is_valid(Avbd2dJointId id);
AVBD2D_API int32_t avbd2d_chain_is_valid(Avbd2dChainId id);

AVBD2D_API uint32_t avbd2d_abi_version(void); /* (major << 16) | minor */
AVBD2D_API const char *avbd2d_result_string(Avbd2dResult r);

/* ----- World -------------------------------------------------------------------------------- */
typedef enum Avbd2dFrictionMix
{
    AVBD2D_FRICTION_GEOMETRIC = 0, /* sqrt(a b), Box2D's default */
    AVBD2D_FRICTION_MIN = 2,
    AVBD2D_FRICTION_MAX = 3,
    AVBD2D_FRICTION_AVERAGE = 4
} Avbd2dFrictionMix;

typedef struct Avbd2dWorldDef
{
    Avbd2dVec2 gravity;          /* m/s^2, default (0, -10) */
    float timeStep;              /* seconds per avbd2d_step, default 1/60 */
    int32_t iterations;          /* primal/dual iterations per step */
    float alpha;                 /* constraint error regularisation, [0, 1] */
    float betaLin;               /* penalty ramp, linear rows */
    float betaAng;               /* penalty ramp, angular rows */
    float gamma;                 /* warm start decay, [0, 1) */
    int32_t autoKStart;          /* nonzero: new contacts start from the effective-mass penalty */
    float dualDamping;           /* [0, 1] */
    float dualDampVel;           /* m/s at which dual damping fades out */
    int32_t killFallen;          /* nonzero: bodies below killY leave the simulation */
    float killY;
    float grabStrength;          /* multiplier on the grab spring stiffness */
    float grabMaxSpeed;          /* m/s the grab may drive a held body */
    int32_t enableSleep;         /* nonzero: quiet bodies fall asleep */
    int32_t sleepFrames;         /* steps in the rest window; 0 disables sleeping */
    float sleepLinVel;           /* m/s; a body under twice this is resting, a mover this fast wakes sleepers */
    float sleepAngVel;           /* rad/s, likewise */
    float sleepDisp;             /* metres a body may travel over the rest window and still sleep */
    float sleepFracture;         /* fraction of a joint's break force that keeps its bodies awake */
    float maxSpeed;              /* m/s; a faster body is treated as exploded and removed */
    int32_t contactCap;          /* max points per manifold (0 = off; only 1 changes anything in 2D) */
    int32_t frictionMix;         /* Avbd2dFrictionMix */
    float hitEventThreshold;     /* m/s of approach speed above which a hit event fires */
    int32_t enableContinuous;    /* nonzero: fast bodies are swept against static geometry */
    int32_t bodyCapacity;        /* hints: expected counts (advisory, the world grows regardless) */
    int32_t shapeCapacity;
    int32_t jointCapacity;
    uint64_t memoryBudget;       /* bytes of device memory (0 = unlimited), see avbd2d_set_memory_budget */
} Avbd2dWorldDef;

AVBD2D_API Avbd2dWorldDef avbd2d_default_world_def(void);

/* Creates a world on its own Vulkan device (default world def). */
AVBD2D_API Avbd2dResult avbd2d_create(Avbd2dWorld **out_world);
/* Creates a world from a def. */
AVBD2D_API Avbd2dResult avbd2d_create_world(const Avbd2dWorldDef *def, Avbd2dWorld **out_world);
AVBD2D_API Avbd2dResult avbd2d_destroy(Avbd2dWorld *world);

/* A Vulkan device the host application created and owns. Handles are passed as opaque
 * values (VkInstance, VkPhysicalDevice, VkDevice) so this header needs no vulkan.h. The
 * device must be Vulkan 1.3 and be created with: shaderInt64, bufferDeviceAddress,
 * scalarBlockLayout, shaderInt8, timelineSemaphore, shaderBufferInt64Atomics,
 * shaderSharedInt64Atomics, hostQueryReset, synchronization2, and compute subgroup
 * operations (basic, ballot, arithmetic). The queue must support compute. */
typedef struct Avbd2dVulkanDevice
{
    uint32_t struct_size;        /* sizeof(Avbd2dVulkanDevice) */
    void *instance;              /* VkInstance */
    void *physical_device;       /* VkPhysicalDevice */
    void *device;                /* VkDevice */
    uint32_t queue_family_index; /* a family with VK_QUEUE_COMPUTE_BIT */
    uint32_t queue_index;        /* the queue within it; the world submits to it from one thread
                                  * at a time, and the host must not use it during a call */
    uint32_t api_version;        /* the apiVersion the instance was created with */
} Avbd2dVulkanDevice;

/* Like avbd2d_create, but runs on the host's device. The world adopts the handles and never
 * destroys them: destroy the world before the device. Returns AVBD2D_ERR_DEVICE if the
 * physical device lacks something the kernels need. */
AVBD2D_API Avbd2dResult avbd2d_create_with_vulkan(const Avbd2dVulkanDevice *device, Avbd2dWorld **out_world);
AVBD2D_API Avbd2dResult avbd2d_create_world_with_vulkan(const Avbd2dWorldDef *def, const Avbd2dVulkanDevice *device,
                                                        Avbd2dWorld **out_world);

/* Caps the device memory the world allocates (0 = unlimited). Growth that would exceed it is
 * refused: avbd2d_step then returns AVBD2D_ERR_CAPACITY (the step clamps, as at a capacity
 * ceiling). May be set at any time. */
AVBD2D_API Avbd2dResult avbd2d_set_memory_budget(Avbd2dWorld *world, uint64_t bytes);

/* Applies the solver fields of a def to the live world (the capacity hints and memoryBudget
 * are ignored here; use avbd2d_set_memory_budget). */
AVBD2D_API Avbd2dResult avbd2d_set_world_def(Avbd2dWorld *world, const Avbd2dWorldDef *def);
AVBD2D_API Avbd2dResult avbd2d_get_world_def(Avbd2dWorld *world, Avbd2dWorldDef *out);
AVBD2D_API Avbd2dResult avbd2d_world_set_gravity(Avbd2dWorld *world, Avbd2dVec2 gravity);
AVBD2D_API Avbd2dResult avbd2d_world_get_gravity(Avbd2dWorld *world, Avbd2dVec2 *out);

/* One timestep (queued creations flushed, queued setters applied, one GPU submission).
 * Returns AVBD2D_ERR_CAPACITY if a hard ceiling clamped it (the step still ran). */
AVBD2D_API Avbd2dResult avbd2d_step(Avbd2dWorld *world);
/* The async halves. step_end with no step in flight returns AVBD2D_OK and changes nothing. */
AVBD2D_API Avbd2dResult avbd2d_step_begin(Avbd2dWorld *world);
AVBD2D_API Avbd2dResult avbd2d_step_end(Avbd2dWorld *world);
/* Wakes every sleeping body at the start of the next step. */
AVBD2D_API Avbd2dResult avbd2d_wake_all(Avbd2dWorld *world);

typedef struct Avbd2dStats
{
    int32_t bodies;         /* live bodies */
    int32_t joints;         /* live joints */
    int32_t pairs;          /* broadphase pairs in the last step */
    int32_t contacts;       /* contact points in the last step */
    int32_t colors;         /* graph colours in the last step */
    int32_t capacity_reruns;/* passes the last step redid after growing a buffer */
    int32_t submits;        /* queue submissions the last step made (1 unless it reran or flushed) */
    int32_t ok;             /* 0 when a hard capacity ceiling clamped the last step */
    float step_ms;          /* host wall clock of the last step */
    int32_t awake;          /* dynamic bodies solved in the last step */
    int32_t asleep;         /* dynamic bodies asleep */
    int32_t removed;        /* bodies removed so far (fell below killY, or exploded) */
    int32_t broken_joints;  /* joints fractured so far */
    float gpu_ms;           /* GPU time of the last step; 0 if timestamps are unavailable */
} Avbd2dStats;

AVBD2D_API Avbd2dResult avbd2d_get_stats(Avbd2dWorld *world, Avbd2dStats *out);

/* What this build supports. 1 = yes, 0 = no. */
typedef struct Avbd2dCapabilities
{
    uint32_t struct_size;          /* set to sizeof(Avbd2dCapabilities) before the call */
    int32_t restitution;           /* shape restitution is stored but has no effect */
    int32_t joint_damping_ratio;   /* dampingRatio is accepted and ignored */
    int32_t determinism;           /* no bit-exact determinism guarantee */
    int32_t continuous;            /* swept collision against static geometry (and bullets) */
    int32_t sensors;
    int32_t contact_events;
    int32_t hit_events;
    int32_t joint_events;          /* joint break events */
    int32_t move_events;           /* body move events (moved-only readback) */
    int32_t joint_forces;          /* constraint force / torque readback */
    int32_t explosions;
    int32_t queries;               /* ray, AABB, shape overlap and cast */
    int32_t chains;
    int32_t live_body_create_destroy;
    int32_t live_shape_add;        /* shape add / destroy on a live body: 1 */
    int32_t per_body_sleep_disable;/* BodyDef.enableSleep is accepted and ignored */
    int32_t set_body_type;
    int32_t kinematic_bodies;
    int32_t grab;
    int32_t max_polygon_vertices;
} Avbd2dCapabilities;

AVBD2D_API Avbd2dResult avbd2d_get_capabilities(Avbd2dWorld *world, Avbd2dCapabilities *out);

/* ----- Bodies ------------------------------------------------------------------------------- */
typedef enum Avbd2dBodyType
{
    AVBD2D_STATIC_BODY = 0,
    AVBD2D_KINEMATIC_BODY = 1,
    AVBD2D_DYNAMIC_BODY = 2
} Avbd2dBodyType;

typedef struct Avbd2dBodyDef
{
    int32_t type;                /* Avbd2dBodyType */
    Avbd2dVec2 position;         /* the body ORIGIN */
    Avbd2dRot rotation;
    Avbd2dVec2 linearVelocity;
    float angularVelocity;
    float linearDamping;
    float angularDamping;
    float gravityScale;
    int32_t enableSleep;         /* accepted; per-body sleep disable is not supported */
    int32_t isAwake;             /* nonzero (default): starts awake */
    int32_t fixedRotation;
    int32_t isBullet;            /* swept against dynamic bodies too */
    int32_t isEnabled;           /* zero: created disabled (no collision, no solve) */
    int32_t lockLinearX;
    int32_t lockLinearY;
    void *userData;
} Avbd2dBodyDef;

AVBD2D_API Avbd2dBodyDef avbd2d_default_body_def(void);

/* The body is created at once (the id is valid), and enters the simulation at the next step. */
AVBD2D_API Avbd2dResult avbd2d_create_body(Avbd2dWorld *world, const Avbd2dBodyDef *def, Avbd2dBodyId *out_id);
/* Destroys the body, its shapes and its joints. */
AVBD2D_API Avbd2dResult avbd2d_destroy_body(Avbd2dBodyId id);

AVBD2D_API Avbd2dResult avbd2d_body_get_position(Avbd2dBodyId id, Avbd2dVec2 *out);     /* origin */
AVBD2D_API Avbd2dResult avbd2d_body_get_rotation(Avbd2dBodyId id, Avbd2dRot *out);
AVBD2D_API Avbd2dResult avbd2d_body_get_angle(Avbd2dBodyId id, float *out);
AVBD2D_API Avbd2dResult avbd2d_body_get_transform(Avbd2dBodyId id, Avbd2dTransform *out);
AVBD2D_API Avbd2dResult avbd2d_body_get_world_center(Avbd2dBodyId id, Avbd2dVec2 *out); /* centre of mass */
AVBD2D_API Avbd2dResult avbd2d_body_get_local_center(Avbd2dBodyId id, Avbd2dVec2 *out); /* in the body frame */
AVBD2D_API Avbd2dResult avbd2d_body_get_linear_velocity(Avbd2dBodyId id, Avbd2dVec2 *out);
AVBD2D_API Avbd2dResult avbd2d_body_get_angular_velocity(Avbd2dBodyId id, float *out);
AVBD2D_API Avbd2dResult avbd2d_body_get_mass(Avbd2dBodyId id, float *out);
AVBD2D_API Avbd2dResult avbd2d_body_get_rotational_inertia(Avbd2dBodyId id, float *out); /* about the centre */
AVBD2D_API Avbd2dResult avbd2d_body_get_type(Avbd2dBodyId id, int32_t *out);
AVBD2D_API Avbd2dResult avbd2d_body_is_awake(Avbd2dBodyId id, int32_t *out);
AVBD2D_API Avbd2dResult avbd2d_body_is_enabled(Avbd2dBodyId id, int32_t *out);
AVBD2D_API Avbd2dResult avbd2d_body_get_user_data(Avbd2dBodyId id, void **out);
AVBD2D_API Avbd2dResult avbd2d_body_set_user_data(Avbd2dBodyId id, void *userData);

AVBD2D_API Avbd2dResult avbd2d_body_set_transform(Avbd2dBodyId id, Avbd2dVec2 position, Avbd2dRot rotation);
AVBD2D_API Avbd2dResult avbd2d_body_set_linear_velocity(Avbd2dBodyId id, Avbd2dVec2 v);
AVBD2D_API Avbd2dResult avbd2d_body_set_angular_velocity(Avbd2dBodyId id, float w);
/* Forces and torques act for the next step only (they become an impulse of force * dt). */
AVBD2D_API Avbd2dResult avbd2d_body_apply_force(Avbd2dBodyId id, Avbd2dVec2 force, Avbd2dVec2 worldPoint, int32_t wake);
AVBD2D_API Avbd2dResult avbd2d_body_apply_force_to_center(Avbd2dBodyId id, Avbd2dVec2 force, int32_t wake);
AVBD2D_API Avbd2dResult avbd2d_body_apply_torque(Avbd2dBodyId id, float torque, int32_t wake);
AVBD2D_API Avbd2dResult avbd2d_body_apply_linear_impulse(Avbd2dBodyId id, Avbd2dVec2 impulse, Avbd2dVec2 worldPoint,
                                                         int32_t wake);
AVBD2D_API Avbd2dResult avbd2d_body_apply_linear_impulse_to_center(Avbd2dBodyId id, Avbd2dVec2 impulse, int32_t wake);
AVBD2D_API Avbd2dResult avbd2d_body_apply_angular_impulse(Avbd2dBodyId id, float impulse, int32_t wake);
AVBD2D_API Avbd2dResult avbd2d_body_set_awake(Avbd2dBodyId id, int32_t awake);
AVBD2D_API Avbd2dResult avbd2d_body_enable(Avbd2dBodyId id);
AVBD2D_API Avbd2dResult avbd2d_body_disable(Avbd2dBodyId id);

/* ----- Shapes ------------------------------------------------------------------------------- */
typedef struct Avbd2dFilter
{
    uint64_t categoryBits;
    uint64_t maskBits;
    int32_t groupIndex;
} Avbd2dFilter;

typedef struct Avbd2dShapeDef
{
    void *userData;
    float friction;
    float restitution;           /* stored, no effect (see capabilities) */
    float rollingResistance;     /* round shapes */
    float tangentSpeed;          /* conveyor surface speed */
    int32_t userMaterialId;
    float density;               /* 0 on a dynamic body's only shapes makes it static-like */
    Avbd2dFilter filter;
    int32_t isSensor;            /* overlaps are reported, nothing collides */
    int32_t enableSensorEvents;
    int32_t enableContactEvents;
    int32_t enableHitEvents;
} Avbd2dShapeDef;

AVBD2D_API Avbd2dShapeDef avbd2d_default_shape_def(void);

#define AVBD2D_MAX_POLYGON_VERTICES 8

typedef struct Avbd2dPolygon
{
    Avbd2dVec2 vertices[AVBD2D_MAX_POLYGON_VERTICES]; /* counter-clockwise */
    Avbd2dVec2 normals[AVBD2D_MAX_POLYGON_VERTICES];  /* outward */
    Avbd2dVec2 centroid;
    float radius;                                     /* rounding radius */
    int32_t count;
} Avbd2dPolygon;

typedef struct Avbd2dCircle
{
    Avbd2dVec2 center;
    float radius;
} Avbd2dCircle;

typedef struct Avbd2dCapsule
{
    Avbd2dVec2 center1, center2;
    float radius;
} Avbd2dCapsule;

typedef struct Avbd2dSegment
{
    Avbd2dVec2 point1, point2;
} Avbd2dSegment;

AVBD2D_API Avbd2dPolygon avbd2d_make_box(float halfWidth, float halfHeight);
AVBD2D_API Avbd2dPolygon avbd2d_make_offset_box(float halfWidth, float halfHeight, Avbd2dVec2 center, Avbd2dRot rotation);
AVBD2D_API Avbd2dPolygon avbd2d_make_rounded_box(float halfWidth, float halfHeight, float radius);
/* The convex hull of up to AVBD2D_MAX_POLYGON_VERTICES points, rounded by radius. A degenerate
 * hull returns AVBD2D_ERR_INVALID_ARG. */
AVBD2D_API Avbd2dResult avbd2d_make_polygon(const Avbd2dVec2 *points, int32_t count, float radius, Avbd2dPolygon *out);

AVBD2D_API Avbd2dResult avbd2d_create_polygon_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                    const Avbd2dPolygon *polygon, Avbd2dShapeId *out_id);
AVBD2D_API Avbd2dResult avbd2d_create_circle_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                   const Avbd2dCircle *circle, Avbd2dShapeId *out_id);
AVBD2D_API Avbd2dResult avbd2d_create_capsule_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                    const Avbd2dCapsule *capsule, Avbd2dShapeId *out_id);
/* A two-sided massless segment. */
AVBD2D_API Avbd2dResult avbd2d_create_segment_shape(Avbd2dBodyId body, const Avbd2dShapeDef *def,
                                                    const Avbd2dSegment *segment, Avbd2dShapeId *out_id);

typedef struct Avbd2dChainDef
{
    void *userData;
    const Avbd2dVec2 *points;
    int32_t count;               /* loop: >= 3, open: >= 2 */
    float friction;
    float restitution;
    Avbd2dFilter filter;
    int32_t isLoop;
} Avbd2dChainDef;

AVBD2D_API Avbd2dChainDef avbd2d_default_chain_def(void);
/* One-sided chain segments (solid on the right of each p_i -> p_i+1, as in Box2D) on `body`,
 * with ghost vertices so a body does not catch on the joints. Massless. */
AVBD2D_API Avbd2dResult avbd2d_create_chain(Avbd2dBodyId body, const Avbd2dChainDef *def, Avbd2dChainId *out_id);

AVBD2D_API Avbd2dResult avbd2d_destroy_shape(Avbd2dShapeId id);
AVBD2D_API Avbd2dResult avbd2d_destroy_chain(Avbd2dChainId id);
AVBD2D_API Avbd2dResult avbd2d_shape_get_body(Avbd2dShapeId id, Avbd2dBodyId *out);
AVBD2D_API Avbd2dResult avbd2d_shape_get_user_data(Avbd2dShapeId id, void **out);

/* ----- Joints ------------------------------------------------------------------------------- */
/* Every def connects two bodies; anchors are relative to each body's origin. breakForce is the
 * load (N on the joint's linear rows) that cuts the joint; >= 1e38 = unbreakable (the default). */
typedef struct Avbd2dRevoluteJointDef
{
    Avbd2dBodyId bodyIdA, bodyIdB;
    Avbd2dVec2 localAnchorA, localAnchorB;
    float referenceAngle;
    int32_t enableSpring;
    float hertz, dampingRatio;
    int32_t enableLimit;
    float lowerAngle, upperAngle;
    int32_t enableMotor;
    float motorSpeed, maxMotorTorque;
    int32_t collideConnected;
    float breakForce;
    void *userData;
} Avbd2dRevoluteJointDef;

typedef struct Avbd2dWeldJointDef
{
    Avbd2dBodyId bodyIdA, bodyIdB;
    Avbd2dVec2 localAnchorA, localAnchorB;
    float referenceAngle;
    float linearHertz, angularHertz;       /* 0 = rigid */
    float linearDampingRatio, angularDampingRatio;
    int32_t collideConnected;
    float breakForce;
    void *userData;
} Avbd2dWeldJointDef;

typedef struct Avbd2dPrismaticJointDef
{
    Avbd2dBodyId bodyIdA, bodyIdB;
    Avbd2dVec2 localAnchorA, localAnchorB;
    Avbd2dVec2 localAxisA;
    float referenceAngle;
    int32_t enableSpring;
    float hertz, dampingRatio;
    int32_t enableLimit;
    float lowerTranslation, upperTranslation;
    int32_t enableMotor;
    float maxMotorForce, motorSpeed;
    int32_t collideConnected;
    float breakForce;
    void *userData;
} Avbd2dPrismaticJointDef;

typedef struct Avbd2dWheelJointDef
{
    Avbd2dBodyId bodyIdA, bodyIdB;
    Avbd2dVec2 localAnchorA, localAnchorB;
    Avbd2dVec2 localAxisA;
    int32_t enableSpring;
    float hertz, dampingRatio;
    int32_t enableLimit;
    float lowerTranslation, upperTranslation;
    int32_t enableMotor;
    float maxMotorTorque, motorSpeed;
    int32_t collideConnected;
    float breakForce;
    void *userData;
} Avbd2dWheelJointDef;

typedef struct Avbd2dDistanceJointDef
{
    Avbd2dBodyId bodyIdA, bodyIdB;
    Avbd2dVec2 localAnchorA, localAnchorB;
    float length;
    int32_t enableSpring;
    float hertz, dampingRatio;
    int32_t enableLimit;
    float minLength, maxLength;
    int32_t enableMotor;
    float maxMotorForce, motorSpeed;
    int32_t collideConnected;
    float breakForce;
    void *userData;
} Avbd2dDistanceJointDef;

typedef struct Avbd2dMotorJointDef
{
    Avbd2dBodyId bodyIdA, bodyIdB;
    Avbd2dVec2 linearOffset;     /* target offset of B in A's frame */
    float angularOffset;
    float maxForce, maxTorque;
    int32_t collideConnected;
    float breakForce;
    void *userData;
} Avbd2dMotorJointDef;

/* Only turns the pair's collision off (b2NullJoint). */
typedef struct Avbd2dFilterJointDef
{
    Avbd2dBodyId bodyIdA, bodyIdB;
    void *userData;
} Avbd2dFilterJointDef;

AVBD2D_API Avbd2dRevoluteJointDef avbd2d_default_revolute_joint_def(void);
AVBD2D_API Avbd2dWeldJointDef avbd2d_default_weld_joint_def(void);
AVBD2D_API Avbd2dPrismaticJointDef avbd2d_default_prismatic_joint_def(void);
AVBD2D_API Avbd2dWheelJointDef avbd2d_default_wheel_joint_def(void);
AVBD2D_API Avbd2dDistanceJointDef avbd2d_default_distance_joint_def(void);
AVBD2D_API Avbd2dMotorJointDef avbd2d_default_motor_joint_def(void);
AVBD2D_API Avbd2dFilterJointDef avbd2d_default_filter_joint_def(void);

AVBD2D_API Avbd2dResult avbd2d_create_revolute_joint(Avbd2dWorld *world, const Avbd2dRevoluteJointDef *def,
                                                     Avbd2dJointId *out_id);
AVBD2D_API Avbd2dResult avbd2d_create_weld_joint(Avbd2dWorld *world, const Avbd2dWeldJointDef *def,
                                                 Avbd2dJointId *out_id);
AVBD2D_API Avbd2dResult avbd2d_create_prismatic_joint(Avbd2dWorld *world, const Avbd2dPrismaticJointDef *def,
                                                      Avbd2dJointId *out_id);
AVBD2D_API Avbd2dResult avbd2d_create_wheel_joint(Avbd2dWorld *world, const Avbd2dWheelJointDef *def,
                                                  Avbd2dJointId *out_id);
AVBD2D_API Avbd2dResult avbd2d_create_distance_joint(Avbd2dWorld *world, const Avbd2dDistanceJointDef *def,
                                                     Avbd2dJointId *out_id);
AVBD2D_API Avbd2dResult avbd2d_create_motor_joint(Avbd2dWorld *world, const Avbd2dMotorJointDef *def,
                                                  Avbd2dJointId *out_id);
AVBD2D_API Avbd2dResult avbd2d_create_filter_joint(Avbd2dWorld *world, const Avbd2dFilterJointDef *def,
                                                   Avbd2dJointId *out_id);
/* The joint stops acting at the next step. A collision pair the joint disabled stays disabled. */
AVBD2D_API Avbd2dResult avbd2d_destroy_joint(Avbd2dJointId id);
AVBD2D_API Avbd2dResult avbd2d_joint_get_user_data(Avbd2dJointId id, void **out);
/* Constraint force on body B and the torque, from the last step (zero before the joint is live). */
AVBD2D_API Avbd2dResult avbd2d_joint_get_constraint_force(Avbd2dJointId id, Avbd2dVec2 *out);
AVBD2D_API Avbd2dResult avbd2d_joint_get_constraint_torque(Avbd2dJointId id, float *out);
AVBD2D_API Avbd2dResult avbd2d_joint_wake_bodies(Avbd2dJointId id);

AVBD2D_API Avbd2dResult avbd2d_revolute_joint_enable_limit(Avbd2dJointId id, int32_t flag);
AVBD2D_API Avbd2dResult avbd2d_revolute_joint_set_limits(Avbd2dJointId id, float lower, float upper);
AVBD2D_API Avbd2dResult avbd2d_revolute_joint_enable_motor(Avbd2dJointId id, int32_t flag);
AVBD2D_API Avbd2dResult avbd2d_revolute_joint_set_motor_speed(Avbd2dJointId id, float speed);
AVBD2D_API Avbd2dResult avbd2d_revolute_joint_set_max_motor_torque(Avbd2dJointId id, float torque);

AVBD2D_API Avbd2dResult avbd2d_prismatic_joint_enable_limit(Avbd2dJointId id, int32_t flag);
AVBD2D_API Avbd2dResult avbd2d_prismatic_joint_set_limits(Avbd2dJointId id, float lower, float upper);
AVBD2D_API Avbd2dResult avbd2d_prismatic_joint_enable_motor(Avbd2dJointId id, int32_t flag);
AVBD2D_API Avbd2dResult avbd2d_prismatic_joint_set_motor_speed(Avbd2dJointId id, float speed);
AVBD2D_API Avbd2dResult avbd2d_prismatic_joint_set_max_motor_force(Avbd2dJointId id, float force);

AVBD2D_API Avbd2dResult avbd2d_wheel_joint_enable_limit(Avbd2dJointId id, int32_t flag);
AVBD2D_API Avbd2dResult avbd2d_wheel_joint_set_limits(Avbd2dJointId id, float lower, float upper);
AVBD2D_API Avbd2dResult avbd2d_wheel_joint_enable_motor(Avbd2dJointId id, int32_t flag);
AVBD2D_API Avbd2dResult avbd2d_wheel_joint_set_motor_speed(Avbd2dJointId id, float speed);
AVBD2D_API Avbd2dResult avbd2d_wheel_joint_set_max_motor_torque(Avbd2dJointId id, float torque);

AVBD2D_API Avbd2dResult avbd2d_distance_joint_enable_limit(Avbd2dJointId id, int32_t flag);
AVBD2D_API Avbd2dResult avbd2d_distance_joint_set_length_range(Avbd2dJointId id, float minLength, float maxLength);
AVBD2D_API Avbd2dResult avbd2d_distance_joint_enable_motor(Avbd2dJointId id, int32_t flag);
AVBD2D_API Avbd2dResult avbd2d_distance_joint_set_motor_speed(Avbd2dJointId id, float speed);
AVBD2D_API Avbd2dResult avbd2d_distance_joint_set_max_motor_force(Avbd2dJointId id, float force);

/* ----- Events ------------------------------------------------------------------------------- */
/* Each list is valid until the next avbd2d_step_end (or avbd2d_step). */
typedef struct Avbd2dBodyMoveEvent
{
    Avbd2dTransform transform;   /* body origin and rotation */
    Avbd2dBodyId bodyId;
    void *userData;
    int32_t fellAsleep;          /* awake at the start of the step, asleep at its end */
} Avbd2dBodyMoveEvent;

/* Bodies that were awake at the start of the step or at its end (and kinematic bodies with a
 * velocity). A world at rest returns none. */
typedef struct Avbd2dBodyEvents
{
    const Avbd2dBodyMoveEvent *moveEvents;
    int32_t moveCount;
} Avbd2dBodyEvents;

typedef struct Avbd2dContactBeginTouchEvent
{
    Avbd2dShapeId shapeIdA, shapeIdB;
} Avbd2dContactBeginTouchEvent;

typedef struct Avbd2dContactEndTouchEvent
{
    Avbd2dShapeId shapeIdA, shapeIdB;
} Avbd2dContactEndTouchEvent;

typedef struct Avbd2dContactHitEvent
{
    Avbd2dShapeId shapeIdA, shapeIdB;
    Avbd2dVec2 point, normal;
    float approachSpeed;
} Avbd2dContactHitEvent;

typedef struct Avbd2dContactEvents
{
    const Avbd2dContactBeginTouchEvent *beginEvents;
    const Avbd2dContactEndTouchEvent *endEvents;
    const Avbd2dContactHitEvent *hitEvents;
    int32_t beginCount, endCount, hitCount;
} Avbd2dContactEvents;

typedef struct Avbd2dSensorBeginTouchEvent
{
    Avbd2dShapeId sensorShapeId, visitorShapeId;
} Avbd2dSensorBeginTouchEvent;

typedef struct Avbd2dSensorEndTouchEvent
{
    Avbd2dShapeId sensorShapeId, visitorShapeId;
} Avbd2dSensorEndTouchEvent;

typedef struct Avbd2dSensorEvents
{
    const Avbd2dSensorBeginTouchEvent *beginEvents;
    const Avbd2dSensorEndTouchEvent *endEvents;
    int32_t beginCount, endCount;
} Avbd2dSensorEvents;

/* A joint cut by its break force (not one the host destroyed). */
typedef struct Avbd2dJointEvent
{
    Avbd2dJointId jointId;
    void *userData;
} Avbd2dJointEvent;

typedef struct Avbd2dJointEvents
{
    const Avbd2dJointEvent *jointEvents;
    int32_t count;
} Avbd2dJointEvents;

AVBD2D_API Avbd2dResult avbd2d_world_get_body_events(Avbd2dWorld *world, Avbd2dBodyEvents *out);
AVBD2D_API Avbd2dResult avbd2d_world_get_contact_events(Avbd2dWorld *world, Avbd2dContactEvents *out);
AVBD2D_API Avbd2dResult avbd2d_world_get_sensor_events(Avbd2dWorld *world, Avbd2dSensorEvents *out);
AVBD2D_API Avbd2dResult avbd2d_world_get_joint_events(Avbd2dWorld *world, Avbd2dJointEvents *out);

/* All live bodies' origins and rotations from the host mirror (no device access). At most
 * `capacity` records are written; `out_count` receives the number of live bodies. */
typedef struct Avbd2dBodyTransform
{
    Avbd2dBodyId bodyId;
    Avbd2dTransform transform;
} Avbd2dBodyTransform;

AVBD2D_API Avbd2dResult avbd2d_world_get_transforms(Avbd2dWorld *world, Avbd2dBodyTransform *out, int32_t capacity,
                                                    int32_t *out_count);

/* ----- Explosions and queries --------------------------------------------------------------- */
typedef struct Avbd2dExplosionDef
{
    uint64_t maskBits;           /* accepted, not applied */
    Avbd2dVec2 position;
    float radius;
    float falloff;
    float impulsePerLength;
} Avbd2dExplosionDef;

AVBD2D_API Avbd2dExplosionDef avbd2d_default_explosion_def(void);
/* Applied at the top of the next step. Settles a step in flight first. */
AVBD2D_API Avbd2dResult avbd2d_world_explode(Avbd2dWorld *world, const Avbd2dExplosionDef *def);

typedef struct Avbd2dQueryFilter
{
    uint64_t categoryBits;
    uint64_t maskBits;
} Avbd2dQueryFilter;

AVBD2D_API Avbd2dQueryFilter avbd2d_default_query_filter(void);

typedef struct Avbd2dRayResult
{
    Avbd2dShapeId shapeId;
    Avbd2dVec2 point, normal;
    float fraction;
    int32_t hit;
} Avbd2dRayResult;

/* b2CastResultFcn: return -1 to ignore the hit, 0 to stop, a fraction to clip the ray, 1 to go on. */
typedef float (*Avbd2dCastResultFcn)(Avbd2dShapeId shapeId, Avbd2dVec2 point, Avbd2dVec2 normal, float fraction,
                                     void *context);
/* Return nonzero to keep going. */
typedef int32_t (*Avbd2dOverlapResultFcn)(Avbd2dShapeId shapeId, void *context);

typedef struct Avbd2dShapeProxy
{
    Avbd2dVec2 points[AVBD2D_MAX_POLYGON_VERTICES]; /* world space */
    int32_t count;
    float radius;
} Avbd2dShapeProxy;

AVBD2D_API Avbd2dResult avbd2d_make_proxy(const Avbd2dVec2 *points, int32_t count, float radius, Avbd2dShapeProxy *out);
AVBD2D_API Avbd2dResult avbd2d_world_cast_ray_closest(Avbd2dWorld *world, Avbd2dVec2 origin, Avbd2dVec2 translation,
                                                      Avbd2dQueryFilter filter, Avbd2dRayResult *out);
AVBD2D_API Avbd2dResult avbd2d_world_cast_ray(Avbd2dWorld *world, Avbd2dVec2 origin, Avbd2dVec2 translation,
                                              Avbd2dQueryFilter filter, Avbd2dCastResultFcn fcn, void *context);

/* Batched queries. Every query function is legal while a step is in flight (avbd2d_world_step_begin
 * without its end): it answers from the poses of the last finished step and does not wait for the
 * step. Outside a step the batches below run on the GPU against those same poses, in one small
 * submission of their own, and are read back before the call returns; in flight they are answered
 * from the host mirror, with the same results. */
typedef struct Avbd2dRayQuery
{
    Avbd2dVec2 origin, translation;
} Avbd2dRayQuery;

/* One closest-hit ray per entry; out[i] is what avbd2d_world_cast_ray_closest would return (hit = 0
 * and a zeroed record for a miss). out has `count` entries. */
AVBD2D_API Avbd2dResult avbd2d_world_cast_rays_batch(Avbd2dWorld *world, const Avbd2dRayQuery *rays, int32_t count,
                                                     Avbd2dQueryFilter filter, Avbd2dRayResult *out);
/* One AABB per entry. The shapes overlapping box i are written to outShapeIds[i * maxPerQuery ...];
 * outCounts[i] is how many there were, which may exceed maxPerQuery (the extras are not written).
 * The order within a box is unspecified. outShapeIds has count * maxPerQuery entries. */
AVBD2D_API Avbd2dResult avbd2d_world_overlap_aabbs_batch(Avbd2dWorld *world, const Avbd2dAABB *aabbs, int32_t count,
                                                         Avbd2dQueryFilter filter, int32_t maxPerQuery,
                                                         Avbd2dShapeId *outShapeIds, int32_t *outCounts);
AVBD2D_API Avbd2dResult avbd2d_world_overlap_aabb(Avbd2dWorld *world, Avbd2dAABB aabb, Avbd2dQueryFilter filter,
                                                  Avbd2dOverlapResultFcn fcn, void *context);
AVBD2D_API Avbd2dResult avbd2d_world_overlap_shape(Avbd2dWorld *world, const Avbd2dShapeProxy *proxy,
                                                   Avbd2dQueryFilter filter, Avbd2dOverlapResultFcn fcn,
                                                   void *context);
AVBD2D_API Avbd2dResult avbd2d_world_cast_shape(Avbd2dWorld *world, const Avbd2dShapeProxy *proxy,
                                                Avbd2dVec2 translation, Avbd2dQueryFilter filter,
                                                Avbd2dCastResultFcn fcn, void *context);

/* ----- Mouse grab --------------------------------------------------------------------------- */
/* A spring joint from the world point to the dynamic body under it. Settles a step in flight and
 * flushes queued creations first. */
AVBD2D_API Avbd2dResult avbd2d_grab_begin(Avbd2dWorld *world, Avbd2dVec2 point, Avbd2dBodyId *out_body);
AVBD2D_API Avbd2dResult avbd2d_grab_move(Avbd2dWorld *world, Avbd2dVec2 point);
AVBD2D_API Avbd2dResult avbd2d_grab_end(Avbd2dWorld *world);

/* Always AVBD2D_ERR_UNSUPPORTED: there is no graphics interop in this build. */
AVBD2D_API Avbd2dResult avbd2d_register_render_target(Avbd2dWorld *world, void *native_handle);

#ifdef __cplusplus
}
#endif

#endif /* AVBD2D_H */
