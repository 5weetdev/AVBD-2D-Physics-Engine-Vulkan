#pragma once

// 2D maths and the solver parameter block for the 2D AVBD solver. Backend-free: scene,
// Vulkan world and C ABI all take these by value. Everything lives in namespace avbd2d.

#include <cmath>
#include <cstdint>

namespace avbd2d
{

struct Vec2
{
    float x = 0.0f, y = 0.0f;
    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}
};

inline Vec2 operator+(Vec2 a, Vec2 b) { return Vec2(a.x + b.x, a.y + b.y); }
inline Vec2 operator-(Vec2 a, Vec2 b) { return Vec2(a.x - b.x, a.y - b.y); }
inline Vec2 operator-(Vec2 a) { return Vec2(-a.x, -a.y); }
inline Vec2 operator*(Vec2 a, float s) { return Vec2(a.x * s, a.y * s); }
inline Vec2 operator*(float s, Vec2 a) { return Vec2(a.x * s, a.y * s); }
inline float dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline float cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
inline float lengthSq(Vec2 a) { return dot(a, a); }
inline float length(Vec2 a) { return std::sqrt(dot(a, a)); }
inline Vec2 perp(Vec2 a) { return Vec2(-a.y, a.x); }
// Rotation by `angle` radians (counter-clockwise).
inline Vec2 rotate(float angle, Vec2 v)
{
    const float c = std::cos(angle), s = std::sin(angle);
    return Vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

constexpr float kPi = 3.14159265358979323846f;

enum Shape2D : uint8_t
{
    SHAPE2D_BOX = 0,     // half extents in Scene2D::half
    SHAPE2D_CIRCLE = 1,  // radius half.x; an offset circle has one vertex (its centre) in the pool
    SHAPE2D_POLYGON = 2, // convex, 3..8 CCW vertices about the centroid, optional rounding radius
    SHAPE2D_CAPSULE = 3, // two vertices (the core segment) and a radius
    SHAPE2D_SEGMENT = 4, // two vertices, radius 0, static
    SHAPE2D_CHAIN = 5,   // one-sided chain segment: ghost1, p1, p2, ghost2; static
};

// Per-shape event flags (Scene2D::shFlags; mirrored in types2d.slang).
enum ShapeFlag2D : uint8_t
{
    SHAPE_FLAG_SENSOR = 1,          // detects overlaps, never collides
    SHAPE_FLAG_CONTACT_EVENTS = 2,  // begin/end touch events (default on)
    SHAPE_FLAG_HIT_EVENTS = 4,      // hit events above the world threshold (default off)
    SHAPE_FLAG_SENSOR_EVENTS = 8,   // a sensor may report this shape as a visitor (default on)
    SHAPE_FLAG_DEFAULT = SHAPE_FLAG_CONTACT_EVENTS | SHAPE_FLAG_SENSOR_EVENTS,
};

constexpr int kMaxPolygonVertices = 8;
constexpr float kLinearSlop = 0.005f; // Box2D's B2_LINEAR_SLOP, the authoring tolerances below use it

// Solver tunables. Gravity is a vector (default (0, -10)).
struct SolverParams2D
{
    float dt = 1.0f / 60.0f;
    Vec2 gravity{0.0f, -10.0f}; // acceleration (m/s^2); the default points down -y
    int iterations = 8;

    float alpha = 0.97f;     // Constraint error regularisation (Eq. 18)
    float betaLin = 10000.0f; // Penalty ramp, linear rows
    float betaAng = 100.0f;   // Penalty ramp, angular rows
    float gamma = 0.99f;      // Dual / penalty warm start decay (Eq. 19)

    bool autoKStart = true;     // New contacts start from the effective-mass penalty
    float dualDamping = 0.1f;   // Damps the dual warm start of highly ramped contacts
    float dualDampVel = 0.1f;

    bool killFallenEnabled = true; // Bodies below killY stop being simulated
    float killY = -100.0f;

    float grabStrength = 1.0f; // Multiplier on the mouse-grab spring stiffness
    // Fastest the grab drives a held body (m/s). Each step every grab spring's target is put at
    // most grabMaxSpeed * dt from where its body actually is, towards the cursor: a flick moves
    // the body no faster than this, and a held body that is blocked does not wind the spring up
    // (and fling everything when it slips free).
    float grabMaxSpeed = 20.0f;

    // Per-body sleeping. A body whose speed stays under twice
    // sleepLinVel / sleepAngVel for sleepFrames steps and whose net movement over that whole
    // window is under sleepDisp (metres, plus gyration-weighted rotation) is a sleep
    // candidate; it sleeps once every body it touches or is jointed to is a candidate too. A
    // sleeper wakes when an awake body touching it moves at sleepLinVel / sleepAngVel or
    // faster, when what supports it changes, or when a jointed partner is awake.
    bool sleepEnabled = true;
    int sleepFrames = 64;
    float sleepLinVel = 0.4f;
    float sleepAngVel = 0.4f;
    float sleepDisp = 0.001f;
    float sleepFracture = 0.8f; // a joint loaded past this fraction of its break force keeps its bodies awake

    // Fidelity trade: keep at most this many points per manifold (the
    // deepest first). 0 = off. A 2D manifold has at most 2 points, so only 1 changes anything;
    // it can make a resting box rock.
    int contactCap = 0;

    // A body faster than this (m/s) is treated as exploded and removed, like one that is
    // non-finite or out of range. Keeps a diverged joint from flooding the broadphase.
    float maxSpeed = 1000.0f;

    // How two shapes' friction coefficients combine into a contact's: 0 and 1 the geometric mean
    // sqrt(a b) (the default), 2 the smaller, 3 the larger, 4 the average. See FrictionMix2D.
    int frictionMix = 0;

    // Continuous collision: after the solve, a body that moved more than half its smallest extent
    // is swept against the static geometry (and, for a bullet body, the other dynamic bodies) and
    // clamped back if it ended up past a surface it should have hit (ccd2d.slang).
    bool enableContinuous = true;
};

enum FrictionMix2D : int
{
    FRICTION_MIX_DEFAULT = 0,
    FRICTION_MIX_GEOMETRIC = 1,
    FRICTION_MIX_MIN = 2,
    FRICTION_MIX_MAX = 3,
    FRICTION_MIX_AVERAGE = 4,
};

// What a body is. Static and kinematic bodies have infinite mass (mass 0 on the device); a
// kinematic body moves at its own velocity and is never pushed, a static one does not move.
enum Body2DType : uint8_t
{
    BODY2D_STATIC = 0,
    BODY2D_KINEMATIC = 1,
    BODY2D_DYNAMIC = 2,
};

} // namespace avbd2d
