#pragma once

// Host-side scene queries for the 2D world: ray casts, AABB and shape overlaps, shape casts
// and the explosion impulse computation. Backend-free (no Vulkan): it reads the authored shape
// table of a Scene2D and the current body poses. A lazily rebuilt BVH of the shapes' world
// AABBs prunes the work; the narrow tests are host ports of Box2D's b2RayCast*, b2ShapeDistance
// (GJK) and b2ShapeCast. Sensor shapes are invisible to every query.

#include "maths2d.h"
#include "scene2d.h"

#include <functional>
#include <vector>

namespace avbd2d
{

// A convex point set (a core) and a rounding radius, in world space: a circle is one point, a
// capsule two, a box or polygon up to 8.
struct Proxy2D
{
    Vec2 pts[8];
    int count = 0;
    float radius = 0.0f;
};

Proxy2D makeCircleProxy(Vec2 center, float radius);
Proxy2D makeCapsuleProxy(Vec2 a, Vec2 b, float radius);
Proxy2D makeBoxProxy(Vec2 center, float angle, Vec2 halfExtents, float radius = 0.0f);
Proxy2D makePolygonProxy(const Vec2 *pts, int count, float radius = 0.0f);

struct DistanceResult2D
{
    Vec2 pointA, pointB; // closest points on the two surfaces (equal when overlapping)
    float distance = 0.0f;
};
// Distance between two proxies' surfaces (radii applied); 0 when they overlap.
DistanceResult2D proxyDistance(const Proxy2D &a, const Proxy2D &b);

struct RayHit2D
{
    int shape = -1;
    int body = -1;
    Vec2 point;
    Vec2 normal;
    float fraction = 1.0f; // of the translation
};

// What a query reads: the shape table and the pose of every body.
struct QueryView2D
{
    const Scene2D *scene = nullptr;
    const Vec2 *pos = nullptr;
    const float *ang = nullptr;
};

struct ExplosionImpulse2D
{
    int body;
    Vec2 dv;
    float dw;
};

class Query2D
{
public:
    // The BVH is rebuilt on the next query after this.
    void invalidate() { m_dirty = true; }

    // Callback result, as Box2D's b2CastResultFcn: -1 ignore this hit, 0 terminate, a fraction
    // clips the ray to it, 1 continue unclipped.
    using CastFn = std::function<float(const RayHit2D &)>;
    void castRay(const QueryView2D &v, Vec2 origin, Vec2 translation, const CastFn &fn);
    bool castRayClosest(const QueryView2D &v, Vec2 origin, Vec2 translation, RayHit2D &out);

    // Sweeps `proxy` along `translation` (the proxy's points are in world space at fraction 0).
    void castShape(const QueryView2D &v, const Proxy2D &proxy, Vec2 translation, const CastFn &fn);
    bool castShapeClosest(const QueryView2D &v, const Proxy2D &proxy, Vec2 translation, RayHit2D &out);

    // Shapes whose world AABB overlaps [lo, hi]; the callback returns false to stop.
    using ShapeFn = std::function<bool(int shape)>;
    void overlapAABB(const QueryView2D &v, Vec2 lo, Vec2 hi, const ShapeFn &fn);
    // Shapes whose surface overlaps the proxy.
    void overlapShape(const QueryView2D &v, const Proxy2D &proxy, const ShapeFn &fn);

    // Box2D's b2World_Explode per dynamic body, summed per body.
    void explosion(const QueryView2D &v, Vec2 pos, float radius, float falloff, float impulsePerLength,
                   std::vector<ExplosionImpulse2D> &out);

private:
    struct Node
    {
        Vec2 lo, hi;
        int left = -1, right = -1, shape = -1;
    };
    void rebuild(const QueryView2D &v);
    int build(std::vector<int> &ids, int begin, int end);

    std::vector<Node> m_nodes;
    std::vector<Vec2> m_lo, m_hi;
    int m_root = -1;
    bool m_dirty = true;
    int m_builtShapes = -1;
};

} // namespace avbd2d
