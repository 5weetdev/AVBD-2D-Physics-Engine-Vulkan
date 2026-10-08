#pragma once

// The 2D scene catalogue. Each scene is a plain function that fills a Scene2D; nothing here
// knows about Vulkan. Gravity is -y, the ground top surface sits at y = 0 unless stated.

#include "scene2d.h"
#include "scenes2d_box2d.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace avbd2d
{
namespace scenes
{

inline uint32_t lcg(uint32_t &s)
{
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}
inline float rand01(uint32_t &s) { return (float)(lcg(s) & 0xFFFF) / 65535.0f; }

inline int addGround(Scene2D &s, float width = 200.0f, float mu = 0.6f)
{
    return s.addBox(Vec2(0.0f, -0.5f), 0.0f, Vec2(width, 1.0f), 0.0f, mu);
}

// 20-row pyramid of 1.0 x 0.6 boxes.
inline void pyramid(Scene2D &s)
{
    s.clear();
    addGround(s);
    const int rows = 20;
    const Vec2 size(1.0f, 0.6f);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < rows - r; c++)
        {
            const float x = ((float)c - 0.5f * (float)(rows - r - 1)) * (size.x + 0.002f);
            const float y = ((float)r + 0.5f) * size.y;
            s.addBox(Vec2(x, y), 0.0f, size, 1.0f, 0.6f);
        }
    s.cameraCenter = Vec2(0.0f, 6.5f);
    s.cameraHeight = 15.0f;
}

// A 16-box tower with a ball thrown at it.
inline void stack(Scene2D &s)
{
    s.clear();
    addGround(s);
    for (int i = 0; i < 16; i++)
        s.addBox(Vec2(0.0f, (float)i + 0.5f), 0.0f, Vec2(1.0f, 1.0f), 1.0f, 0.5f);
    s.addCircle(Vec2(-14.0f, 3.0f), 0.7f, 2.0f, 0.4f, Vec2(14.0f, 3.0f), 0.0f);
    s.cameraCenter = Vec2(-1.0f, 8.0f);
    s.cameraHeight = 18.0f;
}

// Five blocks on a 20 degree ramp, friction 0.1 .. 0.8: tan(20 deg) = 0.36, so the first
// three slide and the last two hold.
inline void frictionRamp(Scene2D &s)
{
    s.clear();
    addGround(s);
    const float theta = 20.0f * kPi / 180.0f;
    const Vec2 c(0.0f, 5.0f);
    s.addBox(c, theta, Vec2(26.0f, 1.0f), 0.0f, 0.5f);
    const Vec2 along(std::cos(theta), std::sin(theta));
    const Vec2 up(-std::sin(theta), std::cos(theta));
    const float mus[5] = {0.1f, 0.25f, 0.35f, 0.5f, 0.8f};
    for (int i = 0; i < 5; i++)
    {
        const Vec2 p = c + along * (-8.0f + 4.0f * (float)i) + up * (0.5f + 0.5f + 0.01f);
        s.addBox(p, theta, Vec2(1.0f, 1.0f), 1.0f, mus[i]);
    }
    s.cameraCenter = Vec2(0.0f, 5.0f);
    s.cameraHeight = 16.0f;
}

// A sagging revolute-jointed plank bridge with boxes on it, and a horizontal chain
// that swings down from a post. Every joint anchor is placed on the joint point exactly.
inline void chainBridge(Scene2D &s)
{
    s.clear();
    addGround(s, 120.0f);

    // Bridge: parabola through the two pins, one plank per segment sized to fit it.
    const int planks = 20;
    const float L = 9.5f, pinY = 7.0f, sag = 1.2f;
    auto curve = [&](float x) { return Vec2(x, pinY - sag * (1.0f - (x / L) * (x / L))); };
    int prev = -1;
    for (int k = 0; k < planks; k++)
    {
        const Vec2 p0 = curve(-L + 2.0f * L * (float)k / (float)planks);
        const Vec2 p1 = curve(-L + 2.0f * L * (float)(k + 1) / (float)planks);
        const Vec2 d = p1 - p0;
        const float len = length(d);
        const int id = s.addBox((p0 + p1) * 0.5f, std::atan2(d.y, d.x), Vec2(len, 0.25f), 1.0f, 0.6f);
        if (k == 0)
            s.addRevoluteAt(-1, id, p0);
        else
            s.addRevoluteAt(prev, id, p0);
        if (k == planks - 1)
            s.addRevoluteAt(-1, id, p1);
        prev = id;
    }
    // Posts the bridge hangs from (static decoration; they also catch dropped boxes).
    s.addBox(Vec2(-L - 1.0f, pinY - 3.0f), 0.0f, Vec2(1.0f, 6.0f), 0.0f, 0.6f);
    s.addBox(Vec2(L + 1.0f, pinY - 3.0f), 0.0f, Vec2(1.0f, 6.0f), 0.0f, 0.6f);

    // Load.
    for (int i = 0; i < 6; i++)
        s.addBox(Vec2(-5.0f + 2.0f * (float)i, pinY + 3.0f + 0.3f * (float)(i % 2)), 0.1f * (float)i,
                 Vec2(0.9f, 0.9f), 1.0f, 0.5f);
    s.addCircle(Vec2(0.0f, pinY + 6.0f), 0.8f, 1.5f, 0.4f);

    // Chain: pinned at (-22, 14), laid out straight toward +x, falls into a swing.
    const int links = 14;
    const float linkLen = 0.6f;
    const Vec2 pin(-22.0f, 14.0f);
    int prevLink = -1;
    for (int k = 0; k < links; k++)
    {
        const Vec2 p0(pin.x + linkLen * (float)k, pin.y);
        const Vec2 p1(pin.x + linkLen * (float)(k + 1), pin.y);
        const int id = s.addBox((p0 + p1) * 0.5f, 0.0f, Vec2(linkLen, 0.2f), 1.0f, 0.4f);
        s.addRevoluteAt(prevLink, id, p0);
        prevLink = id;
    }
    // (no post: the chain swings freely past its pin)

    s.cameraCenter = Vec2(-4.0f, 7.0f);
    s.cameraHeight = 24.0f;
}

// A cols x rows lattice of small boxes and discs (30% discs) dropped into a container.
// Spacing and sizes are chosen so the lattice starts without overlaps.
inline void pileGrid(Scene2D &s, int cols, int rows, float spacing, float camHeight)
{
    s.clear();
    const float width = (float)cols * spacing + 1.0f;
    const float wallHeight = (float)rows * spacing + 60.0f;
    addGround(s, width + 8.0f);
    s.addBox(Vec2(-width * 0.5f - 1.0f, wallHeight * 0.5f), 0.0f, Vec2(2.0f, wallHeight), 0.0f, 0.4f);
    s.addBox(Vec2(width * 0.5f + 1.0f, wallHeight * 0.5f), 0.0f, Vec2(2.0f, wallHeight), 0.0f, 0.4f);
    uint32_t seed = 12345u;
    const float x0 = -0.5f * (float)(cols - 1) * spacing;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
        {
            const float jitter = 0.08f * spacing;
            const Vec2 p(x0 + spacing * (float)c + jitter * (rand01(seed) - 0.5f),
                         2.0f + spacing * (float)r + jitter * (rand01(seed) - 0.5f));
            if (rand01(seed) < 0.3f)
                s.addCircle(p, spacing * (0.23f + 0.09f * rand01(seed)), 1.0f, 0.4f);
            else
                s.addBox(p, kPi * rand01(seed),
                         Vec2(spacing * (0.46f + 0.20f * rand01(seed)), spacing * (0.46f + 0.20f * rand01(seed))),
                         1.0f, 0.5f);
        }
    s.cameraCenter = Vec2(0.0f, camHeight * 0.4f);
    s.cameraHeight = camHeight;
}

inline void pile10k(Scene2D &s) { pileGrid(s, 100, 100, 0.6f, 50.0f); }
inline void pile100k(Scene2D &s) { pileGrid(s, 400, 250, 0.5f, 110.0f); }
inline void pile200k(Scene2D &s) { pileGrid(s, 500, 400, 0.5f, 170.0f); }
inline void pile500k(Scene2D &s) { pileGrid(s, 800, 625, 0.5f, 270.0f); }

// An empty container twice the 500k pile's size in each direction (802 m wide, 745 m of wall):
// a floor and two walls, open at the top, and no bodies.
inline void bigBox(Scene2D &s)
{
    s.clear();
    const float width = 2.0f * (800.0f * 0.5f + 1.0f);
    const float wallHeight = 2.0f * (625.0f * 0.5f + 60.0f);
    addGround(s, width + 8.0f);
    s.addBox(Vec2(-width * 0.5f - 1.0f, wallHeight * 0.5f), 0.0f, Vec2(2.0f, wallHeight), 0.0f, 0.4f);
    s.addBox(Vec2(width * 0.5f + 1.0f, wallHeight * 0.5f), 0.0f, Vec2(2.0f, wallHeight), 0.0f, 0.4f);
    s.cameraCenter = Vec2(0.0f, 216.0f);
    s.cameraHeight = 540.0f;
}

// Ten brick towers, 16 m wide and 125 courses high, 20,000 bricks in all. Every brick is
// welded to its neighbours -- along the course and to the two bricks below -- and the bottom
// course to the ground: a tall welded-brick stress test. The welds are breakable,
// so a heavy hit tears a tower apart instead of just bouncing off.
inline void towers20k(Scene2D &s)
{
    s.clear();
    addGround(s, 400.0f, 0.8f);
    const int towers = 10, layers = 125, wide = 16;
    const float spacing = 20.0f, brickH = 0.5f;
    const float breakForce = 3000.0f;   // N; a base weld carries roughly 300 N
    const float angStiffness = 1.0e6f;  // stiff angular spring

    struct Brick { int id; float x0, x1; };
    for (int t = 0; t < towers; t++)
    {
        const float cx = ((float)t - 0.5f * (float)(towers - 1)) * spacing;
        const float left = cx - 0.5f * (float)wide;
        std::vector<Brick> prev;
        for (int k = 0; k < layers; k++)
        {
            const float y = ((float)k + 0.5f) * brickH;
            // Even courses: 16 whole bricks. Odd courses: a half brick at each end and
            // 15 whole bricks between, so every brick straddles two below it.
            std::vector<Brick> row;
            float x = left;
            auto laid = [&](float w) {
                const int id = s.addBox(Vec2(x + 0.5f * w, y), 0.0f, Vec2(w, brickH), 1.0f, 0.6f);
                row.push_back({id, x, x + w});
                x += w;
            };
            if (k % 2 == 0)
                for (int i = 0; i < wide; i++)
                    laid(1.0f);
            else
            {
                laid(0.5f);
                for (int i = 0; i < wide - 1; i++)
                    laid(1.0f);
                laid(0.5f);
            }
            for (size_t i = 0; i < row.size(); i++)
            {
                if (i > 0)
                    s.addWeldAt(row[i - 1].id, row[i].id, Vec2(row[i].x0, y), false, breakForce, angStiffness);
                if (k == 0)
                    s.addWeldAt(-1, row[i].id, Vec2(0.5f * (row[i].x0 + row[i].x1), 0.0f), false, breakForce,
                                angStiffness);
                for (const Brick &p : prev)
                {
                    const float lo = std::max(p.x0, row[i].x0), hi = std::min(p.x1, row[i].x1);
                    if (hi - lo > 0.05f)
                        s.addWeldAt(p.id, row[i].id, Vec2(0.5f * (lo + hi), y - 0.5f * brickH), false, breakForce,
                                    angStiffness);
                }
            }
            prev = row;
        }
    }
    s.cameraCenter = Vec2(0.0f, 42.0f);
    s.cameraHeight = 112.0f;
}

inline uint32_t rgb8(float r, float g, float b)
{
    auto q = [](float v) { return (uint32_t)(std::min(std::max(v, 0.0f), 1.0f) * 255.0f + 0.5f); };
    return q(r) | (q(g) << 8) | (q(b) << 16) | (255u << 24);
}

// Box2D's ragdoll (shared/human.c, CreateHuman) standing with its feet at `at`: eleven capsule
// bones -- hip, torso, head, and an upper and a lower part per limb -- 1.55 m tall at scale 1,
// joined by ten revolute joints at Box2D's pivots that break when their linear force passes
// `breakForce` (N; kStiffInf = never). Box2D's density (1) and friction (0.2). Every joint has
// Box2D's angle limits, a friction motor (speed 0, max torque frictionScale * frictionTorque *
// scale) and, when jointHertz > 0, a spring back to the standing pose. Parts of one ragdoll do not
// collide with each other (Box2D's negative group index). The lower legs carry Box2D's rounded
// polygon foot (friction 0.05) as a second shape; feet, like the bones, are category 2, and a foot
// only collides with category 1 (the world), never with another ragdoll. The bodies are
// contiguous, hip first; returns the torso.
inline int addRagdoll(Scene2D &s, Vec2 at, float scale, float breakForce, uint32_t shirt, float frictionTorque = 0.03f,
                      float jointHertz = 5.0f, float jointDamping = 0.5f)
{
    const float k = scale;
    const float pi = 3.14159265f;
    auto P = [&](float x, float y) { return Vec2(at.x + x * k, at.y + y * k); };
    const uint32_t skin = rgb8(1.0f, 0.871f, 0.678f);  // b2_colorNavajoWhite
    const uint32_t pants = rgb8(0.118f, 0.565f, 1.0f); // b2_colorDodgerBlue
    const float mu = 0.2f;
    // A vertical bone: capsule core from y0 to y1 (relative to `at`, at scale 1) on the x = 0 line.
    ShapeDef2D boneDef;
    boneDef.friction = mu;
    boneDef.categoryBits = 2;
    boneDef.maskBits = 1 | 2;
    auto bone = [&](float y0, float y1, float r, uint32_t c) {
        const int id = s.addCapsule(P(0.0f, y0), P(0.0f, y1), r * k, 1.0f, mu, Vec2(), 0.0f, c);
        s.setShapeDef(id, boneDef);
        return id;
    };
    // A lower leg: the capsule bone plus the foot (human.c's footPolygon), one body.
    auto lowerLeg = [&](uint32_t c) {
        const Vec2 foot[4] = {Vec2(-0.03f * k, -0.185f * k), Vec2(0.11f * k, -0.185f * k), Vec2(0.11f * k, -0.16f * k),
                              Vec2(-0.03f * k, -0.14f * k)};
        ShapeDef2D footDef = boneDef;
        footDef.friction = 0.05f;
        footDef.maskBits = 1;
        BodyDef2D bd;
        bd.position = P(0.0f, 0.475f);
        bd.color = c;
        s.beginBody(bd);
        s.addCapsuleShape(Vec2(0.0f, -0.155f * k), Vec2(0.0f, 0.125f * k), 0.045f * k, boneDef);
        s.addPolygonShape(Vec2(), 0.0f, foot, 4, 0.015f * k, footDef);
        return s.endBody();
    };
    // A revolute joint with Box2D's limit (in units of pi), friction motor and spring.
    auto joint = [&](int a, int b, Vec2 pivot, float lower, float upper, float frictionScale, float reference = 0.0f) {
        Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Revolute, a, b, pivot);
        d.referenceAngle = reference;
        d.enableLimit = true;
        d.lower = lower * pi;
        d.upper = upper * pi;
        d.enableMotor = true;
        d.motorSpeed = 0.0f;
        d.maxMotorTorque = frictionScale * frictionTorque * k;
        d.enableSpring = jointHertz > 0.0f;
        d.hertz = jointHertz;
        d.dampingRatio = jointDamping;
        d.breakForce = breakForce;
        s.addJoint(d);
    };

    int parts[11];
    int n = 0;
    const int hip = parts[n++] = bone(0.95f - 0.02f, 0.95f + 0.02f, 0.095f, pants);
    const int torso = parts[n++] = bone(1.2f - 0.135f, 1.2f + 0.135f, 0.09f, shirt);
    joint(hip, torso, P(0.0f, 1.0f), -0.25f, 0.0f, 0.5f);
    const int head = parts[n++] = bone(1.475f - 0.038f, 1.475f + 0.039f, 0.075f, skin);
    joint(torso, head, P(0.0f, 1.4f), -0.3f, 0.1f, 0.25f);
    for (int side = 0; side < 2; side++)
    {
        const int upperLeg = parts[n++] = bone(0.775f - 0.125f, 0.775f + 0.125f, 0.06f, pants);
        joint(hip, upperLeg, P(0.0f, 0.9f), -0.05f, 0.4f, 1.0f);
        const int lowerLegId = parts[n++] = lowerLeg(pants);
        joint(upperLeg, lowerLegId, P(0.0f, 0.625f), -0.5f, -0.02f, 0.5f);
        const int upperArm = parts[n++] = bone(1.225f - 0.125f, 1.225f + 0.125f, 0.035f, shirt);
        joint(torso, upperArm, P(0.0f, 1.35f), -0.1f, 0.8f, 0.5f);
        const int lowerArm = parts[n++] = bone(0.975f - 0.125f, 0.975f + 0.125f, 0.03f, skin);
        joint(upperArm, lowerArm, P(0.0f, 1.1f), -0.2f, 0.3f, 0.1f, 0.25f * pi); // frame A turned by pi / 4
    }
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            s.ignoreCollision(parts[i], parts[j]);
    return torso;
}

// A ragdoll standing in an A pose with its feet at `at`: a circle head, a capsule torso and two
// capsules (upper, lower) per arm and per leg, 1.7 m tall at scale 1, arms and legs splayed
// straight from the shoulders and hips. Joined by revolute joints that break past `breakForce`
// (kStiffInf = never). Every part collides with every other part except its direct joint
// neighbours, so limbs hit each other and the torso. Limp: no joint limits or motors yet.
// Bodies are contiguous, torso first; returns the torso.
inline int addPoseRagdoll(Scene2D &s, Vec2 at, float scale, float breakForce, uint32_t shirt)
{
    const float k = scale;
    const uint32_t skin = rgb8(1.0f, 0.871f, 0.678f);
    const uint32_t pants = rgb8(0.118f, 0.565f, 1.0f);
    const float mu = 0.3f;
    const float legA = 0.175f, armA = 0.6f; // angles from straight down (rad): ~10 and ~35 degrees
    const float upLeg = 0.45f, loLeg = 0.45f, rLoLeg = 0.05f;
    const float upArm = 0.28f, loArm = 0.26f;
    const float hipY = rLoLeg + (upLeg + loLeg) * std::cos(legA); // feet rest on `at`
    auto P = [&](float x, float y) { return Vec2(at.x + x * k, at.y + y * k); };
    // Capsule from a joint (x0, y0) running `len` along a direction `ang` off straight down, sign
    // `side` (-1 left, +1 right). Returns the body and writes the far end.
    auto limb = [&](float x0, float y0, float len, float ang, float side, float r, uint32_t c, float &x1, float &y1) {
        x1 = x0 + side * len * std::sin(ang);
        y1 = y0 - len * std::cos(ang);
        // The core stops a radius short of each joint so the rounded ends just touch there
        // rather than overlap: joined parts collide too, without a standing penetration.
        const float dx = (x1 - x0) / len, dy = (y1 - y0) / len;
        return s.addCapsule(P(x0 + dx * r, y0 + dy * r), P(x1 - dx * r, y1 - dy * r), r * k, 1.0f, mu, Vec2(), 0.0f, c);
    };

    const int torso = s.addCapsule(P(0.0f, hipY + 0.10f), P(0.0f, hipY + 0.35f), 0.10f * k, 1.0f, mu, Vec2(), 0.0f, shirt);
    const int head = s.addCircle(P(0.0f, hipY + 0.545f), 0.095f * k, 1.0f, mu, Vec2(), 0.0f, skin);
    s.addRevoluteAt(torso, head, P(0.0f, hipY + 0.45f), true, breakForce);
    for (int i = 0; i < 2; i++)
    {
        const float side = i == 0 ? -1.0f : 1.0f;
        float x, y, x2, y2;
        const int ua = limb(0.10f * side, hipY + 0.32f, upArm, armA, side, 0.04f, shirt, x, y);
        s.addRevoluteAt(torso, ua, P(0.10f * side, hipY + 0.32f), true, breakForce);
        const int la = limb(x, y, loArm, armA, side, 0.033f, skin, x2, y2);
        s.addRevoluteAt(ua, la, P(x, y), true, breakForce);
        const int ul = limb(0.075f * side, hipY, upLeg, legA, side, 0.06f, pants, x, y);
        s.addRevoluteAt(torso, ul, P(0.075f * side, hipY), true, breakForce);
        const int ll = limb(x, y, loLeg, legA, side, rLoLeg, pants, x2, y2);
        s.addRevoluteAt(ul, ll, P(x, y), true, breakForce);
    }
    return torso;
}

// Ragdolls dropped onto a bumpy floor.
inline void ragdolls(Scene2D &s)
{
    s.clear();
    addGround(s);
    // A ramp and two blocks to tumble over.
    s.addBox(Vec2(-6.0f, 1.2f), 0.35f, Vec2(8.0f, 0.4f), 0.0f, 0.5f);
    s.addBox(Vec2(7.0f, 0.6f), 0.0f, Vec2(2.0f, 1.2f), 0.0f, 0.6f);
    uint32_t seed = 11u;
    const int cols = 10, rows = 3;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
        {
            const float x = ((float)c - 0.5f * (float)(cols - 1)) * 2.2f;
            const float y = 6.0f + (float)r * 3.2f + 0.8f * rand01(seed);
            const uint32_t shirt = rgb8(0.4f + 0.6f * rand01(seed), 0.3f + 0.5f * rand01(seed), 0.3f + 0.6f * rand01(seed));
            addRagdoll(s, Vec2(x, y), 1.4f, 150.0f, shirt);
        }
    s.cameraCenter = Vec2(0.0f, 7.0f);
    s.cameraHeight = 18.0f;
}

// Every collider: a stack of capsules, a pile of polygons (some rounded), wheels whose
// circle is off the body origin, all under a funnel of static segments.
inline void shapes(Scene2D &s)
{
    s.clear();
    addGround(s);
    // Capsule stack: twelve 2 m capsules, radius 0.25, laid crosswise in pairs.
    for (int i = 0; i < 12; i++)
    {
        const float y = 0.25f + 0.5f * (float)i;
        const float x = -9.0f + ((i & 1) ? 0.3f : -0.3f);
        s.addCapsule(Vec2(x - 0.75f, y), Vec2(x + 0.75f, y), 0.25f, 1.0f, 0.6f);
    }
    // Polygon pile: 3..8 sides, every third rounded, dropped through the funnel.
    uint32_t seed = 5u;
    for (int r = 0; r < 12; r++)
        for (int c = 0; c < 8; c++)
        {
            const int n = 3 + (r * 8 + c) % 6;
            const float rad = 0.35f + 0.15f * rand01(seed);
            Vec2 pts[kMaxPolygonVertices];
            for (int k = 0; k < n; k++)
            {
                const float a = 2.0f * kPi * (float)k / (float)n;
                pts[k] = Vec2(rad * std::cos(a), rad * std::sin(a));
            }
            const float round = (c % 3 == 0) ? 0.06f : 0.0f;
            s.addPolygon(Vec2(-3.0f + 0.9f * (float)c, 9.0f + 0.9f * (float)r), kPi * rand01(seed), pts, n, round, 1.0f,
                         0.5f);
        }
    // Funnel.
    s.addSegment(Vec2(-5.0f, 8.0f), Vec2(-1.2f, 4.5f));
    s.addSegment(Vec2(5.0f, 8.0f), Vec2(1.2f, 4.5f));
    // Wheels: a 0.6 m circle 0.3 m off the body origin, so they wobble as they roll.
    for (int i = 0; i < 5; i++)
        s.addOffsetCircle(Vec2(7.0f + 1.5f * (float)i, 0.9f + 2.0f * (float)i), 0.0f, Vec2(0.3f, 0.0f), 0.6f, 1.0f, 0.6f,
                          Vec2(-2.0f, 0.0f));
    s.cameraCenter = Vec2(0.0f, 7.0f);
    s.cameraHeight = 22.0f;
}

// Rolling terrain made of one chain (one-sided, ghost vertices, so nothing catches on the
// joints between segments), with a mixed load of shapes dropped on it.
inline void chainTerrain(Scene2D &s)
{
    s.clear();
    const int n = 121;
    std::vector<Vec2> pts((size_t)n);
    for (int i = 0; i < n; i++)
    {
        // Right to left, so the solid side is up (Scene2D::addChain).
        const float x = 30.0f - 0.5f * (float)i;
        pts[(size_t)i] = Vec2(x, 1.5f * std::sin(0.25f * x) + 0.6f * std::sin(0.9f * x + 1.0f));
    }
    s.addChain(pts.data(), n, false, 0.6f);
    // Walls at the ends.
    s.addSegment(Vec2(-30.0f, 0.0f), Vec2(-30.0f, 20.0f));
    s.addSegment(Vec2(30.0f, 0.0f), Vec2(30.0f, 20.0f));
    uint32_t seed = 3u;
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 40; c++)
        {
            const Vec2 p(-27.0f + 1.4f * (float)c, 6.0f + 1.4f * (float)r);
            const float a = kPi * rand01(seed);
            switch ((r * 40 + c) % 4)
            {
            case 0:
                s.addBox(p, a, Vec2(0.8f, 0.6f), 1.0f, 0.6f);
                break;
            case 1:
                s.addCircle(p, 0.4f, 1.0f, 0.6f);
                break;
            case 2:
            {
                const Vec2 d(0.4f * std::cos(a), 0.4f * std::sin(a));
                s.addCapsule(p - d, p + d, 0.25f, 1.0f, 0.6f);
                break;
            }
            default:
            {
                const Vec2 tri[3] = {Vec2(0.5f, 0.0f), Vec2(-0.25f, 0.43f), Vec2(-0.25f, -0.43f)};
                s.addPolygon(p, a, tri, 3, 0.0f, 1.0f, 0.6f);
                break;
            }
            }
        }
    s.cameraCenter = Vec2(0.0f, 5.0f);
    s.cameraHeight = 26.0f;
}

// Box2D's "Compound Shapes": tables (a top and two legs) and ships (a hull, two wings, a pod
// beneath) dropped on the ground and on each other. Each is one body of several shapes.
inline void compound(Scene2D &s)
{
    s.clear();
    addGround(s);
    for (int i = 0; i < 6; i++)
    {
        const float x = -7.0f + 2.8f * (float)i, y = 1.5f + 2.2f * (float)(i % 3);
        s.beginBody(Vec2(x, y), 0.3f * (float)(i - 2));
        s.addBoxShape(Vec2(0.0f, 0.0f), 0.0f, Vec2(1.6f, 0.2f), 1.0f, 0.6f);
        s.addBoxShape(Vec2(-0.7f, -0.5f), 0.0f, Vec2(0.15f, 0.8f), 1.0f, 0.6f);
        s.addBoxShape(Vec2(0.7f, -0.5f), 0.0f, Vec2(0.15f, 0.8f), 1.0f, 0.6f);
        s.endBody(Vec2(), 0.0f, rgb8(0.8f, 0.55f, 0.3f));
    }
    for (int i = 0; i < 5; i++)
    {
        const float x = -5.0f + 2.6f * (float)i, y = 9.0f + 1.6f * (float)(i % 2);
        const Vec2 hull[4] = {Vec2(-0.8f, -0.2f), Vec2(0.8f, -0.2f), Vec2(0.5f, 0.3f), Vec2(-0.5f, 0.3f)};
        const Vec2 wing[3] = {Vec2(0.0f, 0.0f), Vec2(0.9f, 0.0f), Vec2(0.0f, 0.5f)};
        s.beginBody(Vec2(x, y), 0.4f * (float)(i - 2));
        s.addPolygonShape(Vec2(), 0.0f, hull, 4, 0.0f, 1.0f, 0.5f);
        s.addPolygonShape(Vec2(0.8f, 0.0f), 0.0f, wing, 3, 0.0f, 1.0f, 0.5f);
        s.addPolygonShape(Vec2(-0.8f, 0.0f), kPi, wing, 3, 0.0f, 1.0f, 0.5f);
        s.addCircleShape(Vec2(0.0f, -0.4f), 0.2f, 1.0f, 0.5f);
        s.endBody(Vec2(), 0.0f, rgb8(0.45f, 0.65f, 0.9f));
    }
    s.cameraCenter = Vec2(0.0f, 6.0f);
    s.cameraHeight = 18.0f;
}

// Joints: a car on wheel joints over bumps (suspension spring, limit and a driven rear
// wheel), a prismatic elevator with a motor and travel limits carrying crates, a rope of rigid
// distance joints with a heavy ball, a spring distance joint, and a motor joint holding a box.
inline void joints(Scene2D &s)
{
    s.clear();
    addGround(s, 120.0f);

    // Car: chassis on two wheel joints, rear wheel driven (clockwise = forward, +x).
    {
        const float x0 = -14.0f, y0 = 0.9f;
        const int chassis = s.addBox(Vec2(x0, y0 + 0.5f), 0.0f, Vec2(2.6f, 0.5f), 1.0f, 0.4f, Vec2(), 0.0f, rgb8(0.9f, 0.4f, 0.2f));
        for (int i = 0; i < 2; i++)
        {
            const float wx = x0 + (i == 0 ? -0.95f : 0.95f);
            const int wheel = s.addCircle(Vec2(wx, y0), 0.4f, 1.0f, 1.0f, Vec2(), 0.0f, rgb8(0.2f, 0.2f, 0.25f));
            Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Wheel, chassis, wheel, Vec2(wx, y0), Vec2(0.0f, 1.0f));
            d.enableSpring = true;
            d.hertz = 4.0f;
            d.dampingRatio = 0.7f;
            d.enableLimit = true;
            d.lower = -0.25f;
            d.upper = 0.25f;
            d.enableMotor = i == 0;
            d.motorSpeed = -12.0f;
            d.maxMotorTorque = 8.0f;
            s.addJoint(d);
        }
        for (int i = 0; i < 7; i++)
            s.addBox(Vec2(-9.0f + 2.2f * (float)i, 0.08f), 0.0f, Vec2(0.9f, 0.16f + 0.05f * (float)(i % 3)), 0.0f, 0.8f);
    }

    // Elevator: a platform on a prismatic joint to the world, motor driven up to its limit.
    {
        const int base = s.addBox(Vec2(10.0f, 0.15f), 0.0f, Vec2(2.4f, 0.3f), 0.0f);
        (void)base;
        const int platform = s.addBox(Vec2(10.0f, 0.6f), 0.0f, Vec2(2.0f, 0.3f), 2.0f, 0.8f, Vec2(), 0.0f, rgb8(0.3f, 0.6f, 0.9f));
        s.addPrismaticAt(-1, platform, Vec2(10.0f, 0.6f), Vec2(0.0f, 1.0f), true, 0.0f, 5.0f, true, 1.2f, 400.0f);
        for (int i = 0; i < 3; i++)
            s.addBox(Vec2(9.6f + 0.4f * (float)i, 1.1f + 0.0f * (float)i), 0.0f, Vec2(0.5f, 0.5f), 1.0f, 0.6f);
    }

    // Rope: rigid distance joints between small balls, hung from a static anchor, with a ball.
    {
        const Vec2 top(-5.0f, 9.0f);
        int prev = s.addBox(top, 0.0f, Vec2(0.4f, 0.4f), 0.0f);
        const float seg = 0.5f;
        for (int i = 0; i < 8; i++)
        {
            const Vec2 p(top.x + (float)(i + 1) * seg, top.y);
            const int link = s.addCircle(p, 0.12f, 1.0f, 0.3f, Vec2(), 0.0f, rgb8(0.8f, 0.7f, 0.2f));
            s.addDistanceJoint(prev, link, i == 0 ? top : Vec2(p.x - seg, p.y), p, seg);
            prev = link;
        }
        const Vec2 end(top.x + 8.0f * seg + 0.8f, top.y);
        const int ball = s.addCircle(end, 0.7f, 2.0f, 0.4f, Vec2(), 0.0f, rgb8(0.8f, 0.2f, 0.2f));
        s.addDistanceJoint(prev, ball, Vec2(top.x + 8.0f * seg, top.y), end, 0.8f);
    }

    // A spring distance joint (a bouncing box) and a motor joint holding a box near a target.
    {
        const int anchor = s.addBox(Vec2(2.0f, 10.0f), 0.0f, Vec2(0.4f, 0.4f), 0.0f);
        const int bob = s.addBox(Vec2(2.0f, 6.5f), 0.0f, Vec2(0.8f, 0.8f), 1.0f, 0.5f, Vec2(), 0.0f, rgb8(0.4f, 0.8f, 0.4f));
        s.addDistanceJoint(anchor, bob, Vec2(2.0f, 10.0f), Vec2(2.0f, 6.5f), 3.0f, true, 1.5f, 0.5f);

        const int holder = s.addBox(Vec2(5.5f, 10.0f), 0.0f, Vec2(0.4f, 0.4f), 0.0f);
        const int held = s.addBox(Vec2(5.5f, 8.0f), 0.3f, Vec2(0.8f, 0.8f), 1.0f, 0.5f, Vec2(), 0.0f, rgb8(0.8f, 0.5f, 0.8f));
        s.addMotorJoint(holder, held, Vec2(0.0f, -1.5f), 0.0f, 40.0f, 10.0f);
    }
    s.cameraCenter = Vec2(0.0f, 5.5f);
    s.cameraHeight = 20.0f;
}

// Showcase: conveyor belts carrying boxes, a kinematic platform carrying boxes, two
// collision groups passing through each other, and rolling balls with and without rolling
// resistance on a ramp.
inline void materials(Scene2D &s)
{
    s.clear();
    addGround(s);
    // Conveyors: a belt moving right, then one moving left, each with boxes dropped on it.
    for (int k = 0; k < 2; k++)
    {
        const float x = -16.0f + 7.0f * (float)k;
        BodyDef2D bd;
        bd.position = Vec2(x, 1.5f);
        bd.type = BODY2D_STATIC;
        ShapeDef2D sd;
        sd.friction = 0.8f;
        sd.tangentSpeed = k == 0 ? 2.0f : -2.0f;
        s.beginBody(bd);
        s.addBoxShape(Vec2(), 0.0f, Vec2(6.0f, 0.5f), sd);
        s.endBody();
        for (int i = 0; i < 3; i++)
            s.addBox(Vec2(x - 1.5f + 1.5f * (float)i, 2.5f + 1.0f * (float)i), 0.0f, Vec2(0.8f, 0.6f), 1.0f, 0.6f);
    }
    // Kinematic platform moving right with boxes on it.
    {
        BodyDef2D bd;
        bd.position = Vec2(-6.0f, 0.6f);
        bd.type = BODY2D_KINEMATIC;
        bd.linearVelocity = Vec2(1.0f, 0.0f);
        ShapeDef2D sd;
        sd.friction = 0.8f;
        s.beginBody(bd);
        s.addBoxShape(Vec2(), 0.0f, Vec2(4.0f, 0.3f), sd);
        s.endBody();
        for (int i = 0; i < 2; i++)
            s.addBox(Vec2(-6.5f + 1.2f * (float)i, 1.5f), 0.0f, Vec2(0.8f, 0.8f), 1.0f, 0.6f);
    }
    // Filter showcase: red boxes (category 2) and blue discs (category 4) ignore each other but
    // both land on the ground (category 1).
    for (int i = 0; i < 4; i++)
    {
        ShapeDef2D red;
        red.categoryBits = 2;
        red.maskBits = 1 | 2;
        BodyDef2D bd;
        bd.position = Vec2(6.0f + 0.3f * (float)(i % 2), 2.0f + 1.4f * (float)i);
        bd.color = rgb8(0.9f, 0.3f, 0.3f);
        s.beginBody(bd);
        s.addBoxShape(Vec2(), 0.0f, Vec2(1.0f, 0.6f), red);
        s.endBody();
        ShapeDef2D blue;
        blue.categoryBits = 4;
        blue.maskBits = 1 | 4;
        bd.position = Vec2(6.1f, 2.7f + 1.4f * (float)i);
        bd.color = rgb8(0.3f, 0.4f, 0.9f);
        s.beginBody(bd);
        s.addCircleShape(Vec2(), 0.35f, blue);
        s.endBody();
    }
    // Ramp with rolling balls: first no rolling resistance, second 0.3.
    {
        BodyDef2D bd;
        bd.position = Vec2(16.0f, 4.0f);
        bd.angle = -0.35f;
        bd.type = BODY2D_STATIC;
        s.beginBody(bd);
        s.addBoxShape(Vec2(), 0.0f, Vec2(10.0f, 0.4f), ShapeDef2D());
        s.endBody();
        for (int i = 0; i < 2; i++)
        {
            ShapeDef2D sd;
            sd.friction = 0.8f;
            sd.rollingResistance = i == 0 ? 0.0f : 0.3f;
            BodyDef2D b;
            b.position = Vec2(12.0f + 1.6f * (float)i, 6.6f + 0.55f * (float)i);
            b.color = i == 0 ? rgb8(0.9f, 0.8f, 0.3f) : rgb8(0.4f, 0.8f, 0.5f);
            s.beginBody(b);
            s.addCircleShape(Vec2(), 0.4f, sd);
            s.endBody();
        }
    }
    s.cameraCenter = Vec2(0.0f, 5.0f);
    s.cameraHeight = 18.0f;
}

// Balls and boxes drop through a funnel and a static sensor under its mouth (the sensor sees
// every visitor, makes no contact); the balls also enable hit events.
inline void sensorFunnel(Scene2D &s)
{
    s.clear();
    addGround(s);
    s.addBox(Vec2(-4.0f, 9.0f), -0.6f, Vec2(8.0f, 0.4f), 0.0f);
    s.addBox(Vec2(4.0f, 9.0f), 0.6f, Vec2(8.0f, 0.4f), 0.0f);
    const int sensor = s.addBox(Vec2(0.0f, 4.0f), 0.0f, Vec2(3.0f, 1.5f), 0.0f, 0.5f, Vec2(), 0.0f, rgb8(0.3f, 0.8f, 0.4f));
    s.setSensor(sensor);
    uint32_t seed = 17u;
    for (int i = 0; i < 48; i++)
    {
        const float x = -3.0f + 1.0f * (float)(i % 7) + 0.2f * rand01(seed);
        const float y = 12.0f + 1.0f * (float)(i / 7);
        if (i % 3 == 2)
            s.addBox(Vec2(x, y), 0.5f * rand01(seed), Vec2(0.5f, 0.5f), 1.0f, 0.5f);
        else
            s.setHitEvents(s.addCircle(Vec2(x, y), 0.3f, 1.0f, 0.4f, Vec2(), 0.0f, rgb8(0.9f, 0.6f, 0.2f)));
    }
    s.cameraCenter = Vec2(0.0f, 8.0f);
    s.cameraHeight = 22.0f;
}

// Continuous collision: small fast bodies fired at thin static walls. The discs and the
// box would tunnel without CCD; the last disc is a bullet and also stops against the crate.
inline void continuous(Scene2D &s)
{
    s.clear();
    addGround(s, 120.0f);
    s.addBox(Vec2(8.0f, 4.0f), 0.0f, Vec2(0.1f, 8.0f), 0.0f, 0.5f);
    for (int i = 0; i < 3; i++)
        s.addCircle(Vec2(-6.0f, 1.5f + 2.0f * (float)i), 0.15f, 1.0f, 0.3f, Vec2(150.0f, 0.0f), 0.0f,
                    rgb8(0.9f, 0.5f, 0.2f));
    s.addBox(Vec2(-6.0f, 7.5f), 0.0f, Vec2(0.3f, 0.3f), 1.0f, 0.3f, Vec2(120.0f, 0.0f), 0.0f, rgb8(0.3f, 0.6f, 0.9f));
    const int crate = s.addBox(Vec2(-1.0f, 0.5f), 0.0f, Vec2(0.5f, 1.0f), 1.0f, 0.5f);
    (void)crate;
    const int bullet = s.addCircle(Vec2(-6.0f, 0.4f), 0.1f, 1.0f, 0.3f, Vec2(150.0f, 0.0f), 0.0f,
                                   rgb8(0.9f, 0.1f, 0.1f));
    s.setBullet(bullet);
    s.cameraCenter = Vec2(1.0f, 4.0f);
    s.cameraHeight = 14.0f;
}

struct SceneEntry2D
{
    const char *name;
    void (*build)(Scene2D &);
    int approxBodies; // for pacing unattended runs
};

inline const SceneEntry2D *catalogue(int &count)
{
    static const SceneEntry2D entries[] = {
        {"Pyramid", pyramid, 211},
        {"Stack", stack, 18},
        {"Friction Ramp", frictionRamp, 8},
        {"Chain & Bridge", chainBridge, 50},
        {"Ragdolls", ragdolls, 330},

        {"Pile 10k", pile10k, 10000},
        {"Towers 20k", towers20k, 20600},
        {"Pile 100k", pile100k, 100000},
        {"Pile 200k", pile200k, 200000},
        {"Pile 500k", pile500k, 500000},
        {"Shapes", shapes, 120},
        {"Chain Terrain", chainTerrain, 362},
        {"Big Box", bigBox, 3},
        {"Compound Shapes", compound, 11},
        {"Joints", joints, 40},
        {"Materials", materials, 24},
        {"Sensor Funnel", sensorFunnel, 50},
        {"Continuous", continuous, 8},
        // Not registered (bodies tunnel out of the chain loop at every iteration count and motor speed tried, see scenes2d_box2d.h): B2 Spinner.
        {"B2 Vertical Stack", b2VerticalStack, 12},
        {"B2 Tumbler", b2Tumbler, 2026},
        {"B2 Double Domino", b2DoubleDomino, 15},
        {"B2 Bridge", b2Bridge, 165},
        {"B2 Ball & Chain", b2BallAndChain, 31},
        {"B2 Cantilever", b2Cantilever, 8},
        {"B2 Scissor Lift", b2ScissorLift, 10},
        {"B2 Driving", b2Driving, 28},
        {"B2 Conveyor Belt", b2ConveyorBelt, 7},
        {"B2 Rolling Resistance", b2RollingResistance, 40},
        {"B2 Large Pyramid", b2LargePyramid, 5051},
        {"B2 Many Pyramids", b2ManyPyramids, 22020},
        {"B2 Joint Grid", b2JointGrid, 10000},
        {"B2 Smash", b2Smash, 9601},
        {"B2 Card House", b2CardHouse, 41},
        {"B2 Arch", b2Arch, 22},
        {"B2 Confined", b2Confined, 625},
        {"B2 Cliff", b2Cliff, 14},
        {"B2 Capsule Stack", b2CapsuleStack, 21},
        {"B2 HighMassRatio 1", b2HighMassRatio1, 181},
        {"B2 HighMassRatio 2", b2HighMassRatio2, 4},
        {"B2 HighMassRatio 3", b2HighMassRatio3, 4},
        {"B2 Theo Jansen", b2TheoJansen, 60},
        {"B2 Contact Event", b2ContactEvent, 22},
        {"B2 Many Tumblers", b2ManyTumblers, 18050},
        {"B2 Typical Game", b2TypicalGame, 330},
    };
    count = (int)(sizeof(entries) / sizeof(entries[0]));
    return entries;
}

} // namespace scenes
} // namespace avbd2d
