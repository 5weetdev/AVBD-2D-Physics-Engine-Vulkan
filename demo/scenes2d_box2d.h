#pragma once

// Scenes ported from the Box2D v3 samples (samples/sample_stacking.cpp, sample_joints.cpp,
// sample_shapes.cpp and shared/benchmarks.c), same dimensions, densities, frictions and joint
// settings; Box2D is y-up like this solver. Named "B2 ..." in the catalogue. Left out: anything
// driven at runtime (keys, bullets, Rain's spawning), restitution, gear joints, callbacks.
// b2Spinner is kept here but not in the catalogue: the spinner throws bodies out of its chain loop
// at every iteration count (8, 20, 60) and motor speed (5, 1.5) tried. Card House and Arch run with
// more iterations (100 and 40) than the default: at 8 they collapse within 300 steps.
// Differences forced by this solver: joint anchors must coincide in the starting pose (a joint
// takes one world point), spring damping ratios are ignored, per-joint constraint tuning does not
// exist, and the Tumbler spins as a kinematic body instead of through a motor.

#include "scene2d.h"

#include <cmath>
#include <vector>

namespace avbd2d
{
namespace scenes
{

namespace b2s
{

inline int ground(Scene2D &s, Vec2 p1, Vec2 p2, float mu = 0.6f) { return s.addSegment(p1, p2, mu); }

// Box2D's default ground: a static box of half extents (hx, hy) at p.
inline int groundBox(Scene2D &s, Vec2 p, float hx, float hy, float mu = 0.6f)
{
    return s.addBox(p, 0.0f, Vec2(2.0f * hx, 2.0f * hy), 0.0f, mu);
}

inline int revolute(Scene2D &s, int a, int b, Vec2 pt, float frictionTorque = 0.0f, float hertz = 0.0f)
{
    Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Revolute, a, b, pt);
    if (frictionTorque > 0.0f)
    {
        d.enableMotor = true;
        d.motorSpeed = 0.0f;
        d.maxMotorTorque = frictionTorque;
    }
    if (hertz > 0.0f)
    {
        d.enableSpring = true;
        d.hertz = hertz;
        d.dampingRatio = 0.7f;
    }
    return s.addJoint(d);
}

// Box2D's Car (samples/car.cpp): a rounded polygon chassis and two wheels on wheel joints.
// speed is the motor speed (rad/s, negative = forward, +x), torque the motor limit.
inline int car(Scene2D &s, Vec2 pos, float scale, float hertz, float damping, float torque, float speed)
{
    Vec2 v[6] = {Vec2(-1.5f, -0.5f), Vec2(1.5f, -0.5f), Vec2(1.5f, 0.0f),
                 Vec2(0.0f, 0.9f),   Vec2(-1.15f, 0.9f), Vec2(-1.5f, 0.2f)};
    for (Vec2 &p : v)
        p = p * (0.85f * scale);
    const int chassis = s.addPolygon(pos + Vec2(0.0f, scale), 0.0f, v, 6, 0.15f * scale, 1.0f / scale, 0.2f);
    const Vec2 wp[2] = {pos + Vec2(-1.0f * scale, 0.35f * scale), pos + Vec2(1.0f * scale, 0.4f * scale)};
    ShapeDef2D wd;
    wd.friction = 1.5f;
    wd.rollingResistance = 0.1f;
    for (int i = 0; i < 2; i++)
    {
        const int wheel = s.addCircle(wp[i], 0.4f * scale, 2.0f / scale, 1.5f);
        s.setShapeDef(wheel, wd);
        Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Wheel, chassis, wheel, wp[i], Vec2(0.0f, 1.0f));
        d.enableSpring = hertz > 0.0f;
        d.hertz = hertz;
        d.dampingRatio = damping;
        d.enableLimit = true;
        d.lower = -0.25f * scale;
        d.upper = 0.25f * scale;
        d.enableMotor = torque > 0.0f;
        d.motorSpeed = speed;
        d.maxMotorTorque = torque;
        s.addJoint(d);
    }
    return chassis;
}

} // namespace b2s

// sample_stacking.cpp, Vertical Stack: one column of twelve rounded boxes against a wall.
inline void b2VerticalStack(Scene2D &s)
{
    s.clear();
    b2s::ground(s, Vec2(10.0f, 0.0f), Vec2(10.0f, 20.0f), 0.6f);
    b2s::ground(s, Vec2(-30.0f, 0.0f), Vec2(30.0f, 0.0f), 0.6f);
    for (int i = 0; i < 12; i++)
    {
        const float shift = (i % 2 == 0) ? -0.01f : 0.01f;
        s.addRoundedBox(Vec2(8.0f + shift, 0.5f + (float)i), 0.0f, Vec2(0.9f, 0.9f), 0.05f, 1.0f, 0.3f);
    }
    s.cameraCenter = Vec2(-7.0f, 9.0f);
    s.cameraHeight = 28.0f;
}

// benchmarks.c, Tumbler: a hollow box (four slabs, density 50) turning slowly about its centre
// with 2025 small boxes tumbling inside.
inline void b2Tumbler(Scene2D &s)
{
    s.clear();
    BodyDef2D bd;
    bd.position = Vec2(0.0f, 10.0f);
    bd.type = BODY2D_KINEMATIC;
    bd.angularVelocity = (kPi / 180.0f) * 25.0f;
    ShapeDef2D sd;
    sd.density = 50.0f;
    sd.friction = 0.6f;
    s.beginBody(bd);
    s.addBoxShape(Vec2(10.0f, 0.0f), 0.0f, Vec2(1.0f, 20.0f), sd);
    s.addBoxShape(Vec2(-10.0f, 0.0f), 0.0f, Vec2(1.0f, 20.0f), sd);
    s.addBoxShape(Vec2(0.0f, 10.0f), 0.0f, Vec2(20.0f, 1.0f), sd);
    s.addBoxShape(Vec2(0.0f, -10.0f), 0.0f, Vec2(20.0f, 1.0f), sd);
    s.endBody();
    const int grid = 45;
    float y = -0.2f * (float)grid + 10.0f;
    for (int i = 0; i < grid; i++)
    {
        float x = -0.2f * (float)grid;
        for (int j = 0; j < grid; j++)
        {
            s.addBox(Vec2(x, y), 0.0f, Vec2(0.25f, 0.25f), 1.0f, 0.6f);
            x += 0.4f;
        }
        y += 0.4f;
    }
    s.cameraCenter = Vec2(1.5f, 10.0f);
    s.cameraHeight = 30.0f;
}

// sample_stacking.cpp, Card House: 0.4 m cards, 2 mm thick, in a five-storey house of cards.
inline void b2CardHouse(Scene2D &s)
{
    s.clear();
    s.params.iterations = 100;
    b2s::groundBox(s, Vec2(0.0f, -2.0f), 40.0f, 2.0f, 0.7f);
    const float cardHeight = 0.2f, cardThickness = 0.001f;
    const float angle0 = 25.0f * kPi / 180.0f, angle1 = -25.0f * kPi / 180.0f, angle2 = 0.5f * kPi;
    const Vec2 size(2.0f * cardThickness, 2.0f * cardHeight);
    int Nb = 5;
    float z0 = 0.0f, y = cardHeight - 0.02f;
    while (Nb)
    {
        float z = z0;
        for (int i = 0; i < Nb; i++)
        {
            if (i != Nb - 1)
                s.addBox(Vec2(z + 0.25f, y + cardHeight - 0.015f), angle2, size, 1.0f, 0.7f);
            s.addBox(Vec2(z, y), angle1, size, 1.0f, 0.7f);
            z += 0.175f;
            s.addBox(Vec2(z, y), angle0, size, 1.0f, 0.7f);
            z += 0.175f;
        }
        y += cardHeight * 2.0f - 0.03f;
        z0 += 0.175f;
        Nb--;
    }
    s.cameraCenter = Vec2(0.75f, 0.9f);
    s.cameraHeight = 2.5f;
}

// sample_stacking.cpp, Arch: seventeen stone blocks and four boxes on top.
inline void b2Arch(Scene2D &s)
{
    s.clear();
    s.params.iterations = 40;
    Vec2 ps1[9] = {Vec2(16.0f, 0.0f),
                   Vec2(14.93803712795643f, 5.133601056842984f),
                   Vec2(13.79871746027416f, 10.24928069555078f),
                   Vec2(12.56252963284711f, 15.34107019122473f),
                   Vec2(11.20040987372525f, 20.39856541571217f),
                   Vec2(9.66521217819836f, 25.40369899225096f),
                   Vec2(7.87179930638133f, 30.3179337000085f),
                   Vec2(5.635199558196225f, 35.03820717801641f),
                   Vec2(2.405937953536585f, 39.09554102558315f)};
    Vec2 ps2[9] = {Vec2(24.0f, 0.0f),
                   Vec2(22.33619528222415f, 6.02299846205841f),
                   Vec2(20.54936888969905f, 12.00964361211476f),
                   Vec2(18.60854610798073f, 17.9470321677465f),
                   Vec2(16.46769273811807f, 23.81367936585418f),
                   Vec2(14.05325025774858f, 29.57079353071012f),
                   Vec2(11.23551045834022f, 35.13775818285372f),
                   Vec2(7.752568160730571f, 40.30450679009583f),
                   Vec2(3.016931552701656f, 44.28891593799322f)};
    for (int i = 0; i < 9; i++)
    {
        ps1[i] = ps1[i] * 0.25f;
        ps2[i] = ps2[i] * 0.25f;
    }
    const float mu = 0.6f;
    b2s::ground(s, Vec2(-100.0f, 0.0f), Vec2(100.0f, 0.0f), mu);
    for (int i = 0; i < 8; i++)
    {
        const Vec2 p[4] = {ps1[i], ps2[i], ps2[i + 1], ps1[i + 1]};
        s.addPolygon(Vec2(), 0.0f, p, 4, 0.0f, 1.0f, mu);
    }
    for (int i = 0; i < 8; i++)
    {
        const Vec2 p[4] = {Vec2(-ps2[i].x, ps2[i].y), Vec2(-ps1[i].x, ps1[i].y), Vec2(-ps1[i + 1].x, ps1[i + 1].y),
                           Vec2(-ps2[i + 1].x, ps2[i + 1].y)};
        s.addPolygon(Vec2(), 0.0f, p, 4, 0.0f, 1.0f, mu);
    }
    {
        const Vec2 p[4] = {ps1[8], ps2[8], Vec2(-ps2[8].x, ps2[8].y), Vec2(-ps1[8].x, ps1[8].y)};
        s.addPolygon(Vec2(), 0.0f, p, 4, 0.0f, 1.0f, mu);
    }
    for (int i = 0; i < 4; i++)
        s.addBox(Vec2(0.0f, 0.5f + ps2[8].y + 1.0f * (float)i), 0.0f, Vec2(4.0f, 1.0f), 1.0f, mu);
    s.cameraCenter = Vec2(0.0f, 8.0f);
    s.cameraHeight = 20.0f;
}

// sample_stacking.cpp, Double Domino: fifteen dominoes, the first one nudged.
inline void b2DoubleDomino(Scene2D &s)
{
    s.clear();
    b2s::groundBox(s, Vec2(0.0f, -1.0f), 100.0f, 1.0f, 0.6f);
    const int count = 15;
    float x = -0.5f * (float)count;
    for (int i = 0; i < count; i++)
    {
        Vec2 v;
        float w = 0.0f;
        if (i == 0)
        {
            // b2Body_ApplyLinearImpulse({0.2, 0}) at (x, 1.0): the mass is 0.25, the arm 0.5 above the centre.
            const float m = 0.25f, I = m * (0.25f * 0.25f + 1.0f) / 12.0f;
            v = Vec2(0.2f / m, 0.0f);
            w = -0.5f * 0.2f / I;
        }
        s.addBox(Vec2(x, 0.5f), 0.0f, Vec2(0.25f, 1.0f), 1.0f, 0.6f, v, w);
        x += 1.0f;
    }
    s.cameraCenter = Vec2(0.0f, 4.0f);
    s.cameraHeight = 12.5f;
}

// sample_joints.cpp, Bridge: 160 planks on spring and friction revolute joints between two
// anchors 160 m apart, loaded with triangles and balls.
inline void b2Bridge(Scene2D &s)
{
    s.clear();
    const int count = 160;
    ShapeDef2D sd;
    sd.density = 20.0f;
    sd.friction = 0.6f;
    const float xbase = -80.0f;
    int prev = -1;
    for (int i = 0; i < count; i++)
    {
        BodyDef2D bd;
        bd.position = Vec2(xbase + 0.5f + (float)i, 20.0f);
        bd.linearDamping = 0.1f;
        bd.angularDamping = 0.1f;
        s.beginBody(bd);
        s.addBoxShape(Vec2(), 0.0f, Vec2(1.0f, 0.25f), sd);
        const int id = s.endBody();
        b2s::revolute(s, prev, id, Vec2(xbase + (float)i, 20.0f), 200.0f, 2.0f);
        prev = id;
    }
    b2s::revolute(s, -1, prev, Vec2(xbase + (float)count, 20.0f), 200.0f, 2.0f);
    const Vec2 tri[3] = {Vec2(-0.5f, 0.0f), Vec2(0.5f, 0.0f), Vec2(0.0f, 1.5f)};
    for (int i = 0; i < 2; i++)
        s.addPolygon(Vec2(-8.0f + 8.0f * (float)i, 22.0f), 0.0f, tri, 3, 0.0f, 20.0f, 0.6f);
    for (int i = 0; i < 3; i++)
        s.addCircle(Vec2(-6.0f + 6.0f * (float)i, 25.0f), 0.5f, 20.0f, 0.6f);
    s.cameraCenter = Vec2(0.0f, 14.0f);
    s.cameraHeight = 80.0f;
}

// sample_joints.cpp, Ball & Chain: thirty capsules on friction/spring revolute joints from a
// ceiling pin, ending in a radius-4 ball. Links collide only with the ball.
inline void b2BallAndChain(Scene2D &s)
{
    s.clear();
    const int count = 30;
    const float hx = 0.5f;
    const float yTop = (float)count * hx;
    ShapeDef2D link;
    link.density = 20.0f;
    link.friction = 0.6f;
    link.categoryBits = 1;
    link.maskBits = 2;
    int prev = -1;
    for (int i = 0; i < count; i++)
    {
        BodyDef2D bd;
        bd.position = Vec2((1.0f + 2.0f * (float)i) * hx, yTop);
        s.beginBody(bd);
        s.addCapsuleShape(Vec2(-hx, 0.0f), Vec2(hx, 0.0f), 0.125f, link);
        const int id = s.endBody();
        b2s::revolute(s, prev, id, Vec2(2.0f * (float)i * hx, yTop), 100.0f, i > 0 ? 4.0f : 0.0f);
        prev = id;
    }
    ShapeDef2D ball = link;
    ball.categoryBits = 2;
    ball.maskBits = 1;
    BodyDef2D bd;
    bd.position = Vec2((1.0f + 2.0f * (float)count) * hx + 4.0f - hx, yTop);
    s.beginBody(bd);
    s.addCircleShape(Vec2(), 4.0f, ball);
    const int b = s.endBody();
    b2s::revolute(s, prev, b, Vec2(2.0f * (float)count * hx, yTop), 100.0f, 4.0f);
    s.cameraCenter = Vec2(10.0f, 6.0f);
    s.cameraHeight = 55.0f;
}

// sample_joints.cpp, Cantilever: eight capsules welded with soft linear and angular springs.
inline void b2Cantilever(Scene2D &s)
{
    s.clear();
    const float hx = 0.5f;
    ShapeDef2D sd;
    sd.density = 20.0f;
    sd.friction = 0.6f;
    int prev = -1;
    for (int i = 0; i < 8; i++)
    {
        BodyDef2D bd;
        bd.position = Vec2((1.0f + 2.0f * (float)i) * hx, 0.0f);
        s.beginBody(bd);
        s.addCapsuleShape(Vec2(-hx, 0.0f), Vec2(hx, 0.0f), 0.125f, sd);
        const int id = s.endBody();
        Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Weld, prev, id, Vec2(2.0f * (float)i * hx, 0.0f));
        d.linearHertz = 15.0f;
        d.linearDampingRatio = 0.5f;
        d.angularHertz = 5.0f;
        d.angularDampingRatio = 0.5f;
        s.addJoint(d);
        prev = id;
    }
    s.cameraCenter = Vec2(4.0f, -1.0f);
    s.cameraHeight = 17.5f;
}

// sample_joints.cpp, Scissor Lift: three stacked scissor pairs between the ground and a platform
// (one end of each pair on a free-sliding wheel joint), a car on top, and the lift's distance
// joint left unmotorised. The level pitch is 2 * 2.5 * sin(0.15), not Box2D's 1.0, so the pins of
// neighbouring levels start on the same point.
inline void b2ScissorLift(Scene2D &s)
{
    s.clear();
    b2s::ground(s, Vec2(-20.0f, 0.0f), Vec2(20.0f, 0.0f), 0.6f);
    const float ang = 0.15f, c = std::cos(ang), sn = std::sin(ang);
    const float pitch = 5.0f * sn;
    ShapeDef2D sd;
    sd.density = 1.0f;
    sd.friction = 0.6f;
    int base1 = -1, base2 = -1; // bodies the next level's left and right pins attach to
    int linkId1 = -1;
    float y = 0.5f, topEnd = 0.0f;
    auto pin = [&](Scene2D::JointType2D type, int a, int b, Vec2 pt, bool collide) {
        Scene2D::JointDef2D d = s.makeJointDef(type, a, b, pt, Vec2(1.0f, 0.0f));
        d.collideConnected = collide;
        s.addJoint(d);
    };
    for (int i = 0; i < 3; i++)
    {
        int id[2];
        for (int k = 0; k < 2; k++)
        {
            BodyDef2D bd;
            bd.position = Vec2(0.0f, y);
            bd.angle = k == 0 ? ang : -ang;
            s.beginBody(bd);
            s.addCapsuleShape(Vec2(-2.5f, 0.0f), Vec2(2.5f, 0.0f), 0.15f, sd);
            id[k] = s.endBody();
        }
        if (i == 1)
            linkId1 = id[1];
        const float yLow = y - 2.5f * sn;
        pin(Scene2D::JointType2D::Revolute, base1, id[0], Vec2(-2.5f * c, yLow), i == 0);
        pin(i == 0 ? Scene2D::JointType2D::Wheel : Scene2D::JointType2D::Revolute, base2, id[1], Vec2(2.5f * c, yLow),
            i == 0);
        pin(Scene2D::JointType2D::Revolute, id[0], id[1], Vec2(0.0f, y), false);
        base1 = id[1];
        base2 = id[0];
        topEnd = y + 2.5f * sn;
        y += pitch;
    }
    BodyDef2D pbd;
    pbd.position = Vec2(0.0f, topEnd + 0.4f);
    s.beginBody(pbd);
    s.addBoxShape(Vec2(), 0.0f, Vec2(6.0f, 0.4f), sd);
    const int platform = s.endBody();
    pin(Scene2D::JointType2D::Revolute, platform, base1, Vec2(-2.5f * c, topEnd), true);
    pin(Scene2D::JointType2D::Wheel, platform, base2, Vec2(2.5f * c, topEnd), true);

    // The lift's distance joint: slack inside [0.2, 5.5] (a spring of near-zero stiffness).
    const Vec2 a(-2.5f, 0.2f);
    const Vec2 b = s.pos[(size_t)linkId1] + rotate(s.ang[(size_t)linkId1], Vec2(0.5f, 0.0f));
    s.addDistanceJoint(-1, linkId1, a, b, length(b - a), true, 0.01f, 0.0f, 0.2f, 5.5f);

    b2s::car(s, Vec2(0.0f, topEnd + 0.4f + 2.0f), 1.0f, 3.0f, 0.7f, 0.0f, 0.0f);
    s.cameraCenter = Vec2(0.0f, 5.0f);
    s.cameraHeight = 20.0f;
}

// sample_joints.cpp, Driving: Box2D's car on a long course: rolling hills (a chain), a teeter
// totter, a 20-link bridge, a jump ramp and a stack of light boxes. The car drives forward.
inline void b2Driving(Scene2D &s)
{
    s.clear();
    Vec2 pts[25];
    int n = 24;
    pts[n--] = Vec2(-20.0f, -20.0f);
    pts[n--] = Vec2(-20.0f, 0.0f);
    pts[n--] = Vec2(20.0f, 0.0f);
    const float hs[10] = {0.25f, 1.0f, 4.0f, 0.0f, 0.0f, -1.0f, -2.0f, -2.0f, -1.25f, 0.0f};
    float x = 20.0f;
    const float dx = 5.0f;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 10; i++)
        {
            pts[n--] = Vec2(x + dx, hs[i]);
            x += dx;
        }
    pts[n--] = Vec2(x + 40.0f, 0.0f);
    pts[n--] = Vec2(x + 40.0f, -20.0f);
    s.addChain(pts, 25, true, 0.6f);
    x += 80.0f;
    s.addSegment(Vec2(x, 0.0f), Vec2(x + 40.0f, 0.0f));
    x += 40.0f;
    s.addSegment(Vec2(x, 0.0f), Vec2(x + 10.0f, 5.0f));
    x += 20.0f;
    s.addSegment(Vec2(x, 0.0f), Vec2(x + 40.0f, 0.0f));
    x += 40.0f;
    s.addSegment(Vec2(x, 0.0f), Vec2(x, 20.0f));

    // Teeter.
    {
        const int t = s.addBox(Vec2(140.0f, 1.0f), 0.0f, Vec2(20.0f, 0.5f), 1.0f, 0.6f, Vec2(), 1.0f);
        Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Revolute, -1, t, Vec2(140.0f, 1.0f));
        d.enableLimit = true;
        d.lower = -8.0f * kPi / 180.0f;
        d.upper = 8.0f * kPi / 180.0f;
        s.addJoint(d);
    }
    // Bridge: twenty 2 m capsules between the end of the hills and the flat after it.
    {
        const int N = 20;
        ShapeDef2D sd;
        sd.friction = 0.6f;
        int prev = -1;
        for (int i = 0; i < N; i++)
        {
            BodyDef2D bd;
            bd.position = Vec2(161.0f + 2.0f * (float)i, -0.125f);
            s.beginBody(bd);
            s.addCapsuleShape(Vec2(-1.0f, 0.0f), Vec2(1.0f, 0.0f), 0.125f, sd);
            const int id = s.endBody();
            b2s::revolute(s, prev, id, Vec2(160.0f + 2.0f * (float)i, -0.125f));
            prev = id;
        }
        b2s::revolute(s, -1, prev, Vec2(160.0f + 2.0f * (float)N, -0.125f), 50.0f);
    }
    // Boxes.
    for (int i = 0; i < 5; i++)
        s.addBox(Vec2(230.0f, 0.5f + (float)i), 0.0f, Vec2(1.0f, 1.0f), 0.25f, 0.25f);

    b2s::car(s, Vec2(0.0f, 0.0f), 1.0f, 5.0f, 0.7f, 5.0f, -10.0f);
    s.cameraCenter = Vec2(30.0f, 8.0f);
    s.cameraHeight = 40.0f;
}

// sample_shapes.cpp, Conveyor Belt: a 20 m belt (a rounded slab with a tangent speed of 2 m/s)
// carrying five boxes.
inline void b2ConveyorBelt(Scene2D &s)
{
    s.clear();
    b2s::ground(s, Vec2(-20.0f, 0.0f), Vec2(20.0f, 0.0f), 0.6f);
    {
        BodyDef2D bd;
        bd.position = Vec2(-5.0f, 5.0f);
        bd.type = BODY2D_STATIC;
        ShapeDef2D sd;
        sd.friction = 0.8f;
        sd.tangentSpeed = 2.0f;
        const Vec2 pts[4] = {Vec2(-10.0f, -0.25f), Vec2(10.0f, -0.25f), Vec2(10.0f, 0.25f), Vec2(-10.0f, 0.25f)};
        s.beginBody(bd);
        s.addPolygonShape(Vec2(), 0.0f, pts, 4, 0.25f, sd);
        s.endBody();
    }
    for (int i = 0; i < 5; i++)
        s.addBox(Vec2(-10.0f + 2.0f * (float)i, 7.0f), 0.0f, Vec2(1.0f, 1.0f), 1.0f, 0.6f);
    s.cameraCenter = Vec2(2.0f, 7.5f);
    s.cameraHeight = 24.0f;
}

// sample_shapes.cpp, Rolling Resistance: twenty balls spinning down twenty flat tracks, with a
// rolling resistance coefficient of 0.02 * track.
inline void b2RollingResistance(Scene2D &s)
{
    s.clear();
    for (int i = 0; i < 20; i++)
    {
        b2s::ground(s, Vec2(-40.0f, 2.0f * (float)i), Vec2(40.0f, 2.0f * (float)i), 0.6f);
        const int id =
            s.addCircle(Vec2(-39.5f, 2.0f * (float)i + 0.75f), 0.5f, 1.0f, 0.6f, Vec2(5.0f, 0.0f), -10.0f);
        ShapeDef2D sd;
        sd.friction = 0.6f;
        sd.rollingResistance = 0.02f * (float)i;
        s.setShapeDef(id, sd);
    }
    s.cameraCenter = Vec2(5.0f, 20.0f);
    s.cameraHeight = 55.0f;
}

// benchmarks.c, Large Pyramid: 5050 unit boxes, 100 on the base.
inline void b2LargePyramid(Scene2D &s)
{
    s.clear();
    s.params.sleepEnabled = false;
    b2s::groundBox(s, Vec2(0.0f, -1.0f), 100.0f, 1.0f, 0.6f);
    const int baseCount = 100;
    const float a = 0.5f, shift = a;
    for (int i = 0; i < baseCount; i++)
    {
        const float y = (2.0f * (float)i + 1.0f) * shift;
        for (int j = i; j < baseCount; j++)
        {
            const float x = ((float)i + 1.0f) * shift + 2.0f * (float)(j - i) * shift - a * (float)baseCount;
            s.addBox(Vec2(x, y), 0.0f, Vec2(2.0f * a, 2.0f * a), 1.0f, 0.6f);
        }
    }
    s.cameraCenter = Vec2(0.0f, 50.0f);
    s.cameraHeight = 110.0f;
}

// benchmarks.c, Many Pyramids: a 20 x 20 field of ten-box-base pyramids, 22,000 boxes, one
// ground segment under each row.
inline void b2ManyPyramids(Scene2D &s)
{
    s.clear();
    s.params.sleepEnabled = false;
    const int baseCount = 10, rowCount = 20, columnCount = 20;
    const float extent = 0.5f;
    const float groundDeltaY = 2.0f * extent * ((float)baseCount + 1.0f);
    const float groundWidth = 2.0f * extent * (float)columnCount * ((float)baseCount + 1.0f);
    float groundY = 0.0f;
    for (int i = 0; i < rowCount; i++)
    {
        b2s::ground(s, Vec2(-0.5f * groundWidth, groundY), Vec2(0.5f * groundWidth, groundY), 0.6f);
        groundY += groundDeltaY;
    }
    const float baseWidth = 2.0f * extent * (float)baseCount;
    float baseY = 0.0f;
    for (int i = 0; i < rowCount; i++)
    {
        for (int j = 0; j < columnCount; j++)
        {
            const float centerX = -0.5f * groundWidth + (float)j * (baseWidth + 2.0f * extent) + 2.0f * extent;
            for (int r = 0; r < baseCount; r++)
            {
                const float y = (2.0f * (float)r + 1.0f) * extent + baseY;
                for (int k = r; k < baseCount; k++)
                {
                    const float x = ((float)r + 1.0f) * extent + 2.0f * (float)(k - r) * extent + centerX - 0.5f;
                    s.addBox(Vec2(x, y), 0.0f, Vec2(2.0f * extent, 2.0f * extent), 1.0f, 0.6f);
                }
            }
        }
        baseY += groundDeltaY;
    }
    s.cameraCenter = Vec2(0.0f, 115.0f);
    s.cameraHeight = 250.0f;
}

// benchmarks.c, Joint Grid: a 100 x 100 net of circles on revolute joints hanging from seven
// pins on the top row. The circles do not collide with each other.
inline void b2JointGrid(Scene2D &s)
{
    s.clear();
    s.params.sleepEnabled = false;
    s.params.killFallenEnabled = false;
    const int N = 100;
    std::vector<int> bodies((size_t)N * (size_t)N);
    int index = 0;
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
        {
            const bool fixedBody = k >= N / 2 - 3 && k <= N / 2 + 3 && i == 0;
            const Vec2 p((float)k, -(float)i);
            const int id = s.addCircle(p, 0.4f, fixedBody ? 0.0f : 1.0f, 0.6f);
            ShapeDef2D sd;
            sd.friction = 0.6f;
            sd.categoryBits = 2;
            sd.maskBits = ~uint64_t(2);
            s.setShapeDef(id, sd);
            if (i > 0)
                b2s::revolute(s, bodies[(size_t)index - 1], id, p + Vec2(0.0f, 0.5f));
            if (k > 0 && !fixedBody)
                b2s::revolute(s, bodies[(size_t)index - (size_t)N], id, p + Vec2(-0.5f, 0.0f));
            else if (k > 0 && i == 0 && !(k - 1 >= N / 2 - 3 && k - 1 <= N / 2 + 3))
                b2s::revolute(s, bodies[(size_t)index - (size_t)N], id, p + Vec2(-0.5f, 0.0f));
            bodies[(size_t)index++] = id;
        }
    s.cameraCenter = Vec2(50.0f, -45.0f);
    s.cameraHeight = 120.0f;
}

// benchmarks.c, Smash: a heavy 8 m box fired at 40 m/s into a 120 x 80 wall of small squares,
// in zero gravity.
inline void b2Smash(Scene2D &s)
{
    s.clear();
    s.params.gravity = Vec2(0.0f, 0.0f);
    s.params.killFallenEnabled = false;
    s.addBox(Vec2(-20.0f, 0.0f), 0.0f, Vec2(8.0f, 8.0f), 8.0f, 0.6f, Vec2(40.0f, 0.0f));
    const float d = 0.4f;
    const int columns = 120, rows = 80;
    for (int i = 0; i < columns; i++)
        for (int j = 0; j < rows; j++)
            s.addBox(Vec2((float)i * d + 30.0f, ((float)j - (float)rows / 2.0f) * d), 0.0f, Vec2(d, d), 1.0f, 0.6f);
    s.cameraCenter = Vec2(30.0f, 0.0f);
    s.cameraHeight = 70.0f;
}

// benchmarks.c, Spinner: a motor-driven 40 m paddle turning inside a 40 m circle of chain, with
// 6076 capsules, balls and squares riding it.
inline void b2Spinner(Scene2D &s)
{
    s.clear();
    s.params.sleepEnabled = false;
    const int pointCount = 360;
    std::vector<Vec2> pts((size_t)pointCount);
    {
        const float a = -2.0f * kPi / (float)pointCount;
        const float ca = std::cos(a), sa = std::sin(a);
        Vec2 p(40.0f, 0.0f);
        for (int i = 0; i < pointCount; i++)
        {
            pts[(size_t)i] = Vec2(p.x, p.y + 32.0f);
            p = Vec2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);
        }
    }
    s.addChain(pts.data(), pointCount, true, 0.1f);
    {
        const int sp = s.addRoundedBox(Vec2(0.0f, 12.0f), 0.0f, Vec2(0.8f, 40.0f), 0.2f, 1.0f, 0.0f);
        Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Revolute, -1, sp, Vec2(0.0f, 12.0f));
        d.enableMotor = true;
        d.motorSpeed = 5.0f;
        d.maxMotorTorque = 1.0e8f;
        s.addJoint(d);
    }
    float x = -23.0f, y = 2.0f;
    for (int i = 0; i < 2 * 3038; i++)
    {
        int id;
        switch (i % 3)
        {
        case 0:
            id = s.addCapsule(Vec2(x - 0.25f, y), Vec2(x + 0.25f, y), 0.25f, 0.25f, 0.1f);
            break;
        case 1:
            id = s.addCircle(Vec2(x, y), 0.35f, 0.25f, 0.1f);
            break;
        default:
            id = s.addBox(Vec2(x, y), 0.0f, Vec2(0.7f, 0.7f), 0.25f, 0.1f);
            break;
        }
        (void)id;
        x += 0.5f;
        if (x >= 23.0f)
        {
            x = -23.0f;
            y += 0.5f;
        }
    }
    s.cameraCenter = Vec2(0.0f, 32.0f);
    s.cameraHeight = 90.0f;
}

// sample_stacking.cpp, Confined: 625 circles in a closed box of capsules, zero gravity.
inline void b2Confined(Scene2D &s)
{
    s.clear();
    BodyDef2D gd;
    gd.type = BODY2D_STATIC;
    ShapeDef2D wall;
    wall.friction = 0.6f;
    s.beginBody(gd);
    s.addCapsuleShape(Vec2(-10.5f, 0.0f), Vec2(10.5f, 0.0f), 0.5f, wall);
    s.addCapsuleShape(Vec2(-10.5f, 0.0f), Vec2(-10.5f, 20.5f), 0.5f, wall);
    s.addCapsuleShape(Vec2(10.5f, 0.0f), Vec2(10.5f, 20.5f), 0.5f, wall);
    s.addCapsuleShape(Vec2(-10.5f, 20.5f), Vec2(10.5f, 20.5f), 0.5f, wall);
    s.endBody();
    BodyDef2D bd;
    bd.gravityScale = 0.0f;
    const int grid = 25;
    for (int c = 0; c < grid; c++)
        for (int r = 0; r < grid; r++)
        {
            const int id = s.addCircle(Vec2(-8.75f + (float)c * 18.0f / (float)grid, 1.5f + (float)r * 18.0f / (float)grid),
                                       0.5f, 1.0f, 0.6f);
            s.setBodyDef(id, bd);
        }
    s.cameraCenter = Vec2(0.0f, 10.0f);
    s.cameraHeight = 25.0f;
}

// sample_stacking.cpp, Cliff: nine bodies sliding at 1.5 to 2.5 m/s over three ledges (a segment,
// a box and a capsule) and off their edges onto the floor.
inline void b2Cliff(Scene2D &s)
{
    s.clear();
    b2s::groundBox(s, Vec2(0.0f, -1.0f), 100.0f, 1.0f, 0.6f);
    s.addSegment(Vec2(-14.0f, 4.0f), Vec2(-8.0f, 4.0f), 0.6f);
    s.addBox(Vec2(0.0f, 4.0f), 0.0f, Vec2(6.0f, 1.0f), 0.0f, 0.6f);
    s.addCapsule(Vec2(8.5f, 4.0f), Vec2(13.5f, 4.0f), 0.5f, 0.0f, 0.6f);
    const Vec2 sq[4] = {Vec2(-0.5f, -0.5f), Vec2(0.5f, -0.5f), Vec2(0.5f, 0.5f), Vec2(-0.5f, 0.5f)};
    auto capsule = [&](float x, float y, float v) {
        s.addCapsule(Vec2(x - 0.25f, y), Vec2(x + 0.25f, y), 0.25f, 1.0f, 0.01f, Vec2(v, 0.0f));
    };
    capsule(-9.0f, 4.25f, 2.0f);
    capsule(2.0f, 4.75f, 2.0f);
    capsule(13.0f, 4.75f, 2.0f);
    s.addPolygon(Vec2(-11.0f, 4.5f), 0.0f, sq, 4, 0.0f, 1.0f, 0.01f, Vec2(2.5f, 0.0f));
    s.addPolygon(Vec2(0.0f, 5.0f), 0.0f, sq, 4, 0.0f, 1.0f, 0.01f, Vec2(2.5f, 0.0f));
    s.addPolygon(Vec2(11.0f, 5.0f), 0.0f, sq, 4, 0.0f, 1.0f, 0.01f, Vec2(2.5f, 0.0f));
    s.addCircle(Vec2(-13.0f, 4.5f), 0.5f, 1.0f, 0.2f, Vec2(1.5f, 0.0f));
    s.addCircle(Vec2(-2.0f, 5.0f), 0.5f, 1.0f, 0.2f, Vec2(1.5f, 0.0f));
    s.addCircle(Vec2(9.0f, 5.0f), 0.5f, 1.0f, 0.2f, Vec2(1.5f, 0.0f));
    s.cameraCenter = Vec2(0.0f, 5.0f);
    s.cameraHeight = 12.5f;
}

// sample_stacking.cpp, Capsule Stack: twenty 1 m capsules, one on top of the next.
inline void b2CapsuleStack(Scene2D &s)
{
    s.clear();
    b2s::groundBox(s, Vec2(0.0f, -1.0f), 10.0f, 1.0f, 0.6f);
    const float a = 0.25f;
    float y = 2.0f * a;
    for (int i = 0; i < 20; i++)
    {
        s.addCapsule(Vec2(-4.0f * a, y), Vec2(4.0f * a, y), a, 1.0f, 0.6f);
        y += 3.0f * a;
    }
    s.cameraCenter = Vec2(0.0f, 5.0f);
    s.cameraHeight = 12.0f;
}

// sample_robustness.cpp, HighMassRatio1: three ten-box pyramids, the capstone of each 100, 200
// and 300 times heavier than the rest.
inline void b2HighMassRatio1(Scene2D &s)
{
    s.clear();
    b2s::groundBox(s, Vec2(0.0f, -1.0f), 50.0f, 1.0f, 0.6f);
    const float extent = 1.0f;
    for (int j = 0; j < 3; j++)
    {
        int count = 10;
        const float offset = -20.0f * extent + 2.0f * ((float)count + 1.0f) * extent * (float)j;
        float y = extent;
        while (count > 0)
        {
            for (int i = 0; i < count; i++)
            {
                const float coeff = (float)i - 0.5f * (float)count;
                const float yy = count == 1 ? y + 2.0f : y;
                s.addBox(Vec2(2.0f * coeff * extent + offset, yy), 0.0f, Vec2(2.0f * extent, 2.0f * extent),
                         count == 1 ? ((float)j + 1.0f) * 100.0f : 1.0f, 0.6f);
            }
            count--;
            y += 2.0f * extent;
        }
    }
    s.cameraCenter = Vec2(3.0f, 14.0f);
    s.cameraHeight = 28.0f;
}

// sample_robustness.cpp, HighMassRatio2: a 20 m box dropped on two 1 m boxes.
inline void b2HighMassRatio2(Scene2D &s)
{
    s.clear();
    b2s::groundBox(s, Vec2(0.0f, -1.0f), 50.0f, 1.0f, 0.6f);
    s.addBox(Vec2(-9.0f, 0.5f), 0.0f, Vec2(1.0f, 1.0f), 1.0f, 0.6f);
    s.addBox(Vec2(9.0f, 0.5f), 0.0f, Vec2(1.0f, 1.0f), 1.0f, 0.6f);
    s.addBox(Vec2(0.0f, 26.0f), 0.0f, Vec2(20.0f, 20.0f), 1.0f, 0.6f);
    s.cameraCenter = Vec2(0.0f, 16.5f);
    s.cameraHeight = 28.0f;
}

// sample_robustness.cpp, HighMassRatio3: a 20 m box dropped on two small triangles.
inline void b2HighMassRatio3(Scene2D &s)
{
    s.clear();
    b2s::groundBox(s, Vec2(0.0f, -1.0f), 50.0f, 1.0f, 0.6f);
    const Vec2 tri[3] = {Vec2(-0.5f, 0.0f), Vec2(0.5f, 0.0f), Vec2(0.0f, 1.0f)};
    s.addPolygon(Vec2(-9.0f, 0.5f), 0.0f, tri, 3, 0.0f, 1.0f, 0.6f);
    s.addPolygon(Vec2(9.0f, 0.5f), 0.0f, tri, 3, 0.0f, 1.0f, 0.6f);
    s.addBox(Vec2(0.0f, 14.0f), 0.0f, Vec2(20.0f, 20.0f), 1.0f, 0.6f);
    s.cameraCenter = Vec2(0.0f, 16.5f);
    s.cameraHeight = 28.0f;
}

// sample_joints.cpp, Theo Jansen: a motor-driven wheel turning six two-bar legs (soft distance
// joints) under a chassis that walks over forty balls. The chassis, wheel and legs share the
// collision group -1. As in Box2D the wheel is turned to +-120 degrees while the second and third
// leg pairs are attached, then left at -120.
inline void b2TheoJansen(Scene2D &s)
{
    s.clear();
    const Vec2 offset(0.0f, 8.0f);
    const Vec2 pivot(0.0f, 0.8f);
    s.addSegment(Vec2(-50.0f, 0.0f), Vec2(50.0f, 0.0f), 0.6f);
    s.addSegment(Vec2(-50.0f, 0.0f), Vec2(-50.0f, 10.0f), 0.6f);
    s.addSegment(Vec2(50.0f, 0.0f), Vec2(50.0f, 10.0f), 0.6f);
    for (int i = 0; i < 40; i++)
        s.addCircle(Vec2(-40.0f + 2.0f * (float)i, 0.5f), 0.25f, 1.0f, 0.6f);

    ShapeDef2D grp;
    grp.friction = 0.6f;
    grp.groupIndex = -1;
    const int chassis = s.addBox(pivot + offset, 0.0f, Vec2(5.0f, 2.0f), 1.0f, 0.6f);
    s.setShapeDef(chassis, grp);
    const int wheel = s.addCircle(pivot + offset, 1.6f, 1.0f, 0.6f);
    s.setShapeDef(wheel, grp);
    {
        Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Revolute, wheel, chassis, pivot + offset);
        d.enableMotor = true;
        d.motorSpeed = 2.0f;
        d.maxMotorTorque = 400.0f;
        s.addJoint(d);
    }
    const Vec2 wheelAnchor = pivot + Vec2(0.0f, -0.8f);
    auto leg = [&](float sgn) {
        const Vec2 p1(5.4f * sgn, -6.1f), p2(7.2f * sgn, -1.2f), p3(4.3f * sgn, -1.9f);
        const Vec2 p4(3.1f * sgn, 0.8f), p5(6.0f * sgn, 1.5f), p6(2.5f * sgn, 3.7f);
        const Vec2 v1[3] = {p1, p2, p3};
        const Vec2 v2[3] = {Vec2(), p5 - p4, p6 - p4};
        const int b1 = s.addPolygon(offset, 0.0f, v1, 3, 0.0f, 1.0f, 0.6f);
        const int b2 = s.addPolygon(p4 + offset, 0.0f, v2, 3, 0.0f, 1.0f, 0.6f);
        s.setShapeDef(b1, grp);
        s.setShapeDef(b2, grp);
        s.angularDamping[(size_t)b1] = 10.0f;
        s.angularDamping[(size_t)b2] = 10.0f;
        // Soft distance constraints act as the suspension.
        Vec2 aA = p2 + offset, aB = p5 + offset;
        s.addDistanceJoint(b1, b2, aA, aB, length(aA - aB), true, 10.0f, 0.5f);
        aA = p3 + offset;
        aB = p4 + offset;
        s.addDistanceJoint(b1, b2, aA, aB, length(aA - aB), true, 10.0f, 0.5f);
        aB = wheelAnchor + offset;
        s.addDistanceJoint(b1, wheel, aA, aB, length(aA - aB), true, 10.0f, 0.5f);
        aA = p6 + offset;
        s.addDistanceJoint(b2, wheel, aA, aB, length(aA - aB), true, 10.0f, 0.5f);
        s.addJoint(s.makeJointDef(Scene2D::JointType2D::Revolute, b2, chassis, p4 + offset));
    };
    leg(-1.0f);
    leg(1.0f);
    s.ang[(size_t)wheel] = 120.0f * kPi / 180.0f;
    leg(-1.0f);
    leg(1.0f);
    s.ang[(size_t)wheel] = -120.0f * kPi / 180.0f;
    leg(-1.0f);
    leg(1.0f);
    s.cameraCenter = Vec2(0.0f, 6.0f);
    s.cameraHeight = 25.0f;
}

// sample_events.cpp, Contact Event: a bullet "player" circle and twenty drifting debris bodies in
// a 80 m loop of chain, no gravity. Only the player's contacts (and its hits) raise events; the
// debris, which Box2D spawns over time at random, is placed once from a fixed generator.
inline void b2ContactEvent(Scene2D &s)
{
    s.clear();
    const Vec2 pts[4] = {Vec2(40.0f, -40.0f), Vec2(-40.0f, -40.0f), Vec2(-40.0f, 40.0f), Vec2(40.0f, 40.0f)};
    s.addChain(pts, 4, true, 0.6f);
    BodyDef2D nog;
    nog.gravityScale = 0.0f;
    nog.linearDamping = 0.5f;
    nog.angularDamping = 0.5f;
    const int player = s.addCircle(Vec2(0.0f, 0.0f), 1.0f, 1.0f, 0.6f, Vec2(14.0f, 9.0f));
    s.setBodyDef(player, nog);
    s.isBullet[(size_t)player] = 1;
    s.setHitEvents(player, true);
    uint32_t seed = 0x2545F491u;
    auto rnd = [&](float lo, float hi) {
        seed = seed * 1664525u + 1013904223u;
        return lo + (hi - lo) * ((float)(seed >> 8) * (1.0f / 16777216.0f));
    };
    nog.linearDamping = 0.0f;
    nog.angularDamping = 0.0f;
    for (int i = 0; i < 20; i++)
    {
        const Vec2 p(rnd(-38.0f, 38.0f), rnd(-38.0f, 38.0f));
        const float a = rnd(-kPi, kPi);
        const Vec2 v(rnd(-5.0f, 5.0f), rnd(-5.0f, 5.0f));
        const float w = rnd(-1.0f, 1.0f);
        int id;
        if ((i + 1) % 3 == 0)
            id = s.addCircle(p, 0.5f, 1.0f, 0.6f, v, w);
        else if ((i + 1) % 2 == 0)
        {
            const Vec2 ax(-std::sin(a), std::cos(a)); // the capsule's long axis is local y
            id = s.addCapsule(p - ax * 0.25f, p + ax * 0.25f, 0.25f, 1.0f, 0.6f, v, w);
        }
        else
            id = s.addBox(p, a, Vec2(0.8f, 1.2f), 1.0f, 0.6f, v, w);
        s.setBodyDef(id, nog);
        s.setContactEvents(id, false);
    }
    s.cameraCenter = Vec2(0.0f, 0.0f);
    s.cameraHeight = 90.0f;
}

// benchmarks.c, Many Tumblers: a 19 x 19 field of kinematic hollow boxes (four slabs, density 50)
// turning at 25 degrees per second, each holding capsules. Box2D drips fifty capsules into every
// tumbler over time; here forty-nine are placed up front on a 7 x 7 grid.
inline void b2ManyTumblers(Scene2D &s)
{
    s.clear();
    s.params.killFallenEnabled = false;
    ShapeDef2D sd;
    sd.density = 50.0f;
    sd.friction = 0.6f;
    const int rows = 19, cols = 19;
    float x = -4.0f * (float)rows;
    for (int i = 0; i < rows; i++)
    {
        float y = -4.0f * (float)cols;
        for (int j = 0; j < cols; j++)
        {
            BodyDef2D bd;
            bd.position = Vec2(x, y);
            bd.type = BODY2D_KINEMATIC;
            bd.angularVelocity = (kPi / 180.0f) * 25.0f;
            s.beginBody(bd);
            s.addBoxShape(Vec2(2.0f, 0.0f), 0.0f, Vec2(0.5f, 4.0f), sd);
            s.addBoxShape(Vec2(-2.0f, 0.0f), 0.0f, Vec2(0.5f, 4.0f), sd);
            s.addBoxShape(Vec2(0.0f, 2.0f), 0.0f, Vec2(4.0f, 0.5f), sd);
            s.addBoxShape(Vec2(0.0f, -2.0f), 0.0f, Vec2(4.0f, 0.5f), sd);
            s.endBody();
            for (int k = 0; k < 7; k++)
                for (int l = 0; l < 7; l++)
                {
                    const Vec2 p(x + (float)(k - 3) * 0.45f, y + (float)(l - 3) * 0.45f);
                    s.addCapsule(p - Vec2(0.1f, 0.0f), p + Vec2(0.1f, 0.0f), 0.075f, 1.0f, 0.6f);
                }
            y += 8.0f;
        }
        x += 8.0f;
    }
    s.cameraCenter = Vec2(1.0f, -5.5f);
    s.cameraHeight = 170.0f;
}

// Not a Box2D sample: the "typical game" scene of the roadmap, a few hundred mixed bodies. A chain
// terrain with side walls, box stacks, a shower of boxes, circles, capsules and polygons, a rope
// and a pendulum on revolute joints, a seesaw, a car, a kinematic windmill and a kinematic lift.
inline void b2TypicalGame(Scene2D &s)
{
    s.clear();
    auto h = [](float x) { return 1.2f * std::sin(0.25f * x) + 0.6f * std::sin(0.6f * x + 1.0f) + 2.0f; };
    {
        std::vector<Vec2> pts;
        for (float x = 32.0f; x >= -32.0f; x -= 1.0f) // right to left: solid below
            pts.push_back(Vec2(x, h(x)));
        s.addChain(pts.data(), (int)pts.size(), false, 0.6f);
        s.addSegment(Vec2(-32.0f, h(-32.0f)), Vec2(-32.0f, 40.0f), 0.6f);
        s.addSegment(Vec2(32.0f, h(32.0f)), Vec2(32.0f, 40.0f), 0.6f);
    }
    uint32_t seed = 12345u;
    auto rnd = [&](float lo, float hi) {
        seed = seed * 1664525u + 1013904223u;
        return lo + (hi - lo) * ((float)(seed >> 8) * (1.0f / 16777216.0f));
    };
    // Box stacks on the terrain.
    for (int k = 0; k < 8; k++)
    {
        const float x = -28.0f + 3.0f * (float)k;
        for (int i = 0; i < 6; i++)
            s.addBox(Vec2(x, h(x) + 0.9f + 0.82f * (float)i), 0.0f, Vec2(0.8f, 0.8f), 1.0f, 0.6f);
    }
    // A shower of mixed shapes.
    for (int i = 0; i < 200; i++)
    {
        const Vec2 p(rnd(-26.0f, 26.0f), 16.0f + rnd(0.0f, 14.0f));
        const float a = rnd(-kPi, kPi);
        switch (i % 5)
        {
        case 0:
            s.addBox(p, a, Vec2(rnd(0.4f, 0.9f), rnd(0.4f, 0.9f)), 1.0f, 0.5f);
            break;
        case 1:
            s.addCircle(p, rnd(0.2f, 0.45f), 1.0f, 0.4f);
            break;
        case 2:
            s.addCapsule(p - Vec2(0.25f, 0.0f), p + Vec2(0.25f, 0.0f), 0.2f, 1.0f, 0.5f);
            break;
        case 3:
        {
            Vec2 v[5];
            for (int m = 0; m < 5; m++)
            {
                const float t = 2.0f * kPi * (float)m / 5.0f;
                v[m] = Vec2(std::cos(t), std::sin(t)) * 0.4f;
            }
            s.addPolygon(p, a, v, 5, 0.0f, 1.0f, 0.5f);
            break;
        }
        default:
        {
            const Vec2 v[3] = {Vec2(-0.4f, -0.3f), Vec2(0.4f, -0.3f), Vec2(0.0f, 0.5f)};
            s.addPolygon(p, a, v, 3, 0.05f, 1.0f, 0.5f);
            break;
        }
        }
    }
    // A rope of capsules from a world pin, ending in a ball.
    {
        const float x0 = 18.0f, y0 = 24.0f;
        int prev = -1;
        for (int i = 0; i < 10; i++)
        {
            const int id = s.addCapsule(Vec2(x0 + (float)i * 0.5f, y0), Vec2(x0 + (float)i * 0.5f + 0.5f, y0), 0.1f,
                                        2.0f, 0.4f);
            b2s::revolute(s, prev, id, Vec2(x0 + (float)i * 0.5f, y0));
            prev = id;
        }
        const int ball = s.addCircle(Vec2(x0 + 5.5f, y0), 0.5f, 2.0f, 0.4f);
        b2s::revolute(s, prev, ball, Vec2(x0 + 5.0f, y0));
    }
    // A pendulum.
    {
        const int arm = s.addBox(Vec2(-12.0f, 22.0f), 0.0f, Vec2(0.2f, 4.0f), 1.0f, 0.4f);
        b2s::revolute(s, -1, arm, Vec2(-12.0f, 24.0f));
        const int bob = s.addCircle(Vec2(-12.0f, 19.5f), 0.6f, 2.0f, 0.4f);
        b2s::revolute(s, arm, bob, Vec2(-12.0f, 19.5f));
    }
    // A seesaw with limits.
    {
        const float x = 4.0f, y = h(x) + 1.2f;
        const int plank = s.addBox(Vec2(x, y), 0.0f, Vec2(6.0f, 0.2f), 1.0f, 0.6f);
        Scene2D::JointDef2D d = s.makeJointDef(Scene2D::JointType2D::Revolute, -1, plank, Vec2(x, y));
        d.enableLimit = true;
        d.lower = -0.3f;
        d.upper = 0.3f;
        s.addJoint(d);
    }
    // A car.
    b2s::car(s, Vec2(-24.0f, h(-24.0f) + 0.6f), 0.6f, 4.0f, 0.7f, 2.0f, -5.0f);
    // A kinematic windmill and a lift.
    {
        const int mill = s.addBox(Vec2(10.0f, 12.0f), 0.0f, Vec2(8.0f, 0.4f), 1.0f, 0.5f);
        s.setKinematic(mill, Vec2(), 0.8f);
        const int lift = s.addBox(Vec2(-20.0f, 14.0f), 0.0f, Vec2(5.0f, 0.4f), 1.0f, 0.8f);
        s.setKinematic(lift, Vec2(0.5f, 0.0f));
    }
    s.cameraCenter = Vec2(0.0f, 14.0f);
    s.cameraHeight = 45.0f;
}

} // namespace scenes
} // namespace avbd2d
