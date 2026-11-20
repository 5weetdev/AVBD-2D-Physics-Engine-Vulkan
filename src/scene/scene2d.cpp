#include "scene2d.h"

namespace avbd2d
{

int Scene2D::addCapsule(Vec2 p1, Vec2 p2, float r, float density, float mu, Vec2 v, float w, uint32_t rgba)
{
    const Vec2 d = p2 - p1;
    const float L = length(d);
    const float h = 0.5f * L;
    float m = 0.0f, inertia = 0.0f;
    if (density > 0.0f)
    {
        // b2ComputeCapsuleMass.
        const float circleMass = density * kPi * r * r;
        const float boxMass = density * 2.0f * r * L;
        const float lc = 4.0f * r / (3.0f * kPi);
        m = circleMass + boxMass;
        inertia = circleMass * (0.5f * r * r + h * h + 2.0f * h * lc) + boxMass * (4.0f * r * r + L * L) / 12.0f;
    }
    const float angle = L > 0.0f ? std::atan2(d.y, d.x) : 0.0f;
    const int id = addBody((p1 + p2) * 0.5f, angle, Vec2(h + r, r), SHAPE2D_CAPSULE, m, inertia, mu, v, w, rgba);
    const Vec2 pts[2] = {Vec2(-h, 0.0f), Vec2(h, 0.0f)};
    const Vec2 nrm[2] = {Vec2(0.0f, -1.0f), Vec2(0.0f, 1.0f)}; // b2MakeCapsule: rightPerp(axis), its negation
    setShape(id, pts, nrm, 2, r);
    return id;
}

int Scene2D::addChain(const Vec2 *points, int count, bool loop, float mu, uint32_t rgba)
{
    if (count < (loop ? 3 : 2))
        return -1;
    const int n = loop ? count : count - 1;
    auto P = [&](int i)
    {
        if (loop)
            return points[((i % count) + count) % count];
        if (i < 0)
            return points[0] * 2.0f - points[1];
        if (i >= count)
            return points[count - 1] * 2.0f - points[count - 2];
        return points[i];
    };
    int first = -1;
    for (int i = 0; i < n; i++)
    {
        const Vec2 g1 = P(i - 1), p1 = P(i), p2 = P(i + 1), g2 = P(i + 2);
        const Vec2 c = (p1 + p2) * 0.5f;
        const Vec2 h(std::fabs(p2.x - p1.x) * 0.5f, std::fabs(p2.y - p1.y) * 0.5f);
        const int id = addBody(c, 0.0f, h, SHAPE2D_CHAIN, 0.0f, 0.0f, mu, Vec2(), 0.0f, rgba);
        const Vec2 e = p2 - p1;
        const float len = length(e);
        const Vec2 nr = len > 0.0f ? Vec2(e.y / len, -e.x / len) : Vec2(0.0f, 1.0f); // right of p1 -> p2
        const Vec2 pts[4] = {g1 - c, p1 - c, p2 - c, g2 - c};
        const Vec2 nrm[4] = {nr, nr, nr, nr};
        setShape(id, pts, nrm, 4, 0.0f);
        if (first < 0)
            first = id;
    }
    return first;
}

int Scene2D::computeHull(const Vec2 *points, int count, Vec2 *out)
{
    if (count < 3 || count > kMaxPolygonVertices)
        return 0;
    Vec2 ps[kMaxPolygonVertices];
    int n = 0;
    const float tolSq = 0.25f * kLinearSlop * kLinearSlop;
    for (int i = 0; i < count; i++)
    {
        bool unique = true;
        for (int j = 0; j < n && unique; j++)
            unique = lengthSq(points[i] - ps[j]) >= tolSq;
        if (unique)
            ps[n++] = points[i];
    }
    if (n < 3)
        return 0;
    std::sort(ps, ps + n, [](Vec2 a, Vec2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    Vec2 h[2 * kMaxPolygonVertices + 1];
    int k = 0;
    auto turn = [](Vec2 o, Vec2 a, Vec2 b) { return cross(a - o, b - o); };
    for (int i = 0; i < n; i++)
    {
        while (k >= 2 && turn(h[k - 2], h[k - 1], ps[i]) <= 0.0f)
            k--;
        h[k++] = ps[i];
    }
    for (int i = n - 2, lo = k + 1; i >= 0; i--)
    {
        while (k >= lo && turn(h[k - 2], h[k - 1], ps[i]) <= 0.0f)
            k--;
        h[k++] = ps[i];
    }
    k--; // the last point repeats the first
    bool dropped = true;
    while (dropped && k >= 3)
    {
        dropped = false;
        for (int i = 0; i < k && !dropped; i++)
        {
            const Vec2 a = h[(i + k - 1) % k], b = h[i], c = h[(i + 1) % k];
            const float len = length(c - a);
            if (len > 0.0f && std::fabs(cross(b - a, c - a)) / len < 2.0f * kLinearSlop)
            {
                for (int j = i; j < k - 1; j++)
                    h[j] = h[j + 1];
                k--;
                dropped = true;
            }
        }
    }
    if (k < 3)
        return 0;
    for (int i = 0; i < k; i++)
        out[i] = h[i];
    return k;
}

int Scene2D::addJoint(const JointDef2D &d)
{
    JointAxis2 ax[3];
    float frame = std::atan2(d.localAxisA.y, d.localAxisA.x);
    auto lockAxis = [&](int k) { ax[k].pos = JOINT_AXIS_LOCK; };
    auto limitMotorSpring = [&](int k, bool spring, bool limit, bool motor, float maxF)
    {
        if (spring)
        {
            ax[k].pos = JOINT_AXIS_SPRING;
            ax[k].hertz = d.hertz;
            ax[k].stiff = kStiffInf;
        }
        if (limit)
        {
            ax[k].limit = 1;
            ax[k].lo = d.lower;
            ax[k].hi = d.upper;
        }
        if (motor)
        {
            const uint32_t mode = d.motorMode ? d.motorMode : (uint32_t)JOINT_MOTOR_VELOCITY;
            ax[k].motor = mode;
            ax[k].mtarget = mode == JOINT_MOTOR_TARGET ? d.motorTarget : d.motorSpeed;
            ax[k].maxF = maxF;
        }
    };
    auto weldAxis = [&](int k, float hz)
    {
        if (hz > 0.0f)
        {
            ax[k].pos = JOINT_AXIS_SPRING;
            ax[k].hertz = hz;
            ax[k].stiff = kStiffInf;
        }
        else
            ax[k].pos = JOINT_AXIS_LOCK;
    };
    switch (d.type)
    {
    case JointType2D::Revolute:
        lockAxis(0);
        lockAxis(1);
        limitMotorSpring(2, d.enableSpring, d.enableLimit, d.enableMotor, d.maxMotorTorque);
        break;
    case JointType2D::Prismatic:
        limitMotorSpring(0, d.enableSpring, d.enableLimit, d.enableMotor, d.maxMotorForce);
        lockAxis(1);
        lockAxis(2);
        break;
    case JointType2D::Wheel: // suspension spring and limit along the axis, motor spins the wheel
        limitMotorSpring(0, d.enableSpring, d.enableLimit, false, 0.0f);
        lockAxis(1);
        if (d.enableMotor)
        {
            const uint32_t mode = d.motorMode ? d.motorMode : (uint32_t)JOINT_MOTOR_VELOCITY;
            ax[2].motor = mode;
            ax[2].mtarget = mode == JOINT_MOTOR_TARGET ? d.motorTarget : d.motorSpeed;
            ax[2].maxF = d.maxMotorTorque;
        }
        break;
    case JointType2D::Weld:
        weldAxis(0, d.linearHertz);
        weldAxis(1, d.linearHertz);
        weldAxis(2, d.angularHertz);
        break;
    case JointType2D::Motor:
        for (int k = 0; k < 3; k++)
            ax[k].motor = JOINT_MOTOR_TARGET;
        ax[0].mtarget = d.linearOffset.x;
        ax[1].mtarget = d.linearOffset.y;
        ax[2].mtarget = d.angularOffset;
        ax[0].maxF = ax[1].maxF = d.maxMotorForce;
        ax[2].maxF = d.maxMotorTorque;
        break;
    case JointType2D::Distance:
        ax[0].flags = JOINT_FLAG_DISTANCE;
        ax[0].rest = d.length;
        if (d.enableSpring)
        {
            ax[0].pos = JOINT_AXIS_SPRING;
            ax[0].hertz = d.hertz;
            ax[0].stiff = kStiffInf;
        }
        else
            ax[0].pos = JOINT_AXIS_LOCK;
        if (d.enableLimit)
        {
            ax[0].limit = 1;
            ax[0].lo = d.minLength;
            ax[0].hi = d.maxLength;
        }
        if (d.enableMotor)
        {
            const uint32_t mode = d.motorMode ? d.motorMode : (uint32_t)JOINT_MOTOR_VELOCITY;
            ax[0].motor = mode;
            ax[0].mtarget = mode == JOINT_MOTOR_TARGET ? d.motorTarget : d.motorSpeed;
            ax[0].maxF = d.maxMotorForce;
        }
        frame = 0.0f;
        break;
    case JointType2D::Filter:
        if (d.bodyA >= 0 && d.bodyB >= 0)
            ignoreCollision(d.bodyA, d.bodyB);
        break;
    }
    // A per-joint stiff lock for the active ragdoll: hard position rows read `lockScale` as a penalty-cap
    // multiplier. Spring rows ignore it, so this cannot disturb the stock ragdoll.
    if (d.lockScale > 0.0f)
        for (int k = 0; k < 3; k++)
            ax[k].lockScale = d.lockScale;
    return pushJoint(d.bodyA, d.bodyB, d.localAnchorA, d.localAnchorB, frame, d.referenceAngle, ax,
                     d.collideConnected, d.breakForce);
}

void Scene2D::setBodyDef(int id, const BodyDef2D &d)
{
    const size_t b = (size_t)id;
    bodyType[b] = (uint8_t)d.type;
    gravityScale[b] = d.gravityScale;
    linearDamping[b] = d.linearDamping;
    angularDamping[b] = d.angularDamping;
    lockX[b] = d.lockLinearX ? 1 : 0;
    lockY[b] = d.lockLinearY ? 1 : 0;
    lockRot[b] = d.fixedRotation ? 1 : 0;
    if (d.type != BODY2D_DYNAMIC)
    {
        mass[b] = 0.0f;
        moment[b] = 0.0f;
    }
    if (d.type == BODY2D_STATIC)
    {
        vel[b] = Vec2();
        angVel[b] = 0.0f;
    }
}

void Scene2D::addCapsuleShape(Vec2 p1, Vec2 p2, float r, float density, float mu)
{
    const Vec2 d = p2 - p1;
    const float L = length(d);
    const float h = 0.5f * L;
    Pending q = pendingShape(SHAPE2D_CAPSULE, (p1 + p2) * 0.5f, L > 0.0f ? std::atan2(d.y, d.x) : 0.0f,
                             Vec2(h + r, r), r, mu);
    if (density > 0.0f)
    {
        const float circleMass = density * kPi * r * r;
        const float boxMass = density * 2.0f * r * L;
        const float lc = 4.0f * r / (3.0f * kPi);
        q.mass = circleMass + boxMass;
        q.inertia = circleMass * (0.5f * r * r + h * h + 2.0f * h * lc) + boxMass * (4.0f * r * r + L * L) / 12.0f;
    }
    q.n = 2;
    q.v[0] = Vec2(-h, 0.0f);
    q.v[1] = Vec2(h, 0.0f);
    q.nr[0] = Vec2(0.0f, -1.0f);
    q.nr[1] = Vec2(0.0f, 1.0f);
    pending.push_back(q);
}

int Scene2D::endBody()
{
    const BodyDef2D d = pendingBody;
    const int id = endBody(d.linearVelocity, d.angularVelocity, d.color);
    if (id >= 0)
    {
        // A dynamic body whose shapes have no density: Box2D has no default mass, so this solver
        // gives it mass 1 and the inertia of a disc of the body's reach (0.5 * reach^2).
        if (d.type == BODY2D_DYNAMIC && mass[(size_t)id] <= 0.0f)
        {
            mass[(size_t)id] = 1.0f;
            moment[(size_t)id] = 0.5f * half[(size_t)id].x * half[(size_t)id].x;
            vel[(size_t)id] = d.linearVelocity;
            angVel[(size_t)id] = d.angularVelocity;
        }
        const Vec2 v = vel[(size_t)id];
        const float w = angVel[(size_t)id];
        setBodyDef(id, d);
        if (d.type == BODY2D_KINEMATIC)
        {
            vel[(size_t)id] = d.linearVelocity;
            angVel[(size_t)id] = d.angularVelocity;
        }
        else if (d.type == BODY2D_DYNAMIC)
        {
            vel[(size_t)id] = v;
            angVel[(size_t)id] = w;
        }
    }
    pendingBody = BodyDef2D();
    return id;
}

int Scene2D::endBody(Vec2 v, float w, uint32_t rgba)
{
    if (pending.empty())
        return -1;
    float M = 0.0f;
    Vec2 c;
    for (const Pending &q : pending)
    {
        M += q.mass;
        c = c + q.off * q.mass;
    }
    c = M > 0.0f ? c * (1.0f / M) : Vec2();
    float I = 0.0f, reach = 0.0f;
    for (const Pending &q : pending)
    {
        I += q.inertia + q.mass * lengthSq(q.off - c);
        const float own = q.type == SHAPE2D_CIRCLE ? q.rad : length(q.half);
        reach = std::max(reach, length(q.off - c) + own);
    }
    const int id = bodyCount();
    pushBody(pendingP + rotate(pendingAngle, c), pendingAngle, Vec2(reach, reach), M, I, v, w, rgba);
    localCenter[(size_t)id] = c;
    shapeCount[(size_t)id] = (int)pending.size();
    for (const Pending &q : pending)
    {
        const int vo = (int)verts.size();
        verts.insert(verts.end(), q.v, q.v + q.n);
        norms.insert(norms.end(), q.nr, q.nr + q.n);
        pushShape(id, q.type, q.off - c, q.angle, q.half, q.rad, q.mu, vo, q.n);
        applyShapeDef((size_t)shapeTotal() - 1, q.def);
    }
    pending.clear();
    return id;
}

void Scene2D::resizeShapes(int n)
{
    const size_t c = (size_t)n;
    shType.resize(c, (uint8_t)SHAPE2D_BOX);
    shOff.resize(c);
    shHalf.resize(c);
    shAng.resize(c, 0.0f);
    shRad.resize(c, 0.0f);
    shFric.resize(c, 0.5f);
    shVOff.resize(c, 0);
    shVCnt.resize(c, 0);
    shBody.resize(c, -1);
    shCat.resize(c, 1);
    shMask.resize(c, ~uint64_t(0));
    shGroup.resize(c, 0);
    shUserId.resize(c, 0);
    shTangentSpeed.resize(c, 0.0f);
    shRolling.resize(c, 0.0f);
    shRestitution.resize(c, 0.0f);
    shFlags.resize(c, (uint8_t)SHAPE_FLAG_DEFAULT);
}

int Scene2D::installBody(const Scene2D &src, int k, int slot, int shapeOff, int vertOff)
{
    const size_t q = (size_t)k;
    int id = slot;
    if (id < 0)
    {
        id = bodyCount();
        pushBody(src.pos[q], src.ang[q], src.half[q], src.mass[q], src.moment[q], src.vel[q], src.angVel[q],
                 src.color[q]);
    }
    const size_t d = (size_t)id;
    pos[d] = src.pos[q];
    ang[d] = src.ang[q];
    half[d] = src.half[q];
    mass[d] = src.mass[q];
    moment[d] = src.moment[q];
    vel[d] = src.vel[q];
    angVel[d] = src.angVel[q];
    color[d] = src.color[q];
    localCenter[d] = src.localCenter[q];
    bodyType[d] = src.bodyType[q];
    gravityScale[d] = src.gravityScale[q];
    linearDamping[d] = src.linearDamping[q];
    angularDamping[d] = src.angularDamping[q];
    lockX[d] = src.lockX[q];
    lockY[d] = src.lockY[q];
    lockRot[d] = src.lockRot[q];
    isBullet[d] = src.isBullet[q];
    const int n = src.shapeCount[q];
    shapeFirst[d] = shapeOff;
    shapeCount[d] = n;
    if (shapeOff + n > shapeTotal())
        resizeShapes(shapeOff + n);
    const int vt = src.vertTotal(k);
    if (vertOff + vt > (int)verts.size())
    {
        verts.resize((size_t)(vertOff + vt));
        norms.resize((size_t)(vertOff + vt));
    }
    int vo = vertOff;
    for (int i = 0; i < n; i++)
    {
        const size_t s = (size_t)(shapeOff + i), p = (size_t)(src.shapeFirst[q] + i);
        const int cnt = src.shVCnt[p];
        shType[s] = src.shType[p];
        shOff[s] = src.shOff[p];
        shHalf[s] = src.shHalf[p];
        shAng[s] = src.shAng[p];
        shRad[s] = src.shRad[p];
        shFric[s] = src.shFric[p];
        shVOff[s] = cnt > 0 ? vo : 0;
        shVCnt[s] = cnt;
        shBody[s] = id;
        shCat[s] = src.shCat[p];
        shMask[s] = src.shMask[p];
        shGroup[s] = src.shGroup[p];
        shUserId[s] = src.shUserId[p];
        shTangentSpeed[s] = src.shTangentSpeed[p];
        shRolling[s] = src.shRolling[p];
        shRestitution[s] = src.shRestitution[p];
        shFlags[s] = src.shFlags[p];
        for (int c = 0; c < cnt; c++)
        {
            verts[(size_t)(vo + c)] = src.verts[(size_t)(src.shVOff[p] + c)];
            norms[(size_t)(vo + c)] = src.norms[(size_t)(src.shVOff[p] + c)];
        }
        vo += cnt;
    }
    return id;
}

int Scene2D::appendBody(const Scene2D &src, int i, Vec2 offset)
{
    const size_t k = (size_t)i;
    const int id = bodyCount();
    pushBody(src.pos[k] + offset, src.ang[k], src.half[k], src.mass[k], src.moment[k], src.vel[k], src.angVel[k],
             src.color[k]);
    localCenter[(size_t)id] = src.localCenter[k];
    shapeCount[(size_t)id] = src.shapeCount[k];
    bodyType[(size_t)id] = src.bodyType[k];
    gravityScale[(size_t)id] = src.gravityScale[k];
    linearDamping[(size_t)id] = src.linearDamping[k];
    angularDamping[(size_t)id] = src.angularDamping[k];
    lockX[(size_t)id] = src.lockX[k];
    lockY[(size_t)id] = src.lockY[k];
    lockRot[(size_t)id] = src.lockRot[k];
    vel[(size_t)id] = src.vel[k];
    angVel[(size_t)id] = src.angVel[k];
    isBullet[(size_t)id] = src.isBullet[k];
    for (int s = src.shapeFirst[k]; s < src.shapeFirst[k] + src.shapeCount[k]; s++)
    {
        const size_t q = (size_t)s;
        const int o = src.shVOff[q], n = src.shVCnt[q];
        const int vo = (int)verts.size();
        verts.insert(verts.end(), src.verts.begin() + o, src.verts.begin() + o + n);
        norms.insert(norms.end(), src.norms.begin() + o, src.norms.begin() + o + n);
        pushShape(id, src.shType[q], src.shOff[q], src.shAng[q], src.shHalf[q], src.shRad[q], src.shFric[q], vo, n);
        const size_t d = (size_t)shapeTotal() - 1;
        shCat[d] = src.shCat[q];
        shMask[d] = src.shMask[q];
        shGroup[d] = src.shGroup[q];
        shUserId[d] = src.shUserId[q];
        shTangentSpeed[d] = src.shTangentSpeed[q];
        shRolling[d] = src.shRolling[q];
        shRestitution[d] = src.shRestitution[q];
        shFlags.back() = src.shFlags[q];
    }
    return id;
}

bool Scene2D::shapeContains(int s, Vec2 p) const
{
    const size_t k = (size_t)s;
    const Vec2 h = shHalf[k];
    const float r = shRad[k];
    const Vec2 *v = verts.data() + shVOff[k];
    const int n = shVCnt[k];
    switch (shType[k])
    {
    case SHAPE2D_BOX:
        return std::fabs(p.x) <= h.x && std::fabs(p.y) <= h.y;
    case SHAPE2D_CIRCLE:
        return n == 0 ? lengthSq(p) <= h.x * h.x : lengthSq(p - v[0]) <= r * r;
    case SHAPE2D_CAPSULE:
    case SHAPE2D_POLYGON:
    {
        if (n < 2)
            return false;
        const Vec2 *nr = norms.data() + shVOff[k];
        bool inside = n > 2;
        for (int j = 0; j < n && inside; j++)
            inside = dot(nr[j], p - v[j]) <= 0.0f;
        if (inside)
            return true;
        for (int j = 0; j < n; j++)
        {
            const Vec2 a = v[j], e = v[(j + 1) % n] - a;
            const float ee = lengthSq(e);
            const float t = ee > 0.0f ? std::min(std::max(dot(p - a, e) / ee, 0.0f), 1.0f) : 0.0f;
            if (lengthSq(p - (a + e * t)) <= r * r)
                return true;
        }
        return false;
    }
    default:
        return false;
    }
}

bool Scene2D::polyProps(const Vec2 *points, int count, float r, float density, PolyProps &P)
{
    Vec2 hull[kMaxPolygonVertices];
    const int n = computeHull(points, count, hull);
    if (n < 3)
        return false;
    Vec2 nrm[kMaxPolygonVertices];
    for (int i = 0; i < n; i++)
    {
        const Vec2 e = hull[(i + 1) % n] - hull[i];
        const float len = length(e);
        nrm[i] = Vec2(e.y / len, -e.x / len);
    }
    // b2ComputePolygonMass: a rounded polygon's vertices are pushed out along the bisectors.
    Vec2 mv[kMaxPolygonVertices];
    for (int i = 0; i < n; i++)
    {
        mv[i] = hull[i];
        if (r > 0.0f)
        {
            const Vec2 mid = nrm[(i + n - 1) % n] + nrm[i];
            mv[i] = hull[i] + mid * (1.412f * r / length(mid));
        }
    }
    Vec2 center;
    float area = 0.0f, inertia = 0.0f;
    const Vec2 o = mv[0];
    for (int i = 1; i < n - 1; i++)
    {
        const Vec2 e1 = mv[i] - o, e2 = mv[i + 1] - o;
        const float D = cross(e1, e2);
        const float a = 0.5f * D;
        area += a;
        center = center + (e1 + e2) * (a / 3.0f);
        const float intx2 = e1.x * e1.x + e2.x * e1.x + e2.x * e2.x;
        const float inty2 = e1.y * e1.y + e2.y * e1.y + e2.y * e2.y;
        inertia += (0.25f / 3.0f * D) * (intx2 + inty2);
    }
    center = center * (1.0f / area);
    float m = 0.0f, I = 0.0f;
    if (density > 0.0f)
    {
        m = density * area;
        I = density * inertia - m * dot(center, center);
    }
    const Vec2 c = o + center; // the centroid, in the authoring frame
    Vec2 h;
    for (int i = 0; i < n; i++)
    {
        hull[i] = hull[i] - c;
        h = Vec2(std::max(h.x, std::fabs(hull[i].x) + r), std::max(h.y, std::fabs(hull[i].y) + r));
    }
    P.n = n;
    for (int i = 0; i < n; i++)
    {
        P.v[i] = hull[i];
        P.nr[i] = nrm[i];
    }
    P.centroid = c;
    P.half = h;
    P.mass = m;
    P.inertia = I;
    return true;
}

void Scene2D::pushBody(Vec2 p, float angle, Vec2 h, float m, float inertia, Vec2 v, float w, uint32_t rgba)
{
    pos.push_back(p);
    ang.push_back(angle);
    half.push_back(h);
    mass.push_back(m);
    moment.push_back(inertia);
    vel.push_back(m > 0.0f ? v : Vec2());
    angVel.push_back(m > 0.0f ? w : 0.0f);
    color.push_back(rgba);
    localCenter.push_back(Vec2());
    shapeFirst.push_back(shapeTotal());
    shapeCount.push_back(0);
    bodyType.push_back(m > 0.0f ? BODY2D_DYNAMIC : BODY2D_STATIC);
    lockX.push_back(0);
    lockY.push_back(0);
    lockRot.push_back(0);
    gravityScale.push_back(1.0f);
    linearDamping.push_back(0.0f);
    angularDamping.push_back(0.0f);
    isBullet.push_back(0);
}

} // namespace avbd2d
