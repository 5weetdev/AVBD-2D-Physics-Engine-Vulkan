// 2D maths for the 2D solver kernels: vec2/vec3 value types (scalar layout, so they
// match the host's tightly packed float pairs/triples), the handful of vector helpers the
// kernels need, and the 3x3 LDL^T solve.

module maths2d;

public struct vec2
{
    public float x, y;
};

public struct vec3
{
    public float x, y, z;
};

public vec2 v2add(vec2 a, vec2 b) { return vec2(a.x + b.x, a.y + b.y); }
public vec2 v2sub(vec2 a, vec2 b) { return vec2(a.x - b.x, a.y - b.y); }
public vec2 v2mul(vec2 a, float s) { return vec2(a.x * s, a.y * s); }
public vec2 v2neg(vec2 a) { return vec2(-a.x, -a.y); }
public float v2dot(vec2 a, vec2 b) { return a.x * b.x + a.y * b.y; }
public float v2cross(vec2 a, vec2 b) { return a.x * b.y - a.y * b.x; }
public float v2lenSq(vec2 a) { return a.x * a.x + a.y * a.y; }
public float v2len(vec2 a) { return sqrt(a.x * a.x + a.y * a.y); }
public vec2 v2perp(vec2 a) { return vec2(-a.y, a.x); }

// Rotation by `angle` radians (counter-clockwise).
public vec2 v2rot(float angle, vec2 v)
{
    float c = cos(angle);
    float s = sin(angle);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

public float clampf(float x, float lo, float hi) { return max(lo, min(hi, x)); }
// |g| without a sqrt when gravity lies on an axis (the usual case), so the default stays exact.
public float gravityMag(vec2 g)
{
    if (g.x == 0.0f)
        return abs(g.y);
    if (g.y == 0.0f)
        return abs(g.x);
    return sqrt(g.x * g.x + g.y * g.y);
}
public float signf(float x) { return x < 0.0f ? -1.0f : (x > 0.0f ? 1.0f : 0.0f); }
public bool finitef(float x) { return (x == x) && !isinf(x); }

// Pivot guard for the LDL^T factorisation, the
// system is SPD by construction, but float32 cancellation can collapse a pivot.
public float ldlPivot(float d, float eps) { return d > eps ? d : eps; }

// Solves the symmetric 3x3 system A x = b, A = [a00 a01 a02; a01 a11 a12; a02 a12 a22]
// (unknowns: linear x, linear y, angle) by LDL^T with a scale-aware pivot floor.
public void solve3(float a00, float a01, float a02, float a11, float a12, float a22,
                   float b0, float b1, float b2, out float x0, out float x1, out float x2)
{
    float maxDiag = max(max(a00, a11), a22);
    float eps = max(maxDiag * 1.0e-5f, 1.0e-6f);

    float D1 = ldlPivot(a00, eps);
    float L21 = a01 / D1;
    float L31 = a02 / D1;

    float D2 = ldlPivot(a11 - L21 * L21 * D1, eps);
    float L32 = (a12 - L21 * L31 * D1) / D2;

    float D3 = ldlPivot(a22 - (L31 * L31 * D1 + L32 * L32 * D2), eps);

    float y1 = b0;
    float y2 = b1 - L21 * y1;
    float y3 = b2 - L31 * y1 - L32 * y2;

    float z1 = y1 / D1;
    float z2 = y2 / D2;
    float z3 = y3 / D3;

    x2 = z3;
    x1 = z2 - L32 * x2;
    x0 = z1 - L21 * x1 - L31 * x2;
}
