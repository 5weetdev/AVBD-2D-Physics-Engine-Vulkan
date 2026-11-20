// Flat-colour instanced renderer for the 2D demo. Each instance is one primitive; a quad
// (triangle strip, corners generated from the vertex index) covers it and the fragment shader
// shapes it: boxes get an outline, circles are cut out of the quad and get a spoke that shows
// their rotation, capsules are cut to their rounded ends. A polygon instance carries its up-to-8
// local convex CCW vertices and rounding radius; the fragment shader evaluates the exact signed
// distance to the rounded shape (straight edges, arcs at the corners) and outlines its boundary.
// The camera arrives
// as push constants: (centre x, centre y, scale x, scale y), where scale maps world units to
// NDC. World +y points up the screen.

struct PushConstants
{
    float4 cam;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct VsIn
{
    float3 iPose : IPOSE;   // x, y, angle
    float2 iHalf : IHALF;   // box half extents; circle: radius in x; capsule: (half core length, radius);
                            // polygon: local AABB half extents (rounding radius included)
    float4 iColor : ICOLOR;
    uint iShape : ISHAPE;   // 0 box, 1 circle, 2 capsule, 3 rounded polygon
    uint iFlags : IFLAGS;   // bit 0: highlighted (grabbed)
    float iRadius : IRADIUS; // polygon: rounding radius
    uint iCount : ICOUNT;    // polygon: vertex count
    float4 iV01 : IV01;      // polygon: vertices 0 and 1 (local, about the centroid)
    float4 iV23 : IV23;
    float4 iV45 : IV45;
    float4 iV67 : IV67;
};

struct VsOut
{
    float4 clip : SV_Position;
    float2 lp : TEXCOORD0; // body-local position, world units
    float4 color : COLOR0;
    nointerpolation float2 hext : TEXCOORD1;
    nointerpolation uint shape : TEXCOORD2;
    nointerpolation uint flags : TEXCOORD3;
    nointerpolation float radius : TEXCOORD4;
    nointerpolation uint count : TEXCOORD5;
    nointerpolation float4 v01 : TEXCOORD6;
    nointerpolation float4 v23 : TEXCOORD7;
    nointerpolation float4 v45 : TEXCOORD8;
    nointerpolation float4 v67 : TEXCOORD9;
};

[shader("vertex")]
VsOut vsMain(VsIn v, uint vid : SV_VulkanVertexID) // not SV_VertexID: that subtracts BaseVertex and needs shaderDrawParameters
{
    // Triangle strip: (-1,-1) (1,-1) (-1,1) (1,1).
    float2 corner = float2((vid & 1u) != 0u ? 1.0f : -1.0f, (vid & 2u) != 0u ? 1.0f : -1.0f);
    float2 local;
    if (v.iShape == 2u)
        local = corner * float2(v.iHalf.x + v.iHalf.y, v.iHalf.y);
    else // box; circle and polygon (the fragment shader cuts these out of the AABB quad)
        local = corner * v.iHalf;
    float c = cos(v.iPose.z);
    float s = sin(v.iPose.z);
    float2 world = v.iPose.xy + float2(c * local.x - s * local.y, s * local.x + c * local.y);

    VsOut o;
    o.clip = float4((world.x - pc.cam.x) * pc.cam.z, -(world.y - pc.cam.y) * pc.cam.w, 0.0, 1.0);
    o.lp = local;
    o.color = v.iColor;
    o.hext = v.iHalf;
    o.shape = v.iShape;
    o.flags = v.iFlags;
    o.radius = v.iRadius;
    o.count = v.iCount;
    o.v01 = v.iV01;
    o.v23 = v.iV23;
    o.v45 = v.iV45;
    o.v67 = v.iV67;
    return o;
}

[shader("fragment")]
float4 fsMain(VsOut i) : SV_Target
{
    float3 base = i.color.rgb;
    if ((i.flags & 1u) != 0u)
        base = lerp(base, float3(1.0, 1.0, 1.0), 0.45);
    const float3 outline = base * 0.45;
    const float2 p = i.lp;

    if (i.shape == 1u)
    {
        float radius = i.hext.x;
        float r = length(p);
        if (r > radius)
            discard;
        // Outline ring ~0.03 world units wide, and a spoke along the body's +x axis.
        if (radius - r < 0.03)
            return float4(outline, 1.0);
        if (p.x > 0.0 && abs(p.y) < 0.025)
            return float4(outline, 1.0);
        return float4(base, 1.0);
    }

    if (i.shape == 2u)
    {
        float radius = i.hext.y;
        float d = length(float2(p.x - clamp(p.x, -i.hext.x, i.hext.x), p.y));
        if (d > radius)
            discard;
        if (radius - d < 0.03)
            return float4(outline, 1.0);
        return float4(base, 1.0);
    }

    if (i.shape == 3u)
    {
        // Exact signed distance to a convex CCW polygon offset by the rounding radius
        // (straight edges between the offset lines, arcs at the corners). Port of Box2D's
        // sdConvexPolygon (reference box2d, samples/data/solid_polygon.fs): d2 tracks the
        // closest edge segment, `side` flips outside as soon as p is right of any CCW edge.
        float2 v[8];
        v[0] = i.v01.xy;
        v[1] = i.v01.zw;
        v[2] = i.v23.xy;
        v[3] = i.v23.zw;
        v[4] = i.v45.xy;
        v[5] = i.v45.zw;
        v[6] = i.v67.xy;
        v[7] = i.v67.zw;
        const int n = int(i.count);
        float d2 = dot(p - v[0], p - v[0]);
        float side = -1.0;
        int j = n - 1;
        for (int k = 0; k < n; k++)
        {
            const float2 e = v[k] - v[j];
            const float2 w = p - v[j];
            const float2 b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
            d2 = min(d2, dot(b, b));
            if (w.x * e.y - w.y * e.x >= 0.0)
                side = 1.0;
            j = k;
        }
        const float d = side * sqrt(d2) - i.radius;
        if (d > 0.0)
            discard;
        if (d > -0.03)
            return float4(outline, 1.0);
        return float4(base, 1.0);
    }

    float edge = min(i.hext.x - abs(p.x), i.hext.y - abs(p.y));
    if (edge < 0.03)
        return float4(outline, 1.0);
    return float4(base, 1.0);
}
