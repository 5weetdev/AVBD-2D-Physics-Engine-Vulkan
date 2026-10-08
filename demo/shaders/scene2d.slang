// Flat-colour instanced renderer for the 2D demo. Each instance is one primitive; a quad
// (triangle strip, corners generated from the vertex index) covers it and the fragment shader
// shapes it: boxes get an outline, circles are cut out of the quad and get a spoke that shows
// their rotation, capsules are cut to their rounded ends. A triangle instance places its three
// vertices directly (the fourth strip vertex repeats the third) and outlines the edges its
// flags mark as the polygon's boundary; the demo fans a polygon into these. The camera arrives
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
                            // triangle: vertex 0 (local)
    float4 iColor : ICOLOR;
    uint iShape : ISHAPE;   // 0 box, 1 circle, 2 capsule, 3 triangle
    uint iFlags : IFLAGS;   // bit 0: highlighted (grabbed); bits 8..10: triangle edges 01, 12, 20 are outlined
    float4 iExtra : IEXTRA; // triangle: vertices 1 and 2 (local)
};

struct VsOut
{
    float4 clip : SV_Position;
    float2 lp : TEXCOORD0; // body-local position, world units
    float4 color : COLOR0;
    nointerpolation float2 hext : TEXCOORD1;
    nointerpolation uint shape : TEXCOORD2;
    nointerpolation uint flags : TEXCOORD3;
    nointerpolation float4 extra : TEXCOORD4;
};

[shader("vertex")]
VsOut vsMain(VsIn v, uint vid : SV_VulkanVertexID) // not SV_VertexID: that subtracts BaseVertex and needs shaderDrawParameters
{
    // Triangle strip: (-1,-1) (1,-1) (-1,1) (1,1).
    float2 corner = float2((vid & 1u) != 0u ? 1.0f : -1.0f, (vid & 2u) != 0u ? 1.0f : -1.0f);
    float2 local;
    if (v.iShape == 3u)
        local = vid == 0u ? v.iHalf : (vid == 1u ? v.iExtra.xy : v.iExtra.zw);
    else if (v.iShape == 2u)
        local = corner * float2(v.iHalf.x + v.iHalf.y, v.iHalf.y);
    else
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
    o.extra = v.iExtra;
    return o;
}

float lineDist(float2 p, float2 a, float2 b)
{
    float2 e = b - a;
    float len = length(e);
    return len > 0.0 ? abs(e.x * (p.y - a.y) - e.y * (p.x - a.x)) / len : 1e9;
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
        float edge = 1e9;
        if ((i.flags & 0x100u) != 0u)
            edge = min(edge, lineDist(p, i.hext, i.extra.xy));
        if ((i.flags & 0x200u) != 0u)
            edge = min(edge, lineDist(p, i.extra.xy, i.extra.zw));
        if ((i.flags & 0x400u) != 0u)
            edge = min(edge, lineDist(p, i.extra.zw, i.hext));
        if (edge < 0.03)
            return float4(outline, 1.0);
        return float4(base, 1.0);
    }

    float edge = min(i.hext.x - abs(p.x), i.hext.y - abs(p.y));
    if (edge < 0.03)
        return float4(outline, 1.0);
    return float4(base, 1.0);
}
