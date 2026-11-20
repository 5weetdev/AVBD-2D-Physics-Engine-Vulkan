#pragma once

// Flat-colour instanced renderer for the 2D demo: one quad per primitive (box, circle,
// capsule, rounded polygon), an orthographic camera, no depth, no lighting. The instances
// are built on the host from the world's pose read-back and written into a host-visible
// vertex buffer, one per frame in flight.

#include <vulkan/vulkan.h>

#include <cstdint>

#include "vk_buffer.h"

namespace avbdvk
{
class Device;
}

enum : uint32_t
{
    kInstBox = 0,
    kInstCircle = 1,
    kInstCapsule = 2,
    kInstPolygon = 3,
};

struct Instance2D
{
    float x, y, angle;
    float hx, hy;        // box half extents; circle: radius in hx; capsule: (half core length, radius);
                         // polygon: local AABB half extents about the centroid, rounding included
    uint32_t rgba;       // R in the low byte, A in the high byte (255 = opaque; blended)
    uint32_t shape;      // kInst* above
    uint32_t flags;      // bit 0: highlighted (grabbed)
    float radius = 0.0f; // polygon: rounding radius (0 draws a sharp polygon)
    uint32_t count = 0;  // polygon: vertex count
    float v[16] = {};    // polygon: up to 8 local vertices about the centroid (x0,y0,x1,y1,...)
};
static_assert(sizeof(Instance2D) == 104, "Instance2D is mirrored by the vertex input in renderer2d.cpp");

class Renderer2D
{
public:
    static constexpr int kMaxFramesInFlight = 2;

    void init(avbdvk::Device &dev, VkRenderPass pass);
    void destroy();

    // Copies the instances into this frame slot's buffer (growing it if needed -- the caller
    // has waited on the slot's fence) and records the draw inside `pass`, with the viewport
    // and scissor already set. cam = (centre x, centre y, NDC per world unit x, y).
    void draw(VkCommandBuffer cmd, int frameSlot, const Instance2D *instances, uint32_t count, const float cam[4]);

private:
    avbdvk::Device *m_dev = nullptr;
    VkShaderModule m_module = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    avbdvk::MappedBuffer m_inst[kMaxFramesInFlight];
    uint32_t m_capacity[kMaxFramesInFlight] = {0, 0};
};
