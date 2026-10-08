#pragma once

// Flat-colour instanced renderer for the 2D demo: one quad per primitive (box, circle, capsule, polygon-fan triangle), an orthographic camera,
// no depth, no lighting. The instances are built on the host from the world's pose read-back
// and written into a host-visible vertex buffer, one per frame in flight.

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
    kInstTriangle = 3,
};

struct Instance2D
{
    float x, y, angle;
    float hx, hy;     // box half extents; circle: radius in hx; capsule: (half core length, radius);
                      // triangle: vertex 0 (local)
    uint32_t rgba;    // R in the low byte
    uint32_t shape;   // kInst* below
    uint32_t flags;   // bit 0: highlighted; bits 8..10: triangle edges 01, 12, 20 are outlined
    float ex[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // triangle: vertices 1 and 2 (local)
};
static_assert(sizeof(Instance2D) == 48, "Instance2D is mirrored by the vertex input in renderer2d.cpp");

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
