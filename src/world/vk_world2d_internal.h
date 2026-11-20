// Constants and helpers shared by the VkWorld2D translation units (vk_world2d*.cpp). Not a public header.

#pragma once

#include "vk_world2d.h"

#include <algorithm>
#include <chrono>

namespace avbd2d
{

using avbdvk::Buffer;

using Clock = std::chrono::steady_clock;

inline constexpr int kMaxColors = avbdvk::AVBD_MAX_COLORS_VK;
inline constexpr int kAbsMaxPairs = 16 * 1000 * 1000;
inline constexpr int kMinPairCeiling = 65536;
inline constexpr int kPairsPerBodyCeiling = 12; // a real pile needs ~4 per body; this is the runaway cap
inline constexpr int kMaxTimedIterations = 32;  // the command list carries 80 timestamps
inline constexpr int kJointWakeRounds = 6;
inline constexpr int kGrabSlotCount = VkWorld2D::kGrabSlots;
inline constexpr float kGrabTurnRate = 5.0f; // rad/s the grab may turn the held group (45 degrees in ~0.16 s)

inline void sizeBuf(Buffer &b, size_t bytes, bool preserve, VkBufferUsageFlags extra = 0)
{
    bytes = std::max<size_t>(bytes, 16);
    if (preserve)
        b.resizePreserve(bytes, extra);
    else
        b.resize(bytes, extra);
}

// Uploads one element of an array of `elemSize`-byte elements.
inline void putAt(Buffer &b, int index, size_t elemSize, const void *src)
{
    b.upload(src, elemSize, (VkDeviceSize)index * elemSize);
}

// The contact-debug snapshot's layout (see VkWorld2D::setContactDebug): the per-pair arrays
// the overlays consume at offset 0, then the per-contact arrays after them, 8-aligned. The
// step copies these inside its own submission; downloadContacts() reads the same layout.
inline constexpr size_t kDbgPairStride = 8 + 4 + 4 + 4 + 4 + 8 + 4; // key bodyA bodyB count offset normal friction
inline constexpr size_t kDbgContactStride = 8 + 8 + 8 + 8 + 1;      // rA rB lambda penalty stick

inline VkDeviceSize dbgContactBase(int pairCap)
{
    return (VkDeviceSize(pairCap) * kDbgPairStride + 7) & ~VkDeviceSize(7);
}

inline size_t dbgBytes(int pairCap, int contactCap)
{
    return (size_t)(dbgContactBase(pairCap) + VkDeviceSize(contactCap) * kDbgContactStride);
}

// Body / joint patch flags (the mutation queue).
inline constexpr uint32_t BP_SET_POSE = 1, BP_SET_VEL = 2, BP_ADD_VEL = 4, BP_WAKE = 8, BP_DISABLE = 16, BP_ENABLE = 32,
                   BP_SLEEP = 64;
inline constexpr uint32_t JP_MOTOR = 1, JP_LIMIT = 2, JP_LIMIT_ON = 4, JP_WAKE = 16, JP_LOCK_SCALE = 32;

} // namespace avbd2d
