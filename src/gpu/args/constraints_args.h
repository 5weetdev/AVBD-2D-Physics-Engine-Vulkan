#pragma once

// Constraint-graph build argument structs (constraints.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// constraints.slang
// ===========================================================================

struct CsResetColorsArgsGpu
{
    VkDeviceAddress color = 0;
    VkDeviceAddress dyn = 0;
    uint32_t dynCount = 0, _pad0 = 0;
    VkDeviceAddress dynCountPtr = 0; // live count (0: dynCount is exact)
};
static_assert(sizeof(CsResetColorsArgsGpu) % 8 == 0, "layout");

struct CsEmitManifoldArgsGpu
{
    VkDeviceAddress mBodyA = 0, mBodyB = 0, mContactCount = 0;
    uint32_t slots = 0, _pad0 = 0;
    VkDeviceAddress mass = 0, awake = 0;
    VkDeviceAddress keys = 0, isA = 0, counter = 0;
    int32_t capacity = 0, _pad1 = 0;
};
static_assert(sizeof(CsEmitManifoldArgsGpu) % 8 == 0, "layout");

struct CsEmitForceArgsGpu
{
    VkDeviceAddress fBodyA = 0, fBodyB = 0;
    uint32_t count = 0, _pad0 = 0;
    VkDeviceAddress broken = 0; // may be 0 (springs)
    VkDeviceAddress mass = 0, awake = 0;
    VkDeviceAddress keys = 0, isA = 0, counter = 0;
    int32_t capacity = 0;
    int32_t type = 0; // ForceType: FORCE_JOINT=1, FORCE_SPRING=2
};
static_assert(sizeof(CsEmitForceArgsGpu) % 8 == 0, "layout");

struct CsRowsArgsGpu
{
    VkDeviceAddress keys = 0;
    uint32_t n = 0, _pad0 = 0;
    VkDeviceAddress start = 0, end = 0;
    VkDeviceAddress count = 0; // live count; `n` is then the capacity
};
static_assert(sizeof(CsRowsArgsGpu) % 8 == 0, "layout");

struct CsJpRoundArgsGpu
{
    VkDeviceAddress gKey = 0, gIsA = 0, gStart = 0, gEnd = 0;
    VkDeviceAddress mBodyA = 0, mBodyB = 0;
    VkDeviceAddress jBodyA = 0, jBodyB = 0;
    VkDeviceAddress sBodyA = 0, sBodyB = 0;
    VkDeviceAddress mass = 0, awake = 0;
    VkDeviceAddress dyn = 0;
    uint32_t dynCount = 0, _pad0 = 0;
    VkDeviceAddress color = 0, remaining = 0;
    VkDeviceAddress dynCountPtr = 0; // live count (0: dynCount is exact)
};
static_assert(sizeof(CsJpRoundArgsGpu) % 8 == 0, "layout");

struct CsColorKeysArgsGpu
{
    VkDeviceAddress color = 0, dyn = 0;
    uint32_t dynCount = 0, _pad0 = 0;
    VkDeviceAddress keys = 0, vals = 0;
    VkDeviceAddress dynCountPtr = 0; // live count (0: dynCount is exact)
    VkDeviceAddress midTag = 0;      // distance LOD, may be 0
};
static_assert(sizeof(CsColorKeysArgsGpu) % 8 == 0, "layout");

struct CsColorRowsArgsGpu
{
    VkDeviceAddress keys = 0;
    uint32_t n = 0, _pad0 = 0;
    VkDeviceAddress start = 0, end = 0;
    VkDeviceAddress dynCountPtr = 0; // live count (0: n is exact)
    VkDeviceAddress nearEnd = 0;     // distance LOD, may be 0
};
static_assert(sizeof(CsColorRowsArgsGpu) % 8 == 0, "layout");

// csBuildSmall (constraints.slang): the one-workgroup constraint build.
struct CsBuildSmallArgsGpu
{
    VkDeviceAddress mBodyA = 0, mBodyB = 0, mCount = 0;
    VkDeviceAddress jBodyA = 0, jBodyB = 0, jBroken = 0; // jBroken may be 0
    VkDeviceAddress sBodyA = 0, sBodyB = 0;
    VkDeviceAddress mass = 0, awake = 0, dyn = 0;
    VkDeviceAddress keys = 0, isA = 0, rowStart = 0, rowEnd = 0;
    VkDeviceAddress color = 0, bodyByColor = 0;
    VkDeviceAddress ranges = 0; // start[128], end[128], nearEnd[128]
    VkDeviceAddress counter = 0, remaining = 0;
    VkDeviceAddress dynCountPtr = 0; // may be 0
    VkDeviceAddress midTag = 0;      // may be 0
    uint32_t slots = 0, jointSlots = 0, springSlots = 0, bodyCount = 0, dynCount = 0;
    int32_t capacity = 0;
    uint32_t rounds = 0, _pad0 = 0;
};
static_assert(sizeof(CsBuildSmallArgsGpu) == 22 * 8 + 8 * 4,
              "CsBuildSmallArgsGpu drifted from constraints.slang BuildSmallArgs");

} // namespace avbdvk
