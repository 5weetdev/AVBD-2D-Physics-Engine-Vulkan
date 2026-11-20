#pragma once

// Small-world step paths (Phase B of docs/SOLVER_3D_ROADMAP.md): variants of a phase that
// trade the multi-dispatch layout for fewer dispatches and barriers when the world is small
// enough for one workgroup. Every such path is bit-identical to the path it replaces.
//
// AVBD_VK_SMALL_WORLD=0 turns every one of them off (the multi-dispatch path then runs
// everywhere), which is how the numerics oracle compares them: avbd_vk_trajhash must print
// the same final hash either way.

#include <cstdlib>

namespace avbdvk
{

// Read once per process.
inline bool smallWorldEnabled()
{
    static const bool on = []()
    {
        const char *e = std::getenv("AVBD_VK_SMALL_WORLD");
        return !(e && e[0] == '0');
    }();
    return on;
}

// Fused iteration loop (csIterateSmall): the most dynamic bodies, manifold slots and joint
// slots one 256-thread workgroup runs through per step. Colour chunks of 64 bodies are walked
// serially, so a colour of 1024 bodies is 16 chunks; the slot passes stride by 256.
constexpr int kFusedIterMaxBodies = 1024;
constexpr int kFusedIterMaxManifoldSlots = 8192;
constexpr int kFusedIterMaxJointSlots = 4096;

// Small front end (one-workgroup sorts and csBuildSmall).
// Sorts: used when the sort CAPACITY is at most kSmallSortMaxKeys (mirrors SORT_SMALL_MAX in
// sort_small.slang). That module uses 4096 * (8 + 4) = 49152 bytes of workgroup memory
// (kSmallSortSharedBytes), so its pipelines are only created, and the path only taken, on a
// device whose maxComputeSharedMemorySize is at least that; otherwise radix runs.
// csBuildSmall: constraints.slang uses 14344 bytes of workgroup memory (fits the 16 KB Vulkan
// minimum). Used when the dynamic-body bound is at most kSmallBuildMaxDyn (BUILD_SMALL_MAX_DYN
// in the shader) and the body capacity at most kSmallBuildMaxBodies.
constexpr unsigned kSmallSortMaxKeys = 4096;
constexpr unsigned kSmallSortSharedBytes = 49152;
constexpr int kSmallBuildMaxDyn = 2048;
constexpr int kSmallBuildMaxBodies = 16384;

} // namespace avbdvk
