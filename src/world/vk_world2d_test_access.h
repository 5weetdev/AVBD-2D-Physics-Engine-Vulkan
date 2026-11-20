#pragma once

// Test-only access to VkWorld2D internals (capacity forcing, per-pass op counts). Production code
// must not include this header.

#include "vk_world2d.h"

namespace avbd2d
{

struct VkWorld2DTestAccess
{
    // Forces tiny pair/contact capacities so the first steps exercise the rerun path.
    static void forceTinyCapacity(VkWorld2D &w, int pairCap, int contactCap)
    {
        w.debugForceTinyCapacity(pairCap, contactCap);
    }
    // Forces every event kind's buffer to `cap` records at the next build().
    static void forceTinyEventCapacity(VkWorld2D &w, int cap)
    {
        w.m_forcedEventCap = cap;
    }
    // Ops recorded by the last pass (dispatches, compute barriers, transfers).
    static avbdvk::CommandList::OpCounts lastOps(const VkWorld2D &w)
    {
        return w.m_cmd.opCounts();
    }
    static bool lastPassFused(const VkWorld2D &w)
    {
        return w.m_lastFused;
    }
};

} // namespace avbd2d
