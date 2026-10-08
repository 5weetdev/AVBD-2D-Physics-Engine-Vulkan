/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#pragma once

// Vulkan host orchestration for shaders/constraints.slang: CSR entry emission over
// manifolds + joints + springs, the (body<<32)|(type<<29)|index key sort, and the
// batched Jones-Plassmann coloring loop.
//
// Like VkBroadphase/VkNarrowphase, this owns its own CommandList/HostRingBuffer. Since
// M4 a build() is ONE submission: emit, sort and rows run on a device entry count, and
// the Jones-Plassmann rounds are recorded blind to a bound (last build's convergence
// round + 2, at least 8) instead of a host loop that dispatches until the remaining
// counter reads zero. A round past convergence is a no-op, so the colours are the
// host loop's exactly; only a bound that proves too small costs another submission.

#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"
#include "vk_primitives.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace avbdvk
{

// Colors are
// held in a 128-bit mask (two uint64) in csJpRound, so this is a hard limit, not a
// tunable.
static const int AVBD_MAX_COLORS_VK = 128;

struct ConstraintsPipelines
{
    Pipeline resetColors, emitManifold, emitForce, rows, jpRound, colorKeys, colorRows;

    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void destroy();
};

// Device addresses of the body fields the graph build reads: mass (float, >0 = dynamic)
// and awake (uint8, 0/1).
struct VkConstraintsBodyRefs
{
    VkDeviceAddress mass = 0;
    VkDeviceAddress awake = 0;
};

// bodyA/bodyB/contactCount, mirroring the subset of ManifoldSoA k_emit_entries reads.
struct VkConstraintsManifoldRefs
{
    VkDeviceAddress bodyA = 0, bodyB = 0, contactCount = 0;
    int slots = 0;
};

// bodyA/bodyB/broken, mirroring the subset of JointSoA k_emit_force_entries<FORCE_JOINT>
// reads. `broken` may be 0 (no fracture tracked).
struct VkConstraintsJointRefs
{
    VkDeviceAddress bodyA = 0, bodyB = 0, broken = 0;
    int count = 0;
};

// bodyA/bodyB, mirroring the subset of SpringSoA k_emit_force_entries<FORCE_SPRING>
// reads. Springs have no `broken` array.
struct VkConstraintsSpringRefs
{
    VkDeviceAddress bodyA = 0, bodyB = 0;
    int count = 0;
};

// Device addresses of the built graph, as built by VkConstraints.
struct VkConstraintGraph
{
    VkDeviceAddress key = 0;
    VkDeviceAddress isA = 0;
    VkDeviceAddress start = 0;
    VkDeviceAddress end = 0;
};

class VkConstraints
{
public:
    void init(Device &dev);
    void destroy();
    // Frees the capacity-sized buffers and keeps the command pool and ring, so the
    // next reserve() rebuilds from nothing. VkWorld::reset() calls this: a scene
    // switch must not hold the old scene's memory until the new one is built.
    void release();

    // `maxForces` is the joint and spring count, each contributing up to two entries
    // just as a manifold can.
    void reserve(int maxBodies, int maxManifolds, int maxForces = 0);

    void build(ConstraintsPipelines &pl, PrimitivesPipelines &prim, RadixScratch &radix,
               const VkConstraintsBodyRefs &bodies,
               const VkConstraintsManifoldRefs &manifolds,
               const VkConstraintsJointRefs &joints,
               const VkConstraintsSpringRefs &springs,
               const Buffer &dynamicIdx, int dynamicCount, VkDeviceAddress dynamicCountAddr = 0);

    // dynamicCountAddr (M4, here and below): 0 means `dynamicCount` is exact. Non-zero is
    // the device address of an int32 live count and `dynamicCount` is only its upper bound:
    // grids are sized by the bound, every kernel reads min(*addr, bound) and early-outs.

    // build() in pieces (M4). record() zeroes the outputs, resets colours, emits the
    // entries and records the sort and the row ranges into `cmd`, with the entry count
    // kept on device. The caller must have grown `radix` for maxEntries (8-byte keys,
    // 4-byte values) BEFORE recording anything into `cmd` that uses it. finish() runs
    // after the submission and reads back the entry count, the convergence round and
    // the colour ranges.
    void record(CommandList &cmd, HostRingBuffer &ring, ConstraintsPipelines &pl, PrimitivesPipelines &prim,
                RadixScratch &radix, const VkConstraintsBodyRefs &bodies,
                const VkConstraintsManifoldRefs &manifolds, const VkConstraintsJointRefs &joints,
                const VkConstraintsSpringRefs &springs, const Buffer &dynamicIdx, int dynamicCount,
                VkDeviceAddress dynamicCountAddr = 0);
    void finish();
    int entryCapacity() const { return maxEntries; }

    // Jones-Plassmann rounds recorded without a readback between them: `count` rounds
    // starting at round `firstRound`, then the colour grouping. Returns rounds recorded.
    // Needs `radix` grown for dynamicCount (4-byte keys and values) before recording.
    int recordColoring(CommandList &cmd, HostRingBuffer &ring, ConstraintsPipelines &pl, PrimitivesPipelines &prim,
                       RadixScratch &radix, const VkConstraintsBodyRefs &bodies,
                       const VkConstraintsManifoldRefs &manifolds, const VkConstraintsJointRefs &joints,
                       const VkConstraintsSpringRefs &springs, const Buffer &dynamicIdx, int dynamicCount,
                       int firstRound, int count, VkDeviceAddress dynamicCountAddr = 0);
    // The rounds to record blind: a slowly decaying peak of recent convergence rounds + 2,
    // at least 8. Rounds jitter step to step, and every shortfall costs a rerun.
    int jpBound() const { return std::min(std::max(8, (int)ceilf(m_peakRounds) + 2), kMaxJpRounds); }
    // Same cap as the old host loop (AVBD_MAX_COLORS * 4 rounds).
    static const int kMaxJpRounds = AVBD_MAX_COLORS_VK * 4;

    VkConstraintGraph graph() const
    {
        VkConstraintGraph g;
        g.key = sortedKeys.address();
        g.isA = sortedIsA.address();
        g.start = rowStart.address();
        g.end = rowEnd.address();
        return g;
    }

    int entryCount() const { return entries; }

    int colorCount() const { return colors; }
    const Buffer &colorBodies() const { return bodyByColor; }
    const int *colorStart() const { return hColorRanges.data(); }
    const int *colorEnd() const { return hColorRanges.data() + AVBD_MAX_COLORS_VK; }
    // The same ranges on device (AVBD_MAX_COLORS_VK ints each), for indirect dispatch.
    VkDeviceAddress colorStartAddr() const { return colorRanges.address(); }
    VkDeviceAddress colorEndAddr() const
    {
        return colorRanges.address() + (VkDeviceAddress)(AVBD_MAX_COLORS_VK * sizeof(int32_t));
    }
    // Distance LOD. With a mid tag set (one byte per body, nonzero = outside the near
    // tier; may be 0 to turn grouping off) each colour's near bodies sort first and
    // nearEndAddr()[c] is where they end, so [start, nearEnd) is the near-only range.
    // Device-only: the host colour ranges and the readout stay start/end.
    void setMidTag(VkDeviceAddress tag) { m_midTag = tag; }
    VkDeviceAddress nearEndAddr() const
    {
        return colorRanges.address() + (VkDeviceAddress)(2 * AVBD_MAX_COLORS_VK * sizeof(int32_t));
    }

    int coloringRounds() const { return rounds; }

    // Record-into-caller build (step loop). prepare() grows the radix scratch and the
    // readout and must run before anything is recorded into the caller's command list.
    // recordBuild() records a FRESH build (round 0 on) with `jpRounds` blind JP rounds
    // and a copy of the entry counter, the per-round `remaining` slots and the colour
    // ranges into a mapped readout; the caller submits and waits, then finishRecorded()
    // reads it. jpConverged() says whether some recorded round left nothing uncoloured.
    void prepare(RadixScratch &radix, int dynamicBound);
    int recordBuild(CommandList &cmd, HostRingBuffer &ring, ConstraintsPipelines &pl, PrimitivesPipelines &prim,
                    RadixScratch &radix, const VkConstraintsBodyRefs &bodies,
                    const VkConstraintsManifoldRefs &manifolds, const VkConstraintsJointRefs &joints,
                    const VkConstraintsSpringRefs &springs, const Buffer &dynamicIdx, int dynamicBound,
                    VkDeviceAddress dynamicCountAddr, int jpRounds);
    void finishRecorded();
    bool jpConverged() const { return m_jpConverged; }
    // Device address of round r's `remaining` slot (an int32 counter of uncoloured bodies).
    VkDeviceAddress remainingSlotAddr(int r) const
    {
        return remaining.address() + (VkDeviceAddress)r * sizeof(uint32_t);
    }

    const Buffer &bodyColor() const { return color; }

    // Raw buffers, exposed for readback and debugging.
    const Buffer &sortedKeysBuf() const { return sortedKeys; }
    const Buffer &sortedIsABuf() const { return sortedIsA; }
    const Buffer &rowStartBuf() const { return rowStart; }
    const Buffer &rowEndBuf() const { return rowEnd; }

    // Transitive byte total (point 3): every device buffer this class owns, for
    // VkWorld::deviceBytesAllocated()'s memory gate.
    uint64_t bytesAllocated() const
    {
        const Buffer *all[] = {
            &rawKeys, &sortedKeys, &rawIsA, &sortedIsA, &rowStart, &rowEnd, &counter,
            &color, &colorKey, &colorKeySorted, &bodyByColor, &bodyScratch,
            &colorRanges, &remaining,
        };
        uint64_t total = 0;
        for (const Buffer *b : all)
            total += (uint64_t)b->size();
        return total;
    }

private:
    Device *m_dev = nullptr;
    HostRingBuffer ring;
    CommandList cmd;

    Buffer rawKeys, sortedKeys;
    Buffer rawIsA, sortedIsA;
    Buffer rowStart, rowEnd;
    Buffer counter;

    Buffer color;
    Buffer colorKey, colorKeySorted;
    Buffer bodyByColor, bodyScratch;
    Buffer colorRanges;
    Buffer remaining;
    // recordBuild's one host readout: [0] entry counter, [1, 1+kMaxJpRounds) the JP
    // `remaining` slots, then the 2*AVBD_MAX_COLORS_VK colour range ints.
    MappedBuffer readout;
    int m_readoutRounds = 0; // JP rounds the pending recordBuild recorded
    bool m_jpConverged = true;

    std::vector<int> hColorRanges;
    VkDeviceAddress m_midTag = 0; // may be 0

    int maxBodies = 0, maxEntries = 0;
    int entries = 0;
    int colors = 0;
    int rounds = 0;
    bool m_recorded = false;
    int m_recordedRounds = 0; // rounds recorded into the pending submission
    float m_peakRounds = 0.0f; // decaying max of m_lastRounds, for jpBound()
    int m_lastRounds = 0;     // convergence round of the last build, for jpBound()

    void finishColoring(int recordedRounds);
};

} // namespace avbdvk
