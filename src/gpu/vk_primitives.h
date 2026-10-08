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

// GPU primitives: exclusive scan, stream compaction, and a stable LSD radix sort
// (used by the broadphase and the constraint builder).
//
// Shape: a temp-storage size query plus a run call, and the
// run call only *records* into an already-`begin()`-called CommandList -- no host sync
// happens inside any function here, matching the push-constant /
// buffer-device-address dispatch design. The
// temp storage here is a handful of internally-owned, lazily-grown Buffers (a
// DeviceBuffer<T>-style "never shrinks, grows to fit" scratch struct per primitive) --
// simpler to get right than manual byte-offset aliasing into one blob, and just as
// reusable across calls.

#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace avbdvk
{

// One PipelineLayout-compatible pipeline per shader entry point used by the primitives
// below. All share the engine-wide single-push-constant layout, so any of these
// Pipelines may be dispatched interleaved with any other kernel's (integrate, solver,
// and so on) on the same CommandList.
struct PrimitivesPipelines
{
    Pipeline scanBlock, addBlockSums;
    Pipeline flagsToU32, scatterCompact, writeNumSelected;
    Pipeline histU32Pairs, scatterU32Pairs;
    Pipeline histU64Pairs, scatterU64Pairs;
    Pipeline histU64Keys, scatterU64Keys;
    Pipeline reduceOrAndU32, reduceOrAndU64;
    Pipeline finalizeSkipMaskU32, finalizeSkipMaskU64;

    void init(Device &dev, PipelineLayout &layout);
    void destroy();
};

// ---------------------------------------------------------------------------
// Scan: exclusive prefix sum over uint32, n up to ~2^32 (recursion depth is
// log_256(n), so 4 levels covers 20M+ comfortably).
// ---------------------------------------------------------------------------
struct ScanScratch
{
    void init(Device &dev) { m_dev = &dev; }
    void destroy();

    Device *m_dev = nullptr;
    std::vector<std::unique_ptr<Buffer>> levelB; // block sums per recursion level
    std::vector<std::unique_ptr<Buffer>> levelR; // scanned block sums per level (level 0 is the caller's `out`)

    // Sum of every level's device allocation, for VkWorld::deviceBytesAllocated()'s
    // transitive accounting (point 3): this scratch grows lazily on first use and is
    // otherwise invisible to anything outside vk_primitives.
    uint64_t bytesAllocated() const
    {
        uint64_t total = 0;
        for (const auto &b : levelB)
            if (b)
                total += (uint64_t)b->size();
        for (const auto &r : levelR)
            if (r)
                total += (uint64_t)r->size();
        return total;
    }
};

// Grows every level a scan of up to n elements uses, so recording one never allocates.
void reserveScan(ScanScratch &scratch, uint32_t n);

// Informational estimate of scratch bytes scanExclusive will grow ScanScratch to hold
// for an array of `n` uint32s -- mirrors CUB's tempStorageBytes query in shape, though
// (unlike CUB) the real allocation lives inside ScanScratch and grows lazily on use.
size_t scanTempBytes(uint32_t n);

// Exclusive scan of `in[0..n)` into `out[0..n)` (may alias: `in` and `out` may be the
// same buffer, since each block's input is consumed into shared memory before any
// output write). Records into `cmd` (already begin()'d); does not submit or wait.
void scanExclusive(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                    ScanScratch &scratch, const Buffer &in, Buffer &out, uint32_t n);

// ---------------------------------------------------------------------------
// Stream compaction of items selected by a uint8 flag array.
// ---------------------------------------------------------------------------
struct SelectScratch
{
    void init(Device &dev)
    {
        m_dev = &dev;
        scan.init(dev);
        flagsU32.init(dev);
    }
    void destroy();

    Device *m_dev = nullptr;
    ScanScratch scan;
    Buffer flagsU32;
};

size_t selectFlaggedTempBytes(uint32_t n);

// out[] receives the `items[i]` for which `flags[i] != 0`, in original order;
// `numSelected` (a 1-int Buffer) receives the count. `out` must have room for `n`
// elements (the worst case, all flagged).
void selectFlagged(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                    SelectScratch &scratch, const Buffer &items, const Buffer &flags,
                    Buffer &out, Buffer &numSelected, uint32_t n);

// ---------------------------------------------------------------------------
// Radix sort: stable LSD, 8-bit digits, TILE_SIZE (2048) elements/block -- see
// shaders/radix.slang for the tile/wave-rank design and the profiling notes that
// motivated it. One scratch struct serves all three key/value shapes below (fields are
// resized in bytes as needed per call).
// ---------------------------------------------------------------------------
struct RadixScratch
{
    void init(Device &dev)
    {
        m_dev = &dev;
        scan.init(dev);
        digitCount.init(dev);
        digitCountScanned.init(dev);
        keysPing.init(dev);
        keysPong.init(dev);
        valsPing.init(dev);
        valsPong.init(dev);
        reduceOr.init(dev);
        reduceAnd.init(dev);
        skipMask.init(dev);
    }
    void destroy();

    Device *m_dev = nullptr;
    ScanScratch scan;
    Buffer digitCount, digitCountScanned; // RADIX_BUCKETS(256) * numBlocks entries
    Buffer keysPing, keysPong, valsPing, valsPong;
    Buffer reduceOr, reduceAnd; // 1-element OR/AND-of-all-keys, used to skip constant-digit passes
    Buffer skipMask; // numPasses uint32 (0/1), computed entirely on device -- see radix.slang
                      // csFinalizeSkipMaskU32/U64 and vk_primitives.cpp recordSkipMask(). Every
                      // hist/scatter kernel reads this on-device and no-ops/identity-copies for a
                      // skipped pass; nothing here ever does a host readback or its own submit/wait,
                      // so the sort call stays record-only and safe to chain after an unsubmitted
                      // producer dispatch in the same command buffer.

    // Transitive byte total (point 3): every fixed buffer here plus ScanScratch's own
    // lazily-grown levels. VkWorld owns one RadixScratch shared by broadphase and
    // constraints builds, so this is the whole of what that sharing hides from a naive
    // per-VkWorld-member buffer list.
    uint64_t bytesAllocated() const
    {
        const Buffer *fixed[] = {&digitCount, &digitCountScanned, &keysPing, &keysPong,
                                  &valsPing, &valsPong, &reduceOr, &reduceAnd, &skipMask};
        uint64_t total = scan.bytesAllocated();
        for (const Buffer *b : fixed)
            total += (uint64_t)b->size();
        return total;
    }
};

size_t radixSortTempBytes(uint32_t n);

// Grows every scratch buffer a sort of up to n elements (keyBytes/valBytes wide; valBytes
// 0 for a keys-only sort) touches. The sorts grow scratch lazily, and a resize frees the
// old buffer -- so when several sorts share one scratch inside ONE command buffer, a later
// sort's growth would free memory an earlier recorded dispatch still addresses (device
// lost). Callers that record more than one sort per submission reserve first; after that
// the sorts' own growth checks are no-ops.
void reserveRadix(RadixScratch &scratch, uint32_t n, uint32_t keyBytes, uint32_t valBytes);

// Device-count sorts (M4): with countAddr != 0, n is a capacity -- the grid, the scratch and
// the pass schedule are sized by it -- and the live count is read on device as
// min(*countAddr, n). Recording no longer needs the count on the host; the result is
// identical to a sort sized exactly. countAddr == 0 sorts exactly n elements.

// uint32 keys + int32 values --
// the broadphase's body-bucket sort.
void radixSortPairsU32(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                        RadixScratch &scratch, const Buffer &keysIn, Buffer &keysOut,
                        const Buffer &valsIn, Buffer &valsOut, uint32_t n,
                        VkDeviceAddress countAddr = 0);

// uint64 keys + uint32 values -- the constraint builder's CSR entry sort.
//
// Profiling hook (diagnostic only, not used by the solver): when profileBaseIndex is not
// UINT32_MAX, a timestamp is written at profileBaseIndex + pass*4 + {0=before hist,
// 1=after hist/before scan, 2=after scan/before scatter, 3=after scatter} for every
// pass, so the caller can recover a per-pass hist/scan/scatter breakdown with
// cmd.timestampDeltaNs() once the submission has completed. The CommandList must have
// been init()'d with enough timestamp slots (profileBaseIndex + numPasses*4).
void radixSortPairsU64(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                        RadixScratch &scratch, const Buffer &keysIn, Buffer &keysOut,
                        const Buffer &valsIn, Buffer &valsOut, uint32_t n,
                        uint32_t profileBaseIndex = 0xFFFFFFFFu, VkDeviceAddress countAddr = 0);

// uint64 keys only --
// the broadphase's pair-key canonicalization.
void radixSortKeysU64(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                       RadixScratch &scratch, const Buffer &keysIn, Buffer &keysOut, uint32_t n,
                       VkDeviceAddress countAddr = 0);

} // namespace avbdvk
