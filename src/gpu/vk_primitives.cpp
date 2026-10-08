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

#include "vk_primitives.h"
#include "vk_args.h"

#include "scan_spirv.h"
#include "compact_spirv.h"
#include "radix_spirv.h"

#include <algorithm>
#include <cstring>

namespace avbdvk
{

// Args structs -- byte-for-byte mirrors of the Slang structs in shaders/
// {scan,compact,radix}.slang under scalar layout -- live in vk_args.h so each struct is
// defined exactly once.

static const uint32_t kScanBlock = 256;
static const uint32_t kRadixBuckets = 256;
static const uint32_t kTileThreads = 256;
static const uint32_t kKeysPerThread = 8;
static const uint32_t kTileSize = kTileThreads * kKeysPerThread; // 2048, must match radix.slang

static uint32_t divUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// PrimitivesPipelines
// ---------------------------------------------------------------------------

void PrimitivesPipelines::init(Device &dev, PipelineLayout &layout)
{
    scanBlock.init(dev, layout, scan_spirv, scan_spirv_words, "csScanBlock", "csScanBlock");
    addBlockSums.init(dev, layout, scan_spirv, scan_spirv_words, "csAddBlockSums", "csAddBlockSums");

    flagsToU32.init(dev, layout, compact_spirv, compact_spirv_words, "csFlagsToU32", "csFlagsToU32");
    scatterCompact.init(dev, layout, compact_spirv, compact_spirv_words, "csScatterCompact", "csScatterCompact");
    writeNumSelected.init(dev, layout, compact_spirv, compact_spirv_words, "csWriteNumSelected", "csWriteNumSelected");

    histU32Pairs.init(dev, layout, radix_spirv, radix_spirv_words, "csHistU32Pairs", "csHistU32Pairs");
    scatterU32Pairs.init(dev, layout, radix_spirv, radix_spirv_words, "csScatterU32Pairs", "csScatterU32Pairs");
    histU64Pairs.init(dev, layout, radix_spirv, radix_spirv_words, "csHistU64Pairs", "csHistU64Pairs");
    scatterU64Pairs.init(dev, layout, radix_spirv, radix_spirv_words, "csScatterU64Pairs", "csScatterU64Pairs");
    histU64Keys.init(dev, layout, radix_spirv, radix_spirv_words, "csHistU64Keys", "csHistU64Keys");
    scatterU64Keys.init(dev, layout, radix_spirv, radix_spirv_words, "csScatterU64Keys", "csScatterU64Keys");
    reduceOrAndU32.init(dev, layout, radix_spirv, radix_spirv_words, "csReduceOrAndU32", "csReduceOrAndU32");
    reduceOrAndU64.init(dev, layout, radix_spirv, radix_spirv_words, "csReduceOrAndU64", "csReduceOrAndU64");
    finalizeSkipMaskU32.init(dev, layout, radix_spirv, radix_spirv_words, "csFinalizeSkipMaskU32", "csFinalizeSkipMaskU32");
    finalizeSkipMaskU64.init(dev, layout, radix_spirv, radix_spirv_words, "csFinalizeSkipMaskU64", "csFinalizeSkipMaskU64");
}

void PrimitivesPipelines::destroy()
{
    scanBlock.destroy();
    addBlockSums.destroy();
    flagsToU32.destroy();
    scatterCompact.destroy();
    writeNumSelected.destroy();
    histU32Pairs.destroy();
    scatterU32Pairs.destroy();
    histU64Pairs.destroy();
    scatterU64Pairs.destroy();
    histU64Keys.destroy();
    scatterU64Keys.destroy();
    reduceOrAndU32.destroy();
    reduceOrAndU64.destroy();
    finalizeSkipMaskU32.destroy();
    finalizeSkipMaskU64.destroy();
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

void ScanScratch::destroy()
{
    for (auto &b : levelB)
        if (b)
            b->destroy();
    for (auto &b : levelR)
        if (b)
            b->destroy();
    levelB.clear();
    levelR.clear();
}

static Buffer &ensureLevel(std::vector<std::unique_ptr<Buffer>> &levels, size_t idx, Device &dev, VkDeviceSize bytes)
{
    if (levels.size() <= idx)
        levels.resize(idx + 1);
    if (!levels[idx])
    {
        levels[idx] = std::make_unique<Buffer>();
        levels[idx]->init(dev);
    }
    if (levels[idx]->size() < bytes)
        levels[idx]->resize(bytes);
    return *levels[idx];
}

void reserveScan(ScanScratch &scratch, uint32_t n)
{
    // Walks the same level sizes recordScanLevel() asks for, so a scan of up to n
    // elements recorded afterwards never resizes a level.
    uint32_t cur = n;
    for (size_t level = 0; cur > 1; level++)
    {
        uint32_t numBlocks = divUp(cur, kScanBlock);
        ensureLevel(scratch.levelB, level, *scratch.m_dev, (VkDeviceSize)numBlocks * sizeof(uint32_t));
        if (numBlocks > 1)
            ensureLevel(scratch.levelR, level, *scratch.m_dev, (VkDeviceSize)numBlocks * sizeof(uint32_t));
        cur = numBlocks;
    }
}

size_t scanTempBytes(uint32_t n)
{
    size_t total = 0;
    uint32_t cur = n;
    while (cur > 1)
    {
        uint32_t numBlocks = divUp(cur, kScanBlock);
        total += (size_t)numBlocks * sizeof(uint32_t) * 2; // blockSums + scanned-blockSums
        cur = numBlocks;
    }
    return total;
}

static void recordScanLevel(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                             ScanScratch &scratch, size_t level, const Buffer &in, Buffer &out, uint32_t n)
{
    if (n == 0)
        return;
    uint32_t numBlocks = divUp(n, kScanBlock);
    Buffer &blockSums = ensureLevel(scratch.levelB, level, *scratch.m_dev, (VkDeviceSize)numBlocks * sizeof(uint32_t));

    ScanBlockArgs a{in.address(), out.address(), blockSums.address(), n, 0};
    VkDeviceAddress addr = ring.write(&a, sizeof(a));
    cmd.dispatch(pl.scanBlock, numBlocks, addr);

    if (numBlocks > 1)
    {
        Buffer &scannedSums = ensureLevel(scratch.levelR, level, *scratch.m_dev, (VkDeviceSize)numBlocks * sizeof(uint32_t));
        cmd.barrier();
        recordScanLevel(cmd, ring, pl, scratch, level + 1, blockSums, scannedSums, numBlocks);
        cmd.barrier();

        AddBlockSumsArgs b{out.address(), scannedSums.address(), n, 0};
        VkDeviceAddress addr2 = ring.write(&b, sizeof(b));
        cmd.dispatch(pl.addBlockSums, numBlocks, addr2);
    }
}

void scanExclusive(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                    ScanScratch &scratch, const Buffer &in, Buffer &out, uint32_t n)
{
    if (n == 0)
        return;
    recordScanLevel(cmd, ring, pl, scratch, 0, in, out, n);
}

// ---------------------------------------------------------------------------
// Select flagged
// ---------------------------------------------------------------------------

void SelectScratch::destroy()
{
    scan.destroy();
    flagsU32.destroy();
}

size_t selectFlaggedTempBytes(uint32_t n)
{
    return (size_t)n * sizeof(uint32_t) + scanTempBytes(n);
}

void selectFlagged(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                    SelectScratch &scratch, const Buffer &items, const Buffer &flags,
                    Buffer &out, Buffer &numSelected, uint32_t n)
{
    if (n == 0)
    {
        NumSelectedArgs na{0, 0, numSelected.address(), 0, 0};
        VkDeviceAddress addr = ring.write(&na, sizeof(na));
        cmd.dispatch(pl.writeNumSelected, 1, addr);
        return;
    }

    if (scratch.flagsU32.size() < (VkDeviceSize)n * sizeof(uint32_t))
        scratch.flagsU32.resize((VkDeviceSize)n * sizeof(uint32_t));

    uint32_t blocks = divUp(n, kScanBlock);

    FlagsToU32Args fa{flags.address(), scratch.flagsU32.address(), n, 0};
    VkDeviceAddress addr = ring.write(&fa, sizeof(fa));
    cmd.dispatch(pl.flagsToU32, blocks, addr);
    cmd.barrier();

    scanExclusive(cmd, ring, pl, scratch.scan, scratch.flagsU32, scratch.flagsU32, n);
    cmd.barrier();

    ScatterCompactArgs sa{items.address(), flags.address(), scratch.flagsU32.address(), out.address(), n, 0};
    VkDeviceAddress addr2 = ring.write(&sa, sizeof(sa));
    cmd.dispatch(pl.scatterCompact, blocks, addr2);

    NumSelectedArgs na{scratch.flagsU32.address(), flags.address(), numSelected.address(), n, 0};
    VkDeviceAddress addr3 = ring.write(&na, sizeof(na));
    cmd.dispatch(pl.writeNumSelected, 1, addr3);
}

// ---------------------------------------------------------------------------
// Radix sort
// ---------------------------------------------------------------------------

void RadixScratch::destroy()
{
    scan.destroy();
    digitCount.destroy();
    digitCountScanned.destroy();
    keysPing.destroy();
    keysPong.destroy();
    valsPing.destroy();
    valsPong.destroy();
    reduceOr.destroy();
    reduceAnd.destroy();
    skipMask.destroy();
}

size_t radixSortTempBytes(uint32_t n)
{
    uint32_t numBlocks = divUp(std::max(n, 1u), kTileSize);
    size_t bytes = 0;
    bytes += (size_t)kRadixBuckets * numBlocks * sizeof(uint32_t) * 2; // digitCount + scanned
    bytes += (size_t)n * 8 * 4;                                       // ping/pong keys+vals (worst case 8B keys)
    bytes += scanTempBytes(kRadixBuckets * numBlocks);
    return bytes;
}

static void ensureBytes(Buffer &b, Device &dev, VkDeviceSize bytes)
{
    if (b.size() < bytes)
        b.resize(bytes);
    (void)dev;
}

void reserveRadix(RadixScratch &scratch, uint32_t n, uint32_t keyBytes, uint32_t valBytes)
{
    if (n == 0)
        return;
    Device &dev = *scratch.m_dev;
    uint32_t numBlocks = divUp(n, kTileSize);
    ensureBytes(scratch.digitCount, dev, (VkDeviceSize)kRadixBuckets * numBlocks * sizeof(uint32_t));
    ensureBytes(scratch.digitCountScanned, dev, (VkDeviceSize)kRadixBuckets * numBlocks * sizeof(uint32_t));
    ensureBytes(scratch.keysPing, dev, (VkDeviceSize)n * keyBytes);
    ensureBytes(scratch.keysPong, dev, (VkDeviceSize)n * keyBytes);
    if (valBytes > 0)
    {
        ensureBytes(scratch.valsPing, dev, (VkDeviceSize)n * valBytes);
        ensureBytes(scratch.valsPong, dev, (VkDeviceSize)n * valBytes);
    }
    ensureBytes(scratch.reduceOr, dev, 8);
    ensureBytes(scratch.reduceAnd, dev, 8);
    ensureBytes(scratch.skipMask, dev, 8 * sizeof(uint32_t));
    reserveScan(scratch.scan, kRadixBuckets * numBlocks);
}

// ---------------------------------------------------------------------------
// Digit-pass skipping: entirely GPU-resident, recorded into the caller's `cmd`. A global
// OR/AND reduction over the keys (csReduceOrAndU32/U64) followed by a one-thread finalize
// kernel (csFinalizeSkipMaskU32/U64) turns it into a skipMask[pass] buffer; every
// hist/scatter dispatch below reads that flag on device (hist no-ops, scatter performs a
// straight identity copy for a constant digit). No host readback and no separate
// submit/wait -- this must be safe to record immediately after an as-yet-unsubmitted
// producer dispatch writes keysIn in the same command buffer, so nothing here may touch
// the queue. Host-side pass count and ping-pong buffer assignment stay fixed regardless
// of which passes turn out to be skippable; skipping only removes work *inside* a pass,
// never a pass from the schedule.
// ---------------------------------------------------------------------------

template <typename KeyT>
static void recordSkipMask(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                            RadixScratch &scratch, const Buffer &keysIn, uint32_t n, int numPasses,
                            Pipeline &reducePipeline, Pipeline &finalizePipeline, bool isU64,
                            VkDeviceAddress countAddr)
{
    Device &dev = *scratch.m_dev;
    VkDeviceSize keyWordBytes = isU64 ? 8 : 4;
    ensureBytes(scratch.reduceOr, dev, keyWordBytes);
    ensureBytes(scratch.reduceAnd, dev, keyWordBytes);
    ensureBytes(scratch.skipMask, dev, (VkDeviceSize)numPasses * sizeof(uint32_t));

    cmd.fill(scratch.reduceOr, 0u, keyWordBytes);
    cmd.fill(scratch.reduceAnd, 0xFFFFFFFFu, keyWordBytes);

    if (n > 0)
    {
        uint32_t blocks = divUp(n, kTileThreads);
        if (isU64)
        {
            ReduceOrAndArgsU64 a{keysIn.address(), scratch.reduceOr.address(), scratch.reduceAnd.address(), n, 0, countAddr};
            VkDeviceAddress addr = ring.write(&a, sizeof(a));
            cmd.dispatch(reducePipeline, blocks, addr);
        }
        else
        {
            ReduceOrAndArgsU32 a{keysIn.address(), scratch.reduceOr.address(), scratch.reduceAnd.address(), n, 0, countAddr};
            VkDeviceAddress addr = ring.write(&a, sizeof(a));
            cmd.dispatch(reducePipeline, blocks, addr);
        }
    }
    cmd.barrier();

    if (isU64)
    {
        FinalizeSkipMaskArgsU64 fa{scratch.reduceOr.address(), scratch.reduceAnd.address(),
                                    scratch.skipMask.address(), (uint32_t)numPasses, 0};
        VkDeviceAddress addr = ring.write(&fa, sizeof(fa));
        cmd.dispatch(finalizePipeline, 1, addr);
    }
    else
    {
        FinalizeSkipMaskArgsU32 fa{scratch.reduceOr.address(), scratch.reduceAnd.address(),
                                    scratch.skipMask.address(), (uint32_t)numPasses, 0};
        VkDeviceAddress addr = ring.write(&fa, sizeof(fa));
        cmd.dispatch(finalizePipeline, 1, addr);
    }
    cmd.barrier();
}

void radixSortPairsU32(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                        RadixScratch &scratch, const Buffer &keysIn, Buffer &keysOut,
                        const Buffer &valsIn, Buffer &valsOut, uint32_t n, VkDeviceAddress countAddr)
{
    if (n == 0)
        return;
    Device &dev = *scratch.m_dev;
    uint32_t numBlocks = divUp(n, kTileSize);

    ensureBytes(scratch.digitCount, dev, (VkDeviceSize)kRadixBuckets * numBlocks * sizeof(uint32_t));
    ensureBytes(scratch.digitCountScanned, dev, (VkDeviceSize)kRadixBuckets * numBlocks * sizeof(uint32_t));
    ensureBytes(scratch.keysPing, dev, (VkDeviceSize)n * sizeof(uint32_t));
    ensureBytes(scratch.keysPong, dev, (VkDeviceSize)n * sizeof(uint32_t));
    ensureBytes(scratch.valsPing, dev, (VkDeviceSize)n * sizeof(int32_t));
    ensureBytes(scratch.valsPong, dev, (VkDeviceSize)n * sizeof(int32_t));

    const int numPasses = 4;
    recordSkipMask<uint32_t>(cmd, ring, pl, scratch, keysIn, n, numPasses, pl.reduceOrAndU32,
                              pl.finalizeSkipMaskU32, false, countAddr);

    const Buffer *curKeys = &keysIn;
    const Buffer *curVals = &valsIn;
    Buffer *pingPongKeys[2] = {&scratch.keysPing, &scratch.keysPong};
    Buffer *pingPongVals[2] = {&scratch.valsPing, &scratch.valsPong};

    for (int pass = 0; pass < numPasses; pass++)
    {
        bool isLast = (pass == numPasses - 1);
        Buffer &outK = isLast ? keysOut : *pingPongKeys[pass % 2];
        Buffer &outV = isLast ? valsOut : *pingPongVals[pass % 2];

        HistArgsU32Pairs ha{curKeys->address(), scratch.digitCount.address(), scratch.skipMask.address(),
                             n, numBlocks, (uint32_t)pass, 0, countAddr};
        VkDeviceAddress addr = ring.write(&ha, sizeof(ha));
        cmd.dispatch(pl.histU32Pairs, numBlocks, addr);
        cmd.barrier();

        scanExclusive(cmd, ring, pl, scratch.scan, scratch.digitCount, scratch.digitCountScanned,
                       kRadixBuckets * numBlocks);
        cmd.barrier();

        ScatterArgsU32Pairs sa{curKeys->address(), curVals->address(), scratch.digitCountScanned.address(),
                                scratch.skipMask.address(), outK.address(), outV.address(), n, numBlocks,
                                (uint32_t)pass, 0, countAddr};
        VkDeviceAddress addr2 = ring.write(&sa, sizeof(sa));
        cmd.dispatch(pl.scatterU32Pairs, numBlocks, addr2);
        cmd.barrier();

        curKeys = &outK;
        curVals = &outV;
    }
}

void radixSortPairsU64(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                        RadixScratch &scratch, const Buffer &keysIn, Buffer &keysOut,
                        const Buffer &valsIn, Buffer &valsOut, uint32_t n,
                        uint32_t profileBaseIndex, VkDeviceAddress countAddr)
{
    if (n == 0)
        return;
    Device &dev = *scratch.m_dev;
    uint32_t numBlocks = divUp(n, kTileSize);

    ensureBytes(scratch.digitCount, dev, (VkDeviceSize)kRadixBuckets * numBlocks * sizeof(uint32_t));
    ensureBytes(scratch.digitCountScanned, dev, (VkDeviceSize)kRadixBuckets * numBlocks * sizeof(uint32_t));
    ensureBytes(scratch.keysPing, dev, (VkDeviceSize)n * sizeof(uint64_t));
    ensureBytes(scratch.keysPong, dev, (VkDeviceSize)n * sizeof(uint64_t));
    ensureBytes(scratch.valsPing, dev, (VkDeviceSize)n * sizeof(uint32_t));
    ensureBytes(scratch.valsPong, dev, (VkDeviceSize)n * sizeof(uint32_t));

    const int numPasses = 8;
    recordSkipMask<uint64_t>(cmd, ring, pl, scratch, keysIn, n, numPasses, pl.reduceOrAndU64,
                              pl.finalizeSkipMaskU64, true, countAddr);

    const Buffer *curKeys = &keysIn;
    const Buffer *curVals = &valsIn;
    Buffer *pingPongKeys[2] = {&scratch.keysPing, &scratch.keysPong};
    Buffer *pingPongVals[2] = {&scratch.valsPing, &scratch.valsPong};

    bool prof = profileBaseIndex != 0xFFFFFFFFu;

    for (int pass = 0; pass < numPasses; pass++)
    {
        bool isLast = (pass == numPasses - 1);
        Buffer &outK = isLast ? keysOut : *pingPongKeys[pass % 2];
        Buffer &outV = isLast ? valsOut : *pingPongVals[pass % 2];

        if (prof)
            cmd.writeTimestamp(profileBaseIndex + (uint32_t)pass * 4 + 0);

        HistArgsU64Pairs ha{curKeys->address(), scratch.digitCount.address(), scratch.skipMask.address(),
                             n, numBlocks, (uint32_t)pass, 0, countAddr};
        VkDeviceAddress addr = ring.write(&ha, sizeof(ha));
        cmd.dispatch(pl.histU64Pairs, numBlocks, addr);
        cmd.barrier();

        if (prof)
            cmd.writeTimestamp(profileBaseIndex + (uint32_t)pass * 4 + 1);

        scanExclusive(cmd, ring, pl, scratch.scan, scratch.digitCount, scratch.digitCountScanned,
                       kRadixBuckets * numBlocks);
        cmd.barrier();

        if (prof)
            cmd.writeTimestamp(profileBaseIndex + (uint32_t)pass * 4 + 2);

        ScatterArgsU64Pairs sa{curKeys->address(), curVals->address(), scratch.digitCountScanned.address(),
                                scratch.skipMask.address(), outK.address(), outV.address(), n, numBlocks,
                                (uint32_t)pass, 0, countAddr};
        VkDeviceAddress addr2 = ring.write(&sa, sizeof(sa));
        cmd.dispatch(pl.scatterU64Pairs, numBlocks, addr2);
        cmd.barrier();

        if (prof)
            cmd.writeTimestamp(profileBaseIndex + (uint32_t)pass * 4 + 3);

        curKeys = &outK;
        curVals = &outV;
    }
}

void radixSortKeysU64(CommandList &cmd, HostRingBuffer &ring, PrimitivesPipelines &pl,
                       RadixScratch &scratch, const Buffer &keysIn, Buffer &keysOut, uint32_t n,
                       VkDeviceAddress countAddr)
{
    if (n == 0)
        return;
    Device &dev = *scratch.m_dev;
    uint32_t numBlocks = divUp(n, kTileSize);

    ensureBytes(scratch.digitCount, dev, (VkDeviceSize)kRadixBuckets * numBlocks * sizeof(uint32_t));
    ensureBytes(scratch.digitCountScanned, dev, (VkDeviceSize)kRadixBuckets * numBlocks * sizeof(uint32_t));
    ensureBytes(scratch.keysPing, dev, (VkDeviceSize)n * sizeof(uint64_t));
    ensureBytes(scratch.keysPong, dev, (VkDeviceSize)n * sizeof(uint64_t));

    const int numPasses = 8;
    recordSkipMask<uint64_t>(cmd, ring, pl, scratch, keysIn, n, numPasses, pl.reduceOrAndU64,
                              pl.finalizeSkipMaskU64, true, countAddr);

    const Buffer *curKeys = &keysIn;
    Buffer *pingPongKeys[2] = {&scratch.keysPing, &scratch.keysPong};

    for (int pass = 0; pass < numPasses; pass++)
    {
        bool isLast = (pass == numPasses - 1);
        Buffer &outK = isLast ? keysOut : *pingPongKeys[pass % 2];

        HistArgsU64Keys ha{curKeys->address(), scratch.digitCount.address(), scratch.skipMask.address(),
                            n, numBlocks, (uint32_t)pass, 0, countAddr};
        VkDeviceAddress addr = ring.write(&ha, sizeof(ha));
        cmd.dispatch(pl.histU64Keys, numBlocks, addr);
        cmd.barrier();

        scanExclusive(cmd, ring, pl, scratch.scan, scratch.digitCount, scratch.digitCountScanned,
                       kRadixBuckets * numBlocks);
        cmd.barrier();

        ScatterArgsU64Keys sa{curKeys->address(), scratch.digitCountScanned.address(),
                               scratch.skipMask.address(), outK.address(), n, numBlocks, (uint32_t)pass, 0, countAddr};
        VkDeviceAddress addr2 = ring.write(&sa, sizeof(sa));
        cmd.dispatch(pl.scatterU64Keys, numBlocks, addr2);
        cmd.barrier();

        curKeys = &outK;
    }
}

} // namespace avbdvk
