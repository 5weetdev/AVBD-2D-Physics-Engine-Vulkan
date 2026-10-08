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

// Host mirrors of the argument structs the shared kernels read (scan, compact, radix,
// constraint graph, dispatch gate). Field order is a byte-for-byte transcription of the
// Slang counterpart under scalar layout, with VkDeviceAddress standing in for every
// pointer and explicit _padN fields for the padding Slang inserts. Change both together.

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
    VkDeviceAddress dynCountPtr = 0; // M4 live count (0: dynCount is exact)
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
    VkDeviceAddress count = 0; // M4 live count; `n` is then the capacity
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
    VkDeviceAddress dynCountPtr = 0; // M4 live count (0: dynCount is exact)
};
static_assert(sizeof(CsJpRoundArgsGpu) % 8 == 0, "layout");

struct CsColorKeysArgsGpu
{
    VkDeviceAddress color = 0, dyn = 0;
    uint32_t dynCount = 0, _pad0 = 0;
    VkDeviceAddress keys = 0, vals = 0;
    VkDeviceAddress dynCountPtr = 0; // M4 live count (0: dynCount is exact)
    VkDeviceAddress midTag = 0;      // distance LOD, may be 0
};
static_assert(sizeof(CsColorKeysArgsGpu) % 8 == 0, "layout");

struct CsColorRowsArgsGpu
{
    VkDeviceAddress keys = 0;
    uint32_t n = 0, _pad0 = 0;
    VkDeviceAddress start = 0, end = 0;
    VkDeviceAddress dynCountPtr = 0; // M4 live count (0: n is exact)
    VkDeviceAddress nearEnd = 0;     // distance LOD, may be 0
};
static_assert(sizeof(CsColorRowsArgsGpu) % 8 == 0, "layout");

// ===========================================================================
// scan.slang / compact.slang / radix.slang (Phase-2 primitives)
// ===========================================================================

struct ScanBlockArgs
{
    VkDeviceAddress in, out, blockSums;
    uint32_t n, _pad0;
};
static_assert(sizeof(ScanBlockArgs) == 32 && sizeof(ScanBlockArgs) % 8 == 0, "layout");

struct AddBlockSumsArgs
{
    VkDeviceAddress out, blockOffsets;
    uint32_t n, _pad0;
};
static_assert(sizeof(AddBlockSumsArgs) % 8 == 0, "layout");

struct FlagsToU32Args
{
    VkDeviceAddress flags, out;
    uint32_t n, _pad0;
};
static_assert(sizeof(FlagsToU32Args) % 8 == 0, "layout");

struct ScatterCompactArgs
{
    VkDeviceAddress items, flags, scanned, out;
    uint32_t n, _pad0;
};
static_assert(sizeof(ScatterCompactArgs) % 8 == 0, "layout");

struct NumSelectedArgs
{
    VkDeviceAddress scanned, flags, numSelected;
    uint32_t n, _pad0;
};
static_assert(sizeof(NumSelectedArgs) % 8 == 0, "layout");

struct HistArgsU32Pairs
{
    VkDeviceAddress keysIn, digitCount, skipMask;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(HistArgsU32Pairs) % 8 == 0, "layout");

struct ScatterArgsU32Pairs
{
    VkDeviceAddress keysIn, valsIn, digitCountScanned, skipMask, keysOut, valsOut;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ScatterArgsU32Pairs) % 8 == 0, "layout");

struct HistArgsU64Pairs
{
    VkDeviceAddress keysIn, digitCount, skipMask;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(HistArgsU64Pairs) % 8 == 0, "layout");

struct ScatterArgsU64Pairs
{
    VkDeviceAddress keysIn, valsIn, digitCountScanned, skipMask, keysOut, valsOut;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ScatterArgsU64Pairs) % 8 == 0, "layout");

struct HistArgsU64Keys
{
    VkDeviceAddress keysIn, digitCount, skipMask;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(HistArgsU64Keys) % 8 == 0, "layout");

struct ScatterArgsU64Keys
{
    VkDeviceAddress keysIn, digitCountScanned, skipMask, keysOut;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ScatterArgsU64Keys) % 8 == 0, "layout");

struct ReduceOrAndArgsU32
{
    VkDeviceAddress keysIn, outOr, outAnd;
    uint32_t n, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ReduceOrAndArgsU32) % 8 == 0, "layout");

struct ReduceOrAndArgsU64
{
    VkDeviceAddress keysIn, outOr, outAnd;
    uint32_t n, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ReduceOrAndArgsU64) % 8 == 0, "layout");

struct FinalizeSkipMaskArgsU32
{
    VkDeviceAddress orAddr, andAddr, skipMask;
    uint32_t numPasses, _pad0;
};
static_assert(sizeof(FinalizeSkipMaskArgsU32) % 8 == 0, "layout");

struct FinalizeSkipMaskArgsU64
{
    VkDeviceAddress orAddr, andAddr, skipMask;
    uint32_t numPasses, _pad0;
};
static_assert(sizeof(FinalizeSkipMaskArgsU64) % 8 == 0, "layout");

struct GateDispatchArgsGpu
{
    VkDeviceAddress src = 0, dst = 0, flag = 0;
    int32_t count = 0, _pad0 = 0;
};
static_assert(sizeof(GateDispatchArgsGpu) % 8 == 0, "layout");

} // namespace avbdvk
