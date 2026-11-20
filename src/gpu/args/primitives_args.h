#pragma once

// Scan / compact / radix primitive argument structs. Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"

#include <cstdint>

namespace avbdvk
{

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

// Small-world one-workgroup sorts (radix.slang section (e)).
struct SortSmallArgsU32Pairs
{
    VkDeviceAddress keysIn, valsIn, keysOut, valsOut;
    uint32_t n, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(SortSmallArgsU32Pairs) % 8 == 0, "layout");
using SortSmallArgsU64Pairs = SortSmallArgsU32Pairs; // identical field layout

struct SortSmallArgsU64Keys
{
    VkDeviceAddress keysIn, keysOut;
    uint32_t n, _pad0;
    VkDeviceAddress count = 0;
};
static_assert(sizeof(SortSmallArgsU64Keys) % 8 == 0, "layout");

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

} // namespace avbdvk
