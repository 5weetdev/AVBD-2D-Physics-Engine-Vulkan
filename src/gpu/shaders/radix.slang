// Stable LSD radix sort (see the call sites in the broadphase -- uint32 key/int32
// value -- and the constraint builder -- uint64 key/uint32 value, and uint64
// key only, used for graph coloring).
//
// Structural design (rewritten after profiling showed the first cut was superlinear in
// n -- see the profiling notes in vk_primitives.h/.cpp):
//
//  - Each block ("tile") covers TILE_SIZE = TILE_THREADS * KEYS_PER_THREAD elements,
//    processed as KEYS_PER_THREAD sequential 256-wide "chunks" (chunk k covers the
//    contiguous run [tileBase + k*256, tileBase + (k+1)*256)). This is what keeps
//    elements in original order across chunks (thread t handles chunk-relative index t
//    in every chunk, chunks visited k=0..K-1 in increasing order), and it shrinks
//    numBlocks (and therefore digitCount, the array whose stride/size drove the
//    superlinear blowup) by a factor of KEYS_PER_THREAD versus one element per thread.
//
//  - Per-chunk stable rank no longer uses an 8-round shared-memory bit-split network
//    (128 GroupMemoryBarrierWithGroupSync() calls per 256 elements). Instead each
//    subgroup ("wave") computes, via 8 WaveActiveBallot calls (one per digit bit, no
//    barrier -- wave-synchronous), the mask of lanes sharing its exact 8-bit digit, then
//    a stable intra-wave rank via popcount. Wave totals are combined across the block's
//    few waves (4-8, not 256) through a small groupshared table. A per-tile
//    "sCarry[digit]" running total, updated once per chunk, extends that rank across
//    all KEYS_PER_THREAD chunks of the tile. Net: ~4 barriers per chunk instead of 128
//    per 256 elements, and the shared table read pattern is O(waves) not O(1) walked
//    serially by one thread with a stride-numBlocks memory access (the second bug the
//    profiling found).
//
//  - Wave-generic: subgroup size enters only through WaveGetLaneCount()/
//    WaveGetLaneIndex(); MAX_WAVES=8 covers both 32-wide (Nvidia: 256/32=8 waves) and
//    64-wide (AMD: 256/64=4 waves) subgroups, per the migration plan's constraint.
//
//  - Digit-pass skipping (skip passes whose digit is constant across all keys, e.g. the
//    high bytes of a pair key that never leaves a small range) is entirely GPU-resident:
//    a global OR/AND reduction (csReduceOrAndU32/U64 below) is recorded into the same
//    command buffer as the sort, followed by a one-thread csFinalizeSkipMaskU32/U64 that
//    turns it into a per-pass skipMask[pass] flag buffer. Every hist/scatter kernel below
//    takes that buffer and branches on skipMask[pass] itself (uniformly across the whole
//    dispatch, since every invocation reads the same flag for the same pass): hist no-ops,
//    scatter performs a straight per-element identity copy instead of computing ranks.
//    Host-side pass count and buffer ping-pong stay fixed regardless of which passes are
//    skipped -- there is no host readback, so this is safe to record before the dispatch
//    that produced keysIn has even been submitted.
//
// Stability is required and preserved: within a chunk, ties break by lane index (the
// original relative order of that chunk); across chunks, sCarry only ever adds the count
// of *earlier* chunks, so a chunk's ranks never overlap an earlier chunk's.

#define TILE_THREADS 256
#define KEYS_PER_THREAD 8
#define TILE_SIZE (TILE_THREADS * KEYS_PER_THREAD)
#define RADIX_BUCKETS 256
#define MAX_WAVES 8 // 256 / 32 (Nvidia); AMD's 64-wide subgroups only use the first 4 rows

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

uint digitU32(uint key, uint pass) { return (key >> (pass * 8u)) & 0xFFu; }
uint digitU64(uint64_t key, uint pass) { return uint((key >> (uint64_t)(pass * 8u)) & (uint64_t)0xFF); }

// ===========================================================================
// Shared helpers (no groupshared state -- safe to call from any entry point).
// ===========================================================================

// Low `n` bits set (n in [0,64]), split across a uint2 the way WaveActiveBallot's
// returned uint4's .xy holds up to 64 lanes -- z/w are never touched since this port
// only has to support 32- and 64-wide subgroups.
// Device-count sorts (M4): the host sizes the grid by capacity and passes the live element
// count as a device pointer, so the sort can be recorded before the count is known. Blocks
// past the count see no elements -- hist writes zeros, scatter writes nothing -- so the
// scanned offsets and the output are identical to a sort sized exactly.
uint liveCount(uint n, uint *count)
{
    return count != nullptr ? min(count[0], n) : n;
}

uint2 lowBitsMask64(uint n)
{
    uint2 m;
    m.x = (n == 0u) ? 0u : ((n >= 32u) ? 0xFFFFFFFFu : ((1u << n) - 1u));
    uint ny = (n > 32u) ? (n - 32u) : 0u;
    m.y = (ny == 0u) ? 0u : ((ny >= 32u) ? 0xFFFFFFFFu : ((1u << ny) - 1u));
    return m;
}

struct WaveRank
{
    uint laneRank;  // stable rank of this lane among same-digit lanes in its wave
    uint waveCount; // total same-digit lanes in this wave
    uint waveId;    // gi / WaveGetLaneCount()
};

// Stable intra-wave rank of `digit` among the wave's `valid` lanes, via 8 ballots (one
// per digit bit) ANDed down to the exact-match mask, plus one ballot to exclude invalid
// (out-of-range) lanes. No barrier: everything here is wave-synchronous.
WaveRank computeWaveRank(uint gi, uint digit, bool valid)
{
    uint W = WaveGetLaneCount();
    uint laneId = WaveGetLaneIndex();

    uint2 full = lowBitsMask64(W);
    uint2 acc = full;
    [unroll]
    for (uint b = 0; b < 8; b++)
    {
        bool bitSet = ((digit >> b) & 1u) != 0u;
        uint2 ballot = WaveActiveBallot(bitSet).xy;
        uint2 sel = bitSet ? ballot : ((~ballot) & full);
        acc &= sel;
    }
    uint2 validBallot = WaveActiveBallot(valid).xy;
    acc &= validBallot;

    uint2 before = lowBitsMask64(laneId);
    uint2 beforeMatch = acc & before;

    WaveRank r;
    r.laneRank = countbits(beforeMatch.x) + countbits(beforeMatch.y);
    r.waveCount = countbits(acc.x) + countbits(acc.y);
    r.waveId = gi / W;
    return r;
}

// ===========================================================================
// (a) uint32 keys + int32 values (SortPairs) -- broadphase pair sort.
// ===========================================================================

struct HistArgsU32Pairs
{
    uint *keysIn;
    uint *digitCount; // [digit * numBlocks + blockIdx], RADIX_BUCKETS * numBlocks entries
    uint *skipMask;   // numPasses entries, 0/1 -- see csFinalizeSkipMaskU32
    uint n;
    uint numBlocks;
    uint pass;
    uint _pad0;
    uint *count;      // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

groupshared uint sHistA[RADIX_BUCKETS];

[shader("compute")]
[numthreads(TILE_THREADS, 1, 1)]
void csHistU32Pairs(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    HistArgsU32Pairs *args = (HistArgsU32Pairs *)pc.argsAddr;
    uint n = liveCount(args->n, args->count);
    if (args->skipMask[args->pass] != 0u)
        return; // constant digit this pass -- scatter identity-copies, digitCount is unused

    uint gi = gtid.x;
    uint blockIdx = gid.x;
    uint blockBase = blockIdx * TILE_SIZE;

    if (gi < RADIX_BUCKETS)
        sHistA[gi] = 0u;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint k = 0; k < KEYS_PER_THREAD; k++)
    {
        uint idx = blockBase + k * TILE_THREADS + gi;
        if (idx < n)
            InterlockedAdd(sHistA[digitU32(args->keysIn[idx], args->pass)], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi < RADIX_BUCKETS)
        args->digitCount[gi * args->numBlocks + blockIdx] = sHistA[gi];
}

struct ScatterArgsU32Pairs
{
    uint *keysIn;
    int *valsIn;
    uint *digitCountScanned; // exclusive-scanned flattened [digit][block] array
    uint *skipMask;          // numPasses entries, 0/1 -- see csFinalizeSkipMaskU32
    uint *keysOut;
    int *valsOut;
    uint n;
    uint numBlocks;
    uint pass;
    uint _pad0;
    uint *count;      // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

groupshared uint sWaveHistA[MAX_WAVES * RADIX_BUCKETS];
groupshared uint sCarryA[RADIX_BUCKETS];

[shader("compute")]
[numthreads(TILE_THREADS, 1, 1)]
void csScatterU32Pairs(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    ScatterArgsU32Pairs *args = (ScatterArgsU32Pairs *)pc.argsAddr;
    uint n = liveCount(args->n, args->count);
    uint gi = gtid.x;
    uint blockIdx = gid.x;
    uint blockBase = blockIdx * TILE_SIZE;
    uint numBlocks = args->numBlocks;

    if (args->skipMask[args->pass] != 0u)
    {
        // Constant digit this pass -- the permutation is the identity; copy straight through.
        [unroll]
        for (uint k = 0; k < KEYS_PER_THREAD; k++)
        {
            uint idx = blockBase + k * TILE_THREADS + gi;
            if (idx < n)
            {
                args->keysOut[idx] = args->keysIn[idx];
                args->valsOut[idx] = args->valsIn[idx];
            }
        }
        return;
    }

    uint numWaves = TILE_THREADS / WaveGetLaneCount();

    if (gi < RADIX_BUCKETS)
        sCarryA[gi] = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint k = 0; k < KEYS_PER_THREAD; k++)
    {
        uint idx = blockBase + k * TILE_THREADS + gi;
        bool valid = idx < n;
        uint key = valid ? args->keysIn[idx] : 0u;
        int val = valid ? args->valsIn[idx] : 0;
        uint digit = valid ? digitU32(key, args->pass) : 0u;

        [unroll]
        for (uint w = 0; w < MAX_WAVES; w++)
            if (w < numWaves)
                sWaveHistA[w * RADIX_BUCKETS + gi] = 0u;
        GroupMemoryBarrierWithGroupSync();

        WaveRank wr = computeWaveRank(gi, digit, valid);
        if (valid && wr.laneRank == 0u)
            sWaveHistA[wr.waveId * RADIX_BUCKETS + digit] = wr.waveCount;
        GroupMemoryBarrierWithGroupSync();

        uint crossWaveOffset = 0u;
        for (uint w = 0; w < wr.waveId; w++)
            crossWaveOffset += sWaveHistA[w * RADIX_BUCKETS + digit];
        uint localRank = sCarryA[digit] + crossWaveOffset + wr.laneRank;

        if (valid)
        {
            uint base = args->digitCountScanned[digit * numBlocks + blockIdx];
            uint finalPos = base + localRank;
            args->keysOut[finalPos] = key;
            args->valsOut[finalPos] = val;
        }
        GroupMemoryBarrierWithGroupSync();

        if (gi < RADIX_BUCKETS)
        {
            uint total = 0u;
            for (uint w = 0; w < numWaves; w++)
                total += sWaveHistA[w * RADIX_BUCKETS + gi];
            sCarryA[gi] += total;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// (b) uint64 keys + uint32 values (SortPairs) -- constraint-graph CSR entries.
// ===========================================================================

struct HistArgsU64Pairs
{
    uint64_t *keysIn;
    uint *digitCount;
    uint *skipMask;
    uint n;
    uint numBlocks;
    uint pass;
    uint _pad0;
    uint *count;      // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

groupshared uint sHistB[RADIX_BUCKETS];

[shader("compute")]
[numthreads(TILE_THREADS, 1, 1)]
void csHistU64Pairs(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    HistArgsU64Pairs *args = (HistArgsU64Pairs *)pc.argsAddr;
    uint n = liveCount(args->n, args->count);
    if (args->skipMask[args->pass] != 0u)
        return;

    uint gi = gtid.x;
    uint blockIdx = gid.x;
    uint blockBase = blockIdx * TILE_SIZE;

    if (gi < RADIX_BUCKETS)
        sHistB[gi] = 0u;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint k = 0; k < KEYS_PER_THREAD; k++)
    {
        uint idx = blockBase + k * TILE_THREADS + gi;
        if (idx < n)
            InterlockedAdd(sHistB[digitU64(args->keysIn[idx], args->pass)], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi < RADIX_BUCKETS)
        args->digitCount[gi * args->numBlocks + blockIdx] = sHistB[gi];
}

struct ScatterArgsU64Pairs
{
    uint64_t *keysIn;
    uint *valsIn;
    uint *digitCountScanned;
    uint *skipMask;
    uint64_t *keysOut;
    uint *valsOut;
    uint n;
    uint numBlocks;
    uint pass;
    uint _pad0;
    uint *count;      // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

groupshared uint sWaveHistB[MAX_WAVES * RADIX_BUCKETS];
groupshared uint sCarryB[RADIX_BUCKETS];

[shader("compute")]
[numthreads(TILE_THREADS, 1, 1)]
void csScatterU64Pairs(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    ScatterArgsU64Pairs *args = (ScatterArgsU64Pairs *)pc.argsAddr;
    uint n = liveCount(args->n, args->count);
    uint gi = gtid.x;
    uint blockIdx = gid.x;
    uint blockBase = blockIdx * TILE_SIZE;
    uint numBlocks = args->numBlocks;

    if (args->skipMask[args->pass] != 0u)
    {
        [unroll]
        for (uint k = 0; k < KEYS_PER_THREAD; k++)
        {
            uint idx = blockBase + k * TILE_THREADS + gi;
            if (idx < n)
            {
                args->keysOut[idx] = args->keysIn[idx];
                args->valsOut[idx] = args->valsIn[idx];
            }
        }
        return;
    }

    uint numWaves = TILE_THREADS / WaveGetLaneCount();

    if (gi < RADIX_BUCKETS)
        sCarryB[gi] = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint k = 0; k < KEYS_PER_THREAD; k++)
    {
        uint idx = blockBase + k * TILE_THREADS + gi;
        bool valid = idx < n;
        uint64_t key = valid ? args->keysIn[idx] : (uint64_t)0;
        uint val = valid ? args->valsIn[idx] : 0u;
        uint digit = valid ? digitU64(key, args->pass) : 0u;

        [unroll]
        for (uint w = 0; w < MAX_WAVES; w++)
            if (w < numWaves)
                sWaveHistB[w * RADIX_BUCKETS + gi] = 0u;
        GroupMemoryBarrierWithGroupSync();

        WaveRank wr = computeWaveRank(gi, digit, valid);
        if (valid && wr.laneRank == 0u)
            sWaveHistB[wr.waveId * RADIX_BUCKETS + digit] = wr.waveCount;
        GroupMemoryBarrierWithGroupSync();

        uint crossWaveOffset = 0u;
        for (uint w = 0; w < wr.waveId; w++)
            crossWaveOffset += sWaveHistB[w * RADIX_BUCKETS + digit];
        uint localRank = sCarryB[digit] + crossWaveOffset + wr.laneRank;

        if (valid)
        {
            uint base = args->digitCountScanned[digit * numBlocks + blockIdx];
            uint finalPos = base + localRank;
            args->keysOut[finalPos] = key;
            args->valsOut[finalPos] = val;
        }
        GroupMemoryBarrierWithGroupSync();

        if (gi < RADIX_BUCKETS)
        {
            uint total = 0u;
            for (uint w = 0; w < numWaves; w++)
                total += sWaveHistB[w * RADIX_BUCKETS + gi];
            sCarryB[gi] += total;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// (c) uint64 keys only (SortKeys) -- broadphase pair-key canonicalization.
// ===========================================================================

struct HistArgsU64Keys
{
    uint64_t *keysIn;
    uint *digitCount;
    uint *skipMask;
    uint n;
    uint numBlocks;
    uint pass;
    uint _pad0;
    uint *count;      // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

groupshared uint sHistC[RADIX_BUCKETS];

[shader("compute")]
[numthreads(TILE_THREADS, 1, 1)]
void csHistU64Keys(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    HistArgsU64Keys *args = (HistArgsU64Keys *)pc.argsAddr;
    uint n = liveCount(args->n, args->count);
    if (args->skipMask[args->pass] != 0u)
        return;

    uint gi = gtid.x;
    uint blockIdx = gid.x;
    uint blockBase = blockIdx * TILE_SIZE;

    if (gi < RADIX_BUCKETS)
        sHistC[gi] = 0u;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint k = 0; k < KEYS_PER_THREAD; k++)
    {
        uint idx = blockBase + k * TILE_THREADS + gi;
        if (idx < n)
            InterlockedAdd(sHistC[digitU64(args->keysIn[idx], args->pass)], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi < RADIX_BUCKETS)
        args->digitCount[gi * args->numBlocks + blockIdx] = sHistC[gi];
}

struct ScatterArgsU64Keys
{
    uint64_t *keysIn;
    uint *digitCountScanned;
    uint *skipMask;
    uint64_t *keysOut;
    uint n;
    uint numBlocks;
    uint pass;
    uint _pad0;
    uint *count;      // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

groupshared uint sWaveHistC[MAX_WAVES * RADIX_BUCKETS];
groupshared uint sCarryC[RADIX_BUCKETS];

[shader("compute")]
[numthreads(TILE_THREADS, 1, 1)]
void csScatterU64Keys(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    ScatterArgsU64Keys *args = (ScatterArgsU64Keys *)pc.argsAddr;
    uint n = liveCount(args->n, args->count);
    uint gi = gtid.x;
    uint blockIdx = gid.x;
    uint blockBase = blockIdx * TILE_SIZE;
    uint numBlocks = args->numBlocks;

    if (args->skipMask[args->pass] != 0u)
    {
        [unroll]
        for (uint k = 0; k < KEYS_PER_THREAD; k++)
        {
            uint idx = blockBase + k * TILE_THREADS + gi;
            if (idx < n)
                args->keysOut[idx] = args->keysIn[idx];
        }
        return;
    }

    uint numWaves = TILE_THREADS / WaveGetLaneCount();

    if (gi < RADIX_BUCKETS)
        sCarryC[gi] = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint k = 0; k < KEYS_PER_THREAD; k++)
    {
        uint idx = blockBase + k * TILE_THREADS + gi;
        bool valid = idx < n;
        uint64_t key = valid ? args->keysIn[idx] : (uint64_t)0;
        uint digit = valid ? digitU64(key, args->pass) : 0u;

        [unroll]
        for (uint w = 0; w < MAX_WAVES; w++)
            if (w < numWaves)
                sWaveHistC[w * RADIX_BUCKETS + gi] = 0u;
        GroupMemoryBarrierWithGroupSync();

        WaveRank wr = computeWaveRank(gi, digit, valid);
        if (valid && wr.laneRank == 0u)
            sWaveHistC[wr.waveId * RADIX_BUCKETS + digit] = wr.waveCount;
        GroupMemoryBarrierWithGroupSync();

        uint crossWaveOffset = 0u;
        for (uint w = 0; w < wr.waveId; w++)
            crossWaveOffset += sWaveHistC[w * RADIX_BUCKETS + digit];
        uint localRank = sCarryC[digit] + crossWaveOffset + wr.laneRank;

        if (valid)
        {
            uint base = args->digitCountScanned[digit * numBlocks + blockIdx];
            uint finalPos = base + localRank;
            args->keysOut[finalPos] = key;
        }
        GroupMemoryBarrierWithGroupSync();

        if (gi < RADIX_BUCKETS)
        {
            uint total = 0u;
            for (uint w = 0; w < numWaves; w++)
                total += sWaveHistC[w * RADIX_BUCKETS + gi];
            sCarryC[gi] += total;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// (d) Global OR/AND reduction over the keys, used by the host (vk_primitives.cpp) to
// skip a pass whose digit is provably identical on every key (a no-op permutation).
// One block-level shared-memory tree reduction per dispatched block, then one atomic
// pair into the single-element output -- bounds the atomics to numBlocks, not n.
// ===========================================================================

struct ReduceOrAndArgsU32
{
    uint *keysIn;
    uint *outOr;  // 1-element buffer, must be pre-zeroed
    uint *outAnd; // 1-element buffer, must be pre-filled with 0xFFFFFFFF
    uint n;
    uint _pad0;
    uint *count;      // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

groupshared uint sRedOr32[TILE_THREADS];
groupshared uint sRedAnd32[TILE_THREADS];

[shader("compute")]
[numthreads(TILE_THREADS, 1, 1)]
void csReduceOrAndU32(uint3 tid: SV_DispatchThreadID, uint3 gtid: SV_GroupThreadID)
{
    ReduceOrAndArgsU32 *args = (ReduceOrAndArgsU32 *)pc.argsAddr;
    uint n = liveCount(args->n, args->count);
    uint gi = gtid.x;
    bool valid = tid.x < n;
    uint key = valid ? args->keysIn[tid.x] : 0u;
    sRedOr32[gi] = valid ? key : 0u;
    sRedAnd32[gi] = valid ? key : 0xFFFFFFFFu;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint offset = TILE_THREADS / 2; offset > 0; offset >>= 1)
    {
        if (gi < offset)
        {
            sRedOr32[gi] |= sRedOr32[gi + offset];
            sRedAnd32[gi] &= sRedAnd32[gi + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0)
    {
        InterlockedOr(*args->outOr, sRedOr32[0]);
        InterlockedAnd(*args->outAnd, sRedAnd32[0]);
    }
}

struct ReduceOrAndArgsU64
{
    uint64_t *keysIn;
    uint64_t *outOr;
    uint64_t *outAnd;
    uint n;
    uint _pad0;
    uint *count;      // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

groupshared uint64_t sRedOr64[TILE_THREADS];
groupshared uint64_t sRedAnd64[TILE_THREADS];

[shader("compute")]
[numthreads(TILE_THREADS, 1, 1)]
void csReduceOrAndU64(uint3 tid: SV_DispatchThreadID, uint3 gtid: SV_GroupThreadID)
{
    ReduceOrAndArgsU64 *args = (ReduceOrAndArgsU64 *)pc.argsAddr;
    uint n = liveCount(args->n, args->count);
    uint gi = gtid.x;
    bool valid = tid.x < n;
    uint64_t key = valid ? args->keysIn[tid.x] : (uint64_t)0;
    sRedOr64[gi] = valid ? key : (uint64_t)0;
    sRedAnd64[gi] = valid ? key : ~(uint64_t)0;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint offset = TILE_THREADS / 2; offset > 0; offset >>= 1)
    {
        if (gi < offset)
        {
            sRedOr64[gi] |= sRedOr64[gi + offset];
            sRedAnd64[gi] &= sRedAnd64[gi + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0)
    {
        InterlockedOr(*args->outOr, sRedOr64[0]);
        InterlockedAnd(*args->outAnd, sRedAnd64[0]);
    }
}

// ===========================================================================
// (e) Turns the OR/AND reduction into a per-pass skip flag buffer, entirely on device --
// single-thread, single-workgroup, always recorded right after the reduce dispatch (with
// a barrier in between) so every hist/scatter dispatch that follows can read a
// fully-written skipMask. No host involvement: this is what makes the whole sort
// record-only and safe before its input-producing dispatch has been submitted.
// ===========================================================================

struct FinalizeSkipMaskArgsU32
{
    uint *orAddr;
    uint *andAddr;
    uint *skipMask;
    uint numPasses;
    uint _pad0;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csFinalizeSkipMaskU32(uint3 tid: SV_DispatchThreadID)
{
    FinalizeSkipMaskArgsU32 *args = (FinalizeSkipMaskArgsU32 *)pc.argsAddr;
    uint orVal = *args->orAddr;
    uint andVal = *args->andAddr;
    for (uint p = 0; p < args->numPasses; p++)
    {
        uint orByte = (orVal >> (p * 8u)) & 0xFFu;
        uint andByte = (andVal >> (p * 8u)) & 0xFFu;
        args->skipMask[p] = (orByte == andByte) ? 1u : 0u;
    }
}

struct FinalizeSkipMaskArgsU64
{
    uint64_t *orAddr;
    uint64_t *andAddr;
    uint *skipMask;
    uint numPasses;
    uint _pad0;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csFinalizeSkipMaskU64(uint3 tid: SV_DispatchThreadID)
{
    FinalizeSkipMaskArgsU64 *args = (FinalizeSkipMaskArgsU64 *)pc.argsAddr;
    uint64_t orVal = *args->orAddr;
    uint64_t andVal = *args->andAddr;
    for (uint p = 0; p < args->numPasses; p++)
    {
        uint64_t orByte = (orVal >> (uint64_t)(p * 8u)) & (uint64_t)0xFF;
        uint64_t andByte = (andVal >> (uint64_t)(p * 8u)) & (uint64_t)0xFF;
        args->skipMask[p] = (orByte == andByte) ? 1u : 0u;
    }
}
