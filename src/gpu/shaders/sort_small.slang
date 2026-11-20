// Small-world sorts (own module: the groupshared arrays below would otherwise count
// against every radix kernel's shared-memory budget, which is a per-module sum).
// Header copied from radix.slang.

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

uint liveCount(uint n, uint *count)
{
    return count != nullptr ? min(count[0], n) : n;
}


// ===========================================================================
// (e) Small-world sorts: one workgroup sorts up to SORT_SMALL_MAX elements entirely in
// shared memory with a bitonic network, replacing the 4/8 radix passes (hist, scan,
// scatter, skip-mask reduce: dozens of dispatches and barriers) when the CAPACITY is at
// most SORT_SMALL_MAX. The live count is still read on device (min(*count, n)), so no
// overflow path exists: live <= n <= SORT_SMALL_MAX by construction.
//
// The comparison is on (key, original index), a strict total order, so the result is
// exactly the stable radix order (equal keys keep input order); values travel with the
// index. Slots at or past the live count are not written, as in the radix path. The
// network runs over the next power of two, padded with (max key, max index), which sorts
// after every real element. Every thread runs every loop, so barriers are uniform.
// SORT_SMALL_MAX is mirrored by kSmallSortMaxKeys in vk_small_world.h (workgroup memory: 49152 bytes, see there).
// ===========================================================================

static const uint SORT_SMALL_MAX = 4096u;
static const uint SORT_GROUP = 1024u;
static const uint SORT_PAD_IDX = 0xFFFFFFFFu;

groupshared uint64_t sSortK[4096];
groupshared uint sSortI[4096];

bool sortGreater(uint64_t ka, uint ia, uint64_t kb, uint ib)
{
    return ka > kb || (ka == kb && ia > ib);
}

// Bitonic network over sSortK / sSortI[0, size); size a power of two.
void bitonicSortShared(uint size, uint lane)
{
    for (uint k = 2u; k <= size; k <<= 1u)
    {
        for (uint j = k >> 1u; j > 0u; j >>= 1u)
        {
            for (uint i = lane; i < size; i += SORT_GROUP)
            {
                uint l = i ^ j;
                if (l > i)
                {
                    uint64_t ka = sSortK[i];
                    uint64_t kb = sSortK[l];
                    uint ia = sSortI[i];
                    uint ib = sSortI[l];
                    bool up = (i & k) == 0u;
                    bool g = sortGreater(ka, ia, kb, ib);
                    if (up ? g : !g)
                    {
                        sSortK[i] = kb;
                        sSortK[l] = ka;
                        sSortI[i] = ib;
                        sSortI[l] = ia;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

uint sortPow2(uint live)
{
    uint size = 1u;
    while (size < live)
        size <<= 1u;
    return size;
}

struct SortSmallArgsU32Pairs
{
    uint *keysIn;
    uint *valsIn;
    uint *keysOut;
    uint *valsOut;
    uint n;
    uint _pad0;
    uint *count; // 0: n is exact; else n is capacity, the live count is min(*count, n)
};

[shader("compute")]
[numthreads(1024, 1, 1)]
void csSortSmallU32Pairs(uint3 gtid: SV_GroupThreadID)
{
    SortSmallArgsU32Pairs *args = (SortSmallArgsU32Pairs *)pc.argsAddr;
    uint lane = gtid.x;
    uint live = liveCount(args->n, args->count);
    uint size = sortPow2(live);
    for (uint i = lane; i < size; i += SORT_GROUP)
    {
        bool v = i < live;
        sSortK[i] = v ? (uint64_t)args->keysIn[i] : ~(uint64_t)0;
        sSortI[i] = v ? i : SORT_PAD_IDX;
    }
    GroupMemoryBarrierWithGroupSync();
    bitonicSortShared(size, lane);
    for (uint i = lane; i < live; i += SORT_GROUP)
    {
        args->keysOut[i] = (uint)sSortK[i];
        args->valsOut[i] = args->valsIn[sSortI[i]];
    }
}

struct SortSmallArgsU64Pairs
{
    uint64_t *keysIn;
    uint *valsIn;
    uint64_t *keysOut;
    uint *valsOut;
    uint n;
    uint _pad0;
    uint *count;
};

[shader("compute")]
[numthreads(1024, 1, 1)]
void csSortSmallU64Pairs(uint3 gtid: SV_GroupThreadID)
{
    SortSmallArgsU64Pairs *args = (SortSmallArgsU64Pairs *)pc.argsAddr;
    uint lane = gtid.x;
    uint live = liveCount(args->n, args->count);
    uint size = sortPow2(live);
    for (uint i = lane; i < size; i += SORT_GROUP)
    {
        bool v = i < live;
        sSortK[i] = v ? args->keysIn[i] : ~(uint64_t)0;
        sSortI[i] = v ? i : SORT_PAD_IDX;
    }
    GroupMemoryBarrierWithGroupSync();
    bitonicSortShared(size, lane);
    for (uint i = lane; i < live; i += SORT_GROUP)
    {
        args->keysOut[i] = sSortK[i];
        args->valsOut[i] = args->valsIn[sSortI[i]];
    }
}

struct SortSmallArgsU64Keys
{
    uint64_t *keysIn;
    uint64_t *keysOut;
    uint n;
    uint _pad0;
    uint *count;
};

[shader("compute")]
[numthreads(1024, 1, 1)]
void csSortSmallU64Keys(uint3 gtid: SV_GroupThreadID)
{
    SortSmallArgsU64Keys *args = (SortSmallArgsU64Keys *)pc.argsAddr;
    uint lane = gtid.x;
    uint live = liveCount(args->n, args->count);
    uint size = sortPow2(live);
    for (uint i = lane; i < size; i += SORT_GROUP)
    {
        bool v = i < live;
        sSortK[i] = v ? args->keysIn[i] : ~(uint64_t)0;
        sSortI[i] = v ? i : SORT_PAD_IDX; // keys only: ties are indistinguishable, index kept for the shared network
    }
    GroupMemoryBarrierWithGroupSync();
    bitonicSortShared(size, lane);
    for (uint i = lane; i < live; i += SORT_GROUP)
        args->keysOut[i] = sSortK[i];
}
