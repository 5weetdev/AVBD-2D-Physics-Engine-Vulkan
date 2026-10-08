// Exclusive prefix sum (uint32), used as a building block for stream compaction
// (compact.slang) and the cross-block offset computation inside the radix sort
// (radix.slang).
//
// Multi-level block scan, correctness-first (not decoupled look-back):
//   csScanBlock: exclusive-scans each contiguous 256-element chunk in shared memory
//                (Hillis-Steele), writes the local result to `out` and each chunk's
//                total to `blockSums[blockIdx]`.
//   csAddBlockSums: adds a per-block offset (the *scanned* block sums from the next
//                   recursion level) into every element of that block.
// The host (gpu/vk_primitives.cpp, recordScanLevel) recurses: scan the
// n-element array into per-block sums of size ceil(n/256), and if that count is > 1,
// recursively scan IT the same way, then add the result back down. Depth is
// log_256(n), so 3-4 levels covers n up to ~2^32.

#define SCAN_BLOCK 256

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct ScanBlockArgs
{
    uint *in;
    uint *out;
    uint *blockSums; // one entry per dispatched block; always written (block index gid.x)
    uint n;
    uint _pad0;
};

groupshared uint sScan[SCAN_BLOCK];

[shader("compute")]
[numthreads(SCAN_BLOCK, 1, 1)]
void csScanBlock(uint3 tid: SV_DispatchThreadID, uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    ScanBlockArgs *args = (ScanBlockArgs *)pc.argsAddr;
    uint gi = gtid.x;
    uint i = tid.x;

    sScan[gi] = (i < args->n) ? args->in[i] : 0u;
    GroupMemoryBarrierWithGroupSync();

    // Hillis-Steele inclusive scan, in place: each step reads the pre-step values of
    // this thread and (this - offset), barriers, then writes -- so every thread's read
    // in a step always sees last step's fully-written values.
    [unroll]
    for (uint offset = 1; offset < SCAN_BLOCK; offset <<= 1)
    {
        uint t = sScan[gi];
        if (gi >= offset)
            t += sScan[gi - offset];
        GroupMemoryBarrierWithGroupSync();
        sScan[gi] = t;
        GroupMemoryBarrierWithGroupSync();
    }

    if (i < args->n)
        args->out[i] = (gi == 0) ? 0u : sScan[gi - 1];
    if (gi == 0)
        args->blockSums[gid.x] = sScan[SCAN_BLOCK - 1];
}

struct AddBlockSumsArgs
{
    uint *out;
    uint *blockOffsets; // exclusive-scanned block sums; one entry per block
    uint n;
    uint _pad0;
};

[shader("compute")]
[numthreads(SCAN_BLOCK, 1, 1)]
void csAddBlockSums(uint3 tid: SV_DispatchThreadID, uint3 gid: SV_GroupID)
{
    AddBlockSumsArgs *args = (AddBlockSumsArgs *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->n)
        return;
    args->out[i] += args->blockOffsets[gid.x];
}
