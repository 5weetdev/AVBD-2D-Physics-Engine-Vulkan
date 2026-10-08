// Stream compaction of items selected by a uint8 flag array. Built on scan.slang's
// exclusive scan: convert the 0/1 flags to uint32, scan them (host recurses via
// gpu/vk_primitives.cpp), then scatter each flagged item to its scanned
// position.

#define COMPACT_BLOCK 256

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct FlagsToU32Args
{
    uint8_t *flags;
    uint *out;
    uint n;
    uint _pad0;
};

[shader("compute")]
[numthreads(COMPACT_BLOCK, 1, 1)]
void csFlagsToU32(uint3 tid: SV_DispatchThreadID)
{
    FlagsToU32Args *args = (FlagsToU32Args *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->n)
        return;
    args->out[i] = (args->flags[i] != 0) ? 1u : 0u;
}

struct ScatterCompactArgs
{
    int *items;
    uint8_t *flags;
    uint *scanned; // exclusive prefix sum of (flags != 0)
    int *out;
    uint n;
    uint _pad0;
};

[shader("compute")]
[numthreads(COMPACT_BLOCK, 1, 1)]
void csScatterCompact(uint3 tid: SV_DispatchThreadID)
{
    ScatterCompactArgs *args = (ScatterCompactArgs *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->n)
        return;
    if (args->flags[i] != 0)
        args->out[args->scanned[i]] = args->items[i];
}

struct NumSelectedArgs
{
    uint *scanned;
    uint8_t *flags;
    int *numSelected;
    uint n;
    uint _pad0;
};

// Single-thread finisher: numSelected = scanned[n-1] + (flags[n-1] != 0). Cheap enough
// (one dispatch, one thread) that it is not worth folding into the scatter kernel and
// racing the last block's write.
[shader("compute")]
[numthreads(1, 1, 1)]
void csWriteNumSelected(uint3 tid: SV_DispatchThreadID)
{
    NumSelectedArgs *args = (NumSelectedArgs *)pc.argsAddr;
    if (args->n == 0)
    {
        *args->numSelected = 0;
        return;
    }
    uint last = args->n - 1;
    uint total = args->scanned[last] + ((args->flags[last] != 0) ? 1u : 0u);
    *args->numSelected = (int)total;
}
