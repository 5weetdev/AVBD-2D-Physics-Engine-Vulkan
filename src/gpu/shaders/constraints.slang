// CSR constraint-graph builder + Jones-Plassmann coloring. Graph key packing
// is ((body<<32)|(type<<29)|index), with a deterministic jpPriority hash and a batched
// round loop (driven from the host in vk_constraints.cpp, since the "keep
// dispatching until the remaining-counter reads back zero" loop is host orchestration,
// not GPU work). The kernels are k_emit_entries / k_emit_force_entries /
// k_rows / k_reset_colors / k_jp_round / k_color_keys / k_color_rows; the force-entry
// kernel takes a runtime `type` field (joints and springs share one kernel,
// csEmitForceEntries).
//
// warpReserve<2> (gpu_warp's ballot-loop aggregation) becomes subgroupReserve, the
// same WavePrefixSum/WaveActiveSum construction narrowphase.slang and broadphase.slang
// already use, generic over WaveGetLaneCount() rather than assuming 32 lanes.
//
// Host orchestration (buffer reserve/grow, the batched JP round readback loop, the two
// radix sorts) is gpu/vk_constraints.h/.cpp.


struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// ---------------------------------------------------------------------------
// Constraint graph key: (body<<32)|(type<<29)|index -- the device structs' makeGraphKey/
// graphBody/graphType/graphIndex, transcribed bit for bit. FORCE_MANIFOLD=0,
// FORCE_JOINT=1, FORCE_SPRING=2 (gpu_forces's ForceType).
// ---------------------------------------------------------------------------
static const int AVBD_FORCE_INDEX_BITS_S = 29;
static const uint AVBD_MAX_FORCE_INDEX_S = (uint(1) << AVBD_FORCE_INDEX_BITS_S) - 1u;
static const int FORCE_MANIFOLD_S = 0;
static const int FORCE_JOINT_S = 1;
static const int FORCE_SPRING_S = 2;
static const int AVBD_MAX_COLORS_S = 128; // the constraint builder: AVBD_MAX_COLORS

uint64_t makeGraphKeyS(int body, int type, int index)
{
    return (uint64_t(uint(body)) << 32) |
           (uint64_t(uint(type)) << AVBD_FORCE_INDEX_BITS_S) |
           uint64_t(uint(index));
}

int graphBodyS(uint64_t k) { return (int)(uint)(k >> 32); }
int graphTypeS(uint64_t k) { return (int)((uint(k & uint64_t(0xFFFFFFFFu))) >> AVBD_FORCE_INDEX_BITS_S); }
int graphIndexS(uint64_t k) { return (int)(uint(k & uint64_t(0xFFFFFFFFu)) & AVBD_MAX_FORCE_INDEX_S); }

// Subgroup-aggregated contiguous-slot reservation -- see narrowphase.slang's
// subgroupReserve (identical construction, redeclared here per this tree's convention
// of each shader file owning its own small helper copies rather than sharing across
// modules that have no `import`able surface for them).
int subgroupReserve(int *counter, int n)
{
    int prefix = WavePrefixSum(n);
    int total = WaveActiveSum(n);
    int base = 0;
    if (WaveIsFirstLane())
        InterlockedAdd(*counter, total, base);
    base = WaveReadLaneFirst(base);
    return base + prefix;
}

// ---------------------------------------------------------------------------
// csResetColors -- k_reset_colors. Colors do not persist across steps (greedy-from-scratch
// beats incremental repair here).
// ---------------------------------------------------------------------------
struct ResetColorsArgs
{
    int *color;
    int *dyn;
    uint dynCount;
    uint _pad0;
    uint *dynCountPtr; // when set, dynCount is a bound and the live count is min(*ptr, dynCount)
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csResetColors(uint3 tid: SV_DispatchThreadID)
{
    ResetColorsArgs *args = (ResetColorsArgs *)pc.argsAddr;
    uint t = tid.x;
    uint n = args->dynCountPtr != nullptr ? min(args->dynCountPtr[0], args->dynCount) : args->dynCount;
    if (t >= n)
        return;
    args->color[args->dyn[t]] = -1;
}

// ---------------------------------------------------------------------------
// csEmitManifoldEntries -- k_emit_entries. One entry per (solvable body, manifold)
// incidence; "solvable" is mass>0 and awake, as in the live-body test.
// ---------------------------------------------------------------------------
struct EmitManifoldArgs
{
    int *mBodyA;
    int *mBodyB;
    int *mContactCount;
    uint slots;
    uint _pad0;
    float *mass;
    uint8_t *awake;
    uint64_t *keys;
    int *isA;
    int *counter;
    int capacity;
    int _pad1;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csEmitManifoldEntries(uint3 tid: SV_DispatchThreadID)
{
    EmitManifoldArgs *args = (EmitManifoldArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->slots || args->mContactCount[t] <= 0)
        return;

    int a = args->mBodyA[t], b = args->mBodyB[t];
    bool liveA = args->mass[a] > 0.0f && args->awake[a] != 0;
    bool liveB = args->mass[b] > 0.0f && args->awake[b] != 0;
    int n = (liveA ? 1 : 0) + (liveB ? 1 : 0);
    if (n == 0)
        return;

    int base = subgroupReserve(args->counter, n);
    if (base + n > args->capacity)
        return;

    if (liveA)
    {
        args->keys[base] = makeGraphKeyS(a, FORCE_MANIFOLD_S, (int)t);
        args->isA[base] = 1;
        base++;
    }
    if (liveB)
    {
        args->keys[base] = makeGraphKeyS(b, FORCE_MANIFOLD_S, (int)t);
        args->isA[base] = 0;
    }
}

// ---------------------------------------------------------------------------
// csEmitForceEntries -- k_emit_force_entries<TYPE>, with TYPE folded into a runtime
// `type` field instead of a template parameter (joints and springs share this one
// kernel; `broken` is null for springs).
// A world-anchored joint (bodyA < 0) contributes only bodyB's entry.
// ---------------------------------------------------------------------------
struct EmitForceArgs
{
    int *fBodyA;
    int *fBodyB;
    uint count;
    uint _pad0;
    uint8_t *broken; // may be null (springs)
    float *mass;
    uint8_t *awake;
    uint64_t *keys;
    int *isA;
    int *counter;
    int capacity;
    int type;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csEmitForceEntries(uint3 tid: SV_DispatchThreadID)
{
    EmitForceArgs *args = (EmitForceArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->count)
        return;
    if (args->broken != nullptr && args->broken[t] != 0)
        return;

    int a = args->fBodyA[t], b = args->fBodyB[t];
    bool liveA = a >= 0 && args->mass[a] > 0.0f && args->awake[a] != 0;
    bool liveB = b >= 0 && args->mass[b] > 0.0f && args->awake[b] != 0;
    int n = (liveA ? 1 : 0) + (liveB ? 1 : 0);
    if (n == 0)
        return;

    int base = subgroupReserve(args->counter, n);
    if (base + n > args->capacity)
        return;

    if (liveA)
    {
        args->keys[base] = makeGraphKeyS(a, args->type, (int)t);
        args->isA[base] = 1;
        base++;
    }
    if (liveB)
    {
        args->keys[base] = makeGraphKeyS(b, args->type, (int)t);
        args->isA[base] = 0;
    }
}

// ---------------------------------------------------------------------------
// csConstraintsRows -- k_rows. Row boundaries over the sorted (body,type,index) entry
// list. start/end are zeroed by the host before this dispatch, so a body with no
// constraints keeps an empty range.
// ---------------------------------------------------------------------------
struct RowsArgs
{
    uint64_t *keys;
    uint n;
    uint _pad0;
    int *start;
    int *end;
    // when set, `n` is the capacity and the live entry count is min(*count, n).
    uint *count;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csConstraintsRows(uint3 tid: SV_DispatchThreadID)
{
    RowsArgs *args = (RowsArgs *)pc.argsAddr;
    uint n = args->count != nullptr ? min(args->count[0], args->n) : args->n;
    uint e = tid.x;
    if (e >= n)
        return;
    int body = graphBodyS(args->keys[e]);
    if (e == 0 || graphBodyS(args->keys[e - 1]) != body)
        args->start[body] = (int)e;
    if (e == n - 1 || graphBodyS(args->keys[e + 1]) != body)
        args->end[body] = (int)e + 1;
}

// ---------------------------------------------------------------------------
// csJpRound -- k_jp_round. One Jones-Plassmann round: a body takes the smallest color
// unused by its neighbours as soon as it outranks every neighbour still uncolored.
// The graph's `other` lookup (graphOther in the constraint builder) needs the manifold/
// joint/spring bodyA/bodyB arrays, passed in directly since this kernel has no access
// to the real ManifoldSoA/JointSoA/SpringSoA types.
// ---------------------------------------------------------------------------
struct JpRoundArgs
{
    uint64_t *gKey;
    int *gIsA;
    int *gStart;
    int *gEnd;
    int *mBodyA;
    int *mBodyB;
    int *jBodyA;
    int *jBodyB;
    int *sBodyA;
    int *sBodyB;
    float *mass;
    uint8_t *awake;
    int *dyn;
    uint dynCount;
    uint _pad0;
    int *color;
    uint *remaining;
    uint *dynCountPtr; // live-count pointer, see ResetColorsArgs
};

int graphOtherS(JpRoundArgs *args, uint64_t key, int isA)
{
    int idx = graphIndexS(key);
    int type = graphTypeS(key);
    if (type == FORCE_JOINT_S)
        return isA != 0 ? args->jBodyB[idx] : args->jBodyA[idx];
    if (type == FORCE_SPRING_S)
        return isA != 0 ? args->sBodyB[idx] : args->sBodyA[idx];
    return isA != 0 ? args->mBodyB[idx] : args->mBodyA[idx];
}

// Deterministic priority for Jones-Plassmann -- a hash of the body index rather than a
// random draw, so the coloring is reproducible run to run. Ties break on the index
// itself. Bit-for-bit the same finalizer as the constraint builder's jpPriority.
uint jpPriorityS(int i)
{
    uint h = (uint)i;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

bool jpOutranksS(int j, int i)
{
    uint pj = jpPriorityS(j), pi = jpPriorityS(i);
    return pj > pi || (pj == pi && j > i);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csJpRound(uint3 tid: SV_DispatchThreadID)
{
    JpRoundArgs *args = (JpRoundArgs *)pc.argsAddr;
    uint t = tid.x;
    uint n = args->dynCountPtr != nullptr ? min(args->dynCountPtr[0], args->dynCount) : args->dynCount;
    if (t >= n)
        return;
    int i = args->dyn[t];
    if (args->color[i] >= 0)
        return;

    uint64_t used0 = uint64_t(0), used1 = uint64_t(0);

    for (int e = args->gStart[i]; e < args->gEnd[i]; e++)
    {
        uint64_t key = args->gKey[e];
        int isA = args->gIsA[e];
        int other = graphOtherS(args, key, isA);
        if (other < 0 || args->mass[other] <= 0.0f || args->awake[other] == 0)
            continue; // Statics and sleepers are not in the graph: never written.

        int c = args->color[other];
        if (c < 0)
        {
            if (jpOutranksS(other, i))
            {
                uint dummy;
                InterlockedAdd(*args->remaining, 1u, dummy);
                return;
            }
            continue;
        }
        if (c < 64)
            used0 |= (uint64_t(1) << c);
        else if (c < 128)
            used1 |= (uint64_t(1) << (c - 64));
    }

    int c = 0;
    while (c < AVBD_MAX_COLORS_S)
    {
        uint64_t bit = (c < 64) ? (used0 >> c) : (used1 >> (c - 64));
        if ((bit & uint64_t(1)) == uint64_t(0))
            break;
        c++;
    }
    args->color[i] = (c < AVBD_MAX_COLORS_S) ? c : AVBD_MAX_COLORS_S - 1;
}

// ---------------------------------------------------------------------------
// csColorKeys / csConstraintsColorRows -- k_color_keys / k_color_rows. Group the
// dynamic bodies by color, keyed by color alone so within a color bodies land in
// ascending body-index order (matters for the primal sweep's coalesced self-access;
// spatial reordering was tried and rejected).
// ---------------------------------------------------------------------------
struct ColorKeysArgs
{
    int *color;
    int *dyn;
    uint dynCount;
    uint _pad0;
    uint *keys;
    int *vals;
    uint *dynCountPtr; // live-count pointer, see ResetColorsArgs
    // Distance LOD, may be null: key = color * 2 + (body outside the near tier), so
    // within each colour the near bodies come first and a near-only iteration is a
    // prefix of the colour's range (ColorRowsArgs::nearEnd).
    uint8_t *midTag;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csColorKeys(uint3 tid: SV_DispatchThreadID)
{
    ColorKeysArgs *args = (ColorKeysArgs *)pc.argsAddr;
    uint t = tid.x;
    uint n = args->dynCountPtr != nullptr ? min(args->dynCountPtr[0], args->dynCount) : args->dynCount;
    if (t >= n)
        return;
    int i = args->dyn[t];
    int c = args->color[i];
    uint key = (uint)(c < 0 ? 0 : c);
    if (args->midTag != nullptr)
        key = key * 2u + (args->midTag[i] != 0 ? 1u : 0u);
    args->keys[t] = key;
    args->vals[t] = i;
}

struct ColorRowsArgs
{
    uint *keys;
    uint n;
    uint _pad0;
    int *start;
    int *end;
    uint *dynCountPtr; // when set, `n` is a bound and the live count is min(*ptr, n)
    // Distance LOD, may be null: keys carry the near/mid bit (ColorKeysArgs::midTag),
    // the colour is key >> 1, and nearEnd[c] is where colour c's near bodies end.
    int *nearEnd;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csConstraintsColorRows(uint3 tid: SV_DispatchThreadID)
{
    ColorRowsArgs *args = (ColorRowsArgs *)pc.argsAddr;
    uint n = args->dynCountPtr != nullptr ? min(args->dynCountPtr[0], args->n) : args->n;
    uint e = tid.x;
    if (e >= n)
        return;
    uint k = args->keys[e];
    if (args->nearEnd == nullptr)
    {
        if (e == 0 || args->keys[e - 1] != k)
            args->start[k] = (int)e;
        if (e == n - 1 || args->keys[e + 1] != k)
            args->end[k] = (int)e + 1;
        return;
    }
    uint c = k >> 1;
    bool first = e == 0 || (args->keys[e - 1] >> 1) != c;
    bool last = e == n - 1 || (args->keys[e + 1] >> 1) != c;
    if (first)
        args->start[c] = (int)e;
    if (last)
        args->end[c] = (int)e + 1;
    // One writer per colour: the last near body, or, when the colour has none, its
    // first (mid) body. A colour with no bodies keeps the zero fill, as start/end do.
    if ((k & 1u) == 0u && (e == n - 1 || args->keys[e + 1] != k))
        args->nearEnd[c] = (int)e + 1;
    else if ((k & 1u) != 0u && first)
        args->nearEnd[c] = (int)e;
}

// ---------------------------------------------------------------------------
// csBuildSmall -- the small-world constraint build: CSR rows, Jones-Plassmann colouring
// and colour grouping in ONE workgroup / ONE dispatch, replacing the ~10-dispatch
// build (3 emits, a 4/8-pass radix sort, rows, JP rounds, colour keys, a second radix
// sort, colour rows) plus its fills. It writes exactly what that build writes, for the
// same inputs:
//   - rowStart/rowEnd/sortedKeys/sortedIsA: a counting sort by body, then a shell sort
//     of each row on its unique 64-bit key (a sort on unique keys has one result, so it
//     is the radix order). A body with no entries keeps start = end = 0, like the fill.
//   - counter: the TOTAL entry count, including entries past `capacity` (those are not
//     stored, as in the emit kernels; the host reruns the pass with a bigger capacity).
//   - color: Jones-Plassmann is deterministic (a body is coloured once every outranking
//     neighbour is, with the smallest free colour), so the colours equal the dispatch
//     version's whatever the interleaving. Rounds run to convergence in-kernel.
//   - remaining[r] for r < rounds (the host's blind round count): nonzero for every round
//     before the converging one and zero from it on, which is what the dispatch version
//     leaves, so the host's convergence round and the device-side step flag
//     (launchStepFlagIfSet on the last slot) see the same thing.
//   - bodyByColor / colorRanges (start, end, and nearEnd when midTag is set): the stable
//     group by colour (and near-before-mid within a colour), in dyn-list order.
// Gated by the host to dynBound <= BUILD_SMALL_MAX_DYN (the shared bucket array).
// ---------------------------------------------------------------------------
static const uint BUILD_SMALL_GROUP = 1024u;
static const uint BUILD_SMALL_MAX_DYN = 2048u;
groupshared uint sBsChunk[1024];
groupshared uint sBsFlag;
groupshared uint sBsRounds;
groupshared uint sBsBucket[2048];
groupshared int sBsCnt[256];
groupshared int sBsStart[256];

struct BuildSmallArgs
{
    int *mBodyA;
    int *mBodyB;
    int *mCount;
    int *jBodyA;
    int *jBodyB;
    uint8_t *jBroken; // may be null (no fracture tracked)
    int *sBodyA;
    int *sBodyB;
    float *mass;
    uint8_t *awake;
    int *dyn;
    uint64_t *keys;
    int *isA;
    int *rowStart;
    int *rowEnd;
    int *color;
    int *bodyByColor;
    int *ranges;       // start[128], end[128], nearEnd[128]
    int *counter;
    uint *remaining;
    uint *dynCountPtr; // may be null: dynCount is exact
    uint8_t *midTag;   // may be null
    uint slots;
    uint jointSlots;
    uint springSlots;
    uint bodyCount;
    uint dynCount;
    int capacity;
    uint rounds;       // blind round slots of `remaining` to fill
    uint _pad0;
};

bool bsLive(BuildSmallArgs *a, int b)
{
    return b >= 0 && a->mass[b] > 0.0f && a->awake[b] != 0;
}

// One entry for (body, type, index) when the body is live: counted (store = false) or
// stored into its row (store = true).
void bsEntry(BuildSmallArgs *a, int body, int type, int index, int isA, bool store)
{
    if (!bsLive(a, body))
        return;
    if (!store)
    {
        InterlockedAdd(a->rowEnd[body], 1);
        return;
    }
    int p;
    InterlockedAdd(a->rowEnd[body], 1, p);
    if (p < a->capacity)
    {
        a->keys[p] = makeGraphKeyS(body, type, index);
        a->isA[p] = isA;
    }
}

void bsEmitAll(BuildSmallArgs *a, uint lane, bool store)
{
    for (uint s = lane; s < a->slots; s += BUILD_SMALL_GROUP)
    {
        if (a->mCount[s] <= 0)
            continue;
        bsEntry(a, a->mBodyA[s], FORCE_MANIFOLD_S, (int)s, 1, store);
        bsEntry(a, a->mBodyB[s], FORCE_MANIFOLD_S, (int)s, 0, store);
    }
    for (uint j = lane; j < a->jointSlots; j += BUILD_SMALL_GROUP)
    {
        if (a->jBroken != nullptr && a->jBroken[j] != 0)
            continue;
        bsEntry(a, a->jBodyA[j], FORCE_JOINT_S, (int)j, 1, store);
        bsEntry(a, a->jBodyB[j], FORCE_JOINT_S, (int)j, 0, store);
    }
    for (uint q = lane; q < a->springSlots; q += BUILD_SMALL_GROUP)
    {
        bsEntry(a, a->sBodyA[q], FORCE_SPRING_S, (int)q, 1, store);
        bsEntry(a, a->sBodyB[q], FORCE_SPRING_S, (int)q, 0, store);
    }
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void csBuildSmall(uint3 gtid: SV_GroupThreadID)
{
    BuildSmallArgs *a = (BuildSmallArgs *)pc.argsAddr;
    const uint lane = gtid.x;
    const uint T = BUILD_SMALL_GROUP;
    const uint dynN = a->dynCountPtr != nullptr ? min(a->dynCountPtr[0], a->dynCount) : a->dynCount;

    // Zero the per-body counts and the colour ranges, reset the colours.
    for (uint b = lane; b < a->bodyCount; b += T)
    {
        a->rowEnd[b] = 0;
        a->rowStart[b] = 0;
    }
    for (uint r = lane; r < 3u * 128u; r += T)
        a->ranges[r] = 0;
    for (uint t = lane; t < dynN; t += T)
        a->color[a->dyn[t]] = -1;
    for (uint c = lane; c < 256u; c += T)
        sBsCnt[c] = 0;
    if (lane == 0u)
    {
        sBsFlag = 0u;
        sBsRounds = 0u;
    }
    AllMemoryBarrierWithGroupSync();

    // Count each body's entries.
    bsEmitAll(a, lane, false);
    AllMemoryBarrierWithGroupSync();

    // Exclusive prefix over the bodies: a chunk per thread, a Hillis-Steele scan of the sums.
    uint per = (a->bodyCount + T - 1u) / T;
    uint lo = min(lane * per, a->bodyCount);
    uint hi = min(lo + per, a->bodyCount);
    uint sum = 0u;
    for (uint b = lo; b < hi; b++)
        sum += (uint)a->rowEnd[b];
    sBsChunk[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1u; off < T; off <<= 1u)
    {
        uint v = lane >= off ? sBsChunk[lane - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sBsChunk[lane] += v;
        GroupMemoryBarrierWithGroupSync();
    }
    uint run = sBsChunk[lane] - sum;
    if (lane == T - 1u)
        a->counter[0] = (int)sBsChunk[lane]; // total, including entries past capacity
    for (uint b = lo; b < hi; b++)
    {
        uint c = (uint)a->rowEnd[b];
        a->rowStart[b] = (int)run;
        a->rowEnd[b] = (int)run; // fill cursor; ends at start + count
        run += c;
    }
    AllMemoryBarrierWithGroupSync();

    // Store the entries into their rows (any order inside a row).
    bsEmitAll(a, lane, true);
    AllMemoryBarrierWithGroupSync();

    // Sort each row by key (shell sort; keys are unique so the order is the radix order),
    // and give an empty row start = end = 0 like the zero fill.
    for (uint b = lane; b < a->bodyCount; b += T)
    {
        int s0 = a->rowStart[b];
        int e0 = a->rowEnd[b];
        if (e0 == s0)
        {
            a->rowStart[b] = 0;
            a->rowEnd[b] = 0;
            continue;
        }
        e0 = min(e0, a->capacity);
        int len = e0 - s0;
        int gaps[8] = {701, 301, 132, 57, 23, 10, 4, 1};
        for (int gi = 0; gi < 8; gi++)
        {
            int gap = gaps[gi];
            if (gap >= len && gap != 1)
                continue;
            for (int i = s0 + gap; i < e0; i++)
            {
                uint64_t k = a->keys[i];
                int f = a->isA[i];
                int j = i;
                while (j - gap >= s0 && a->keys[j - gap] > k)
                {
                    a->keys[j] = a->keys[j - gap];
                    a->isA[j] = a->isA[j - gap];
                    j -= gap;
                }
                a->keys[j] = k;
                a->isA[j] = f;
            }
        }
    }
    AllMemoryBarrierWithGroupSync();

    // Jones-Plassmann to convergence.
    for (;;)
    {
        for (uint t = lane; t < dynN; t += T)
        {
            int i = a->dyn[t];
            if (a->color[i] >= 0)
                continue;
            uint64_t used0 = uint64_t(0), used1 = uint64_t(0);
            bool wait = false;
            int e1 = min(a->rowEnd[i], a->capacity);
            for (int e = a->rowStart[i]; e < e1; e++)
            {
                uint64_t key = a->keys[e];
                int idx = graphIndexS(key);
                int type = graphTypeS(key);
                int isA = a->isA[e];
                int other;
                if (type == FORCE_JOINT_S)
                    other = isA != 0 ? a->jBodyB[idx] : a->jBodyA[idx];
                else if (type == FORCE_SPRING_S)
                    other = isA != 0 ? a->sBodyB[idx] : a->sBodyA[idx];
                else
                    other = isA != 0 ? a->mBodyB[idx] : a->mBodyA[idx];
                if (other < 0 || a->mass[other] <= 0.0f || a->awake[other] == 0)
                    continue;
                int c = a->color[other];
                if (c < 0)
                {
                    if (jpOutranksS(other, i))
                    {
                        wait = true;
                        break;
                    }
                    continue;
                }
                if (c < 64)
                    used0 |= (uint64_t(1) << c);
                else if (c < 128)
                    used1 |= (uint64_t(1) << (c - 64));
            }
            if (wait)
            {
                InterlockedOr(sBsFlag, 1u);
                continue;
            }
            int c = 0;
            while (c < AVBD_MAX_COLORS_S)
            {
                uint64_t bit = (c < 64) ? (used0 >> c) : (used1 >> (c - 64));
                if ((bit & uint64_t(1)) == uint64_t(0))
                    break;
                c++;
            }
            a->color[i] = (c < AVBD_MAX_COLORS_S) ? c : AVBD_MAX_COLORS_S - 1;
        }
        AllMemoryBarrierWithGroupSync();
        uint again = sBsFlag;
        // Round r left `again` uncoloured: the value the dispatch version holds in slot r.
        if (lane == 0u)
        {
            uint r = sBsRounds;
            if (r < a->rounds)
                a->remaining[r] = again;
            sBsRounds = r + 1u;
        }
        AllMemoryBarrierWithGroupSync();
        if (again == 0u)
            break;
        if (lane == 0u)
            sBsFlag = 0u;
        AllMemoryBarrierWithGroupSync();
    }
    // Slots past the converging round read zero in the dispatch version too.
    for (uint r = sBsRounds + lane; r < a->rounds; r += T)
        a->remaining[r] = 0u;

    // Group by colour (bucket = colour * 2 + outside-near bit), stable in dyn order.
    for (uint t = lane; t < dynN; t += T)
    {
        int i = a->dyn[t];
        int c = max(a->color[i], 0);
        uint bk = (uint)c * 2u + ((a->midTag != nullptr && a->midTag[i] != 0) ? 1u : 0u);
        sBsBucket[t] = bk;
        InterlockedAdd(sBsCnt[bk], 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u)
    {
        int acc = 0;
        for (int k = 0; k < 256; k++)
        {
            sBsStart[k] = acc;
            acc += sBsCnt[k];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < 256u)
    {
        int n = sBsCnt[lane];
        if (n > 0)
        {
            int p = sBsStart[lane];
            for (uint t = 0; t < dynN; t++)
                if (sBsBucket[t] == lane)
                    a->bodyByColor[p++] = a->dyn[t];
        }
    }
    if (lane < 128u)
    {
        int c = (int)lane;
        int nNear = sBsCnt[2 * c];
        int nAll = nNear + sBsCnt[2 * c + 1];
        if (nAll > 0)
        {
            a->ranges[c] = sBsStart[2 * c];
            a->ranges[128 + c] = sBsStart[2 * c] + nAll;
            if (a->midTag != nullptr)
                a->ranges[256 + c] = sBsStart[2 * c] + nNear;
        }
    }
}
