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
    uint *dynCountPtr; // M4: when set, dynCount is a bound and the live count is min(*ptr, dynCount)
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
    // M4: when set, `n` is the capacity and the live entry count is min(*count, n).
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
    uint *dynCountPtr; // M4: live-count pointer, see ResetColorsArgs
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
    uint *dynCountPtr; // M4: live-count pointer, see ResetColorsArgs
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
    uint *dynCountPtr; // M4: when set, `n` is a bound and the live count is min(*ptr, n)
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
