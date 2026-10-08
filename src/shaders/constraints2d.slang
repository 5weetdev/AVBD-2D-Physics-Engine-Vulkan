// Small-world constraint graph for the 2D solver: the CSR body->constraint rows, the
// Jones-Plassmann colouring and the colour grouping, built by ONE workgroup in ONE dispatch.
//
// It produces exactly what the shared multi-dispatch build does (emit, radix sort by the
// (body, type, index) key, row ranges, JP rounds, stable sort by colour): the same sorted keys,
// the same row ranges, the same colour per body and the same per-colour body lists.
//   - Rows: a counting sort by body (count, prefix, scatter) followed by an insertion sort of
//     each row on its unique 64-bit key. A sort on a unique key has one result.
//   - Colours: Jones-Plassmann is deterministic. A body is coloured only once every neighbour
//     that outranks it is coloured, so the result equals greedy colouring in descending
//     (priority, index) order whatever the interleaving; rounds run to convergence in-kernel
//     (so the host never sees a short JP bound) with the same smallest-free-colour rule and the
//     same clamp to the last colour.
//   - Grouping: stable by colour in dyn-list order, like the stable radix sort it replaces.
// The colour count lands in counters[5].

module constraints2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct BuildArgs2
{
    int *mBodyA;
    int *mBodyB;
    int *mCount;      // contacts per manifold slot
    int *jBodyA;
    int *jBodyB;
    uint8_t *jBroken;
    float *mass;
    uint8_t *awake;
    int *dyn;
    uint64_t *keys;   // sorted entries out
    int *isA;
    int *rowStart;
    int *rowEnd;
    int *color;
    int *colorBodies;
    int *colorStart;  // start[128], end[128] follow at colorStart + 128
    int *counters;
    uint slots;
    uint jointSlots;
    uint bodyCount;
    uint dynCount;
    int capacity;     // entries the key/isA arrays hold
    int _pad0;
};

static const uint BUILD_GROUP = 1024u;
static const int MAXC = 128;
groupshared uint sChunk[BUILD_GROUP];
groupshared uint sFlag;
groupshared int sCnt[128];
groupshared int sStart[128];
groupshared uint sMaxColor;

uint64_t makeKey(int body, int type, int index)
{
    return (uint64_t(uint(body)) << 32) | (uint64_t(uint(type)) << 29) | uint64_t(uint(index));
}

uint jpPrio(int i)
{
    uint h = (uint)i;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

bool outranks(int j, int i)
{
    uint pj = jpPrio(j), pi = jpPrio(i);
    return pj > pi || (pj == pi && j > i);
}

bool isLive(BuildArgs2 *a, int b)
{
    return b >= 0 && a->mass[b] > 0.0f && a->awake[b] != 0;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void csBuildSmall2D(uint3 gtid: SV_GroupThreadID)
{
    BuildArgs2 *a = (BuildArgs2 *)pc.argsAddr;
    uint lane = gtid.x;
    uint T = BUILD_GROUP;

    // Zero the per-body counts, reset the colours.
    for (uint b = lane; b < a->bodyCount; b += T)
        a->rowEnd[b] = 0;
    for (uint t = lane; t < a->dynCount; t += T)
        a->color[a->dyn[t]] = -1;
    if (lane < 128u)
        sCnt[lane] = 0;
    if (lane == 0u)
    {
        sFlag = 0u;
        sMaxColor = 0u;
    }
    AllMemoryBarrierWithGroupSync();

    // Count the entries of every body: one per live end of a manifold with contacts / unbroken joint.
    for (uint s = lane; s < a->slots; s += T)
    {
        if (a->mCount[s] <= 0)
            continue;
        int ba = a->mBodyA[s], bb = a->mBodyB[s];
        if (isLive(a, ba))
            InterlockedAdd(a->rowEnd[ba], 1);
        if (isLive(a, bb))
            InterlockedAdd(a->rowEnd[bb], 1);
    }
    for (uint j = lane; j < a->jointSlots; j += T)
    {
        if (a->jBroken[j] != 0)
            continue;
        int ba = a->jBodyA[j], bb = a->jBodyB[j];
        if (isLive(a, ba))
            InterlockedAdd(a->rowEnd[ba], 1);
        if (isLive(a, bb))
            InterlockedAdd(a->rowEnd[bb], 1);
    }
    AllMemoryBarrierWithGroupSync();

    // Exclusive prefix over the bodies: a chunk per thread, a scan of the chunk sums.
    uint per = (a->bodyCount + T - 1u) / T;
    uint lo = min(lane * per, a->bodyCount);
    uint hi = min(lo + per, a->bodyCount);
    uint sum = 0u;
    for (uint b = lo; b < hi; b++)
        sum += (uint)a->rowEnd[b];
    sChunk[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1u; off < T; off <<= 1u)
    {
        uint v = lane >= off ? sChunk[lane - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sChunk[lane] += v;
        GroupMemoryBarrierWithGroupSync();
    }
    uint run = sChunk[lane] - sum;
    for (uint b = lo; b < hi; b++)
    {
        uint c = (uint)a->rowEnd[b];
        a->rowStart[b] = (int)run;
        a->rowEnd[b] = (int)run; // the fill cursor; ends at start + count
        run += c;
    }
    AllMemoryBarrierWithGroupSync();

    // Scatter the entries into their rows (any order inside a row).
    for (uint s = lane; s < a->slots; s += T)
    {
        if (a->mCount[s] <= 0)
            continue;
        int ba = a->mBodyA[s], bb = a->mBodyB[s];
        if (isLive(a, ba))
        {
            int p;
            InterlockedAdd(a->rowEnd[ba], 1, p);
            if (p < a->capacity)
            {
                a->keys[p] = makeKey(ba, 0, (int)s);
                a->isA[p] = 1;
            }
        }
        if (isLive(a, bb))
        {
            int p;
            InterlockedAdd(a->rowEnd[bb], 1, p);
            if (p < a->capacity)
            {
                a->keys[p] = makeKey(bb, 0, (int)s);
                a->isA[p] = 0;
            }
        }
    }
    for (uint j = lane; j < a->jointSlots; j += T)
    {
        if (a->jBroken[j] != 0)
            continue;
        int ba = a->jBodyA[j], bb = a->jBodyB[j];
        if (isLive(a, ba))
        {
            int p;
            InterlockedAdd(a->rowEnd[ba], 1, p);
            if (p < a->capacity)
            {
                a->keys[p] = makeKey(ba, 1, (int)j);
                a->isA[p] = 1;
            }
        }
        if (isLive(a, bb))
        {
            int p;
            InterlockedAdd(a->rowEnd[bb], 1, p);
            if (p < a->capacity)
            {
                a->keys[p] = makeKey(bb, 1, (int)j);
                a->isA[p] = 0;
            }
        }
    }
    AllMemoryBarrierWithGroupSync();

    // Sort each row by key (insertion; rows are short, keys unique).
    for (uint b = lane; b < a->bodyCount; b += T)
    {
        int s0 = a->rowStart[b];
        int e0 = min(a->rowEnd[b], a->capacity);
        for (int i = s0 + 1; i < e0; i++)
        {
            uint64_t k = a->keys[i];
            int f = a->isA[i];
            int j = i - 1;
            while (j >= s0 && a->keys[j] > k)
            {
                a->keys[j + 1] = a->keys[j];
                a->isA[j + 1] = a->isA[j];
                j--;
            }
            a->keys[j + 1] = k;
            a->isA[j + 1] = f;
        }
    }
    AllMemoryBarrierWithGroupSync();

    // Jones-Plassmann to convergence.
    for (;;)
    {
        for (uint t = lane; t < a->dynCount; t += T)
        {
            int i = a->dyn[t];
            if (a->color[i] >= 0)
                continue;
            uint64_t used0 = 0, used1 = 0;
            bool wait = false;
            int e1 = min(a->rowEnd[i], a->capacity);
            for (int e = a->rowStart[i]; e < e1; e++)
            {
                uint64_t key = a->keys[e];
                int idx = (int)((uint)(key & 0xFFFFFFFFu) & ((1u << 29) - 1u));
                int type = (int)((uint)(key & 0xFFFFFFFFu) >> 29);
                int isA = a->isA[e];
                int other;
                if (type == 1)
                    other = isA != 0 ? a->jBodyB[idx] : a->jBodyA[idx];
                else
                    other = isA != 0 ? a->mBodyB[idx] : a->mBodyA[idx];
                if (other < 0 || a->mass[other] <= 0.0f || a->awake[other] == 0)
                    continue;
                int c = a->color[other];
                if (c < 0)
                {
                    if (outranks(other, i))
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
                InterlockedOr(sFlag, 1u);
                continue;
            }
            int c = 0;
            while (c < MAXC)
            {
                uint64_t bit = (c < 64) ? (used0 >> c) : (used1 >> (c - 64));
                if ((bit & uint64_t(1)) == uint64_t(0))
                    break;
                c++;
            }
            a->color[i] = c < MAXC ? c : MAXC - 1;
        }
        AllMemoryBarrierWithGroupSync();
        uint again = sFlag;
        AllMemoryBarrierWithGroupSync();
        if (again == 0u)
            break;
        if (lane == 0u)
            sFlag = 0u;
        AllMemoryBarrierWithGroupSync();
    }

    // Group by colour, stable in dyn order.
    for (uint t = lane; t < a->dynCount; t += T)
    {
        int c = max(a->color[a->dyn[t]], 0);
        InterlockedAdd(sCnt[c], 1);
        InterlockedMax(sMaxColor, (uint)c);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u)
    {
        int acc = 0;
        for (int c = 0; c < MAXC; c++)
        {
            sStart[c] = acc;
            acc += sCnt[c];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < 128u)
    {
        int c = (int)lane;
        int n = sCnt[c];
        a->colorStart[c] = n > 0 ? sStart[c] : 0;
        a->colorStart[128 + c] = n > 0 ? sStart[c] + n : 0;
        int p = sStart[c];
        if (n > 0)
        {
            for (uint t = 0; t < a->dynCount; t++)
            {
                int i = a->dyn[t];
                if (max(a->color[i], 0) == c)
                    a->colorBodies[p++] = i;
            }
        }
    }
    if (lane == 0u)
        a->counters[5] = a->dynCount > 0 ? (int)sMaxColor + 1 : 0;
}
