// 2D broadphase, on shapes (a body owns one or more; the shapes of one body never pair). The
// grid list (dynamic shapes, then the statics no bigger than the largest of them;
// vk_world2d.cpp uploadLists) is hashed into a uniform grid of linked lists (cell size at
// least the largest grid body's bounding diameter, so overlapping bodies sit in the same or an
// adjacent cell) and each dynamic grid body walks the cells around it. Statics too big for the
// cell (the ground), few and huge, are tested by brute force, and so are the few dynamic bodies
// far bigger than the rest (csOversizePairs2D), which would otherwise blow the cell up for all.
//
// Bounds are swept, because the narrowphase collides at the x^t positions with a speculative
// reach of this step's relative motion (narrowphase2d.slang): a pair is reported when the two
// boxes (at x^t, at the warm-started angles) overlap at some point of their *relative* sweep
// from x^t to the warm-started pose. A body falling at 50 m/s moves twice its own size per step,
// so a test at the warm pose alone lets it pass through the pile unseen; and a block of bodies
// falling together has no relative sweep, so it adds no pairs. A body is binned by the midpoint
// of its sweep and walks 1 + ceil(sweep / cell) rings of cells (at most MAX_RING); of two
// candidates the one with the wider walk reports the pair (the lower index on a tie), because
// the wider walk is the one guaranteed to reach the other. Pairs are appended with an atomic
// counter that always counts, so the host sees how many were required even when the buffer was
// too small, and the keys are then sorted by the shared radix sort.

import maths2d;
import types2d;
import collide2d;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const float PAIR_PAD = 0.03f; // AABB slack: a little over the collision margin
// Candidates one body will examine. A healthy neighbourhood holds tens; a diverged pile can put
// thousands on one point, and an unbounded walk would turn that into an N^2 kernel that stalls
// the whole machine. Past the cap the rest of the cell is ignored (the guard removes the culprit).
static const int MAX_VISITS = 1024;
// Widest walk, in rings of cells around the body's own (1 = the 3x3 block). A body sweeping
// further than MAX_RING - 1 cells in one step is only partly protected.
static const int MAX_RING = 4;

// Half extents of shape i's axis-aligned bounding box when the body is at angle `a`.
vec2 aabbHalfAt(World2 *w, uint i, float a)
{
    vec2 h = w->sHalf[i];
    if (w->sType[i] == SHAPE_CIRCLE)
        return vec2(h.x, h.x);
    a = a + w->sAng[i];
    float c = abs(cos(a));
    float s = abs(sin(a));
    return vec2(c * h.x + s * h.y, s * h.x + c * h.y);
}

// Where shape i sits relative to its body's centre of mass, in world orientation at angle `a`.
vec2 shapeArm(World2 *w, uint i, float a)
{
    vec2 o = w->sOff[i];
    if (o.x == 0.0f && o.y == 0.0f)
        return o;
    return v2rot(a, o);
}

// One shape's sweep: its box at x^t (warm-started angle), the motion to the warm-started pose,
// the midpoint it is binned by, and how many rings of cells it walks.
struct Swept
{
    vec2 p0;  // x^t position
    vec2 s;   // warm-started position minus x^t
    vec2 e;   // half extents at the warm-started angle
    vec2 mid; // p0 + s / 2
    int ring;
};

Swept sweptBox(World2 *w, uint i)
{
    uint b = (uint)w->sBody[i];
    Swept r;
    r.p0 = v2add(w->initPos[b], shapeArm(w, i, w->ang[b]));
    r.s = v2sub(w->pos[b], w->initPos[b]);
    r.e = aabbHalfAt(w, i, w->ang[b]);
    r.mid = v2add(r.p0, v2mul(r.s, 0.5f));
    float reach = max(abs(r.s.x), abs(r.s.y)) / w->cellSize;
    r.ring = finitef(reach) ? min(1 + (int)ceil(reach), MAX_RING) : MAX_RING;
    return r;
}

// Do the boxes overlap anywhere along the relative sweep? Per axis, the relative centre moves
// from d to d + s; the interval test is exact for each axis, so this is a conservative bound.
bool sweptOverlap(Swept a, Swept b)
{
    vec2 s = v2sub(a.s, b.s);
    vec2 d = v2add(v2sub(a.p0, b.p0), v2mul(s, 0.5f));
    return abs(d.x) <= a.e.x + b.e.x + 0.5f * abs(s.x) + PAIR_PAD &&
           abs(d.y) <= a.e.y + b.e.y + 0.5f * abs(s.y) + PAIR_PAD;
}

// A static shape does not move, so this is its box; a kinematic one sweeps from x^t to where its
// own velocity takes it (integrate2d.slang), which sweptBox measures.
Swept staticBox(World2 *w, uint j)
{
    Swept r;
    uint b = (uint)w->sBody[j];
    if ((asuint(w->bProps[b].w) & BODY_KINEMATIC) != 0u)
        return sweptBox(w, j);
    r.p0 = v2add(w->pos[b], shapeArm(w, j, w->ang[b]));
    r.s = vec2(0.0f, 0.0f);
    r.e = aabbHalfAt(w, j, w->ang[b]);
    r.mid = r.p0;
    r.ring = 1;
    return r;
}

int cellCoord(float x, float cell) { return (int)floor(x / cell); }

uint cellSlot(int cx, int cy, uint mask)
{
    uint h = ((uint)cx * 73856093u) ^ ((uint)cy * 19349663u);
    h ^= h >> 15;
    return h & mask;
}

bool isIgnored(World2 *w, uint64_t key)
{
    int lo = 0;
    int hi = (int)w->ignoreCount - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        uint64_t k = w->ignore[mid];
        if (k == key)
            return true;
        if (k < key)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return false;
}

// Box2D's filter: shapes sharing a nonzero group collide when it is positive and never when it is
// negative; otherwise each one's category must be in the other's mask.
bool filterAllows(World2 *w, uint a, uint b)
{
    int ga = w->sGroup[a];
    if (ga != 0 && ga == w->sGroup[b])
        return ga > 0;
    return (w->sMask[b] & w->sCat[a]) != 0 && (w->sMask[a] & w->sCat[b]) != 0;
}

void emitPair(World2 *w, uint a, uint b)
{
    uint lo = min(a, b);
    uint hi = max(a, b);
    uint64_t key = (uint64_t(lo) << 32) | uint64_t(hi);
    uint ba = (uint)w->sBody[a];
    uint bb = (uint)w->sBody[b];
    if (ba == bb)
        return; // a body's own shapes never collide
    if (!filterAllows(w, a, b))
        return;
    // The ignore list is of body pairs.
    if (w->ignoreCount > 0 && isIgnored(w, (uint64_t(min(ba, bb)) << 32) | uint64_t(max(ba, bb))))
        return;
    int idx;
    InterlockedAdd(w->counters[0], 1, idx);
    if ((uint)idx < w->pairCap)
        w->pairsRaw[idx] = key;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csGridClear2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint i = tid.x;
    if (i > w->tableMask)
        return;
    w->gridHead[i] = -1;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csGridInsert2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->gridCount)
        return;
    int i = w->grid[t];
    if (w->sleepTimer[w->sBody[i]] < 0)
        return; // removed bodies are not collidable; sleeping ones are (they act as anchors)
    vec2 p = sweptBox(w, (uint)i).mid;
    uint slot = cellSlot(cellCoord(p.x, w->cellSize), cellCoord(p.y, w->cellSize), w->tableMask);
    int prev;
    InterlockedExchange(w->gridHead[slot], i, prev);
    w->gridNext[i] = prev;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csFindPairs2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint t = tid.x;
    if (t >= w->gridDyn)
        return;
    int i = w->grid[t];
    if (w->sleepTimer[w->sBody[i]] < 0)
        return; // removed bodies are not collidable

    Swept si = sweptBox(w, (uint)i);
    int cx = cellCoord(si.mid.x, w->cellSize);
    int cy = cellCoord(si.mid.y, w->cellSize);

    // Dynamic-dynamic. Every body is binned in exactly one cell, so a candidate is examined only
    // when the cell being walked is its own: that skips both a hash collision's strangers and
    // the repeats of a slot two walked cells share. Sleepers scan too, so a
    // sleeper keeps the contact fingerprint that tells it when its support has changed.
    int visits = 0;
    int R = si.ring;
    for (int dy = -R; dy <= R && visits <= MAX_VISITS; dy++)
    {
        for (int dx = -R; dx <= R && visits <= MAX_VISITS; dx++)
        {
            int qx = cx + dx;
            int qy = cy + dy;
            for (int j = w->gridHead[cellSlot(qx, qy, w->tableMask)]; j >= 0; j = w->gridNext[j])
            {
                if (++visits > MAX_VISITS)
                    break;
                if (j == i || w->sBody[j] == w->sBody[i])
                    continue;
                Swept sj = sweptBox(w, (uint)j);
                if (cellCoord(sj.mid.x, w->cellSize) != qx || cellCoord(sj.mid.y, w->cellSize) != qy)
                    continue;
                // The wider walk reports; on a tie the lower index. A static never walks, so the
                // dynamic body always reports it.
                if (w->mass[w->sBody[j]] > 0.0f && (sj.ring > R || (sj.ring == R && j < i)))
                    continue;
                if (sweptOverlap(si, sj))
                    emitPair(w, (uint)i, (uint)j);
            }
        }
    }

    // Dynamic-static.
    for (uint s = 0; s < w->statCount; s++)
    {
        int j = w->stat[s];
        if (w->sleepTimer[w->sBody[j]] < 0)
            continue; // a disabled static is gone
        Swept sj = staticBox(w, (uint)j);
        if (sweptOverlap(si, sj))
            emitPair(w, (uint)i, (uint)j);
    }
}

// Oversize dynamic bodies, by brute force: one thread per (oversize body, candidate), the
// candidates being every grid body, every big static and the oversize bodies after this one.
[shader("compute")]
[numthreads(128, 1, 1)]
void csOversizePairs2D(uint3 tid: SV_DispatchThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint per = w->gridCount + w->statCount + w->overCount;
    uint t = tid.x;
    if (per == 0 || t >= w->overCount * per)
        return;
    uint o = t / per;
    uint c = t - o * per;
    int i = w->over[o];
    int j;
    if (c < w->gridCount)
        j = w->grid[c];
    else if (c < w->gridCount + w->statCount)
        j = w->stat[c - w->gridCount];
    else
    {
        uint q = c - w->gridCount - w->statCount;
        if (q <= o)
            return; // each oversize pair once
        j = w->over[q];
    }
    if (w->sleepTimer[w->sBody[i]] < 0 || w->sleepTimer[w->sBody[j]] < 0)
        return;
    Swept si = sweptBox(w, (uint)i);
    Swept sj = w->mass[w->sBody[j]] > 0.0f ? sweptBox(w, (uint)j) : staticBox(w, (uint)j);
    if (sweptOverlap(si, sj))
        emitPair(w, (uint)i, (uint)j);
}

// ---------------------------------------------------------------------------
// The pair-key sort for a small world, in ONE workgroup (replaces the ~50-dispatch radix sort when
// the pair capacity is at most SORT_SMALL_MAX). The keys are a multiset, so any correct sort writes
// the same sorted[0, live): equal keys are indistinguishable. Slots past the live count are not
// written, as in the radix path. A bitonic network over a power-of-two padded with the maximum
// key; every thread runs every loop, so the barriers are uniform.
// ---------------------------------------------------------------------------
static const uint SORT_SMALL_MAX = 4096u;
static const uint SORT_GROUP = 1024u;
groupshared uint64_t sSortKeys[4096];

[shader("compute")]
[numthreads(1024, 1, 1)]
void csSortPairsSmall2D(uint3 gtid: SV_GroupThreadID)
{
    Args2 *args = (Args2 *)pc.argsAddr;
    World2 *w = args->w;
    uint lane = gtid.x;
    uint live = min((uint)w->counters[0], w->pairCap);
    uint size = 1u;
    while (size < live)
        size <<= 1u;
    for (uint i = lane; i < size; i += SORT_GROUP)
        sSortKeys[i] = i < live ? w->pairsRaw[i] : ~(uint64_t)0;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 2u; k <= size; k <<= 1u)
    {
        for (uint j = k >> 1u; j > 0u; j >>= 1u)
        {
            for (uint i = lane; i < size; i += SORT_GROUP)
            {
                uint l = i ^ j;
                if (l > i)
                {
                    uint64_t a = sSortKeys[i];
                    uint64_t b = sSortKeys[l];
                    bool up = (i & k) == 0u;
                    if (up ? (a > b) : (a < b))
                    {
                        sSortKeys[i] = b;
                        sSortKeys[l] = a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for (uint i = lane; i < live; i += SORT_GROUP)
        w->pairsSorted[i] = sSortKeys[i];
}
