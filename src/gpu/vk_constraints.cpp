#include "vk_constraints.h"
#include "vk_args.h"

namespace avbdvk
{

// Every entry point of this module, named once; init and destroy walk the same table.
static std::array<PipelineEntry, 8> entries(ConstraintsPipelines &p)
{
    return {{
        {&p.resetColors, "csResetColors"},
        {&p.emitManifold, "csEmitManifoldEntries"},
        {&p.emitForce, "csEmitForceEntries"},
        {&p.rows, "csConstraintsRows"},
        {&p.jpRound, "csJpRound"},
        {&p.colorKeys, "csColorKeys"},
        {&p.colorRows, "csConstraintsColorRows"},
        {&p.buildSmall, "csBuildSmall"},
    }};
}

void ConstraintsPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    initPipelines(dev, layout, spirv, spirvWords, entries(*this));
}

void ConstraintsPipelines::destroy()
{
    destroyPipelines(entries(*this));
}

// Device-side argument layout lives in vk_args.h (Cs*ArgsGpu), so every
// dispatched struct is defined exactly once (see that header's comment).

void VkConstraints::init(Device &dev)
{
    m_dev = &dev;
    m_ring.init(dev, 1 << 20);
    m_cmd.init(dev, 4);

    rawKeys.init(dev);
    sortedKeys.init(dev);
    rawIsA.init(dev);
    sortedIsA.init(dev);
    rowStart.init(dev);
    rowEnd.init(dev);
    counter.init(dev);

    color.init(dev);
    colorKey.init(dev);
    colorKeySorted.init(dev);
    bodyByColor.init(dev);
    bodyScratch.init(dev);
    colorRanges.init(dev);
    remaining.init(dev);
    readout.init(dev, (VkDeviceSize)(1 + kMaxJpRounds + 2 * AVBD_MAX_COLORS_VK) * sizeof(int32_t),
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
}

void VkConstraints::destroy()
{
    release();
    readout.destroy();
    m_cmd.destroy();
    m_ring.destroy();
}

void VkConstraints::release()
{
    rawKeys.destroy();
    sortedKeys.destroy();
    rawIsA.destroy();
    sortedIsA.destroy();
    rowStart.destroy();
    rowEnd.destroy();
    counter.destroy();

    color.destroy();
    colorKey.destroy();
    colorKeySorted.destroy();
    bodyByColor.destroy();
    bodyScratch.destroy();
    colorRanges.destroy();
    remaining.destroy();
}

void VkConstraints::reserve(int maxBodies_, int maxManifolds, int maxForces)
{
    maxBodies = maxBodies_;
    maxEntries = (maxManifolds + maxForces) * 2;
    if (maxEntries < 1)
        maxEntries = 1; // Buffer::resize(0) is legal but pointless; keep it >=1 so a
                        // capacity check (base + n > capacity) is always meaningful.

    rawKeys.resize((VkDeviceSize)maxEntries * sizeof(uint64_t));
    sortedKeys.resize((VkDeviceSize)maxEntries * sizeof(uint64_t));
    rawIsA.resize((VkDeviceSize)maxEntries * sizeof(int32_t));
    sortedIsA.resize((VkDeviceSize)maxEntries * sizeof(int32_t));
    rowStart.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    rowEnd.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    counter.resize(sizeof(int32_t));

    color.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    colorKey.resize((VkDeviceSize)maxBodies * sizeof(uint32_t));
    colorKeySorted.resize((VkDeviceSize)maxBodies * sizeof(uint32_t));
    bodyByColor.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    bodyScratch.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    // start, end, then the LOD near-only ends (nearEndAddr()).
    colorRanges.resize((VkDeviceSize)3 * AVBD_MAX_COLORS_VK * sizeof(int32_t));
    remaining.resize((VkDeviceSize)kMaxJpRounds * sizeof(uint32_t));

    hColorRanges.assign(2 * AVBD_MAX_COLORS_VK, 0);
}

void VkConstraints::build(ConstraintsPipelines &pl, PrimitivesPipelines &prim, RadixScratch &radix,
                          const VkConstraintsBodyRefs &bodies, const VkConstraintsManifoldRefs &manifolds,
                          const VkConstraintsJointRefs &joints, const VkConstraintsSpringRefs &springs,
                          const Buffer &dynamicIdx, int dynamicCount, VkDeviceAddress dynamicCountAddr)
{
    entries = 0;
    colors = 0;
    rounds = 0;

    // Emit, sort and rows share one submission; the sort's scratch must be grown before
    // anything is recorded (see reserveRadix), and the colour sort's too, since the JP
    // tail below records into the same RadixScratch.
    reserveRadix(radix, (uint32_t)maxEntries, 8, 4);
    reserveRadix(radix, (uint32_t)dynamicCount, 4, 4);

    m_ring.reset();
    m_cmd.begin();
    int first;
    if (smallBuildEligible(dynamicCount))
    {
        first = recordSmall(m_cmd, m_ring, pl, bodies, manifolds, joints, springs, dynamicIdx, dynamicCount,
                            dynamicCountAddr, jpBound());
    }
    else
    {
        smallSortStats().multiBuilds++;
        record(m_cmd, m_ring, pl, prim, radix, bodies, manifolds, joints, springs, dynamicIdx, dynamicCount,
               dynamicCountAddr);
        first = recordColoring(m_cmd, m_ring, pl, prim, radix, bodies, manifolds, joints, springs, dynamicIdx,
                               dynamicCount, 0, jpBound(), dynamicCountAddr);
    }
    m_cmd.submit();
    m_cmd.wait();
    finish();

    // Rare path: the bound was too small (a scene switch, a sudden pile-up). JP resumes
    // from the colours already on device -- extra rounds are idempotent -- and the
    // grouping is redone, until it converges or hits the same cap the host loop had.
    int recorded = first;
    while (dynamicCount > 0 && rounds < 0 && recorded < kMaxJpRounds)
    {
        const int more = std::min(recorded, kMaxJpRounds - recorded);
        m_ring.reset();
        m_cmd.begin();
        recorded += recordColoring(m_cmd, m_ring, pl, prim, radix, bodies, manifolds, joints, springs, dynamicIdx,
                                   dynamicCount, recorded, more, dynamicCountAddr);
        m_cmd.submit();
        m_cmd.wait();
        finishColoring(recorded);
    }
    if (rounds < 0)
        rounds = recorded; // Cap hit without converging: same outcome as the host loop.
    m_lastRounds = rounds;
    m_peakRounds = std::max((float)rounds, m_peakRounds * 0.98f);
}

void VkConstraints::record(CommandList &cmd, HostRingBuffer &ring, ConstraintsPipelines &pl, PrimitivesPipelines &prim,
                           RadixScratch &radix, const VkConstraintsBodyRefs &bodies,
                           const VkConstraintsManifoldRefs &manifolds, const VkConstraintsJointRefs &joints,
                           const VkConstraintsSpringRefs &springs, const Buffer &dynamicIdx, int dynamicCount,
                           VkDeviceAddress dynamicCountAddr)
{
    const uint32_t block = 256;
    m_recordedRounds = 0;

    // Recorded, not the old synchronous zero()s: each of those was a submit+wait.
    cmd.fill(rowStart, 0u);
    cmd.fill(rowEnd, 0u);
    cmd.fill(colorRanges, 0u);
    cmd.fill(counter, 0u, sizeof(int32_t));

    if (dynamicCount == 0)
        return;

    const int forceCount = joints.count + springs.count;

    // ---- Reset colors, then emit every CSR entry (manifolds, joints, springs) into
    // one shared counter -- same shape as Constraints::build's first block. ----
    CsResetColorsArgsGpu rc{color.address(), dynamicIdx.address(), (uint32_t)dynamicCount, 0u, dynamicCountAddr};
    cmd.dispatchArgs(pl.resetColors, (dynamicCount + block - 1) / block, ring, rc);

    if (manifolds.slots > 0 || forceCount > 0)
    {
        if (manifolds.slots > 0)
        {
            CsEmitManifoldArgsGpu em{};
            em.mBodyA = manifolds.bodyA;
            em.mBodyB = manifolds.bodyB;
            em.mContactCount = manifolds.contactCount;
            em.slots = (uint32_t)manifolds.slots;
            em.mass = bodies.mass;
            em.awake = bodies.awake;
            em.keys = rawKeys.address();
            em.isA = rawIsA.address();
            em.counter = counter.address();
            em.capacity = maxEntries;
            cmd.dispatchArgs(pl.emitManifold, ((uint32_t)manifolds.slots + block - 1) / block, ring, em);
        }
        if (joints.count > 0)
        {
            CsEmitForceArgsGpu ej{};
            ej.fBodyA = joints.bodyA;
            ej.fBodyB = joints.bodyB;
            ej.count = (uint32_t)joints.count;
            ej.broken = joints.broken;
            ej.mass = bodies.mass;
            ej.awake = bodies.awake;
            ej.keys = rawKeys.address();
            ej.isA = rawIsA.address();
            ej.counter = counter.address();
            ej.capacity = maxEntries;
            ej.type = 1; // FORCE_JOINT
            cmd.dispatchArgs(pl.emitForce, ((uint32_t)joints.count + block - 1) / block, ring, ej);
        }
        if (springs.count > 0)
        {
            CsEmitForceArgsGpu es{};
            es.fBodyA = springs.bodyA;
            es.fBodyB = springs.bodyB;
            es.count = (uint32_t)springs.count;
            es.broken = 0;
            es.mass = bodies.mass;
            es.awake = bodies.awake;
            es.keys = rawKeys.address();
            es.isA = rawIsA.address();
            es.counter = counter.address();
            es.capacity = maxEntries;
            es.type = 2; // FORCE_SPRING
            cmd.dispatchArgs(pl.emitForce, ((uint32_t)springs.count + block - 1) / block, ring, es);
        }
        cmd.barrier();

        // ---- Sort entries by (body,type,index), then derive per-body row ranges. The
        // entry count stays on device: both run over maxEntries and read the live count
        // from the emit counter, clamped to the capacity exactly as the old readback was.
        radixSortPairsU64(cmd, ring, prim, radix, rawKeys, sortedKeys, rawIsA, sortedIsA, (uint32_t)maxEntries,
                          0xFFFFFFFFu, counter.address());
        cmd.barrier();

        CsRowsArgsGpu ra{sortedKeys.address(), (uint32_t)maxEntries, 0u,
                         rowStart.address(),   rowEnd.address(),     counter.address()};
        cmd.dispatchArgs(pl.rows, ((uint32_t)maxEntries + block - 1) / block, ring, ra);
        cmd.barrier();
    }
    m_recorded = true;
}

void VkConstraints::finish()
{
    entries = 0;
    if (!m_recorded)
        return;
    m_recorded = false;
    int32_t e = 0;
    counter.readback(&e, sizeof(int32_t));
    entries = e < maxEntries ? e : maxEntries;
    finishColoring(m_recordedRounds);
}

void VkConstraints::finishColoring(int recordedRounds)
{
    colors = 0;
    rounds = 0;
    if (recordedRounds <= 0)
        return;

    // rounds = the first round that left nothing uncoloured, +1; -1 if none did.
    std::vector<uint32_t> left((size_t)recordedRounds);
    remaining.readback(left.data(), (VkDeviceSize)recordedRounds * sizeof(uint32_t));
    rounds = -1;
    for (int r = 0; r < recordedRounds; r++)
        if (left[r] == 0)
        {
            rounds = r + 1;
            break;
        }

    // One contiguous readback: start and end live in the same allocation.
    colorRanges.readback(hColorRanges.data(), (VkDeviceSize)2 * AVBD_MAX_COLORS_VK * sizeof(int32_t));
    for (int c = 0; c < AVBD_MAX_COLORS_VK; c++)
        if (hColorRanges[AVBD_MAX_COLORS_VK + c] > hColorRanges[c])
            colors = c + 1;
}

int VkConstraints::recordColoring(CommandList &cmd, HostRingBuffer &ring, ConstraintsPipelines &pl,
                                  PrimitivesPipelines &prim, RadixScratch &radix, const VkConstraintsBodyRefs &bodies,
                                  const VkConstraintsManifoldRefs &manifolds, const VkConstraintsJointRefs &joints,
                                  const VkConstraintsSpringRefs &springs, const Buffer &dynamicIdx, int dynamicCount,
                                  int firstRound, int count, VkDeviceAddress dynamicCountAddr)
{
    const uint32_t block = 256;
    m_recordedRounds = 0;
    if (dynamicCount == 0 || count <= 0)
        return 0;

    // ---- Jones-Plassmann, `count` rounds back to back. Each round counts the bodies
    // it left uncoloured into its OWN slot of `remaining`, so one readback afterwards
    // says which round converged. Past convergence a round is a no-op (every body is
    // coloured and returns at once), so over-recording changes no colour: the result
    // is exactly the host loop's "dispatch until the counter reads zero". ----
    if (firstRound == 0)
        cmd.fill(remaining, 0u);
    VkConstraintGraph g = graph();
    for (int r = 0; r < count; r++)
    {
        CsJpRoundArgsGpu jp{};
        jp.gKey = g.key;
        jp.gIsA = g.isA;
        jp.gStart = g.start;
        jp.gEnd = g.end;
        jp.mBodyA = manifolds.bodyA;
        jp.mBodyB = manifolds.bodyB;
        jp.jBodyA = joints.bodyA;
        jp.jBodyB = joints.bodyB;
        jp.sBodyA = springs.bodyA;
        jp.sBodyB = springs.bodyB;
        jp.mass = bodies.mass;
        jp.awake = bodies.awake;
        jp.dyn = dynamicIdx.address();
        jp.dynCount = (uint32_t)dynamicCount;
        jp.color = color.address();
        jp.remaining = remaining.address() + (VkDeviceAddress)(firstRound + r) * sizeof(uint32_t);
        jp.dynCountPtr = dynamicCountAddr;
        cmd.dispatchArgs(pl.jpRound, (dynamicCount + block - 1) / block, ring, jp);
        cmd.barrier();
    }

    // ---- Group dynamic bodies by color so each color is one contiguous dispatch. ----
    CsColorKeysArgsGpu ck{color.address(),    dynamicIdx.address(),  (uint32_t)dynamicCount, 0u,
                          colorKey.address(), bodyScratch.address(), dynamicCountAddr,       m_midTag};
    cmd.dispatchArgs(pl.colorKeys, (dynamicCount + block - 1) / block, ring, ck);
    cmd.barrier();

    radixSortPairsU32(cmd, ring, prim, radix, colorKey, colorKeySorted, bodyScratch, bodyByColor,
                      (uint32_t)dynamicCount, dynamicCountAddr);
    cmd.barrier();

    // colorRows writes only the colours present, so a rerun must clear the last one's.
    if (firstRound > 0)
        cmd.fill(colorRanges, 0u);
    VkDeviceAddress startAddr = colorRanges.address();
    VkDeviceAddress endAddr = colorRanges.address() + (VkDeviceAddress)(AVBD_MAX_COLORS_VK * sizeof(int32_t));
    CsColorRowsArgsGpu cr{colorKeySorted.address(), (uint32_t)dynamicCount,      0u, startAddr, endAddr,
                          dynamicCountAddr,         m_midTag ? nearEndAddr() : 0};
    cmd.dispatchArgs(pl.colorRows, (dynamicCount + block - 1) / block, ring, cr);
    cmd.barrier();

    m_recordedRounds = firstRound + count;
    return count;
}

bool VkConstraints::smallBuildEligible(int dynamicBound) const
{
    return smallWorldEnabled() && dynamicBound > 0 && dynamicBound <= kSmallBuildMaxDyn &&
           maxBodies <= kSmallBuildMaxBodies;
}

int VkConstraints::recordSmall(CommandList &cmd, HostRingBuffer &ring, ConstraintsPipelines &pl,
                               const VkConstraintsBodyRefs &bodies, const VkConstraintsManifoldRefs &manifolds,
                               const VkConstraintsJointRefs &joints, const VkConstraintsSpringRefs &springs,
                               const Buffer &dynamicIdx, int dynamicBound, VkDeviceAddress dynamicCountAddr,
                               int blindRounds)
{
    smallSortStats().smallBuilds++;
    CsBuildSmallArgsGpu a{};
    a.mBodyA = manifolds.bodyA;
    a.mBodyB = manifolds.bodyB;
    a.mCount = manifolds.contactCount;
    a.jBodyA = joints.bodyA;
    a.jBodyB = joints.bodyB;
    a.jBroken = joints.broken;
    a.sBodyA = springs.bodyA;
    a.sBodyB = springs.bodyB;
    a.mass = bodies.mass;
    a.awake = bodies.awake;
    a.dyn = dynamicIdx.address();
    a.keys = sortedKeys.address();
    a.isA = sortedIsA.address();
    a.rowStart = rowStart.address();
    a.rowEnd = rowEnd.address();
    a.color = color.address();
    a.bodyByColor = bodyByColor.address();
    a.ranges = colorRanges.address();
    a.counter = counter.address();
    a.remaining = remaining.address();
    a.dynCountPtr = dynamicCountAddr;
    a.midTag = m_midTag;
    a.slots = (uint32_t)std::max(manifolds.slots, 0);
    a.jointSlots = (uint32_t)std::max(joints.count, 0);
    a.springSlots = (uint32_t)std::max(springs.count, 0);
    a.bodyCount = (uint32_t)maxBodies;
    a.dynCount = (uint32_t)dynamicBound;
    a.capacity = maxEntries;
    a.rounds = (uint32_t)blindRounds;
    cmd.barrier(); // producers of the manifolds, bodies and dyn list (the multi path opens with a fill)
    cmd.dispatchArgs(pl.buildSmall, 1, ring, a);
    cmd.barrier();
    m_recorded = true;
    m_recordedRounds = blindRounds;
    return blindRounds;
}

void VkConstraints::prepare(RadixScratch &radix, int dynamicBound)
{
    reserveRadix(radix, (uint32_t)maxEntries, 8, 4);
    reserveRadix(radix, (uint32_t)dynamicBound, 4, 4);
}

int VkConstraints::recordBuild(CommandList &cmd, HostRingBuffer &ring, ConstraintsPipelines &pl,
                               PrimitivesPipelines &prim, RadixScratch &radix, const VkConstraintsBodyRefs &bodies,
                               const VkConstraintsManifoldRefs &manifolds, const VkConstraintsJointRefs &joints,
                               const VkConstraintsSpringRefs &springs, const Buffer &dynamicIdx, int dynamicBound,
                               VkDeviceAddress dynamicCountAddr, int jpRounds)
{
    int recorded;
    if (smallBuildEligible(dynamicBound))
    {
        recorded = recordSmall(cmd, ring, pl, bodies, manifolds, joints, springs, dynamicIdx, dynamicBound,
                               dynamicCountAddr, jpRounds);
    }
    else
    {
        smallSortStats().multiBuilds++;
        record(cmd, ring, pl, prim, radix, bodies, manifolds, joints, springs, dynamicIdx, dynamicBound,
               dynamicCountAddr);
        recorded = recordColoring(cmd, ring, pl, prim, radix, bodies, manifolds, joints, springs, dynamicIdx,
                                  dynamicBound, 0, jpRounds, dynamicCountAddr);
    }
    m_readoutRounds = recorded;
    cmd.barrier();
    // With no dynamic bodies record() returned early: the counter and ranges were still
    // zero-filled, and no rounds were recorded, so the same three copies stay valid.
    cmd.copyToHost(counter, readout, sizeof(int32_t), 0, 0);
    if (recorded > 0)
        cmd.copyToHost(remaining, readout, (VkDeviceSize)recorded * sizeof(uint32_t), 0, sizeof(int32_t));
    cmd.copyToHost(colorRanges, readout, (VkDeviceSize)2 * AVBD_MAX_COLORS_VK * sizeof(int32_t), 0,
                   (VkDeviceSize)(1 + kMaxJpRounds) * sizeof(int32_t));
    return recorded;
}

void VkConstraints::finishRecorded()
{
    entries = 0;
    colors = 0;
    rounds = 0;
    m_jpConverged = true;
    const bool recorded = m_recorded;
    m_recorded = false;
    if (!recorded)
    {
        m_lastRounds = 0; // dynamicBound == 0: nothing was built, as build() ends
        return;
    }
    const int32_t *ro = readout.mappedAs<int32_t>();
    entries = ro[0] < maxEntries ? ro[0] : maxEntries;

    const int n = m_readoutRounds;
    if (n <= 0)
        return;
    rounds = -1;
    for (int r = 0; r < n; r++)
        if (ro[1 + r] == 0)
        {
            rounds = r + 1;
            break;
        }
    std::copy(ro + 1 + kMaxJpRounds, ro + 1 + kMaxJpRounds + 2 * AVBD_MAX_COLORS_VK, hColorRanges.begin());
    for (int c = 0; c < AVBD_MAX_COLORS_VK; c++)
        if (hColorRanges[AVBD_MAX_COLORS_VK + c] > hColorRanges[c])
            colors = c + 1;
    m_jpConverged = rounds >= 0;
    if (rounds < 0)
        rounds = n; // As build(): an unconverged result is the round count recorded.
    m_lastRounds = rounds;
    m_peakRounds = std::max((float)rounds, m_peakRounds * 0.98f);
}

} // namespace avbdvk
