#include "vk_world2d.h"
#include "vk_world2d_internal.h"

#include "vk_capacity.h"
#include "vk_util.h"

#include "broadphase2d_spirv.h"
#include "ccd2d_spirv.h"
#include "constraints_spirv.h"
#include "constraints2d_spirv.h"
#include "events2d_spirv.h"
#include "integrate2d_spirv.h"
#include "narrowphase2d_spirv.h"
#include "query2d_spirv.h"
#include "sleep2d_spirv.h"
#include "solver2d_spirv.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace avbdvk;

namespace avbd2d
{


QueryView2D VkWorld2D::queryView()
{
    QueryView2D v;
    v.scene = &m_scene;
    v.pos = m_hostPos.data();
    v.ang = m_hostAng.data();
    return v;
}


bool VkWorld2D::castRayClosest(Vec2 origin, Vec2 translation, RayHit2D &out)
{
    return m_init && m_query.castRayClosest(queryView(), origin, translation, out);
}

void VkWorld2D::castRay(Vec2 origin, Vec2 translation, const Query2D::CastFn &fn)
{
    if (m_init)
        m_query.castRay(queryView(), origin, translation, fn);
}

void VkWorld2D::overlapAABB(Vec2 lo, Vec2 hi, const Query2D::ShapeFn &fn)
{
    if (m_init)
        m_query.overlapAABB(queryView(), lo, hi, fn);
}

void VkWorld2D::overlapShape(const Proxy2D &proxy, const Query2D::ShapeFn &fn)
{
    if (m_init)
        m_query.overlapShape(queryView(), proxy, fn);
}

bool VkWorld2D::castShapeClosest(const Proxy2D &proxy, Vec2 translation, RayHit2D &out)
{
    return m_init && m_query.castShapeClosest(queryView(), proxy, translation, out);
}

void VkWorld2D::castShape(const Proxy2D &proxy, Vec2 translation, const Query2D::CastFn &fn)
{
    if (m_init)
        m_query.castShape(queryView(), proxy, translation, fn);
}


// ---------------------------------------------------------------------------
// Batched queries
// ---------------------------------------------------------------------------
namespace
{
void growMapped(MappedBuffer &b, Device &dev, VkDeviceSize bytes, MappedBuffer::HostAccess access)
{
    bytes = std::max<VkDeviceSize>(bytes, 256);
    if (b.size() >= bytes)
        return;
    b.destroy();
    b.init(dev, bytes + bytes / 2, 0, access);
}
} // namespace


// Runs the clear / insert / (rays|boxes) kernels in one submission and waits. The caller has
// checked that no step is in flight. Results land in m_qOutHit / m_qOutIds / m_qOutCounts.
bool VkWorld2D::runQueryBatch(bool rays, const float *in, int count, uint64_t cat, uint64_t mask, int maxPer)
{
    const int shapes = m_scene.shapeTotal();
    const uint32_t insertCount = (uint32_t)(m_gridCount + m_overCount + m_statCount);
    uint32_t table = 64;
    while (table < 2u * (uint32_t)std::max(m_gridCount, 1))
        table <<= 1;

    auto sz = [&](Buffer &b, size_t bytes)
    {
        if (b.size() < bytes)
            b.resize(std::max<size_t>(bytes + bytes / 2, 256));
    };
    sz(m_qBox, (size_t)shapes * 16);
    sz(m_qHead, (size_t)table * 4);
    sz(m_qNext, (size_t)shapes * 16);
    sz(m_qBig, (size_t)shapes * 4);
    sz(m_qCount, 16);
    if (rays)
    {
        growMapped(m_qRays, m_dev, (VkDeviceSize)count * 16, MappedBuffer::HostAccess::Write);
        growMapped(m_qOutHit, m_dev, (VkDeviceSize)count * 32, MappedBuffer::HostAccess::Read);
        growMapped(m_qAabbs, m_dev, 16, MappedBuffer::HostAccess::Write);
        growMapped(m_qOutIds, m_dev, 16, MappedBuffer::HostAccess::Read);
        growMapped(m_qOutCounts, m_dev, 16, MappedBuffer::HostAccess::Read);
        memcpy(m_qRays.mapped(), in, (size_t)count * 16);
    }
    else
    {
        growMapped(m_qAabbs, m_dev, (VkDeviceSize)count * 16, MappedBuffer::HostAccess::Write);
        growMapped(m_qOutIds, m_dev, (VkDeviceSize)count * (size_t)std::max(maxPer, 1) * 4,
                   MappedBuffer::HostAccess::Read);
        growMapped(m_qOutCounts, m_dev, (VkDeviceSize)count * 4, MappedBuffer::HostAccess::Read);
        growMapped(m_qRays, m_dev, 16, MappedBuffer::HostAccess::Write);
        growMapped(m_qOutHit, m_dev, 32, MappedBuffer::HostAccess::Read);
        memcpy(m_qAabbs.mapped(), in, (size_t)count * 16);
    }
    // The kernels write device-local buffers; the copy engine moves them to the mapped ones.
    // Shader stores straight into host-cached memory cost ~1 us each (see csCompactMoved2D).
    const size_t hitBytes = rays ? (size_t)count * 32 : 32;
    const size_t idBytes = rays ? 16 : (size_t)count * (size_t)std::max(maxPer, 1) * 4;
    const size_t countBytes = rays ? 16 : (size_t)count * 4;
    sz(m_qOutHitDev, hitBytes);
    sz(m_qOutIdsDev, idBytes);
    sz(m_qOutCountsDev, countBytes);

    m_ring.reset();
    m_cmd.begin();
    const World2Gpu W = makeWorld(4);
    memcpy(m_worldStage.mapped(), &W, sizeof(W));
    m_cmd.copy(m_worldStage, m_worldDev, sizeof(W));
    m_cmd.barrier();

    QueryBatch2Gpu Q{};
    Q.w = m_worldDev.address();
    Q.qBox = m_qBox.address();
    Q.qHead = m_qHead.address();
    Q.qNext = m_qNext.address();
    Q.qBig = m_qBig.address();
    Q.qCount = m_qCount.address();
    Q.rays = m_qRays.address();
    Q.aabbs = m_qAabbs.address();
    Q.outHit = m_qOutHitDev.address();
    Q.outIds = m_qOutIdsDev.address();
    Q.outCounts = m_qOutCountsDev.address();
    Q.cat = cat;
    Q.mask = mask;
    Q.headMask = table - 1;
    Q.count = (uint32_t)count;
    Q.maxPer = (uint32_t)std::max(maxPer, 1);
    Q.insertCount = insertCount;
    const VkDeviceAddress qa = m_ring.write(&Q, sizeof(Q));
    auto groups = [](uint32_t n, uint32_t g) { return (n + g - 1) / g; };
    m_cmd.dispatch(m_queryClear, groups(table, 256), qa);
    m_cmd.barrier();
    if (insertCount > 0)
    {
        m_cmd.dispatch(m_queryInsert, groups(insertCount, 256), qa);
        m_cmd.barrier();
    }
    m_cmd.dispatch(rays ? m_queryRays : m_queryAabbs, groups((uint32_t)count, 64), qa);
    if (rays)
        m_cmd.copyToHost(m_qOutHitDev, m_qOutHit, hitBytes);
    else
    {
        m_cmd.copyToHost(m_qOutCountsDev, m_qOutCounts, countBytes);
        m_cmd.copyToHost(m_qOutIdsDev, m_qOutIds, idBytes);
    }
    m_cmd.submit();
    m_cmd.wait();
    return true;
}


bool VkWorld2D::castRaysBatch(const RayQuery2D *rays, int count, uint64_t cat, uint64_t mask, RayBatchHit2D *out)
{
    static_assert(sizeof(RayQuery2D) == 16, "RayQuery2D must pack as float4");
    for (int i = 0; i < count; i++)
        out[i] = RayBatchHit2D();
    if (!m_init || count <= 0)
        return false;
    const bool device = !m_stepActive && !uploadsPending() && m_bodyCount > 0 && m_scene.shapeTotal() > 0;
    if (!device)
    {
        for (int i = 0; i < count; i++)
        {
            RayHit2D best;
            bool found = false;
            m_query.castRay(queryView(), rays[i].origin, rays[i].translation,
                            [&](const RayHit2D &h)
                            {
                                const size_t s = (size_t)h.shape;
                                if ((m_scene.shCat[s] & mask) == 0 || (cat & m_scene.shMask[s]) == 0)
                                    return -1.0f;
                                best = h;
                                found = true;
                                return h.fraction;
                            });
            if (found)
            {
                out[i].shape = best.shape;
                out[i].fraction = best.fraction;
                out[i].point = best.point;
                out[i].normal = best.normal;
            }
        }
        return false;
    }
    runQueryBatch(true, reinterpret_cast<const float *>(rays), count, cat, mask, 1);
    const float *r = m_qOutHit.mappedAs<float>();
    for (int i = 0; i < count; i++)
    {
        int shape;
        memcpy(&shape, r + (size_t)i * 8, 4);
        if (shape < 0)
            continue;
        out[i].shape = shape;
        out[i].fraction = r[(size_t)i * 8 + 1];
        out[i].point = Vec2(r[(size_t)i * 8 + 2], r[(size_t)i * 8 + 3]);
        out[i].normal = Vec2(r[(size_t)i * 8 + 4], r[(size_t)i * 8 + 5]);
    }
    return true;
}


bool VkWorld2D::overlapAabbsBatch(const Vec2 *lo, const Vec2 *hi, int count, uint64_t cat, uint64_t mask, int maxPer,
                                  int *outIds, int *outCounts)
{
    for (int i = 0; i < count; i++)
        outCounts[i] = 0;
    if (!m_init || count <= 0)
        return false;
    maxPer = std::max(maxPer, 1);
    const bool device = !m_stepActive && !uploadsPending() && m_bodyCount > 0 && m_scene.shapeTotal() > 0;
    if (!device)
    {
        for (int i = 0; i < count; i++)
        {
            int n = 0;
            m_query.overlapAABB(queryView(), lo[i], hi[i],
                                [&](int s)
                                {
                                    if ((m_scene.shCat[(size_t)s] & mask) != 0 &&
                                        (cat & m_scene.shMask[(size_t)s]) != 0)
                                    {
                                        if (n < maxPer)
                                            outIds[(size_t)i * (size_t)maxPer + (size_t)n] = s;
                                        n++;
                                    }
                                    return true;
                                });
            outCounts[i] = n;
        }
        return false;
    }
    std::vector<float> boxes((size_t)count * 4);
    for (int i = 0; i < count; i++)
    {
        boxes[(size_t)i * 4 + 0] = lo[i].x;
        boxes[(size_t)i * 4 + 1] = lo[i].y;
        boxes[(size_t)i * 4 + 2] = hi[i].x;
        boxes[(size_t)i * 4 + 3] = hi[i].y;
    }
    runQueryBatch(false, boxes.data(), count, cat, mask, maxPer);
    memcpy(outCounts, m_qOutCounts.mapped(), (size_t)count * 4);
    const int *ids = m_qOutIds.mappedAs<int>();
    for (int i = 0; i < count; i++)
    {
        const int n = std::min(outCounts[i], maxPer);
        for (int k = 0; k < n; k++)
            outIds[(size_t)i * (size_t)maxPer + (size_t)k] = ids[(size_t)i * (size_t)maxPer + (size_t)k];
    }
    return true;
}

} // namespace avbd2d
