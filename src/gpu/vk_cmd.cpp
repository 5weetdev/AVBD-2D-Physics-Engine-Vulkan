/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#include "vk_cmd.h"
#include "vk_gate.h"
#include "vk_util.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>

namespace avbdvk
{

namespace
{
struct ProfRegistry
{
    std::mutex mu;
    std::map<std::string, CommandList::ProfileEntry> entries;
    uint64_t overflow = 0;
};
ProfRegistry &registry()
{
    static ProfRegistry r;
    return r;
}
std::atomic<int> &profOverride() // 0 (default): off, 1: on
{
    static std::atomic<int> v{0};
    return v;
}
bool profWanted()
{
    return profOverride().load() > 0;
}
} // namespace

void CommandList::setProfiling(bool on) { profOverride().store(on ? 1 : 0); }

void CommandList::profileReset()
{
    ProfRegistry &r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    r.entries.clear();
    r.overflow = 0;
}

std::vector<CommandList::ProfileEntry> CommandList::profileSnapshot()
{
    ProfRegistry &r = registry();
    std::vector<ProfileEntry> v;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        for (auto &kv : r.entries)
            v.push_back(kv.second);
    }
    std::sort(v.begin(), v.end(), [](const ProfileEntry &a, const ProfileEntry &b) { return a.totalNs > b.totalNs; });
    return v;
}

uint64_t CommandList::profileOverflow()
{
    ProfRegistry &r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    return r.overflow;
}

void CommandList::profileDump(FILE *f, int steps)
{
    std::vector<ProfileEntry> v = profileSnapshot();
    double total = 0;
    for (auto &e : v)
        total += e.totalNs;
    const double n = steps > 0 ? steps : 1;
    fprintf(f, "%-28s %10s %10s %10s %8s\n", "kernel", "ms/step", "calls/step", "mean us", "%");
    for (auto &e : v)
        fprintf(f, "%-28s %10.4f %10.1f %10.2f %7.2f%%\n", e.name.c_str(), e.totalNs * 1e-6 / n,
                (double)e.count / n, e.count ? e.totalNs * 1e-3 / (double)e.count : 0.0,
                total > 0 ? 100.0 * e.totalNs / total : 0.0);
    fprintf(f, "%-28s %10.4f   (profiled total per step; overflowed ops: %llu)\n", "TOTAL", total * 1e-6 / n,
            (unsigned long long)profileOverflow());
}

void CommandList::profStamp(const char *label)
{
    if (!m_profPool)
        return;
    if (m_profUsed >= kProfQueries)
    {
        ProfRegistry &r = registry();
        std::lock_guard<std::mutex> lk(r.mu);
        r.overflow++;
        return;
    }
    vkCmdWriteTimestamp(m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_profPool, m_profUsed++);
    m_profLabels.push_back(label);
}

void CommandList::profCollect()
{
    if (!m_profPending)
        return;
    m_profPending = false;
    if (m_profUsed < 2)
        return;
    std::vector<uint64_t> ts(m_profUsed);
    if (vkGetQueryPoolResults(m_dev->device(), m_profPool, 0, m_profUsed, ts.size() * sizeof(uint64_t), ts.data(),
                              sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
        return;
    const double period = m_dev->timestampPeriodNs();
    ProfRegistry &r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    for (uint32_t i = 1; i < m_profUsed; i++)
    {
        ProfileEntry &e = r.entries[m_profLabels[i - 1]];
        if (e.name.empty())
            e.name = m_profLabels[i - 1];
        e.totalNs += (double)(ts[i] - ts[i - 1]) * period;
        e.count++;
    }
}

void CommandList::init(Device &dev, uint32_t maxTimestamps)
{
    m_dev = &dev;
    m_maxTimestamps = maxTimestamps;

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = dev.commandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev.device(), &ai, &m_cmd));

    VkSemaphoreTypeCreateInfo typeInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = 0;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    si.pNext = &typeInfo;
    VK_CHECK(vkCreateSemaphore(dev.device(), &si, nullptr, &m_timeline));

    if (maxTimestamps > 0 && dev.timestampsSupported())
    {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = maxTimestamps;
        if (vkCreateQueryPool(dev.device(), &qi, nullptr, &m_queryPool) == VK_SUCCESS)
            m_timestampsValid = true;
    }

    if (profWanted() && dev.timestampsSupported())
    {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kProfQueries;
        if (vkCreateQueryPool(dev.device(), &qi, nullptr, &m_profPool) != VK_SUCCESS)
            m_profPool = VK_NULL_HANDLE;
        else
            m_profLabels.reserve(kProfQueries);
    }
}

void CommandList::destroy()
{
    if (!m_dev)
        return;
    if (m_queryPool)
        vkDestroyQueryPool(m_dev->device(), m_queryPool, nullptr);
    if (m_profPool)
        vkDestroyQueryPool(m_dev->device(), m_profPool, nullptr);
    m_profPool = VK_NULL_HANDLE;
    if (m_timeline)
        vkDestroySemaphore(m_dev->device(), m_timeline, nullptr);
    if (m_cmd)
        vkFreeCommandBuffers(m_dev->device(), m_dev->commandPool(), 1, &m_cmd);
    m_queryPool = VK_NULL_HANDLE;
    m_timeline = VK_NULL_HANDLE;
    m_cmd = VK_NULL_HANDLE;
}

void CommandList::begin()
{
    VK_CHECK(vkResetCommandBuffer(m_cmd, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(m_cmd, &bi));
    m_boundCompute = VK_NULL_HANDLE;
    m_ops = OpCounts();
    if (m_timestampsValid)
        vkCmdResetQueryPool(m_cmd, m_queryPool, 0, m_maxTimestamps);
    if (m_profPool)
    {
        vkCmdResetQueryPool(m_cmd, m_profPool, 0, kProfQueries);
        vkCmdWriteTimestamp(m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_profPool, 0);
        m_profLabels.clear();
        m_profUsed = 1;
        m_profPending = true;
    }
}

void CommandList::dispatch(Pipeline &pipeline, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ,
                            VkDeviceAddress argsAddr)
{
    m_ops.dispatches++;
    if (m_gate)
    {
        m_gate->dispatch(*this, pipeline, groupsX, groupsY, groupsZ, argsAddr);
        return;
    }
    bindCompute(pipeline);
    struct
    {
        uint64_t addr;
    } pc{argsAddr};
    vkCmdPushConstants(m_cmd, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(m_cmd, groupsX, groupsY, groupsZ);
    profStamp(pipeline.name());
}

void CommandList::dispatchIndirect(Pipeline &pipeline, const Buffer &argsBuf, VkDeviceSize offset,
                                   VkDeviceAddress argsAddr)
{
    m_ops.dispatches++;
    bindCompute(pipeline);
    struct
    {
        uint64_t addr;
    } pc{argsAddr};
    vkCmdPushConstants(m_cmd, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatchIndirect(m_cmd, argsBuf.handle(), offset);
    profStamp(pipeline.name());
}

// Consecutive dispatches of one pipeline (the primal colours, JP rounds, radix passes)
// skip the rebind; the push constant is still set per dispatch. Only this class binds
// compute pipelines, and begin() forgets the binding with the buffer's reset.
void CommandList::bindCompute(Pipeline &pipeline)
{
    if (pipeline.handle() == m_boundCompute)
        return;
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    m_boundCompute = pipeline.handle();
}

void CommandList::barrierIndirect()
{
    m_ops.barriers++;
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                       VK_ACCESS_2_SHADER_WRITE_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(m_cmd, &dep);
}

void CommandList::barrier()
{
    m_ops.barriers++;
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(m_cmd, &dep);
}

namespace
{
void transferBarrier(VkCommandBuffer cmd, bool before)
{
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    if (before)
    {
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    }
    else
    {
        mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
}
} // namespace

void CommandList::fill(Buffer &buf, uint32_t word, VkDeviceSize bytes, VkDeviceSize offset)
{
    if (offset >= buf.size())
        return;
    if (bytes == ~0ull)
        bytes = buf.size() - offset;
    bytes = std::min((bytes + 3) & ~VkDeviceSize(3), buf.size() - offset);
    if (bytes == 0)
        return;
    m_ops.transfers++;
    transferBarrier(m_cmd, true);
    vkCmdFillBuffer(m_cmd, buf.handle(), offset, bytes, word);
    transferBarrier(m_cmd, false);
    profStamp("fill");
}

void CommandList::copy(const Buffer &src, Buffer &dst, VkDeviceSize bytes, VkDeviceSize srcOffset,
                       VkDeviceSize dstOffset)
{
    if (bytes == 0)
        return;
    m_ops.transfers++;
    transferBarrier(m_cmd, true);
    VkBufferCopy region{srcOffset, dstOffset, bytes};
    vkCmdCopyBuffer(m_cmd, src.handle(), dst.handle(), 1, &region);
    transferBarrier(m_cmd, false);
    profStamp("copy");
}

void CommandList::copy(const MappedBuffer &src, Buffer &dst, VkDeviceSize bytes, VkDeviceSize srcOffset,
                       VkDeviceSize dstOffset)
{
    if (bytes == 0)
        return;
    m_ops.transfers++;
    transferBarrier(m_cmd, true);
    VkBufferCopy region{srcOffset, dstOffset, bytes};
    vkCmdCopyBuffer(m_cmd, src.handle(), dst.handle(), 1, &region);
    transferBarrier(m_cmd, false);
    profStamp("copy");
}

void CommandList::copyToHost(const Buffer &src, MappedBuffer &dst, VkDeviceSize bytes, VkDeviceSize srcOffset,
                             VkDeviceSize dstOffset)
{
    if (bytes == 0)
        return;
    m_ops.transfers++;
    transferBarrier(m_cmd, true);
    VkBufferCopy region{srcOffset, dstOffset, bytes};
    vkCmdCopyBuffer(m_cmd, src.handle(), dst.handle(), 1, &region);

    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(m_cmd, &dep);
    profStamp("copyToHost");
}

void CommandList::beginLabel(const char *name)
{
    if (m_dev->cmdBeginDebugLabel)
    {
        VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        label.pLabelName = name;
        m_dev->cmdBeginDebugLabel(m_cmd, &label);
    }
}

void CommandList::endLabel()
{
    if (m_dev->cmdEndDebugLabel)
        m_dev->cmdEndDebugLabel(m_cmd);
}

void CommandList::writeTimestamp(uint32_t index, VkPipelineStageFlagBits stage)
{
    if (m_timestampsValid && index < m_maxTimestamps)
        vkCmdWriteTimestamp(m_cmd, stage, m_queryPool, index);
}

void CommandList::submit()
{
    VK_CHECK(vkEndCommandBuffer(m_cmd));

    m_timelineValue++;
    VkTimelineSemaphoreSubmitInfo timelineInfo{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &m_timelineValue;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &timelineInfo;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &m_timeline;
    Device::noteSubmit();
    m_dev->queueSubmit(si, true);
}

void CommandList::wait()
{
    VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &m_timeline;
    waitInfo.pValues = &m_timelineValue;
    VK_CHECK(vkWaitSemaphores(m_dev->device(), &waitInfo, UINT64_MAX));
    profCollect();
}

double CommandList::timestampDeltaNs(uint32_t startIndex, uint32_t endIndex) const
{
    if (!m_timestampsValid)
        return -1.0;
    uint64_t values[2] = {0, 0};
    VkResult r0 = vkGetQueryPoolResults(m_dev->device(), m_queryPool, startIndex, 1,
                                         sizeof(uint64_t), &values[0], sizeof(uint64_t),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    VkResult r1 = vkGetQueryPoolResults(m_dev->device(), m_queryPool, endIndex, 1,
                                         sizeof(uint64_t), &values[1], sizeof(uint64_t),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r0 != VK_SUCCESS || r1 != VK_SUCCESS)
        return -1.0;
    return (double)(values[1] - values[0]) * m_dev->timestampPeriodNs();
}

} // namespace avbdvk
