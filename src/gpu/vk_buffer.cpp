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

#include "vk_buffer.h"
#include "vk_util.h"

#include <algorithm>

namespace avbdvk
{

uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw DeviceError("[avbd_vk] no suitable memory type");
}

bool tryFindMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props, uint32_t &outIndex)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
        {
            outIndex = i;
            return true;
        }
    }
    return false;
}

static VkCommandBuffer beginOneShot(Device &dev)
{
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = dev.commandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(dev.device(), &ai, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

static void endOneShotAndWait(Device &dev, VkCommandBuffer cmd)
{
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    Device::noteSubmit();
    dev.queueSubmit(si, false);
    VK_CHECK(vkQueueWaitIdle(dev.computeQueue()));
    vkFreeCommandBuffers(dev.device(), dev.commandPool(), 1, &cmd);
}

void Buffer::destroy()
{
    if (!m_dev)
        return;
    if (m_buffer)
        vkDestroyBuffer(m_dev->device(), m_buffer, nullptr);
    if (m_memory)
        vkFreeMemory(m_dev->device(), m_memory, nullptr);
    m_dev->refundBytes(m_charged);
    m_charged = 0;
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_address = 0;
    m_size = 0;
}

// vkCmdFillBuffer (used by zero()) requires its size to be a multiple of 4; some
// callers size a Buffer at 1 byte/element (e.g. the per-body awake-flag buffer), so
// bodyCount() itself need not be a multiple of 4. Pad every allocation up to a multiple
// of 4 bytes so zero()'s rounded-up fill size is always within the real allocation --
// root-fixed here rather than at each 1-byte-stride call site.
static VkDeviceSize alignUp4(VkDeviceSize bytes)
{
    return (bytes + 3) & ~VkDeviceSize(3);
}

void Buffer::resize(VkDeviceSize bytes, VkBufferUsageFlags extraUsage)
{
    if (bytes != 0)
        m_dev->checkBudget(alignUp4(bytes), m_charged); // throws before anything is freed or allocated
    destroy();
    if (bytes == 0)
        return;
    bytes = alignUp4(bytes);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraUsage;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_dev->device(), &bi, nullptr, &m_buffer));
    Device::noteAlloc();

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_dev->device(), m_buffer, &req);

    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flagsInfo;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(m_dev->physicalDevice(), req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_dev->device(), &ai, nullptr, &m_memory));
    Device::noteAlloc();
    VK_CHECK(vkBindBufferMemory(m_dev->device(), m_buffer, m_memory, 0));

    VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(m_dev->device(), &addrInfo);
    m_size = bytes;
    m_charged = req.size;
    m_dev->chargeBytes(m_charged);
}

void Buffer::resizePreserve(VkDeviceSize bytes, VkBufferUsageFlags extraUsage)
{
    if (m_size == 0 || bytes == 0)
    {
        // Nothing to preserve (first allocation, or shrinking to empty) -- same as
        // plain resize().
        resize(bytes, extraUsage);
        return;
    }

    VkBuffer oldBuffer = m_buffer;
    VkDeviceMemory oldMemory = m_memory;
    VkDeviceAddress oldAddress = m_address;
    VkDeviceSize oldSize = m_size;
    VkDeviceSize oldCharged = m_charged;
    Device *dev = m_dev;

    // Detach so destroy() inside resize() below does not free the old allocation --
    // it is still needed as the copy source. Its charge stays on the device until the copy
    // is done, so the budget check inside resize() sees the old + new peak.
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_address = 0;
    m_size = 0;
    m_charged = 0;

    try
    {
        resize(bytes, extraUsage);
        VkDeviceSize copyBytes = std::min(oldSize, bytes);
        VkCommandBuffer cmd = beginOneShot(*dev);
        VkBufferCopy region{0, 0, copyBytes};
        vkCmdCopyBuffer(cmd, oldBuffer, m_buffer, 1, &region);
        endOneShotAndWait(*dev, cmd);
    }
    catch (...)
    {
        // Growth refused or failed: drop whatever was half built and keep the old buffer.
        destroy();
        m_buffer = oldBuffer;
        m_memory = oldMemory;
        m_address = oldAddress;
        m_size = oldSize;
        m_charged = oldCharged;
        throw;
    }

    vkDestroyBuffer(dev->device(), oldBuffer, nullptr);
    vkFreeMemory(dev->device(), oldMemory, nullptr);
    dev->refundBytes(oldCharged);
}

void Buffer::zero(VkDeviceSize bytes)
{
    if (bytes == ~0ull)
        bytes = m_size;
    // vkCmdFillBuffer requires a multiple of 4; m_size is always padded to one (see
    // alignUp4 in resize()/resizePreserve()), so rounding a smaller logical `bytes` up
    // never reads/writes past the real allocation.
    bytes = std::min(alignUp4(bytes), m_size);
    if (bytes == 0)
        return;
    VkCommandBuffer cmd = beginOneShot(*m_dev);
    vkCmdFillBuffer(cmd, m_buffer, 0, bytes, 0);
    endOneShotAndWait(*m_dev, cmd);
}

void Buffer::upload(const void *src, VkDeviceSize bytes, VkDeviceSize dstOffset)
{
    if (bytes == 0)
        return; // a zero-size staging allocation is invalid (empty scene)
    VkBuffer staging;
    VkDeviceMemory stagingMem;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_dev->device(), &bi, nullptr, &staging));
    Device::noteAlloc();

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_dev->device(), staging, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(m_dev->physicalDevice(), req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(m_dev->device(), &ai, nullptr, &stagingMem));
    Device::noteAlloc();
    VK_CHECK(vkBindBufferMemory(m_dev->device(), staging, stagingMem, 0));

    void *mapped;
    VK_CHECK(vkMapMemory(m_dev->device(), stagingMem, 0, bytes, 0, &mapped));
    memcpy(mapped, src, (size_t)bytes);
    vkUnmapMemory(m_dev->device(), stagingMem);

    VkCommandBuffer cmd = beginOneShot(*m_dev);
    VkBufferCopy region{0, dstOffset, bytes};
    vkCmdCopyBuffer(cmd, staging, m_buffer, 1, &region);
    endOneShotAndWait(*m_dev, cmd);

    vkDestroyBuffer(m_dev->device(), staging, nullptr);
    vkFreeMemory(m_dev->device(), stagingMem, nullptr);
}

void Buffer::readback(void *dst, VkDeviceSize bytes, VkDeviceSize srcOffset) const
{
    if (bytes == 0)
        return; // a zero-size staging allocation is invalid (empty scene)
    VkBuffer staging;
    VkDeviceMemory stagingMem;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_dev->device(), &bi, nullptr, &staging));
    Device::noteAlloc();

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_dev->device(), staging, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(m_dev->physicalDevice(), req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(m_dev->device(), &ai, nullptr, &stagingMem));
    Device::noteAlloc();
    VK_CHECK(vkBindBufferMemory(m_dev->device(), staging, stagingMem, 0));

    VkCommandBuffer cmd = beginOneShot(*m_dev);
    VkBufferCopy region{srcOffset, 0, bytes};
    vkCmdCopyBuffer(cmd, m_buffer, staging, 1, &region);
    endOneShotAndWait(*m_dev, cmd);

    void *mapped;
    VK_CHECK(vkMapMemory(m_dev->device(), stagingMem, 0, bytes, 0, &mapped));
    memcpy(dst, mapped, (size_t)bytes);
    vkUnmapMemory(m_dev->device(), stagingMem);

    vkDestroyBuffer(m_dev->device(), staging, nullptr);
    vkFreeMemory(m_dev->device(), stagingMem, nullptr);
}

// ---------------------------------------------------------------------------
// HostRingBuffer
// ---------------------------------------------------------------------------

void HostRingBuffer::init(Device &dev, VkDeviceSize bytes)
{
    dev.checkBudget(bytes);
    m_dev = &dev;
    m_size = bytes;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev.device(), &bi, nullptr, &m_buffer));
    Device::noteAlloc();

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev.device(), m_buffer, &req);

    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flagsInfo;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(dev.physicalDevice(), req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(dev.device(), &ai, nullptr, &m_memory));
    Device::noteAlloc();
    VK_CHECK(vkBindBufferMemory(dev.device(), m_buffer, m_memory, 0));
    VK_CHECK(vkMapMemory(dev.device(), m_memory, 0, bytes, 0, &m_mapped));

    VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(dev.device(), &addrInfo);
    m_charged = req.size;
    dev.chargeBytes(m_charged);
}

void HostRingBuffer::destroy()
{
    if (!m_dev)
        return;
    if (m_mapped)
        vkUnmapMemory(m_dev->device(), m_memory);
    if (m_buffer)
        vkDestroyBuffer(m_dev->device(), m_buffer, nullptr);
    if (m_memory)
        vkFreeMemory(m_dev->device(), m_memory, nullptr);
    m_dev->refundBytes(m_charged);
    m_charged = 0;
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_mapped = nullptr;
}

VkDeviceAddress HostRingBuffer::alloc(VkDeviceSize size)
{
    // 16-byte alignment: comfortably covers every scalar-layout struct field width
    // this port uses (uint64_t addresses, vec3, quat), and matches std430-style
    // alignment rules generally.
    const VkDeviceSize align = 16;
    m_cursor = (m_cursor + align - 1) & ~(align - 1);
    if (m_cursor + size > m_size)
    {
        m_cursor = 0;
        m_wrapped = true;
    }
    if (size > m_size)
    {
        char msg[160];
        snprintf(msg, sizeof(msg), "[avbd_vk] HostRingBuffer::write: %llu bytes exceeds ring size %llu",
                 (unsigned long long)size, (unsigned long long)m_size);
        fprintf(stderr, "%s\n", msg);
        throw VkError(VK_ERROR_OUT_OF_HOST_MEMORY, msg);
    }
    VkDeviceAddress addr = m_address + m_cursor;
    m_cursor += size;
    return addr;
}

VkDeviceAddress HostRingBuffer::write(const void *data, VkDeviceSize size)
{
    VkDeviceAddress addr = alloc(size);
    memcpy(hostPtr(addr), data, (size_t)size);
    return addr;
}

// ---------------------------------------------------------------------------
// MappedBuffer
// ---------------------------------------------------------------------------

void MappedBuffer::init(Device &dev, VkDeviceSize bytes, VkBufferUsageFlags extraUsage, HostAccess access)
{
    dev.checkBudget(bytes);
    m_dev = &dev;
    m_size = bytes;

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraUsage;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev.device(), &bi, nullptr, &m_buffer));
    Device::noteAlloc();

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev.device(), m_buffer, &req);

    VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    uint32_t typeIndex;
    m_deviceLocal = false;
    if (access == HostAccess::Read &&
        tryFindMemoryType(dev.physicalDevice(), req.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                          typeIndex))
    {
        // Cached system memory: the GPU writes it over PCIe once, the host reads it at
        // cache speed. Coherent, so mapped() stays readable without an invalidate.
    }
    else if (access == HostAccess::Write &&
             tryFindMemoryType(dev.physicalDevice(), req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, typeIndex))
    {
        m_deviceLocal = true;
    }
    else
    {
        m_deviceLocal = false;
        typeIndex = findMemoryType(dev.physicalDevice(), req.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flagsInfo;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = typeIndex;
    VK_CHECK(vkAllocateMemory(dev.device(), &ai, nullptr, &m_memory));
    Device::noteAlloc();
    VK_CHECK(vkBindBufferMemory(dev.device(), m_buffer, m_memory, 0));
    VK_CHECK(vkMapMemory(dev.device(), m_memory, 0, bytes, 0, &m_mapped));

    VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(dev.device(), &addrInfo);
    m_charged = req.size;
    dev.chargeBytes(m_charged);
}

void MappedBuffer::destroy()
{
    if (!m_dev)
        return;
    if (m_mapped)
        vkUnmapMemory(m_dev->device(), m_memory);
    if (m_buffer)
        vkDestroyBuffer(m_dev->device(), m_buffer, nullptr);
    if (m_memory)
        vkFreeMemory(m_dev->device(), m_memory, nullptr);
    m_dev->refundBytes(m_charged);
    m_charged = 0;
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_mapped = nullptr;
    m_address = 0;
    m_size = 0;
}

} // namespace avbdvk
