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

#pragma once

// Device-local buffers with SHADER_DEVICE_ADDRESS usage, plus a staging upload/readback
// path. No VMA in this SDK install, so allocation is a direct vkAllocateMemory per
// buffer -- fine for the handful of large allocations this engine makes.

#include "vk_device.h"

#include <cstdint>
#include <cstring>

namespace avbdvk
{

// A single device-local allocation with a stable buffer device address. Grows like
// DeviceBuffer<T>: never shrinks, doubles with headroom on overflow.
class Buffer
{
public:
    Buffer() = default;
    ~Buffer() { destroy(); }

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    void init(Device &dev) { m_dev = &dev; }
    void destroy();

    // (Re)allocates to at least `bytes`. Existing contents are NOT preserved.
    void resize(VkDeviceSize bytes, VkBufferUsageFlags extraUsage = 0);

    // Like resize(), but copies min(old size, bytes) bytes of the OLD contents into
    // the new allocation first (device-to-device vkCmdCopyBuffer, one-shot submit +
    // wait), the growth-preserving counterpart to resize(). Used where warm-started state (contact cache, dual
    // variables) must survive a mid-run capacity grow -- see VkNarrowphase::grow().
    // A no-op copy (same behavior as resize()) if there was nothing to preserve.
    void resizePreserve(VkDeviceSize bytes, VkBufferUsageFlags extraUsage = 0);

    VkBuffer handle() const { return m_buffer; }
    VkDeviceAddress address() const { return m_address; }
    VkDeviceSize size() const { return m_size; }

    // Host-visible persistently-mapped staging buffer helpers. upload() copies from a
    // host pointer into this device-local buffer via a temporary staging buffer and a
    // one-shot command buffer + fence wait; readback() is the reverse. Simple and
    // synchronous -- correctness first, fine for rare host<->device round trips.
    void upload(const void *src, VkDeviceSize bytes, VkDeviceSize dstOffset = 0);
    void readback(void *dst, VkDeviceSize bytes, VkDeviceSize srcOffset = 0) const;

    void zero(VkDeviceSize bytes = ~0ull);

private:
    Device *m_dev = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceAddress m_address = 0;
    VkDeviceSize m_size = 0;
    VkDeviceSize m_charged = 0; // bytes charged to Device's memory budget accounting
};

// Host-visible, persistently mapped buffer -- the "argument ring" the push-constant
// dispatch design needs: the host writes a small struct per
// dispatch into this ring and passes its device address in an 8-byte push constant.
class HostRingBuffer
{
public:
    HostRingBuffer() = default;
    ~HostRingBuffer() { destroy(); }

    HostRingBuffer(const HostRingBuffer &) = delete;
    HostRingBuffer &operator=(const HostRingBuffer &) = delete;

    void init(Device &dev, VkDeviceSize bytes);
    void destroy();

    // Writes `size` bytes at the ring's current cursor (advancing it, wrapping and
    // 16-byte-aligning), returns the device address of where they landed. The ring is
    // sized generously enough per frame that wraparound within one frame's dispatches
    // does not clobber an argument struct still in flight (a throwaway caller can use a fresh
    // ring per call for exactly this reason).
    VkDeviceAddress write(const void *data, VkDeviceSize size);

    // Reserves `size` bytes at the cursor with write()'s alignment/wrap rules, copying
    // nothing. The caller fills them through hostPtr() (bytes are not cleared).
    VkDeviceAddress alloc(VkDeviceSize size);
    // Host pointer to a device address this ring handed out.
    void *hostPtr(VkDeviceAddress addr) const { return (uint8_t *)m_mapped + (addr - m_address); }

    void reset() { m_cursor = 0; m_wrapped = false; }
    // True once the cursor has wrapped since the last reset(). A caller that records a
    // whole submission between resets must treat that as fatal: the wrap overwrote
    // argument structs the same submission still reads.
    bool wrapped() const { return m_wrapped; }

private:
    Device *m_dev = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceAddress m_address = 0;
    void *m_mapped = nullptr;
    VkDeviceSize m_size = 0;
    VkDeviceSize m_cursor = 0;
    VkDeviceSize m_charged = 0;
    bool m_wrapped = false;
};

// A persistently-mapped, device-address-bearing buffer for the "no readback copy"
// pose path: a kernel writes into it as an ordinary storage buffer via
// its device address, and the host reads results straight out of the mapped pointer
// after waiting on the submission's fence -- no staging buffer, no vkCmdCopyBuffer.
// Tries DEVICE_LOCAL | HOST_VISIBLE first (ReBAR / resizable BAR: a device-local heap
// the host can also map directly, so kernel writes land in the same memory the host
// reads, no PCIe copy either way); falls back to plain HOST_VISIBLE | HOST_COHERENT
// (system memory the GPU writes over PCIe) when no such heap exists.
class MappedBuffer
{
public:
    MappedBuffer() = default;
    ~MappedBuffer() { destroy(); }

    MappedBuffer(const MappedBuffer &) = delete;
    MappedBuffer &operator=(const MappedBuffer &) = delete;

    // Who reads the mapping picks the heap. Write (the default, above) is for data the
    // host fills and the GPU reads. Read is for data the GPU writes and the host reads
    // back every frame (the pose buffer): host reads of BAR memory are uncached PCIe
    // reads, so it prefers HOST_CACHED | HOST_COHERENT system memory instead, and falls
    // back to plain HOST_COHERENT where no cached type exists.
    enum class HostAccess { Write, Read };

    void init(Device &dev, VkDeviceSize bytes, VkBufferUsageFlags extraUsage = 0,
              HostAccess access = HostAccess::Write);
    void destroy();

    VkBuffer handle() const { return m_buffer; }
    VkDeviceAddress address() const { return m_address; }
    VkDeviceSize size() const { return m_size; }

    // Raw mapped host pointer. Valid to read/write directly once the caller has waited
    // on the fence/queue for any GPU work touching this buffer -- HOST_COHERENT memory
    // needs no explicit flush/invalidate, and the ReBAR path is coherent the same way.
    void *mapped() const { return m_mapped; }
    template <typename T> T *mappedAs() const { return reinterpret_cast<T *>(m_mapped); }

    // True if the allocation landed in DEVICE_LOCAL | HOST_VISIBLE memory (ReBAR) rather
    // than the HOST_VISIBLE | HOST_COHERENT fallback. Informational only.
    bool isDeviceLocal() const { return m_deviceLocal; }

private:
    Device *m_dev = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceAddress m_address = 0;
    void *m_mapped = nullptr;
    VkDeviceSize m_size = 0;
    VkDeviceSize m_charged = 0;
    bool m_deviceLocal = false;
};

uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props);
bool tryFindMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props, uint32_t &outIndex);

} // namespace avbdvk
