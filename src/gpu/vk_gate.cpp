#include "vk_gate.h"

#include "vk_args.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace avbdvk
{

void DispatchGate::reserve(int slots)
{
    const VkDeviceSize need = (VkDeviceSize)(slots > 0 ? slots : 0) * 12;
    if (need > args.size())
        args.resize(need, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
}

void DispatchGate::begin(CommandList &cmd, HostRingBuffer &ring, Pipeline &gatePipeline, VkDeviceAddress flag,
                         int slots)
{
    assert((VkDeviceSize)slots * 12 <= args.size());
    m_slots = slots;
    m_used = 0;
    m_src = ring.alloc((VkDeviceSize)slots * 12);
    m_host = (uint32_t *)ring.hostPtr(m_src);
    memset(m_host, 0, (size_t)slots * 12);

    GateDispatchArgsGpu a{};
    a.src = m_src;
    a.dst = args.address();
    a.flag = flag;
    a.count = slots;
    cmd.dispatchArgs(gatePipeline, (uint32_t)((slots + 127) / 128), ring, a);
    cmd.barrierIndirect();
}

void DispatchGate::dispatch(CommandList &cmd, Pipeline &p, uint32_t gx, uint32_t gy, uint32_t gz,
                            VkDeviceAddress argsAddr)
{
    if (gx == 0 || gy == 0 || gz == 0)
        return;
    if (m_used >= m_slots)
    {
        fprintf(stderr, "[avbd_vk] DispatchGate: out of slots (%d)\n", m_slots);
        abort();
    }
    const int i = m_used++;
    m_host[3 * i + 0] = gx;
    m_host[3 * i + 1] = gy;
    m_host[3 * i + 2] = gz;
    cmd.dispatchIndirect(p, args, (VkDeviceSize)i * 12, argsAddr);
}

} // namespace avbdvk
