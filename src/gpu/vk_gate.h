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

// Dispatch gate: while installed on a CommandList (CommandList::setGate), every
// dispatch() becomes an indirect dispatch whose {x,y,z} lives in a device buffer. A
// gate kernel (csGateDispatchArgs) copies the host-written commands into that buffer,
// or zeroes them when the flag word on the device is non-zero, so a run of phases can
// be skipped on the GPU (e.g. after an earlier phase overflowed) inside one submission.

#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>

namespace avbdvk
{

class DispatchGate
{
public:
    void init(Device &dev) { m_dev = &dev; args.init(dev); }
    void destroy() { args.destroy(); m_src = 0; m_host = nullptr; m_slots = 0; m_used = 0; }
    uint64_t bytesAllocated() const { return (uint64_t)args.size(); }

    // Grows the indirect-command buffer to `slots` * 12 bytes. Call BEFORE recording.
    void reserve(int slots);

    // Records the gate kernel (gate NOT yet installed) and the fence for the indirect
    // read. `flag` is the device address of an int32; non-zero closes the gate.
    void begin(CommandList &cmd, HostRingBuffer &ring, Pipeline &gatePipeline, VkDeviceAddress flag, int slots);

    // Gated CommandList::dispatch. Zero-sized dispatches are dropped.
    void dispatch(CommandList &cmd, Pipeline &p, uint32_t gx, uint32_t gy, uint32_t gz, VkDeviceAddress argsAddr);

    int used() const { return m_used; }

private:
    Device *m_dev = nullptr;
    Buffer args;
    VkDeviceAddress m_src = 0;
    uint32_t *m_host = nullptr;
    int m_slots = 0;
    int m_used = 0;
};

} // namespace avbdvk
