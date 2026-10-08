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

// Command buffer recording: one dispatch() call per kernel launch (the analogue of a kernel launch), a full compute->compute memory barrier between dependent
// dispatches (good enough until the real port needs finer-grained sync), a
// timeline-semaphore submit + wait, timestamp queries and debug labels for
// RenderDoc/Nsight.

#include "vk_buffer.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace avbdvk
{

class DispatchGate;

class CommandList
{
public:
    CommandList() = default;
    ~CommandList() { destroy(); }

    CommandList(const CommandList &) = delete;
    CommandList &operator=(const CommandList &) = delete;

    void init(Device &dev, uint32_t maxTimestamps = 64);
    void destroy();

    // Begins recording. Call once per submission.
    void begin();

    // Records a dispatch. `argsAddr` is the device address of the kernel's argument
    // struct, written earlier this frame via a HostRingBuffer or a device Buffer --
    // exactly the "one uint64 in push constants" design in the migration plan.
    void dispatch(Pipeline &pipeline, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ,
                  VkDeviceAddress argsAddr);
    void dispatch(Pipeline &pipeline, uint32_t groupsX, VkDeviceAddress argsAddr)
    {
        dispatch(pipeline, groupsX, 1, 1, argsAddr);
    }

    // A full compute-shader-write -> compute-shader-read memory barrier. Call between
    // any two dispatches where the second reads what the first wrote.
    void barrier();

    // Routes dispatch() through `g` (see vk_gate.h) until reset with nullptr.
    // dispatchIndirect() is never gated.
    void setGate(DispatchGate *g) { m_gate = g; }

    // M4: a dispatch whose group count is read from `argsBuf` at `offset` (a
    // VkDispatchIndirectCommand; the buffer needs VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT).
    // The kernel that wrote the command must be fenced with barrierIndirect(), since
    // the indirect read is not a compute-shader read.
    void dispatchIndirect(Pipeline &pipeline, const Buffer &argsBuf, VkDeviceSize offset, VkDeviceAddress argsAddr);
    void barrierIndirect();

    // Recorded transfers, each fenced by a full barrier on both sides so it can sit
    // between any two dispatches: the in-step replacements for the synchronous
    // Buffer::zero()/upload()/readback() round trips (M4). `bytes` is rounded up to a
    // multiple of 4 and clamped to the buffer, as Buffer::zero() does; 0 is a no-op.
    void fill(Buffer &buf, uint32_t word, VkDeviceSize bytes = ~0ull, VkDeviceSize offset = 0);
    void copy(const Buffer &src, Buffer &dst, VkDeviceSize bytes, VkDeviceSize srcOffset = 0,
              VkDeviceSize dstOffset = 0);
    // Host-filled staging -> device: `src` must have been made with TRANSFER_SRC usage
    // and written before submit() (queue submission makes host writes visible).
    void copy(const MappedBuffer &src, Buffer &dst, VkDeviceSize bytes, VkDeviceSize srcOffset = 0,
              VkDeviceSize dstOffset = 0);
    // A recorded readback: copies into a mapped buffer (made with TRANSFER_DST usage)
    // and makes the write visible to the HOST stage, which ALL_COMMANDS does not cover.
    // Read dst.mapped() after wait(). Replaces a Buffer::readback() one-shot submit.
    void copyToHost(const Buffer &src, MappedBuffer &dst, VkDeviceSize bytes, VkDeviceSize srcOffset = 0,
                    VkDeviceSize dstOffset = 0);

    void beginLabel(const char *name);
    void endLabel();

    // Writes a timestamp into the query pool at `index` (0-based, must be < maxTimestamps
    // passed to init()). Read back with getTimestampsNs() after the submission's fence/
    // semaphore has signalled.
    void writeTimestamp(uint32_t index, VkPipelineStageFlagBits stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    // Ends recording and submits, signalling the internal timeline semaphore to
    // `nextValue`. Returns immediately; call wait() to block until it is done.
    void submit();

    // The timeline value the most recent submit() signals (0 before the first).
    uint64_t lastSubmitValue() const { return m_timelineValue; }

    // Blocks until the most recent submit() has completed.
    void wait();

    // Valid only after wait(). Returns (end - start) in nanoseconds for two indices
    // written with writeTimestamp(), or -1 if timestamps are unsupported/not written.
    double timestampDeltaNs(uint32_t startIndex, uint32_t endIndex) const;

    // ---- Per-kernel GPU profiler (opt-in) -------------------------------------------
    // Enabled per CommandList at init() when setProfiling(true) was called before it
    // (process-wide; CommandLists created earlier are unaffected). Uses its OWN large query pool, separate from the phase timestamps.
    // begin() writes timestamp 0; every dispatch / dispatchIndirect / fill / copy then writes
    // one more, labelled with the pipeline name ("fill"/"copy"/"copyToHost" for transfers).
    // Interval i = ts[i] - ts[i-1], which is the op's own cost PLUS the barrier that precedes
    // it (and any drain of the previous op): an approximation, not an isolated kernel time.
    // Results are read in wait() and accumulated into a process-wide registry. Off: no extra
    // timestamps, no allocations, no pool.
    static void setProfiling(bool on);
    bool profiling() const { return m_profPool != VK_NULL_HANDLE; }
    struct ProfileEntry
    {
        std::string name;
        double totalNs = 0;
        uint64_t count = 0;
    };
    static void profileReset();
    static std::vector<ProfileEntry> profileSnapshot(); // sorted by totalNs, descending
    static uint64_t profileOverflow();                  // ops that did not fit in the pool
    static void profileDump(FILE *f, int steps = 1);    // steps: divide totals for per-step means

    // Ops recorded since begin(): dispatches (direct + indirect), compute barriers, transfers.
    struct OpCounts { uint32_t dispatches = 0, barriers = 0, transfers = 0; };
    OpCounts opCounts() const { return m_ops; }

    VkCommandBuffer handle() const { return m_cmd; }
    uint32_t maxTimestamps() const { return m_maxTimestamps; }

private:
    Device *m_dev = nullptr;
    VkCommandBuffer m_cmd = VK_NULL_HANDLE;
    VkPipeline m_boundCompute = VK_NULL_HANDLE; // last compute pipeline bound since begin()
    void bindCompute(Pipeline &pipeline);
    VkSemaphore m_timeline = VK_NULL_HANDLE;
    uint64_t m_timelineValue = 0;
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    uint32_t m_maxTimestamps = 0;
    bool m_timestampsValid = false;
    DispatchGate *m_gate = nullptr;
    OpCounts m_ops;

    static constexpr uint32_t kProfQueries = 16384;
    VkQueryPool m_profPool = VK_NULL_HANDLE;
    std::vector<const char *> m_profLabels; // label of op i (interval i+1); reserved at init
    uint32_t m_profUsed = 0;                // timestamps written this submission (incl. #0)
    bool m_profPending = false;
    void profStamp(const char *label);
    void profCollect();
};

} // namespace avbdvk
