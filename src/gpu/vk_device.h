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

// Instance + physical/logical device setup for the Vulkan backend. Gives the engine
// a context bound to a chosen GPU and one queue.
//
// Deliberately minimal: one compute queue, validation layers gated by build config
// or an env var, and the feature set the kernels need: bufferDeviceAddress, shaderInt64,
// shaderBufferInt64Atomics, shaderBufferFloat32AtomicAdd, timelineSemaphore,
// subgroup size control.

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace avbdvk
{

// Set by the environment (AVBD_VK_VALIDATION=1) or forced on in a debug build, unless
// setValidationOverride() has been called (process-wide; call before Device::init()).
bool wantValidation();
void setValidationOverride(bool on);

class Device
{
public:
    Device();
    ~Device();

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    // Creates the instance, picks a physical device (discrete NVIDIA preferred,
    // else the first discrete GPU, else whatever is available), and creates a
    // logical device with one compute queue and the required features enabled.
    // Throws avbdvk::DeviceError (what() names the missing device, feature or extension) or
    // VkError on failure -- there is no fallback path. An uncaught throw terminates the
    // process, which is what the old abort did; the 2D C ABI turns it into AVBD2D_ERR_DEVICE.
    void init();
    // Adopts a device the host application created (the Vulkan-game path). Nothing is created
    // except a command pool and pipeline cache; shutdown() never destroys the instance,
    // physical device, device or queue. The device must have been created with the features
    // and extensions init() would enable (only what the physical device SUPPORTS can be
    // checked here); throws DeviceError if the physical device lacks one, the API version
    // is below 1.3, or the queue family/index is invalid.
    void adopt(VkInstance instance, VkPhysicalDevice physical, VkDevice device, uint32_t queueFamily,
               uint32_t queueIndex, uint32_t apiVersion);
    void shutdown();

    VkInstance instance() const { return m_instance; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    VkDevice device() const { return m_device; }
    // Debug counter of vkQueueSubmit calls made through this engine (CommandList::submit
    // and Buffer's one-shot submits). Process-wide; read deltas around a step.
    static uint64_t submitCount() { return submitCounter().load(std::memory_order_relaxed); }
    static void noteSubmit() { submitCounter().fetch_add(1, std::memory_order_relaxed); }
    // Debug counter of vkCreateBuffer + vkAllocateMemory calls made by Buffer (and its
    // staging one-shots). Process-wide; read deltas around a step.
    static uint64_t allocCount() { return allocCounter().load(std::memory_order_relaxed); }
    static void noteAlloc() { allocCounter().fetch_add(1, std::memory_order_relaxed); }

    // The single place the engine submits. Counts submits made through CommandList and, when
    // setDebugFailSubmit(N) is set, makes the Nth CommandList submit throw VK_ERROR_DEVICE_LOST
    // instead of reaching the driver (a debug override for exercising the error path). `step` is true for those.
    void queueSubmit(const VkSubmitInfo &si, bool step);
    // N >= 1: the Nth step submit of this device reports VK_ERROR_DEVICE_LOST. 0 = off.
    void setDebugFailSubmit(uint64_t n) { m_failSubmitAt = n; }

    // --- Memory budget ----------------------------------------------------------------------
    // 0 = unlimited. Buffer/MappedBuffer/HostRingBuffer allocations that would push
    // allocatedBytes() past the budget throw BudgetExceeded BEFORE touching the driver.
    void setMemoryBudget(uint64_t bytes) { m_budget = bytes; }
    uint64_t memoryBudget() const { return m_budget; }
    uint64_t allocatedBytes() const { return m_allocated.load(std::memory_order_relaxed); }
    // Throws BudgetExceeded unless `bytes` more (on top of what is allocated, minus `reuse`
    // that the caller is about to free) fits the budget.
    void checkBudget(uint64_t bytes, uint64_t reuse = 0) const;
    void chargeBytes(uint64_t bytes) { m_allocated.fetch_add(bytes, std::memory_order_relaxed); }
    void refundBytes(uint64_t bytes) { m_allocated.fetch_sub(bytes, std::memory_order_relaxed); }

    // --- Portability ------------------------------------------------------------------------
    // Subgroup policy, before init(). Ignore (the default) changes nothing.
    // Generic3264 (the 2D solver; its kernels, and the shared radix/constraints kernels, are
    // written for subgroup size 32 or 64 with ballot + arithmetic ops): init() refuses a device
    // whose subgroup ops are missing or whose native size is neither, unless the device can
    // pin the size to 32 (subgroup size control), in which case every compute pipeline is
    // created with required size 32. AVBD_VK_PIN_SUBGROUP=32 forces the pin (debug override).
    enum class SubgroupPolicy { Ignore, Generic3264 };
    void setSubgroupPolicy(SubgroupPolicy p) { m_subgroupPolicy = p; }
    // Required subgroup size to put on compute pipelines, 0 = leave to the driver.
    uint32_t pinnedSubgroupSize() const { return m_pinnedSubgroup; }
    // VK_EXT_shader_atomic_float (float32 atomic add): required by default;
    // the 2D solver turns this off since it does not use it. When not required it is still enabled if supported.
    void setRequireAtomicFloat(bool on) { m_requireAtomicFloat = on; }
    // Prefer a COMPUTE queue family without GRAPHICS (async compute / compute-only) when not
    // presenting. Before init().
    void setPreferDedicatedCompute(bool on) { m_preferDedicatedCompute = on; }
    bool adopted() const { return m_adopted; }
    // False if the queue family reports no valid timestamp bits.
    bool timestampsSupported() const { return m_timestampValidBits > 0; }

    VkQueue computeQueue() const { return m_computeQueue; }
    uint32_t computeQueueFamily() const { return m_computeQueueFamily; }
    VkCommandPool commandPool() const { return m_commandPool; }
    VkPipelineCache pipelineCache() const { return m_pipelineCache; }

    // --- sdl_vk demo interop (docs/VK_DEMO_INTERFACE.md) -----------------------------
    // Compute and graphics share the single queue created above, so these are plain
    // aliases -- named to match the demo's expectations, not a second queue.
    VkQueue queue() const { return m_computeQueue; }
    uint32_t queueFamily() const { return m_computeQueueFamily; }
    VkSurfaceKHR surface() const { return m_surface; }

    // Must be called before init(). Appends to the instance extension list that
    // createInstance() enables -- e.g. the result of SDL_Vulkan_GetInstanceExtensions,
    // so the instance created here can also present to an SDL-owned VkSurfaceKHR.
    void requireInstanceExtensions(const std::vector<const char *> &extensions);

    // Must be called before init(). `makeSurface` is invoked once, right after the
    // instance is created, and must return a VkSurfaceKHR made against THIS instance
    // (e.g. SDL_Vulkan_CreateSurface(window, instance())). init() then restricts
    // physical-device and queue-family selection to a family that supports
    // GRAPHICS|COMPUTE and present to that surface, and enables VK_KHR_swapchain on
    // the logical device. A no-op for every existing headless caller that never sets
    // this hook (surface() stays VK_NULL_HANDLE and selection is unchanged).
    void enablePresentation(std::function<VkSurfaceKHR(VkInstance)> makeSurface);

    // Debug-utils function pointers, null if VK_EXT_debug_utils was not enabled.
    PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugLabel = nullptr;
    PFN_vkSetDebugUtilsObjectNameEXT setDebugObjectName = nullptr;

    // --- External-memory interop hooks -----------------------------------------------
    // Both must be called before init(); both are no-ops by default,
    // so nothing above this comment changes behavior.

    // Pin physical-device selection to the adapter a D3D11 device already chose
    // (its DXGI_ADAPTER_DESC::AdapterLuid), instead of the discrete-GPU heuristic in
    // pickPhysicalDevice(). Adapter pinning matters because on a multi-GPU box, picking
    // independently can put
    // Vulkan and D3D11 to different physical GPUs, and external-memory import then
    // fails silently or refuses.
    void requireAdapterLuid(const uint8_t luid[8]);

    // Enables VK_KHR_external_memory_win32 / VK_KHR_external_semaphore_win32 (their
    // capability-extension prerequisites are core in the 1.3 instance this backend
    // already requests, so no instance extensions are needed). init() aborts if the
    // selected device does not support them.
    void enableExternalWin32();
    bool hasExternalWin32() const { return m_externalWin32; }

    // The physical device's VkPhysicalDeviceIDProperties.deviceLUID, valid after
    // init(). Lets an interop backend double check (or log) which adapter Vulkan
    // actually landed on.
    const uint8_t *adapterLuid() const { return m_adapterLuid; }

    // GL/EGL-side adapter pinning: GL_EXT_memory_object exposes GL_DEVICE_UUID_EXT
    // (glGetUnsignedBytei_vEXT), matched against VkPhysicalDeviceIDProperties.deviceUUID.
    // D3D11 has an AdapterLuid; GL has no LUID query at all but does have this UUID,
    // which is the same idea one layer down.
    void requireDeviceUUID(const uint8_t uuid[16]);
    const uint8_t *deviceUUID() const { return m_deviceUUID; }

    // Device limits/props the rest of the backend cares about.
    uint32_t subgroupSize() const { return m_subgroupSize; }
    float timestampPeriodNs() const { return m_timestampPeriodNs; }

    // Prints the device-selection line to stderr during init(). Off by default; call
    // before init().
    void setVerbose(bool on) { m_verbose = on; }
    // Name of the selected physical device ("" before init()).
    const char *deviceName() const { return m_deviceName; }

    // Message counter incremented by the debug messenger callback -- a clean run
    // should leave it at 0.
    static uint32_t validationMessageCount();
    static void resetValidationMessageCount();

private:
    void createInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createPoolAndCache();
    // Limits, subgroup properties and the feature/extension floor of one physical device.
    struct DeviceInfo
    {
        uint32_t apiVersion = 0;
        uint32_t subgroupSize = 32;
        float timestampPeriodNs = 1.0f;
        bool atomicFloat = false;
        bool subgroupSizeControl = false;
        uint32_t pinSubgroup = 0;
        std::vector<std::string> extensions;
    };
    // True if `d` can run the solver; otherwise `why` names what is missing.
    bool evaluateDevice(VkPhysicalDevice d, DeviceInfo &info, std::string &why) const;
    void applyInfo(const DeviceInfo &info);

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    static std::atomic<uint64_t> &submitCounter()
    {
        static std::atomic<uint64_t> c{0};
        return c;
    }
    static std::atomic<uint64_t> &allocCounter()
    {
        static std::atomic<uint64_t> c{0};
        return c;
    }
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    uint32_t m_computeQueueFamily = ~0u;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;

    uint32_t m_subgroupSize = 32;
    float m_timestampPeriodNs = 1.0f;
    bool m_validation = false;
    bool m_verbose = false;
    char m_deviceName[256] = {};

    bool m_hasRequiredLuid = false;
    uint8_t m_requiredLuid[8] = {};
    uint8_t m_adapterLuid[8] = {};
    bool m_hasRequiredUUID = false;
    uint8_t m_requiredUUID[16] = {};
    uint8_t m_deviceUUID[16] = {};
    bool m_externalWin32 = false;

    std::vector<const char *> m_extraInstanceExtensions;
    std::function<VkSurfaceKHR(VkInstance)> m_makeSurface;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    bool m_wantPresent = false;

    std::atomic<uint64_t> m_allocated{0};
    uint64_t m_budget = 0;
    uint64_t m_failSubmitAt = 0;
    uint64_t m_stepSubmits = 0;
    SubgroupPolicy m_subgroupPolicy = SubgroupPolicy::Ignore;
    uint32_t m_pinnedSubgroup = 0;
    bool m_requireAtomicFloat = true;
    bool m_haveAtomicFloat = false;
    bool m_haveSubgroupSizeControl = false;
    bool m_preferDedicatedCompute = false;
    bool m_adopted = false;
    uint32_t m_timestampValidBits = 64;
    uint32_t m_apiVersion = 0;
    std::vector<std::string> m_extensions; // supported device extensions
};

} // namespace avbdvk
