// External-memory interop: VK_KHR_external_memory_win32's
// extension name macro and structs live in vulkan_win32.h, which vulkan.h only pulls
// in when VK_USE_PLATFORM_WIN32_KHR is defined before its first inclusion in this
// translation unit -- must come before "vk_device.h", which is what includes
// vulkan.h. Harmless when enableExternalWin32() is never called.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "vk_device.h"
#include "vk_util.h"

#ifndef _WIN32
#include <unistd.h>
#endif
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>

namespace avbdvk
{

static unsigned long processId()
{
#ifdef _WIN32
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}

static std::atomic<uint32_t> g_validationMessageCount{0};

uint32_t Device::validationMessageCount()
{
    return g_validationMessageCount.load();
}
void Device::resetValidationMessageCount()
{
    g_validationMessageCount.store(0);
}

static std::atomic<int> g_validationOverride{-1}; // -1: follow build/env, 0/1: forced

void setValidationOverride(bool on)
{
    g_validationOverride.store(on ? 1 : 0);
}

bool wantValidation()
{
    const int o = g_validationOverride.load();
    if (o >= 0)
        return o != 0;
#ifndef NDEBUG
    return true;
#else
    const char *env = std::getenv("AVBD_VK_VALIDATION");
    return env && env[0] == '1';
#endif
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
                                                    const VkDebugUtilsMessengerCallbackDataEXT *data,
                                                    void * /*userData*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        g_validationMessageCount.fetch_add(1);
        fprintf(stderr, "[vk-validation] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

Device::Device() {}
Device::~Device()
{
    shutdown();
}

void Device::requireAdapterLuid(const uint8_t luid[8])
{
    m_hasRequiredLuid = true;
    memcpy(m_requiredLuid, luid, 8);
}

void Device::enableExternalWin32()
{
    m_externalWin32 = true;
}

void Device::requireInstanceExtensions(const std::vector<const char *> &extensions)
{
    for (auto *e : extensions)
        m_extraInstanceExtensions.push_back(e);
}

void Device::enablePresentation(std::function<VkSurfaceKHR(VkInstance)> makeSurface)
{
    m_wantPresent = true;
    m_makeSurface = std::move(makeSurface);
}

void Device::requireDeviceUUID(const uint8_t uuid[16])
{
    m_hasRequiredUUID = true;
    memcpy(m_requiredUUID, uuid, 16);
}

void Device::init()
{
    m_validation = wantValidation();
    createInstance();
    if (m_wantPresent)
    {
        m_surface = m_makeSurface(m_instance);
        if (m_surface == VK_NULL_HANDLE)
        {
            throw DeviceError("[avbd_vk] enablePresentation's makeSurface() returned VK_NULL_HANDLE");
        }
    }
    pickPhysicalDevice();
    createLogicalDevice();
    createPoolAndCache();
}

void Device::adopt(VkInstance instance, VkPhysicalDevice physical, VkDevice device, uint32_t queueFamily,
                   uint32_t queueIndex, uint32_t apiVersion)
{
    if (!instance || !physical || !device)
        throw DeviceError("[avbd_vk] adopt: null VkInstance / VkPhysicalDevice / VkDevice");
    if (apiVersion < VK_API_VERSION_1_3)
        throw DeviceError("[avbd_vk] adopt: the Vulkan API version must be 1.3 or newer");
    m_adopted = true;
    m_instance = instance;
    m_physicalDevice = physical;
    m_device = device;
    m_validation = false;

    uint32_t famCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &famCount, nullptr);
    std::vector<VkQueueFamilyProperties> fams(famCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &famCount, fams.data());
    if (queueFamily >= famCount || !(fams[queueFamily].queueFlags & VK_QUEUE_COMPUTE_BIT) ||
        queueIndex >= fams[queueFamily].queueCount)
        throw DeviceError("[avbd_vk] adopt: the queue family/index is not a compute queue of the device");
    m_computeQueueFamily = queueFamily;
    m_timestampValidBits = fams[queueFamily].timestampValidBits;

    DeviceInfo info;
    std::string why;
    if (!evaluateDevice(physical, info, why))
        throw DeviceError("[avbd_vk] adopt: " + why);
    applyInfo(info);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);
    snprintf(m_deviceName, sizeof(m_deviceName), "%s", props.deviceName);
    m_maxSharedMemory = props.limits.maxComputeSharedMemorySize;
    vkGetDeviceQueue(device, queueFamily, queueIndex, &m_computeQueue);
    createPoolAndCache();
}

// The driver recompiles every SPIR-V module on every launch unless the application keeps a
// pipeline cache: csCcdToi alone is ~13 s on an RTX 4070 SUPER, ~22 s for the whole kernel set,
// which left the demo's window unresponsive for that long. The cache is a file per device
// (vendor, device id, driver version in the name; the driver also rejects a stale blob by its
// own header), at AVBD_PIPELINE_CACHE_DIR, %LOCALAPPDATA%/avbd3d, $XDG_CACHE_HOME/avbd3d or ~/.cache/avbd3d. Best effort throughout.
static std::string pipelineCachePath(VkPhysicalDevice pd)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    std::string dir;
    if (const char *e = getenv("AVBD_PIPELINE_CACHE_DIR"))
        dir = e;
    else if (const char *la = getenv("LOCALAPPDATA"))
        dir = std::string(la) + "/avbd3d";
    else if (const char *xdg = getenv("XDG_CACHE_HOME"))
        dir = std::string(xdg) + "/avbd3d";
    else if (const char *home = getenv("HOME"))
        dir = std::string(home) + "/.cache/avbd3d";
    if (dir.empty() || dir == "off")
        return {};
    char name[96];
    snprintf(name, sizeof(name), "/pipeline_cache_%04x_%04x_%08x.bin", props.vendorID, props.deviceID,
             props.driverVersion);
    return dir + name;
}

void Device::createPoolAndCache()
{
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_computeQueueFamily;
    VK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool));

    m_pipelineCachePath = pipelineCachePath(m_physicalDevice);
    std::vector<char> blob;
    if (!m_pipelineCachePath.empty())
    {
        if (FILE *f = fopen(m_pipelineCachePath.c_str(), "rb"))
        {
            fseek(f, 0, SEEK_END);
            const long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (n > 0)
            {
                blob.resize((size_t)n);
                if (fread(blob.data(), 1, blob.size(), f) != blob.size())
                    blob.clear();
            }
            fclose(f);
        }
    }
    VkPipelineCacheCreateInfo cacheInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    cacheInfo.initialDataSize = blob.size();
    cacheInfo.pInitialData = blob.empty() ? nullptr : blob.data();
    VK_CHECK(vkCreatePipelineCache(m_device, &cacheInfo, nullptr, &m_pipelineCache));
}

void Device::savePipelineCache()
{
    if (!m_pipelineCache || m_pipelineCachePath.empty())
        return;
    size_t n = 0;
    if (vkGetPipelineCacheData(m_device, m_pipelineCache, &n, nullptr) != VK_SUCCESS || n == 0)
        return;
    std::vector<char> blob(n);
    if (vkGetPipelineCacheData(m_device, m_pipelineCache, &n, blob.data()) != VK_SUCCESS)
        return;
    const size_t slash = m_pipelineCachePath.find_last_of("/\\");
    if (slash != std::string::npos)
        std::filesystem::create_directories(m_pipelineCachePath.substr(0, slash));
    // Write beside and rename, so a concurrent process (ctest runs several) never reads half a file.
    char tmp[16];
    snprintf(tmp, sizeof(tmp), ".%u.tmp", (unsigned)processId());
    const std::string tmpPath = m_pipelineCachePath + tmp;
    if (FILE *f = fopen(tmpPath.c_str(), "wb"))
    {
        const bool ok = fwrite(blob.data(), 1, n, f) == n;
        fclose(f);
        std::error_code ec;
        if (ok)
            std::filesystem::rename(tmpPath, m_pipelineCachePath, ec);
        if (!ok || ec)
            std::filesystem::remove(tmpPath, ec);
    }
}

void Device::shutdown()
{
    if (m_adopted)
    {
        // Host-owned handles: destroy only what adopt() created.
        if (m_device)
        {
            vkDeviceWaitIdle(m_device);
            savePipelineCache();
            if (m_pipelineCache)
                vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
            if (m_commandPool)
                vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        }
        m_pipelineCache = VK_NULL_HANDLE;
        m_commandPool = VK_NULL_HANDLE;
        m_device = VK_NULL_HANDLE;
        m_instance = VK_NULL_HANDLE;
        m_physicalDevice = VK_NULL_HANDLE;
        m_computeQueue = VK_NULL_HANDLE;
        m_sideQueue = m_copyQueue = VK_NULL_HANDLE;
        m_sideFamily = m_copyFamily = ~0u;
        m_sharedFamilies.clear();
        m_adopted = false;
        return;
    }
    if (m_device)
    {
        vkDeviceWaitIdle(m_device);
        savePipelineCache();
        if (m_pipelineCache)
            vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
        if (m_commandPool)
            vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
        m_sideQueue = m_copyQueue = VK_NULL_HANDLE;
        m_sideFamily = m_copyFamily = ~0u;
        m_sharedFamilies.clear();
        m_pipelineCache = VK_NULL_HANDLE;
        m_commandPool = VK_NULL_HANDLE;
    }
    if (m_debugMessenger)
    {
        auto destroy =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy)
            destroy(m_instance, m_debugMessenger, nullptr);
        m_debugMessenger = VK_NULL_HANDLE;
    }
    if (m_surface)
    {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }
    if (m_instance)
    {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

void Device::createInstance()
{
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "avbd_vk";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char *> layers;
    std::vector<const char *> extensions;

    if (m_validation)
    {
        // Only enabled if actually present -- do not hard-fail a release box that
        // lacks the validation layer package just because AVBD_VK_VALIDATION=1 leaked
        // into its environment.
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> avail(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, avail.data());
        bool has = false;
        for (auto &l : avail)
            if (strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0)
                has = true;
        if (has)
        {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        else
        {
            fprintf(stderr, "[avbd_vk] validation requested but VK_LAYER_KHRONOS_validation not found\n");
            m_validation = false;
        }
    }

    for (auto *e : m_extraInstanceExtensions)
        extensions.push_back(e);

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &appInfo;
    ci.enabledLayerCount = (uint32_t)layers.size();
    ci.ppEnabledLayerNames = layers.data();
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();

    // Synchronization validation rides along with the core layer: the one-submit step is
    // all barriers, and a missing one shows up as a hazard here long before it shows up
    // as a wrong pose.
    VkValidationFeatureEnableEXT syncval = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    features.enabledValidationFeatureCount = 1;
    features.pEnabledValidationFeatures = &syncval;
    if (m_validation)
        ci.pNext = &features;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));

    if (m_validation)
    {
        auto create =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
        if (create)
        {
            VkDebugUtilsMessengerCreateInfoEXT dci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            dci.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debugCallback;
            VK_CHECK(create(m_instance, &dci, nullptr, &m_debugMessenger));
        }
    }
}

void Device::pickPhysicalDevice()
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0)
    {
        throw DeviceError("[avbd_vk] no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    // With enablePresentation() active, a candidate must have SOME queue family that
    // is both GRAPHICS|COMPUTE and can present to m_surface -- same-queue compute +
    // graphics + present, so the caller never juggles ownership transfers.
    auto supportsPresent = [&](VkPhysicalDevice d) -> bool
    {
        if (!m_wantPresent)
            return true;
        uint32_t fc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &fc, nullptr);
        std::vector<VkQueueFamilyProperties> fams(fc);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &fc, fams.data());
        for (uint32_t i = 0; i < fc; i++)
        {
            if (!(fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !(fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
                continue;
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, m_surface, &present);
            if (present)
                return true;
        }
        return false;
    };

    VkPhysicalDevice best = VK_NULL_HANDLE;
    std::string rejected;
    int bestScore = -1;
    uint8_t bestLuid[8] = {};
    for (auto d : devices)
    {
        if (!supportsPresent(d))
            continue;

        DeviceInfo info;
        std::string why;
        if (!evaluateDevice(d, info, why))
        {
            VkPhysicalDeviceProperties p0;
            vkGetPhysicalDeviceProperties(d, &p0);
            rejected += std::string("  ") + p0.deviceName + ": " + why + "\n";
            continue;
        }

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);

        VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(d, &props2);

        // Adapter pinning: when a caller wants Vulkan on
        // the same physical GPU as an existing D3D11 device, LUID match is not
        // a tiebreaker among heuristics, it is the whole selection -- picking the
        // "best" GPU independently is exactly the bug this avoids.
        if (m_hasRequiredLuid)
        {
            if (!idProps.deviceLUIDValid || memcmp(idProps.deviceLUID, m_requiredLuid, 8) != 0)
                continue;
            best = d;
            memcpy(bestLuid, idProps.deviceLUID, 8);
            bestScore = 1;
            break;
        }
        if (m_hasRequiredUUID)
        {
            if (memcmp(idProps.deviceUUID, m_requiredUUID, 16) != 0)
                continue;
            best = d;
            if (idProps.deviceLUIDValid)
                memcpy(bestLuid, idProps.deviceLUID, 8);
            bestScore = 1;
            break;
        }

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            score += 100;
        if (strstr(props.deviceName, "NVIDIA"))
            score += 10;
        if (score > bestScore)
        {
            bestScore = score;
            best = d;
            if (idProps.deviceLUIDValid)
                memcpy(bestLuid, idProps.deviceLUID, 8);
        }
    }

    if (best == VK_NULL_HANDLE)
    {
        std::string msg = m_hasRequiredLuid ? "[avbd_vk] no Vulkan physical device matches the requested adapter LUID"
                                            : "[avbd_vk] no suitable Vulkan physical device found";
        if (!rejected.empty())
            msg += "; rejected devices:\n" + rejected;
        fprintf(stderr, "%s\n", msg.c_str());
        throw DeviceError(msg);
    }
    m_physicalDevice = best;
    memcpy(m_adapterLuid, bestLuid, 8);
    {
        VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props2ForId{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2ForId.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2ForId);
        memcpy(m_deviceUUID, idProps.deviceUUID, 16);
    }

    DeviceInfo info;
    std::string why;
    evaluateDevice(m_physicalDevice, info, why); // passed in the loop above
    applyInfo(info);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    snprintf(m_deviceName, sizeof(m_deviceName), "%s", props.deviceName);
    m_maxSharedMemory = props.limits.maxComputeSharedMemorySize;
    if (m_verbose)
        fprintf(stderr, "[avbd_vk] using device: %s (subgroup size %u%s)\n", props.deviceName, m_subgroupSize,
                m_pinnedSubgroup ? ", pinned to 32" : "");
}

void Device::applyInfo(const DeviceInfo &info)
{
    m_apiVersion = info.apiVersion;
    m_subgroupSize = info.subgroupSize;
    m_timestampPeriodNs = info.timestampPeriodNs;
    m_haveAtomicFloat = info.atomicFloat;
    m_haveSubgroupSizeControl = info.subgroupSizeControl;
    m_pinnedSubgroup = info.pinSubgroup;
    m_extensions = info.extensions;
}

bool Device::evaluateDevice(VkPhysicalDevice d, DeviceInfo &info, std::string &why) const
{
    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    p11.pNext = &p13;
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &p11;
    vkGetPhysicalDeviceProperties2(d, &props2);
    info.apiVersion = props2.properties.apiVersion;
    info.timestampPeriodNs = props2.properties.limits.timestampPeriod;
    info.subgroupSize = p11.subgroupSize;

    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, exts.data());
    auto has = [&](const char *name)
    {
        for (auto &e : exts)
            if (strcmp(e.extensionName, name) == 0)
                return true;
        return false;
    };
    for (auto &e : exts)
        info.extensions.push_back(e.extensionName);

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT fAF{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    const bool afExt = has(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    if (afExt)
        f13.pNext = &fAF;
    vkGetPhysicalDeviceFeatures2(d, &f2);
    info.atomicFloat = afExt && fAF.shaderBufferFloat32AtomicAdd;
    info.subgroupSizeControl =
        f13.subgroupSizeControl && (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;

    std::string missing;
    auto need = [&](bool ok, const std::string &what)
    {
        if (!ok)
            missing += std::string(missing.empty() ? "" : ", ") + what;
    };
    need(info.apiVersion >= VK_API_VERSION_1_3, "Vulkan 1.3 (the device reports an older API version)");
    need(f2.features.shaderInt64, "shaderInt64");
    need(f12.bufferDeviceAddress, "bufferDeviceAddress");
    need(f12.scalarBlockLayout, "scalarBlockLayout");
    need(f12.shaderInt8, "shaderInt8");
    need(f12.timelineSemaphore, "timelineSemaphore");
    need(f12.hostQueryReset, "hostQueryReset");
    need(f13.synchronization2, "synchronization2");
    if (m_requireAtomicFloat)
        need(info.atomicFloat, "VK_EXT_shader_atomic_float (shaderBufferFloat32AtomicAdd)");
#ifdef _WIN32
    if (m_externalWin32)
        need(has(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) && has(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME),
             "VK_KHR_external_memory_win32 / VK_KHR_external_semaphore_win32");
#else
    if (m_externalWin32)
        need(false, "VK_KHR_external_memory_win32 (Windows only)");
#endif
    if (m_wantPresent)
        need(has(VK_KHR_SWAPCHAIN_EXTENSION_NAME), "VK_KHR_swapchain");

    if (m_subgroupPolicy == SubgroupPolicy::Generic3264)
    {
        const VkSubgroupFeatureFlags ops =
            VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        need((p11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                 (p11.subgroupSupportedOperations & ops) == ops,
             "compute subgroup operations (basic, ballot, arithmetic)");
        const bool native = info.subgroupSize == 32 || info.subgroupSize == 64;
        const char *force = std::getenv("AVBD_VK_PIN_SUBGROUP"); // debug override: exercise the pinned path
        const bool forcePin = force && force[0] == '3' && force[1] == '2';
        if (!native || forcePin)
        {
            if (info.subgroupSizeControl && p13.minSubgroupSize <= 32 && 32 <= p13.maxSubgroupSize)
                info.pinSubgroup = 32;
            else if (!native)
                need(false, "subgroup size 32 or 64 (the device's is " + std::to_string(info.subgroupSize) +
                                " and it cannot pin 32 through subgroup size control)");
        }
    }

    if (!missing.empty())
    {
        why = "missing " + missing;
        return false;
    }
    return true;
}

void Device::createLogicalDevice()
{
    uint32_t famCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &famCount, nullptr);
    std::vector<VkQueueFamilyProperties> fams(famCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &famCount, fams.data());

    for (uint32_t i = 0; i < famCount; i++)
    {
        if (m_wantPresent)
        {
            if (!(fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !(fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
                continue;
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(m_physicalDevice, i, m_surface, &present);
            if (!present)
                continue;
            m_computeQueueFamily = i;
            break;
        }
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            // A compute-only family is preferred when asked for; otherwise the first compute one.
            if (m_preferDedicatedCompute && (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                if (m_computeQueueFamily == ~0u)
                    m_computeQueueFamily = i; // fallback, keep looking
                continue;
            }
            m_computeQueueFamily = i;
            break;
        }
    }
    if (m_computeQueueFamily == ~0u)
    {
        throw DeviceError(m_wantPresent ? "[avbd_vk] no GRAPHICS|COMPUTE queue family with present support found"
                                        : "[avbd_vk] no compute-capable queue family found");
    }
    m_timestampValidBits = fams[m_computeQueueFamily].timestampValidBits;

    // Side queues: a second compute queue (another family, preferably without GRAPHICS, else
    // a second queue of the main family) and a transfer-only family for DMA copies.
    uint32_t sideIndex = 0;
    for (uint32_t i = 0; i < famCount; i++)
    {
        if (i == m_computeQueueFamily || !(fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
            continue;
        if (m_sideFamily == ~0u || !(fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            m_sideFamily = i;
    }
    if (m_sideFamily == ~0u && fams[m_computeQueueFamily].queueCount >= 2)
    {
        m_sideFamily = m_computeQueueFamily;
        sideIndex = 1;
    }
    for (uint32_t i = 0; i < famCount; i++)
        if (i != m_computeQueueFamily && i != m_sideFamily && (fams[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(fams[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
        {
            m_copyFamily = i;
            break;
        }

    // The main queue keeps the top priority; the side queues serve small latency-bound reads.
    const float priorities[2] = {1.0f, 1.0f};
    std::vector<VkDeviceQueueCreateInfo> qcis;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = m_computeQueueFamily;
    qci.queueCount = sideIndex == 1 ? 2 : 1;
    qci.pQueuePriorities = priorities;
    qcis.push_back(qci);
    if (m_sideFamily != ~0u && sideIndex == 0)
    {
        qci.queueFamilyIndex = m_sideFamily;
        qci.queueCount = 1;
        qcis.push_back(qci);
    }
    if (m_copyFamily != ~0u)
    {
        qci.queueFamilyIndex = m_copyFamily;
        qci.queueCount = 1;
        qcis.push_back(qci);
    }

    // Feature chain: Vulkan 1.2 (BDA, timeline semaphores, shaderInt64, atomics) +
    // Vulkan 1.3 (sync2, though we don't strictly require it yet) + subgroup size
    // control + the shader atomic float extension.
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.hostQueryReset = VK_TRUE;
    // slangc's -fvk-use-scalar-layout emits scalar (std430-relaxed) offsets for ordinary
    // SSBO blocks, but buffer_reference (PhysicalStorageBuffer) blocks -- e.g.
    // WriteInstancesArgs.out's BodyInstance* in render.slang -- follow the SPIR-V
    // relaxed-block-layout rule instead, which BodyInstance's {vec3 pos; vec4 rot; ...}
    // fails (rot's vec4 at offset 12 straddles the 16-byte boundary). BodyInstance's
    // 44-byte layout is a fixed ABI (matches BodyInstance and
    // Unity's generated struct byte-for-byte), so it cannot be padded/reordered to fit
    // relaxed rules; enabling the core (1.2) scalarBlockLayout feature makes that same
    // straddling layout legal for buffer_reference blocks too, which is what the
    // validator's own message above suggests ("may be allowed if you enable the
    // scalarBlockLayout feature").
    f12.scalarBlockLayout = VK_TRUE;

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.synchronization2 = VK_TRUE;
    f13.subgroupSizeControl = m_haveSubgroupSizeControl ? VK_TRUE : VK_FALSE;
    f13.pNext = &f12;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT fAtomicFloat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    fAtomicFloat.shaderBufferFloat32AtomicAdd = VK_TRUE;
    fAtomicFloat.pNext = &f13;

    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.features.shaderInt64 = VK_TRUE;
    features2.pNext = m_haveAtomicFloat ? (void *)&fAtomicFloat : (void *)&f13;

    auto supported = [&](const char *name)
    {
        for (auto &e : m_extensions)
            if (e == name)
                return true;
        return false;
    };
    std::vector<const char *> deviceExtensions;
    // Each is enabled only if the device lists it (core in 1.2/1.3 or optional).
    for (const char *name :
         {VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
          // Diagnostic-only: VkPhysicalDeviceMemoryBudgetPropertiesEXT.
          VK_EXT_MEMORY_BUDGET_EXTENSION_NAME})
        if (supported(name))
            deviceExtensions.push_back(name);
    if (m_haveAtomicFloat)
        deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

    if (m_wantPresent)
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

#ifdef _WIN32
    if (m_externalWin32)
    {
        // Capability extensions (VK_KHR_external_memory_capabilities /
        // _external_semaphore_capabilities) are instance-level and promoted to core
        // in the 1.3 instance this backend creates, so only the win32-handle halves
        // need requesting here.
        deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    }
#endif

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &features2;
    dci.queueCreateInfoCount = (uint32_t)qcis.size();
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = (uint32_t)deviceExtensions.size();
    dci.ppEnabledExtensionNames = deviceExtensions.data();
    VK_CHECK(vkCreateDevice(m_physicalDevice, &dci, nullptr, &m_device));

    vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);
    if (m_sideFamily != ~0u)
        vkGetDeviceQueue(m_device, m_sideFamily, sideIndex, &m_sideQueue);
    if (m_copyFamily != ~0u)
        vkGetDeviceQueue(m_device, m_copyFamily, 0, &m_copyQueue);
    m_sharedFamilies.clear();
    for (uint32_t f : {m_computeQueueFamily, m_sideFamily, m_copyFamily})
        if (f != ~0u && std::find(m_sharedFamilies.begin(), m_sharedFamilies.end(), f) == m_sharedFamilies.end())
            m_sharedFamilies.push_back(f);

    if (m_validation)
    {
        cmdBeginDebugLabel =
            (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(m_instance, "vkCmdBeginDebugUtilsLabelEXT");
        cmdEndDebugLabel =
            (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(m_instance, "vkCmdEndDebugUtilsLabelEXT");
        setDebugObjectName =
            (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectNameEXT");
    }
}

void Device::queueSubmit(const VkSubmitInfo &si, bool step)
{
    if (step)
    {
        const uint64_t n = ++m_stepSubmits;
        if (m_failSubmitAt != 0 && n >= m_failSubmitAt)
        {
            // Test hook (AVBD2D_DEBUG_FAIL_SUBMIT): behave as if the driver reported device loss.
            fprintf(stderr, "[avbd_vk] debug: simulating VK_ERROR_DEVICE_LOST on submit %llu\n", (unsigned long long)n);
            throw VkError(VK_ERROR_DEVICE_LOST, "simulated device loss (debug hook)");
        }
    }
    VK_CHECK(vkQueueSubmit(m_computeQueue, 1, &si, VK_NULL_HANDLE));
}

void Device::checkBudget(uint64_t bytes, uint64_t reuse) const
{
    if (m_budget == 0)
        return;
    const uint64_t cur = m_allocated.load();
    const uint64_t after = cur - (reuse < cur ? reuse : cur) + bytes;
    if (after > m_budget)
        throw BudgetExceeded("[avbd_vk] memory budget exceeded: " + std::to_string(after) + " bytes needed, budget " +
                             std::to_string(m_budget));
}

} // namespace avbdvk
