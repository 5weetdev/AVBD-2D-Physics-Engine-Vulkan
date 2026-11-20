#pragma once

// Exception-to-error translation shared by the 2D and 3D C APIs. Each API keeps its own error enum and
// maps the Failure kind to it; the classification of what went wrong lives here once.

#include "vk_util.h"

#include <new>
#include <string>

namespace avbdvk
{

enum class Failure
{
    HostOutOfMemory,   // std::bad_alloc
    DeviceOutOfMemory, // a Vulkan call or a memory budget ran out of memory
    DeviceUnusable,    // the device cannot run the solver (DeviceError)
    DeviceLost,        // VK_ERROR_DEVICE_LOST
    VulkanError,       // any other failed Vulkan call
    Exception,         // another std::exception
    Unknown            // not an exception type we know
};

struct Classified
{
    Failure kind;
    std::string what;
};

inline bool isOutOfMemory(const VkError &e)
{
    const VkResult r = e.result();
    return r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY;
}

// Classifies the exception being handled. Call it from inside a catch block.
inline Classified classifyCurrentException()
{
    try
    {
        throw;
    }
    catch (const std::bad_alloc &)
    {
        return {Failure::HostOutOfMemory, "out of host memory"};
    }
    catch (const DeviceError &e)
    {
        return {Failure::DeviceUnusable, e.what()};
    }
    catch (const VkError &e)
    {
        if (isOutOfMemory(e))
            return {Failure::DeviceOutOfMemory, e.what()};
        return {e.result() == VK_ERROR_DEVICE_LOST ? Failure::DeviceLost : Failure::VulkanError, e.what()};
    }
    catch (const std::exception &e)
    {
        return {Failure::Exception, e.what()};
    }
    catch (...)
    {
        return {Failure::Unknown, "unknown exception"};
    }
}

// Runs f(); on any exception returns onFailure(classification).
template <class R, class F, class OnFailure> R guardedCall(F &&f, OnFailure &&onFailure)
{
    try
    {
        return f();
    }
    catch (...)
    {
        return onFailure(classifyCurrentException());
    }
}

} // namespace avbdvk
