#pragma once

// Small helpers shared by the Vulkan backend translation units: a check macro that
// reports and throws avbdvk::VkError. A caller that does not catch it terminates the
// process (std::terminate -> abort), the old behaviour; the C ABIs and VkWorld::stepBegin/
// stepEnd catch it and turn it into an error (the world goes dead, see StepResult::vkError).
// Destroy paths never use VK_CHECK, so unwinding cannot throw again.
// AVBD_VK_ABORT_ON_ERROR=1 restores print + abort() at the failing call (debugging: the
// device-lost recipe in docs/TESTING.md wants the failing stack, not an error code).

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>

namespace avbdvk
{

// Workgroup counts: ceil(n / d). The signed form returns 0 for n <= 0 (an empty phase dispatches
// nothing); the unsigned form is the plain ceiling.
constexpr uint32_t ceilDiv(uint32_t n, uint32_t d)
{
    return (n + d - 1) / d;
}
constexpr uint32_t ceilDiv(int n, int d)
{
    return n <= 0 ? 0u : (uint32_t)((n + d - 1) / d);
}

// A failed Vulkan call (or an unusable device / exhausted budget, see below).
class VkError : public std::runtime_error
{
public:
    VkError(VkResult r, const std::string &what) : std::runtime_error(what), m_result(r) {}
    VkResult result() const { return m_result; }

private:
    VkResult m_result;
};

// The device (or an adopted external device) cannot run the solver; what() names why.
class DeviceError : public VkError
{
public:
    explicit DeviceError(const std::string &what) : VkError(VK_ERROR_INITIALIZATION_FAILED, what) {}
};

// An allocation would exceed Device::setMemoryBudget(). Nothing was allocated.
class BudgetExceeded : public VkError
{
public:
    explicit BudgetExceeded(const std::string &what) : VkError(VK_ERROR_OUT_OF_DEVICE_MEMORY, what) {}
};

[[noreturn]] inline void vkCheckFail(VkResult r, const char *file, int line, const char *expr)
{
    fprintf(stderr, "Vulkan error %d at %s:%d\n    %s\n", (int)r, file, line, expr);
    throw VkError(r, std::string("Vulkan error ") + std::to_string((int)r) + " at " + file + ":" +
                         std::to_string(line) + " (" + expr + ")");
}

} // namespace avbdvk

#define VK_CHECK(expr)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        VkResult avbd_vk_res_ = (expr);                                                                                \
        if (avbd_vk_res_ != VK_SUCCESS)                                                                                \
            ::avbdvk::vkCheckFail(avbd_vk_res_, __FILE__, __LINE__, #expr);                                            \
    } while (0)
