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

// Small helpers shared by the Vulkan backend translation units: a check macro that
// reports and throws avbdvk::VkError. A caller that does not catch it terminates the
// process (std::terminate -> abort), the old behaviour; the 2D C ABI catches it and
// returns an error code instead. Destroy paths never use VK_CHECK, so unwinding cannot
// throw again.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>

namespace avbdvk
{

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

#define VK_CHECK(expr)                                                                   \
    do                                                                                   \
    {                                                                                    \
        VkResult avbd_vk_res_ = (expr);                                                  \
        if (avbd_vk_res_ != VK_SUCCESS)                                                  \
            ::avbdvk::vkCheckFail(avbd_vk_res_, __FILE__, __LINE__, #expr);              \
    } while (0)
