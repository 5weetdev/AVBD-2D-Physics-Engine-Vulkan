#pragma once

#include "vk_buffer.h"

#include <cstdint>

namespace avbdvk
{

struct GateDispatchArgsGpu
{
    VkDeviceAddress src = 0, dst = 0, flag = 0;
    int32_t count = 0, _pad0 = 0;
};
static_assert(sizeof(GateDispatchArgsGpu) % 8 == 0, "layout");

} // namespace avbdvk
