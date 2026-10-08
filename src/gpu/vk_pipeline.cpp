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

#include "vk_pipeline.h"
#include "vk_util.h"

namespace avbdvk
{

void PipelineLayout::init(Device &dev)
{
    m_dev = &dev;

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = 8; // one uint64 device address

    VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ci.setLayoutCount = 0;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(dev.device(), &ci, nullptr, &m_layout));
}

void PipelineLayout::destroy()
{
    if (m_dev && m_layout)
        vkDestroyPipelineLayout(m_dev->device(), m_layout, nullptr);
    m_layout = VK_NULL_HANDLE;
}

void Pipeline::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords,
                     const char *entryPoint, const char *debugName)
{
    m_dev = &dev;
    m_layout = layout.handle();
    m_name = debugName ? debugName : (entryPoint ? entryPoint : "unnamed");

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirvWords * sizeof(uint32_t);
    smci.pCode = spirv;
    VK_CHECK(vkCreateShaderModule(dev.device(), &smci, nullptr, &m_module));

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = m_module;
    stage.pName = entryPoint;
    // Subgroup width policy (Device::setSubgroupPolicy): pin 32 when the device's native size
    // is not one the kernels support, or when the debug override forces it. 0 leaves it to the driver.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo pin{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (dev.pinnedSubgroupSize() != 0)
    {
        pin.requiredSubgroupSize = dev.pinnedSubgroupSize();
        stage.pNext = &pin;
    }

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = layout.handle();
    VK_CHECK(vkCreateComputePipelines(dev.device(), dev.pipelineCache(), 1, &ci, nullptr, &m_pipeline));

    if (debugName && dev.setDebugObjectName)
    {
        VkDebugUtilsObjectNameInfoEXT nameInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        nameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
        nameInfo.objectHandle = (uint64_t)m_pipeline;
        nameInfo.pObjectName = debugName;
        dev.setDebugObjectName(dev.device(), &nameInfo);
    }
}

void Pipeline::destroy()
{
    if (!m_dev)
        return;
    if (m_pipeline)
        vkDestroyPipeline(m_dev->device(), m_pipeline, nullptr);
    if (m_module)
        vkDestroyShaderModule(m_dev->device(), m_module, nullptr);
    m_pipeline = VK_NULL_HANDLE;
    m_module = VK_NULL_HANDLE;
}

} // namespace avbdvk
