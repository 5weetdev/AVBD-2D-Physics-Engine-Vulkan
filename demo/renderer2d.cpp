#include "renderer2d.h"

#include "vk_device.h"
#include "vk_util.h"

#include "scene2d_spirv.h" // generated: scene2d_spirv / scene2d_spirv_words

#include <algorithm>
#include <cstddef>
#include <cstring>

void Renderer2D::init(avbdvk::Device &dev, VkRenderPass pass)
{
    m_dev = &dev;
    VkDevice d = dev.device();

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = scene2d_spirv_words * sizeof(uint32_t);
    smci.pCode = scene2d_spirv;
    VK_CHECK(vkCreateShaderModule(d, &smci, nullptr, &m_module));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = 16;
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(d, &plci, nullptr, &m_layout));

    VkVertexInputBindingDescription binding{0, sizeof(Instance2D), VK_VERTEX_INPUT_RATE_INSTANCE};
    VkVertexInputAttributeDescription attrs[6]{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Instance2D, x)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT, (uint32_t)offsetof(Instance2D, hx)};
    attrs[2] = {2, 0, VK_FORMAT_R8G8B8A8_UNORM, (uint32_t)offsetof(Instance2D, rgba)};
    attrs[3] = {3, 0, VK_FORMAT_R32_UINT, (uint32_t)offsetof(Instance2D, shape)};
    attrs[4] = {4, 0, VK_FORMAT_R32_UINT, (uint32_t)offsetof(Instance2D, flags)};
    attrs[5] = {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(Instance2D, ex)};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 6;
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                         VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    const VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = m_module;
    stages[0].pName = "vsMain";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = m_module;
    stages[1].pName = "fsMain";

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dyn;
    gpci.layout = m_layout;
    gpci.renderPass = pass;
    gpci.subpass = 0;
    VK_CHECK(vkCreateGraphicsPipelines(d, dev.pipelineCache(), 1, &gpci, nullptr, &m_pipeline));
}

void Renderer2D::destroy()
{
    if (!m_dev)
        return;
    VkDevice d = m_dev->device();
    vkDeviceWaitIdle(d);
    for (int i = 0; i < kMaxFramesInFlight; i++)
    {
        m_inst[i].destroy();
        m_capacity[i] = 0;
    }
    if (m_pipeline)
        vkDestroyPipeline(d, m_pipeline, nullptr);
    if (m_layout)
        vkDestroyPipelineLayout(d, m_layout, nullptr);
    if (m_module)
        vkDestroyShaderModule(d, m_module, nullptr);
    m_pipeline = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
    m_module = VK_NULL_HANDLE;
    m_dev = nullptr;
}

void Renderer2D::draw(VkCommandBuffer cmd, int frameSlot, const Instance2D *instances, uint32_t count,
                      const float cam[4])
{
    if (!m_dev || count == 0)
        return;
    const int slot = frameSlot % kMaxFramesInFlight;
    if (m_capacity[slot] < count)
    {
        // The slot's previous frame has finished (the caller waits on its fence), so its
        // buffer can be replaced.
        const uint32_t cap = std::max(count + count / 2, 1024u);
        m_inst[slot].destroy();
        m_inst[slot].init(*m_dev, (VkDeviceSize)cap * sizeof(Instance2D), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          avbdvk::MappedBuffer::HostAccess::Write);
        m_capacity[slot] = cap;
    }
    memcpy(m_inst[slot].mapped(), instances, (size_t)count * sizeof(Instance2D));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdPushConstants(cmd, m_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, cam);
    VkBuffer vb = m_inst[slot].handle();
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
    vkCmdDraw(cmd, 4, count, 0, 0);
}
