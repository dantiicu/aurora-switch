#include "gx_pipeline.hpp"

#include "gpu.hpp"
#include "shader_compiler.hpp"
#include "../internal.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace aurora::vk {
namespace {
Module Log("aurora::vk::gx");

VkDescriptorSetLayout g_staticLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout g_uniformLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout g_textureLayout = VK_NULL_HANDLE;
VkPipelineLayout g_pipelineLayout = VK_NULL_HANDLE;

bool vk_check(VkResult result, const char* call) {
  if (result == VK_SUCCESS) {
    return true;
  }
  Log.error("{} failed: {}", call, static_cast<int>(result));
  return false;
}

template <typename T>
uint64_t handle_value(T handle) {
#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
#else
  return static_cast<uint64_t>(handle);
#endif
}

#define AURORA_VK_GX_CHECK(call)                                                                                      \
  do {                                                                                                                \
    if (!vk_check((call), #call)) {                                                                                    \
      return false;                                                                                                   \
    }                                                                                                                 \
  } while (false)

VkBlendFactor to_blend_factor(GXBlendFactor factor, bool isDst) {
  switch (factor) {
  case GX_BL_ZERO:
    return VK_BLEND_FACTOR_ZERO;
  case GX_BL_ONE:
    return VK_BLEND_FACTOR_ONE;
  case GX_BL_SRCCLR:
    return isDst ? VK_BLEND_FACTOR_SRC_COLOR : VK_BLEND_FACTOR_DST_COLOR;
  case GX_BL_INVSRCCLR:
    return isDst ? VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
  case GX_BL_SRCALPHA:
    return VK_BLEND_FACTOR_SRC_ALPHA;
  case GX_BL_INVSRCALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  case GX_BL_DSTALPHA:
    return VK_BLEND_FACTOR_DST_ALPHA;
  case GX_BL_INVDSTALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  default:
    Log.error("Unsupported GX blend factor {}", underlying(factor));
    return VK_BLEND_FACTOR_ONE;
  }
}

VkCompareOp to_compare_op(GXCompare func) {
  switch (func) {
  case GX_NEVER:
    return VK_COMPARE_OP_NEVER;
  case GX_LESS:
    return gx::UseReversedZ ? VK_COMPARE_OP_GREATER : VK_COMPARE_OP_LESS;
  case GX_EQUAL:
    return VK_COMPARE_OP_EQUAL;
  case GX_LEQUAL:
    return gx::UseReversedZ ? VK_COMPARE_OP_GREATER_OR_EQUAL : VK_COMPARE_OP_LESS_OR_EQUAL;
  case GX_GREATER:
    return gx::UseReversedZ ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_GREATER;
  case GX_NEQUAL:
    return VK_COMPARE_OP_NOT_EQUAL;
  case GX_GEQUAL:
    return gx::UseReversedZ ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_GREATER_OR_EQUAL;
  case GX_ALWAYS:
    return VK_COMPARE_OP_ALWAYS;
  default:
    Log.error("Unsupported GX compare op {}", underlying(func));
    return VK_COMPARE_OP_ALWAYS;
  }
}

VkBlendOp to_color_blend_op(GXBlendMode mode, GXLogicOp op) {
  switch (mode) {
  case GX_BM_SUBTRACT:
    return VK_BLEND_OP_REVERSE_SUBTRACT;
  case GX_BM_LOGIC:
    switch (op) {
    case GX_LO_CLEAR:
    case GX_LO_COPY:
    case GX_LO_NOOP:
      return VK_BLEND_OP_ADD;
    default:
      Log.error("Unsupported GX logic op {}", underlying(op));
      return VK_BLEND_OP_ADD;
    }
  case GX_BM_NONE:
  case GX_BM_BLEND:
    return VK_BLEND_OP_ADD;
  default:
    Log.error("Unsupported GX blend mode {}", underlying(mode));
    return VK_BLEND_OP_ADD;
  }
}

VkPipelineColorBlendAttachmentState to_color_blend_attachment(const gx::PipelineConfig& config) {
  VkPipelineColorBlendAttachmentState attachment{};
  attachment.blendEnable = VK_TRUE;
  attachment.colorBlendOp = to_color_blend_op(config.blendMode, config.blendOp);
  attachment.alphaBlendOp = VK_BLEND_OP_ADD;

  switch (config.blendMode) {
  case GX_BM_NONE:
    attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    break;
  case GX_BM_BLEND:
    attachment.srcColorBlendFactor = to_blend_factor(config.blendFacSrc, false);
    attachment.dstColorBlendFactor = to_blend_factor(config.blendFacDst, true);
    break;
  case GX_BM_SUBTRACT:
    attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    break;
  case GX_BM_LOGIC:
    switch (config.blendOp) {
    case GX_LO_CLEAR:
      attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      break;
    case GX_LO_COPY:
      attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      break;
    case GX_LO_NOOP:
      attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      break;
    default:
      attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      break;
    }
    break;
  default:
    attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    break;
  }

  if (config.dstAlpha != UINT32_MAX) {
    attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
    attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  } else {
    attachment.srcAlphaBlendFactor = attachment.srcColorBlendFactor;
    attachment.dstAlphaBlendFactor = attachment.dstColorBlendFactor;
  }

  attachment.colorWriteMask = 0;
  if (config.colorUpdate) {
    attachment.colorWriteMask |= VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
  }
  if (config.alphaUpdate) {
    attachment.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
  }
  return attachment;
}

VkCullModeFlags to_cull_mode(GXCullMode mode) {
  switch (mode) {
  case GX_CULL_NONE:
    return VK_CULL_MODE_NONE;
  case GX_CULL_FRONT:
    return VK_CULL_MODE_FRONT_BIT;
  case GX_CULL_BACK:
    return VK_CULL_MODE_BACK_BIT;
  default:
    Log.error("Unsupported GX cull mode {}", underlying(mode));
    return VK_CULL_MODE_NONE;
  }
}

VkSampleCountFlagBits to_sample_count(uint32_t samples) {
  switch (samples) {
  case 1:
    return VK_SAMPLE_COUNT_1_BIT;
  case 2:
    return VK_SAMPLE_COUNT_2_BIT;
  case 4:
    return VK_SAMPLE_COUNT_4_BIT;
  case 8:
    return VK_SAMPLE_COUNT_8_BIT;
  default:
    Log.error("Unsupported sample count {}", samples);
    return VK_SAMPLE_COUNT_1_BIT;
  }
}

bool create_descriptor_layouts() {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE) {
    Log.error("Cannot create GX pipeline layout before Vulkan device initialization");
    return false;
  }

  const std::array staticBindings{
      VkDescriptorSetLayoutBinding{
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      VkDescriptorSetLayoutBinding{
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      },
  };
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(staticBindings.size());
  layoutInfo.pBindings = staticBindings.data();
  AURORA_VK_GX_CHECK(vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &g_staticLayout));

  const VkDescriptorSetLayoutBinding uniformBinding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
  };
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &uniformBinding;
  AURORA_VK_GX_CHECK(vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &g_uniformLayout));

  std::array<VkDescriptorSetLayoutBinding, gx::MaxTextures * 2> textureBindings{};
  for (uint32_t i = 0; i < gx::MaxTextures; ++i) {
    textureBindings[i * 2] = {
        .binding = i * 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    textureBindings[i * 2 + 1] = {
        .binding = i * 2 + 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
  }
  layoutInfo.bindingCount = static_cast<uint32_t>(textureBindings.size());
  layoutInfo.pBindings = textureBindings.data();
  AURORA_VK_GX_CHECK(vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &g_textureLayout));

  const std::array layouts{g_staticLayout, g_uniformLayout, g_textureLayout};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
  pipelineLayoutInfo.pSetLayouts = layouts.data();
  AURORA_VK_GX_CHECK(vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, nullptr, &g_pipelineLayout));
  return true;
}
} // namespace

VkPipelineLayout gx_pipeline_layout() noexcept { return g_pipelineLayout; }
VkDescriptorSetLayout gx_static_descriptor_layout() noexcept { return g_staticLayout; }
VkDescriptorSetLayout gx_uniform_descriptor_layout() noexcept { return g_uniformLayout; }
VkDescriptorSetLayout gx_texture_descriptor_layout() noexcept { return g_textureLayout; }

bool initialize_gx_pipeline_layout() noexcept {
  if (g_pipelineLayout != VK_NULL_HANDLE) {
    return true;
  }
  return create_descriptor_layouts();
}

void shutdown_gx_pipeline_layout() noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE) {
    g_staticLayout = VK_NULL_HANDLE;
    g_uniformLayout = VK_NULL_HANDLE;
    g_textureLayout = VK_NULL_HANDLE;
    g_pipelineLayout = VK_NULL_HANDLE;
    return;
  }
  if (g_pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(vkDevice, g_pipelineLayout, nullptr);
    g_pipelineLayout = VK_NULL_HANDLE;
  }
  if (g_textureLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(vkDevice, g_textureLayout, nullptr);
    g_textureLayout = VK_NULL_HANDLE;
  }
  if (g_uniformLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(vkDevice, g_uniformLayout, nullptr);
    g_uniformLayout = VK_NULL_HANDLE;
  }
  if (g_staticLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(vkDevice, g_staticLayout, nullptr);
    g_staticLayout = VK_NULL_HANDLE;
  }
}

std::optional<GxPipeline> create_gx_pipeline(const gx::PipelineConfig& config, VkRenderPass renderPass,
                                             bool hasDepthAttachment, const char* label) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    Log.error("Cannot create GX pipeline without Vulkan device/render pass");
    return std::nullopt;
  }
  if (!initialize_gx_pipeline_layout()) {
    return std::nullopt;
  }

  const auto shaderSource = gx::build_shader_source(config.shaderConfig);
  const std::string pipelineLabel = label != nullptr ? label : "GX Pipeline";
  Log.debug("[DirectVK] {} build begin shaderBytes={} msaa={} depthCompare={} depthWrite={} blend={} cull={}",
            pipelineLabel, shaderSource.size(), config.msaaSamples, config.depthCompare, config.depthUpdate,
            underlying(config.blendMode), underlying(config.cullMode));
  Log.debug("[DirectVK] {} vertex shader module begin", pipelineLabel);
  auto vertexShader = create_shader_module_from_wgsl(shaderSource, "vs_main", pipelineLabel + " VS");
  Log.debug("[DirectVK] {} vertex shader module done handle={}", pipelineLabel, handle_value(vertexShader));
  Log.debug("[DirectVK] {} fragment shader module begin", pipelineLabel);
  auto fragmentShader = create_shader_module_from_wgsl(shaderSource, "fs_main", pipelineLabel + " FS");
  Log.debug("[DirectVK] {} fragment shader module done handle={}", pipelineLabel, handle_value(fragmentShader));
  GxPipeline out{
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
  };
  if (out.vertexShader == VK_NULL_HANDLE || out.fragmentShader == VK_NULL_HANDLE) {
    destroy_gx_pipeline(out);
    return std::nullopt;
  }

  const std::array stages{
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = out.vertexShader,
          .pName = "vs_main",
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = out.fragmentShader,
          .pName = "fs_main",
      },
  };

  VkPipelineVertexInputStateCreateInfo vertexInput{};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.cullMode = to_cull_mode(config.cullMode);
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.lineWidth = 1.f;

  VkPipelineMultisampleStateCreateInfo multisample{};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = to_sample_count(config.msaaSamples);

  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = hasDepthAttachment && config.depthCompare ? VK_TRUE : VK_FALSE;
  depthStencil.depthWriteEnable = hasDepthAttachment && config.depthCompare && config.depthUpdate ? VK_TRUE : VK_FALSE;
  depthStencil.depthCompareOp = config.depthCompare ? to_compare_op(config.depthFunc) : VK_COMPARE_OP_ALWAYS;

  const auto colorBlendAttachment = to_color_blend_attachment(config);
  VkPipelineColorBlendStateCreateInfo colorBlend{};
  colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &colorBlendAttachment;

  const std::array dynamicStates{
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_BLEND_CONSTANTS,
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages = stages.data();
  pipelineInfo.pVertexInputState = &vertexInput;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisample;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlend;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = g_pipelineLayout;
  pipelineInfo.renderPass = renderPass;

  Log.debug("[DirectVK] {} vkCreateGraphicsPipelines begin", pipelineLabel);
  const VkResult result = vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &out.pipeline);
  if (result != VK_SUCCESS) {
    Log.error("vkCreateGraphicsPipelines failed for {}: {}", pipelineLabel, static_cast<int>(result));
    destroy_gx_pipeline(out);
    return std::nullopt;
  }
  Log.debug("[DirectVK] {} vkCreateGraphicsPipelines done handle={}", pipelineLabel, handle_value(out.pipeline));

  return out;
}

void destroy_gx_pipeline(GxPipeline& pipeline) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE) {
    pipeline = {};
    return;
  }
  if (pipeline.pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(vkDevice, pipeline.pipeline, nullptr);
  }
  if (pipeline.fragmentShader != VK_NULL_HANDLE) {
    vkDestroyShaderModule(vkDevice, pipeline.fragmentShader, nullptr);
  }
  if (pipeline.vertexShader != VK_NULL_HANDLE) {
    vkDestroyShaderModule(vkDevice, pipeline.vertexShader, nullptr);
  }
  pipeline = {};
}

} // namespace aurora::vk
