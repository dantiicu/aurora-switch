#pragma once

#include "../gx/pipeline.hpp"

#include <optional>

#include <vulkan/vulkan.h>

namespace aurora::vk {

struct GxPipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkShaderModule vertexShader = VK_NULL_HANDLE;
  VkShaderModule fragmentShader = VK_NULL_HANDLE;
};

VkPipelineLayout gx_pipeline_layout() noexcept;
VkDescriptorSetLayout gx_static_descriptor_layout() noexcept;
VkDescriptorSetLayout gx_uniform_descriptor_layout() noexcept;
VkDescriptorSetLayout gx_texture_descriptor_layout() noexcept;
bool initialize_gx_pipeline_layout() noexcept;
void shutdown_gx_pipeline_layout() noexcept;

std::optional<GxPipeline> create_gx_pipeline(const gx::PipelineConfig& config, VkRenderPass renderPass,
                                             bool hasDepthAttachment, const char* label = nullptr) noexcept;
void destroy_gx_pipeline(GxPipeline& pipeline) noexcept;

} // namespace aurora::vk
