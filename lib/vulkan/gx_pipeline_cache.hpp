#pragma once

#include "gx_pipeline.hpp"
#include "../gfx/common.hpp"

#include <vulkan/vulkan.h>

namespace aurora::vk {

void remember_gx_pipeline_config(gfx::PipelineRef ref, const gx::PipelineConfig& config) noexcept;
GxPipeline* get_or_create_gx_pipeline(gfx::PipelineRef ref, VkRenderPass renderPass,
                                       bool hasDepthAttachment) noexcept;
void shutdown_gx_pipeline_cache() noexcept;

} // namespace aurora::vk
