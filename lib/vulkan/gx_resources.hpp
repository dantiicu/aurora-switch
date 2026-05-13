#pragma once

#include "resources.hpp"

#include <array>

#include <vulkan/vulkan.h>

namespace aurora::gx {
struct GXBindGroups;
} // namespace aurora::gx

namespace aurora::vk {

inline constexpr uint32_t GxStagingBufferCount = 3;

struct GxResources {
  Buffer vertexBuffer;
  Buffer uniformBuffer;
  Buffer indexBuffer;
  Buffer storageBuffer;
  std::array<Buffer, GxStagingBufferCount> stagingBuffers;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet staticDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet uniformDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet emptyTextureDescriptorSet = VK_NULL_HANDLE;
  VkImage emptyTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory emptyTextureMemory = VK_NULL_HANDLE;
  VkImageView emptyTextureView = VK_NULL_HANDLE;
  VkSampler emptyTextureSampler = VK_NULL_HANDLE;
  bool emptyTextureReady = false;
};

bool create_gx_resources(GxResources& out) noexcept;
bool prepare_gx_resources_for_frame(GxResources& resources, VkCommandBuffer commandBuffer) noexcept;
VkDescriptorSet get_gx_texture_descriptor_set(GxResources& resources,
                                              const aurora::gx::GXBindGroups& bindGroups) noexcept;
void destroy_gx_resources(GxResources& resources) noexcept;

} // namespace aurora::vk
