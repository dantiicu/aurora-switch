#include "gx_resources.hpp"

#include "gpu.hpp"
#include "gx_pipeline.hpp"
#include "../gfx/common.hpp"
#include "../gx/gx.hpp"
#include "../internal.hpp"

#include <array>
#include <limits>

#include <absl/container/flat_hash_map.h>

namespace aurora::vk {
namespace {
Module Log("aurora::vk::gx_resources");

constexpr VkDeviceSize StagingBufferSize =
    gfx::UniformBufferSize + gfx::VertexBufferSize + gfx::IndexBufferSize + gfx::StorageBufferSize +
    (gfx::UseTextureBuffer ? gfx::TextureUploadSize : 0);
constexpr uint32_t MaxTextureDescriptorSets = 4096;

struct TextureDescriptorEntry {
  VkDescriptorSet set = VK_NULL_HANDLE;
  std::array<VkSampler, gx::MaxTextures> samplers{};
  std::array<gfx::TextureHandle, gx::MaxTextures> textures{};
};

absl::flat_hash_map<gfx::BindGroupRef, TextureDescriptorEntry> g_textureDescriptorSets;

bool vk_check(VkResult result, const char* call) {
  if (result == VK_SUCCESS) {
    return true;
  }
  Log.error("{} failed: {}", call, static_cast<int>(result));
  return false;
}

#define AURORA_VK_GX_RES_CHECK(call)                                                                                  \
  do {                                                                                                                \
    if (!vk_check((call), #call)) {                                                                                    \
      return false;                                                                                                   \
    }                                                                                                                 \
  } while (false)

bool create_descriptor_sets(GxResources& out) noexcept {
  const auto vkDevice = device();
  if (!initialize_gx_pipeline_layout()) {
    return false;
  }

  const std::array poolSizes{
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 2,
      },
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 1,
      },
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = gx::MaxTextures * (MaxTextureDescriptorSets + 1),
      },
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = gx::MaxTextures * (MaxTextureDescriptorSets + 1),
      },
  };
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 3 + MaxTextureDescriptorSets;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  AURORA_VK_GX_RES_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &out.descriptorPool));

  const std::array layouts{
      gx_static_descriptor_layout(),
      gx_uniform_descriptor_layout(),
      gx_texture_descriptor_layout(),
  };
  VkDescriptorSetAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.descriptorPool = out.descriptorPool;
  allocateInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
  allocateInfo.pSetLayouts = layouts.data();
  std::array<VkDescriptorSet, 3> sets{};
  AURORA_VK_GX_RES_CHECK(vkAllocateDescriptorSets(vkDevice, &allocateInfo, sets.data()));
  out.staticDescriptorSet = sets[0];
  out.uniformDescriptorSet = sets[1];
  out.emptyTextureDescriptorSet = sets[2];

  const auto vertexInfo = descriptor_buffer_info(out.vertexBuffer);
  const auto storageInfo = descriptor_buffer_info(out.storageBuffer);
  const auto uniformInfo = descriptor_buffer_info(out.uniformBuffer, gx::MaxUniformSize);
  const std::array writes{
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = out.staticDescriptorSet,
          .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &vertexInfo,
      },
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = out.staticDescriptorSet,
          .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &storageInfo,
      },
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = out.uniformDescriptorSet,
          .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .pBufferInfo = &uniformInfo,
      },
  };
  vkUpdateDescriptorSets(vkDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  std::array<VkDescriptorImageInfo, gx::MaxTextures> imageInfos{};
  std::array<VkDescriptorImageInfo, gx::MaxTextures> samplerInfos{};
  std::array<VkWriteDescriptorSet, gx::MaxTextures * 2> textureWrites{};
  for (uint32_t i = 0; i < gx::MaxTextures; ++i) {
    imageInfos[i] = {
        .imageView = out.emptyTextureView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    samplerInfos[i] = {
        .sampler = out.emptyTextureSampler,
    };
    textureWrites[i * 2] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = out.emptyTextureDescriptorSet,
        .dstBinding = i * 2,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = &imageInfos[i],
    };
    textureWrites[i * 2 + 1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = out.emptyTextureDescriptorSet,
        .dstBinding = i * 2 + 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = &samplerInfos[i],
    };
  }
  vkUpdateDescriptorSets(vkDevice, static_cast<uint32_t>(textureWrites.size()), textureWrites.data(), 0, nullptr);
  return true;
}

VkSamplerAddressMode to_address_mode(GXTexWrapMode mode) noexcept {
  switch (mode) {
  case GX_CLAMP:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case GX_REPEAT:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case GX_MIRROR:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  default:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

VkFilter to_min_filter(GXTexFilter filter) noexcept {
  switch (filter) {
  case GX_LINEAR:
  case GX_LIN_MIP_NEAR:
  case GX_LIN_MIP_LIN:
    return VK_FILTER_LINEAR;
  default:
    return VK_FILTER_NEAREST;
  }
}

VkSamplerMipmapMode to_mipmap_mode(GXTexFilter filter) noexcept {
  switch (filter) {
  case GX_NEAR_MIP_LIN:
  case GX_LIN_MIP_LIN:
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  default:
    return VK_SAMPLER_MIPMAP_MODE_NEAREST;
  }
}

u32 tex_bits(u32 reg, u32 size, u32 shift) noexcept { return (reg >> shift) & ((1u << size) - 1); }

GXTexWrapMode tex_wrap_s(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  return static_cast<GXTexWrapMode>(tex_bits(texture.mode0, 2, 0));
}

GXTexWrapMode tex_wrap_t(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  return static_cast<GXTexWrapMode>(tex_bits(texture.mode0, 2, 2));
}

GXTexFilter tex_min_filter(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  constexpr GXTexFilter kHwToGxFilter[8] = {
      GX_NEAR, GX_NEAR_MIP_NEAR, GX_LIN_MIP_NEAR, GX_NEAR, GX_LINEAR, GX_NEAR_MIP_LIN, GX_LIN_MIP_LIN, GX_NEAR,
  };
  return kHwToGxFilter[tex_bits(texture.mode0, 3, 5)];
}

GXTexFilter tex_mag_filter(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  return tex_bits(texture.mode0, 1, 4) != 0 ? GX_LINEAR : GX_NEAR;
}

bool tex_has_mips(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  return tex_min_filter(texture) != GX_NEAR && tex_min_filter(texture) != GX_LINEAR;
}

float tex_lod_bias(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  return static_cast<float>(static_cast<int8_t>(tex_bits(texture.mode0, 8, 9))) / 32.0f;
}

float tex_min_lod(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  return static_cast<float>(tex_bits(texture.mode1, 8, 0)) / 16.0f;
}

float tex_max_lod(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  return static_cast<float>(tex_bits(texture.mode1, 8, 8)) / 16.0f;
}

VkSampler create_sampler(const aurora::gx::GXBindGroups::VkTextureBinding& texture) noexcept {
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = tex_mag_filter(texture) == GX_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  samplerInfo.minFilter = to_min_filter(tex_min_filter(texture));
  samplerInfo.mipmapMode = to_mipmap_mode(tex_min_filter(texture));
  samplerInfo.addressModeU = to_address_mode(tex_wrap_s(texture));
  samplerInfo.addressModeV = to_address_mode(tex_wrap_t(texture));
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.minLod = tex_has_mips(texture) ? tex_min_lod(texture) : 0.f;
  samplerInfo.maxLod = tex_has_mips(texture) ? tex_max_lod(texture) : 0.f;
  samplerInfo.mipLodBias = tex_lod_bias(texture);

  VkSampler sampler = VK_NULL_HANDLE;
  if (!vk_check(vkCreateSampler(device(), &samplerInfo, nullptr, &sampler), "vkCreateSampler")) {
    return VK_NULL_HANDLE;
  }
  return sampler;
}

void clear_texture_descriptor_cache() noexcept {
  const auto vkDevice = device();
  if (vkDevice != VK_NULL_HANDLE) {
    for (auto& [_, entry] : g_textureDescriptorSets) {
      for (auto sampler : entry.samplers) {
        if (sampler != VK_NULL_HANDLE) {
          vkDestroySampler(vkDevice, sampler, nullptr);
        }
      }
    }
  }
  g_textureDescriptorSets.clear();
  gx::clear_direct_vulkan_texture_handles();
}

bool create_empty_texture(GxResources& out) noexcept {
  const auto vkDevice = device();

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.extent = {1, 1, 1};
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  AURORA_VK_GX_RES_CHECK(vkCreateImage(vkDevice, &imageInfo, nullptr, &out.emptyTextureImage));

  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(vkDevice, out.emptyTextureImage, &requirements);
  const uint32_t memoryType = find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (memoryType == std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  VkMemoryAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.allocationSize = requirements.size;
  allocateInfo.memoryTypeIndex = memoryType;
  AURORA_VK_GX_RES_CHECK(vkAllocateMemory(vkDevice, &allocateInfo, nullptr, &out.emptyTextureMemory));
  AURORA_VK_GX_RES_CHECK(vkBindImageMemory(vkDevice, out.emptyTextureImage, out.emptyTextureMemory, 0));

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = out.emptyTextureImage;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.layerCount = 1;
  AURORA_VK_GX_RES_CHECK(vkCreateImageView(vkDevice, &viewInfo, nullptr, &out.emptyTextureView));

  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.maxLod = 1000.f;
  AURORA_VK_GX_RES_CHECK(vkCreateSampler(vkDevice, &samplerInfo, nullptr, &out.emptyTextureSampler));
  return true;
}
} // namespace

bool create_gx_resources(GxResources& out) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE) {
    Log.error("Cannot create GX resources before Vulkan device initialization");
    return false;
  }

  destroy_gx_resources(out);

  constexpr VkMemoryPropertyFlags deviceMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (!create_buffer(out.uniformBuffer, gfx::UniformBufferSize,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceMemory, 0,
                     "GX Uniform Buffer") ||
      !create_buffer(out.vertexBuffer, gfx::VertexBufferSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceMemory, 0,
                     "GX Vertex Buffer") ||
      !create_buffer(out.indexBuffer, gfx::IndexBufferSize,
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceMemory, 0,
                     "GX Index Buffer") ||
      !create_buffer(out.storageBuffer, gfx::StorageBufferSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceMemory, 0,
                     "GX Storage Buffer")) {
    destroy_gx_resources(out);
    return false;
  }

  for (uint32_t i = 0; i < out.stagingBuffers.size(); ++i) {
    const auto label = fmt::format("GX Staging Buffer {}", i);
    if (!create_buffer(out.stagingBuffers[i], StagingBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, label.c_str())) {
      destroy_gx_resources(out);
      return false;
    }
  }

  if (!create_empty_texture(out)) {
    destroy_gx_resources(out);
    return false;
  }

  if (!create_descriptor_sets(out)) {
    destroy_gx_resources(out);
    return false;
  }

  Log.info("Created GX Vulkan resources");
  return true;
}

bool prepare_gx_resources_for_frame(GxResources& resources, VkCommandBuffer commandBuffer) noexcept {
  if (resources.emptyTextureReady) {
    return true;
  }

  VkImageMemoryBarrier toTransfer{};
  toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toTransfer.srcAccessMask = 0;
  toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toTransfer.image = resources.emptyTextureImage;
  toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  toTransfer.subresourceRange.levelCount = 1;
  toTransfer.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &toTransfer);

  const VkClearColorValue white{{1.f, 1.f, 1.f, 1.f}};
  const VkImageSubresourceRange range{
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1,
  };
  vkCmdClearColorImage(commandBuffer, resources.emptyTextureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1,
                       &range);

  VkImageMemoryBarrier toShader{};
  toShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  toShader.image = resources.emptyTextureImage;
  toShader.subresourceRange = range;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &toShader);

  resources.emptyTextureReady = true;
  return true;
}

VkDescriptorSet get_gx_texture_descriptor_set(GxResources& resources,
                                              const aurora::gx::GXBindGroups& bindGroups) noexcept {
  if (bindGroups.textureBindGroup == 0) {
    return resources.emptyTextureDescriptorSet;
  }
  if (const auto it = g_textureDescriptorSets.find(bindGroups.textureBindGroup); it != g_textureDescriptorSets.end()) {
    return it->second.set;
  }

  if (g_textureDescriptorSets.size() >= MaxTextureDescriptorSets) {
    Log.warn("GX Vulkan texture descriptor cache full; using empty texture set");
    return resources.emptyTextureDescriptorSet;
  }

  VkDescriptorSetAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.descriptorPool = resources.descriptorPool;
  const VkDescriptorSetLayout layout = gx_texture_descriptor_layout();
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = &layout;

  TextureDescriptorEntry entry{};
  if (!vk_check(vkAllocateDescriptorSets(device(), &allocateInfo, &entry.set), "vkAllocateDescriptorSets")) {
    return resources.emptyTextureDescriptorSet;
  }

  std::array<VkDescriptorImageInfo, gx::MaxTextures> imageInfos{};
  std::array<VkDescriptorImageInfo, gx::MaxTextures> samplerInfos{};
  std::array<VkWriteDescriptorSet, gx::MaxTextures * 2> writes{};
  const auto* retainedTextures = gx::direct_vulkan_texture_handles(bindGroups.textureBindGroup);
  for (uint32_t i = 0; i < gx::MaxTextures; ++i) {
    const bool sampledTexture = (bindGroups.vkSampledTextureMask & (1u << i)) != 0;
    const auto& texture = bindGroups.vkTextures[i];
    const gfx::TextureHandle textureRef = retainedTextures ? (*retainedTextures)[i] : nullptr;
    const VkImageView textureView = textureRef ? textureRef->vkImageView : VK_NULL_HANDLE;
    const bool validTexture = sampledTexture && textureView != VK_NULL_HANDLE;
    if (validTexture) {
      entry.textures[i] = textureRef;
    }
    imageInfos[i] = {
        .imageView = validTexture ? textureView : resources.emptyTextureView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkSampler sampler = resources.emptyTextureSampler;
    if (validTexture) {
      sampler = create_sampler(texture);
      if (sampler == VK_NULL_HANDLE) {
        sampler = resources.emptyTextureSampler;
      } else {
        entry.samplers[i] = sampler;
      }
    }
    samplerInfos[i] = {
        .sampler = sampler,
    };

    writes[i * 2] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = entry.set,
        .dstBinding = i * 2,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = &imageInfos[i],
    };
    writes[i * 2 + 1] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = entry.set,
        .dstBinding = i * 2 + 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = &samplerInfos[i],
    };
  }
  vkUpdateDescriptorSets(device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  const auto [it, inserted] = g_textureDescriptorSets.emplace(bindGroups.textureBindGroup, entry);
  if (!inserted) {
    for (auto sampler : entry.samplers) {
      if (sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device(), sampler, nullptr);
      }
    }
  }
  return it->second.set;
}

void destroy_gx_resources(GxResources& resources) noexcept {
  const auto vkDevice = device();
  clear_texture_descriptor_cache();
  if (vkDevice != VK_NULL_HANDLE && resources.descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(vkDevice, resources.descriptorPool, nullptr);
  }
  resources.descriptorPool = VK_NULL_HANDLE;
  resources.staticDescriptorSet = VK_NULL_HANDLE;
  resources.uniformDescriptorSet = VK_NULL_HANDLE;
  resources.emptyTextureDescriptorSet = VK_NULL_HANDLE;
  resources.emptyTextureReady = false;

  if (vkDevice != VK_NULL_HANDLE) {
    if (resources.emptyTextureSampler != VK_NULL_HANDLE) {
      vkDestroySampler(vkDevice, resources.emptyTextureSampler, nullptr);
    }
    if (resources.emptyTextureView != VK_NULL_HANDLE) {
      vkDestroyImageView(vkDevice, resources.emptyTextureView, nullptr);
    }
    if (resources.emptyTextureImage != VK_NULL_HANDLE) {
      vkDestroyImage(vkDevice, resources.emptyTextureImage, nullptr);
    }
    if (resources.emptyTextureMemory != VK_NULL_HANDLE) {
      vkFreeMemory(vkDevice, resources.emptyTextureMemory, nullptr);
    }
  }
  resources.emptyTextureSampler = VK_NULL_HANDLE;
  resources.emptyTextureView = VK_NULL_HANDLE;
  resources.emptyTextureImage = VK_NULL_HANDLE;
  resources.emptyTextureMemory = VK_NULL_HANDLE;

  for (auto& buffer : resources.stagingBuffers) {
    destroy_buffer(buffer);
  }
  destroy_buffer(resources.storageBuffer);
  destroy_buffer(resources.indexBuffer);
  destroy_buffer(resources.uniformBuffer);
  destroy_buffer(resources.vertexBuffer);
}

} // namespace aurora::vk
