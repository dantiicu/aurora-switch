#include "common.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "aurora/aurora.h"
#include "texture.hpp"
#include "texture_convert.hpp"
#include "../gx/gx_fmt.hpp"
#ifdef AURORA_ENABLE_DIRECT_VULKAN
#include "../vulkan/gpu.hpp"
#include "../vulkan/resources.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <magic_enum.hpp>
#include <tracy/Tracy.hpp>
#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx {
using webgpu::g_device;
using webgpu::g_queue;

namespace {
Module Log("aurora::gfx");

constexpr u32 div_ceil(u32 value, u32 divisor) noexcept { return (value + divisor - 1) / divisor; }

wgpu::Extent3D physical_size(wgpu::Extent3D size, TextureFormatInfo info) {
  const uint32_t width = ((size.width + info.blockWidth - 1) / info.blockWidth) * info.blockWidth;
  const uint32_t height = ((size.height + info.blockHeight - 1) / info.blockHeight) * info.blockHeight;
  return {.width = width, .height = height, .depthOrArrayLayers = size.depthOrArrayLayers};
}

#ifdef AURORA_ENABLE_DIRECT_VULKAN
VkFormat to_vk(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::R8Unorm:
    return VK_FORMAT_R8_UNORM;
  case wgpu::TextureFormat::RG8Unorm:
    return VK_FORMAT_R8G8_UNORM;
  case wgpu::TextureFormat::R16Sint:
    return VK_FORMAT_R16_SINT;
  case wgpu::TextureFormat::RGBA8Unorm:
    return VK_FORMAT_R8G8B8A8_UNORM;
  case wgpu::TextureFormat::BGRA8Unorm:
    return VK_FORMAT_B8G8R8A8_UNORM;
  default:
    Log.error("Unsupported direct Vulkan texture format {}", magic_enum::enum_name(format));
    return VK_FORMAT_UNDEFINED;
  }
}

bool vk_check(VkResult result, const char* call) {
  if (result == VK_SUCCESS) {
    return true;
  }
  Log.error("{} failed: {}", call, static_cast<int>(result));
  return false;
}

#define AURORA_VK_TEX_CHECK(call)                                                                                     \
  do {                                                                                                                \
    if (!vk_check((call), #call)) {                                                                                    \
      return false;                                                                                                   \
    }                                                                                                                 \
  } while (false)

bool create_vk_texture(TextureRef& ref, const char* label, VkImageUsageFlags usage) noexcept {
  const auto vkDevice = vk::device();
  if (vkDevice == VK_NULL_HANDLE) {
    Log.error("Cannot create direct Vulkan texture before device initialization");
    return false;
  }

  ref.vkFormat = to_vk(ref.format);
  if (ref.vkFormat == VK_FORMAT_UNDEFINED) {
    return false;
  }

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.format = ref.vkFormat;
  imageInfo.extent = {ref.size.width, ref.size.height, ref.size.depthOrArrayLayers};
  imageInfo.mipLevels = ref.mipCount;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = usage;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  AURORA_VK_TEX_CHECK(vkCreateImage(vkDevice, &imageInfo, nullptr, &ref.vkImage));

  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(vkDevice, ref.vkImage, &requirements);
  const uint32_t memoryType = vk::find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (memoryType == std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  VkMemoryAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.allocationSize = requirements.size;
  allocateInfo.memoryTypeIndex = memoryType;
  AURORA_VK_TEX_CHECK(vkAllocateMemory(vkDevice, &allocateInfo, nullptr, &ref.vkMemory));
  AURORA_VK_TEX_CHECK(vkBindImageMemory(vkDevice, ref.vkImage, ref.vkMemory, 0));

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = ref.vkImage;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = ref.vkFormat;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.levelCount = ref.mipCount;
  viewInfo.subresourceRange.layerCount = 1;
  AURORA_VK_TEX_CHECK(vkCreateImageView(vkDevice, &viewInfo, nullptr, &ref.vkImageView));
  ref.vkLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (label != nullptr) {
    Log.debug("Created direct Vulkan texture {} {}x{} mips={}", label, ref.size.width, ref.size.height, ref.mipCount);
  }
  return true;
}

bool upload_vk_texture(TextureRef& ref, ArrayRef<uint8_t> data) noexcept {
  if (ref.vkImage == VK_NULL_HANDLE) {
    return false;
  }

  std::vector<VkBufferImageCopy> copies;
  copies.reserve(ref.mipCount);
  VkDeviceSize stagingOffset = 0;
  uint32_t dataOffset = 0;
  for (uint32_t mip = 0; mip < ref.mipCount; ++mip) {
    const wgpu::Extent3D mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
        .depthOrArrayLayers = ref.size.depthOrArrayLayers,
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks * mipSize.depthOrArrayLayers;
    CHECK(dataOffset + dataSize <= data.size(), "direct Vulkan texture upload: expected at least {} bytes, got {}",
          dataOffset + dataSize, data.size());

    stagingOffset = AURORA_ALIGN(stagingOffset, 4);
    copies.push_back(VkBufferImageCopy{
        .bufferOffset = stagingOffset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageExtent = {physicalSize.width, physicalSize.height, mipSize.depthOrArrayLayers},
    });
    stagingOffset += dataSize;
    dataOffset += dataSize;
  }

  vk::Buffer uploadBuffer;
  if (!vk::create_buffer(uploadBuffer, stagingOffset, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    return false;
  }
  if (!vk::map_buffer(uploadBuffer)) {
    vk::destroy_buffer(uploadBuffer);
    return false;
  }

  dataOffset = 0;
  for (uint32_t mip = 0; mip < ref.mipCount; ++mip) {
    const auto info = format_info(ref.format);
    const uint32_t mipWidth = std::max(ref.size.width >> mip, 1u);
    const uint32_t mipHeight = std::max(ref.size.height >> mip, 1u);
    const auto physicalSize = physical_size({mipWidth, mipHeight, ref.size.depthOrArrayLayers}, info);
    const uint32_t dataSize = (physicalSize.width / info.blockWidth) * (physicalSize.height / info.blockHeight) *
                              info.blockSize * ref.size.depthOrArrayLayers;
    std::memcpy(static_cast<uint8_t*>(uploadBuffer.mapped) + copies[mip].bufferOffset, data.data() + dataOffset,
                dataSize);
    dataOffset += dataSize;
  }
  vk::flush_mapped_buffer(uploadBuffer);

  const auto recordUpload = [&](VkCommandBuffer cmd) {
    const VkBufferMemoryBarrier uploadReadBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = uploadBuffer.buffer,
        .offset = 0,
        .size = uploadBuffer.size,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                         &uploadReadBarrier, 0, nullptr);

    VkImageMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = ref.vkLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_ACCESS_SHADER_READ_BIT : 0;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransfer.oldLayout = ref.vkLayout;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.image = ref.vkImage;
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransfer.subresourceRange.levelCount = ref.mipCount;
    toTransfer.subresourceRange.layerCount = 1;
    const VkPipelineStageFlags srcStage =
        ref.vkLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                                                 : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    vkCmdPipelineBarrier(cmd, srcStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);

    vkCmdCopyBufferToImage(cmd, uploadBuffer.buffer, ref.vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(copies.size()), copies.data());

    VkImageMemoryBarrier toShader{};
    toShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShader.image = ref.vkImage;
    toShader.subresourceRange = toTransfer.subresourceRange;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &toShader);
  };

  const auto frameCmd = vk::current_frame().commandBuffer;
  if (frameCmd != VK_NULL_HANDLE) {
    recordUpload(frameCmd);
    vk::defer_destroy_buffer(uploadBuffer);
  } else {
    const bool ok = vk::submit_immediate(recordUpload);
    vk::destroy_buffer(uploadBuffer);
    if (!ok) {
      return false;
    }
  }

  ref.vkLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  return true;
}
#endif
} // namespace

TextureFormatInfo format_info(wgpu::TextureFormat format) noexcept {
  switch (format) {
    DEFAULT_FATAL("unimplemented texture format {}", magic_enum::enum_name(format));
  case wgpu::TextureFormat::R8Unorm:
    return {1, 1, 1, false};
  case wgpu::TextureFormat::RG8Unorm:
  case wgpu::TextureFormat::R16Sint:
    return {1, 1, 2, false};
  case wgpu::TextureFormat::RGBA8Unorm:
  case wgpu::TextureFormat::BGRA8Unorm:
  case wgpu::TextureFormat::R32Float:
    return {1, 1, 4, false};
  case wgpu::TextureFormat::BC1RGBAUnorm:
    return {4, 4, 8, true};
  case wgpu::TextureFormat::BC3RGBAUnorm:
  case wgpu::TextureFormat::BC5RGUnorm:
  case wgpu::TextureFormat::BC7RGBAUnorm:
    return {4, 4, 16, true};
  }
}

uint64_t calc_texture_size(wgpu::TextureFormat format, u32 width, u32 height, u32 mips) noexcept {
  const auto info = format_info(format);
  uint64_t total = 0;
  for (uint32_t mip = 0; mip < mips; ++mip) {
    const uint32_t mipWidth = std::max(width >> mip, 1u);
    const uint32_t mipHeight = std::max(height >> mip, 1u);
    const uint64_t widthBlocks = div_ceil(mipWidth, info.blockWidth);
    const uint64_t heightBlocks = div_ceil(mipHeight, info.blockHeight);
    const uint64_t mipBytes = widthBlocks * heightBlocks * info.blockSize;
    total += mipBytes;
  }
  return total;
}

TextureRef::~TextureRef() {
#ifdef AURORA_ENABLE_DIRECT_VULKAN
  const auto vkDevice = vk::device();
  if (vkDevice != VK_NULL_HANDLE) {
    if (vkImageView != VK_NULL_HANDLE) {
      vkDestroyImageView(vkDevice, vkImageView, nullptr);
    }
    if (vkImage != VK_NULL_HANDLE) {
      vkDestroyImage(vkDevice, vkImage, nullptr);
    }
    if (vkMemory != VK_NULL_HANDLE) {
      vkFreeMemory(vkDevice, vkMemory, nullptr);
    }
  }
#endif
}

TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 format, ArrayRef<uint8_t> data,
                                    bool tlut, const char* label) noexcept {
  ZoneScoped;

  auto handle = new_dynamic_texture_2d(width, height, mips, format, label);
  auto& ref = *handle;

  ConvertedTexture converted;
  if (ref.gxFormat != InvalidTextureFormat) {
    if (tlut) {
      CHECK(ref.size.height == 1, "new_static_texture_2d[{}]: expected tlut height 1, got {}", label, ref.size.height);
      CHECK(ref.mipCount == 1, "new_static_texture_2d[{}]: expected tlut mipCount 1, got {}", label, ref.mipCount);
      converted = convert_tlut(ref.gxFormat, ref.size.width, data);
    } else {
      converted = convert_texture(ref.gxFormat, ref.size.width, ref.size.height, ref.mipCount, data);
    }
    if (!converted.data.empty()) {
      data = converted.data;
      ref.hasArbitraryMips = converted.hasArbitraryMips;
    }
  }

#ifdef AURORA_USE_DIRECT_VULKAN_GX
  upload_vk_texture(ref, data);
  return handle;
#endif

  uint32_t offset = 0;
  for (uint32_t mip = 0; mip < mips; ++mip) {
    const wgpu::Extent3D mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
        .depthOrArrayLayers = ref.size.depthOrArrayLayers,
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks * mipSize.depthOrArrayLayers;
    CHECK(offset + dataSize <= data.size(), "new_static_texture_2d[{}]: expected at least {} bytes, got {}", label,
          offset + dataSize, data.size());
    const wgpu::TexelCopyTextureInfo dstView{
        .texture = ref.texture,
        .mipLevel = mip,
    };
    if constexpr (UseTextureBuffer) {
      const auto range = push_texture_data(data.data() + offset, dataSize, bytesPerRow, heightBlocks);
      const wgpu::TexelCopyBufferLayout dataLayout{
          .offset = range.offset,
          .bytesPerRow = bytesPerRow,
          .rowsPerImage = heightBlocks,
      };
      g_textureUploads.emplace_back(dataLayout, std::move(dstView), physicalSize);
    } else {
      const wgpu::TexelCopyBufferLayout dataLayout{
          .bytesPerRow = bytesPerRow,
          .rowsPerImage = heightBlocks,
      };
      g_queue.WriteTexture(&dstView, data.data() + offset, dataSize, &dataLayout, &physicalSize);
    }
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("new_static_texture_2d[{}]: texture used {} bytes, but given {} bytes", label, offset, data.size());
  }
  return handle;
}

TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                     const char* label) noexcept {
  ZoneScopedS(3);
  const auto wgpuFormat = to_wgpu(gxFormat);
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  auto handle = std::make_shared<TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{}, size,
                                             wgpuFormat, mips, gxFormat);
  create_vk_texture(*handle, label, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  return handle;
#else
  const wgpu::TextureDescriptor textureDescriptor{
      .label = label,
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = wgpuFormat,
      .mipLevelCount = mips,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);
  const auto viewLabel = fmt::format("{} view", label);
  wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = wgpuFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = mips,
  };
  auto textureView = texture.CreateView(&textureViewDescriptor);
  return std::make_shared<TextureRef>(std::move(texture), std::move(textureView), wgpu::TextureView{}, size, wgpuFormat,
                                      mips, gxFormat);
#endif
}

TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  ZoneScoped;

  const auto wgpuFormat =
#ifdef AURORA_USE_DIRECT_VULKAN_GX
      to_wgpu(gxFormat);
#else
      webgpu::g_graphicsConfig.surfaceConfiguration.format;
#endif
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  auto handle = std::make_shared<TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{}, size,
                                             wgpuFormat, 1, gxFormat);
  create_vk_texture(*handle, label,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  return handle;
#else
  const wgpu::TextureDescriptor textureDescriptor{
      .label = label,
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = wgpuFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);

  // Create texture view for color attachments
  const auto viewLabel = fmt::format("{} view", label);
  wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = wgpuFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
  };
  auto attachmentTextureView = texture.CreateView(&textureViewDescriptor);
  wgpu::TextureView sampleTextureView = attachmentTextureView;
  return std::make_shared<TextureRef>(std::move(texture), std::move(sampleTextureView),
                                      std::move(attachmentTextureView), size, wgpuFormat, 1, gxFormat);
#endif
}

TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  ZoneScoped;

  const auto wgpuFormat = to_wgpu(gxFormat);
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  auto handle = std::make_shared<TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{}, size,
                                             wgpuFormat, 1, gxFormat);
  create_vk_texture(*handle, label,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  return handle;
#else
  const wgpu::TextureDescriptor textureDescriptor{
      .label = label,
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = wgpuFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);

  // Create texture view for color attachments
  const auto viewLabel = fmt::format("{} view", label);
  wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = wgpuFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
  };
  auto attachmentTextureView = texture.CreateView(&textureViewDescriptor);
  wgpu::TextureView sampleTextureView = attachmentTextureView;
  return std::make_shared<TextureRef>(std::move(texture), std::move(sampleTextureView),
                                      std::move(attachmentTextureView), size, wgpuFormat, 1, gxFormat);
#endif
}

void write_texture(TextureRef& ref, ArrayRef<uint8_t> data) noexcept {
  ZoneScoped;

  ConvertedTexture converted;
  if (ref.gxFormat != InvalidTextureFormat) {
    converted = convert_texture(ref.gxFormat, ref.size.width, ref.size.height, ref.mipCount, data);
    ref.hasArbitraryMips = converted.hasArbitraryMips;
    if (!converted.data.empty()) {
      data = converted.data;
    }
  }

#ifdef AURORA_USE_DIRECT_VULKAN_GX
  upload_vk_texture(ref, data);
  return;
#endif

  uint32_t offset = 0;
  for (uint32_t mip = 0; mip < ref.mipCount; ++mip) {
    const wgpu::Extent3D mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
        .depthOrArrayLayers = ref.size.depthOrArrayLayers,
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks * mipSize.depthOrArrayLayers;
    CHECK(offset + dataSize <= data.size(), "write_texture: expected at least {} bytes, got {}", offset + dataSize,
          data.size());
    const wgpu::TexelCopyTextureInfo dstView{
        .texture = ref.texture,
        .mipLevel = mip,
    };
    if constexpr (UseTextureBuffer) {
      const auto range = push_texture_data(data.data() + offset, dataSize, bytesPerRow, heightBlocks);
      const wgpu::TexelCopyBufferLayout dataLayout{
          .offset = range.offset,
          .bytesPerRow = bytesPerRow,
          .rowsPerImage = heightBlocks,
      };
      g_textureUploads.emplace_back(dataLayout, std::move(dstView), physicalSize);
    } else {
      const wgpu::TexelCopyBufferLayout dataLayout{
          .bytesPerRow = bytesPerRow,
          .rowsPerImage = heightBlocks,
      };
      g_queue.WriteTexture(&dstView, data.data() + offset, dataSize, &dataLayout, &physicalSize);
    }
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("write_texture: texture used {} bytes, but given {} bytes", offset, data.size());
  }
}
} // namespace aurora::gfx
