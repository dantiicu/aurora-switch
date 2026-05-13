#include "resources.hpp"

#include "gpu.hpp"
#include "../internal.hpp"

#include <cstring>
#include <limits>

namespace aurora::vk {
namespace {
Module Log("aurora::vk::resources");

bool vk_check(VkResult result, const char* call) {
  if (result == VK_SUCCESS) {
    return true;
  }
  Log.error("{} failed: {}", call, static_cast<int>(result));
  return false;
}

#define AURORA_VK_RES_CHECK(call)                                                                                     \
  do {                                                                                                                \
    if (!vk_check((call), #call)) {                                                                                    \
      return false;                                                                                                   \
    }                                                                                                                 \
  } while (false)

bool has_flags(VkMemoryPropertyFlags flags, VkMemoryPropertyFlags required) noexcept {
  return (flags & required) == required;
}
} // namespace

uint32_t find_memory_type(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) noexcept {
  const auto vkPhysicalDevice = physical_device();
  if (vkPhysicalDevice == VK_NULL_HANDLE) {
    Log.error("Cannot find memory type before Vulkan physical device initialization");
    return std::numeric_limits<uint32_t>::max();
  }

  VkPhysicalDeviceMemoryProperties memoryProperties{};
  vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &memoryProperties);

  auto find = [&](VkMemoryPropertyFlags flags) -> uint32_t {
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
      if ((typeBits & (1u << i)) == 0) {
        continue;
      }
      if (has_flags(memoryProperties.memoryTypes[i].propertyFlags, flags)) {
        return i;
      }
    }
    return std::numeric_limits<uint32_t>::max();
  };

  if (preferred != 0) {
    const uint32_t preferredType = find(required | preferred);
    if (preferredType != std::numeric_limits<uint32_t>::max()) {
      return preferredType;
    }
  }

  const uint32_t requiredType = find(required);
  if (requiredType == std::numeric_limits<uint32_t>::max()) {
    Log.error("No Vulkan memory type found for bits={:#x} required={:#x} preferred={:#x}", typeBits, required,
              preferred);
  }
  return requiredType;
}

bool create_buffer(Buffer& out, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags requiredMemory,
                   VkMemoryPropertyFlags preferredMemory, const char* label) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE) {
    Log.error("Cannot create buffer before Vulkan device initialization");
    return false;
  }

  destroy_buffer(out);

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  AURORA_VK_RES_CHECK(vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &out.buffer));

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(vkDevice, out.buffer, &requirements);
  const uint32_t memoryType = find_memory_type(requirements.memoryTypeBits, requiredMemory, preferredMemory);
  if (memoryType == std::numeric_limits<uint32_t>::max()) {
    destroy_buffer(out);
    return false;
  }

  VkPhysicalDeviceMemoryProperties memoryProperties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device(), &memoryProperties);

  VkMemoryAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.allocationSize = requirements.size;
  allocateInfo.memoryTypeIndex = memoryType;
  AURORA_VK_RES_CHECK(vkAllocateMemory(vkDevice, &allocateInfo, nullptr, &out.memory));
  AURORA_VK_RES_CHECK(vkBindBufferMemory(vkDevice, out.buffer, out.memory, 0));

  out.size = size;
  out.memoryFlags = memoryProperties.memoryTypes[memoryType].propertyFlags;
  if (label != nullptr) {
    Log.info("Created Vulkan buffer {} size={} usage={:#x} memory={:#x}", label, size, usage, out.memoryFlags);
  }
  return true;
}

void destroy_buffer(Buffer& buffer) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE) {
    buffer = {};
    return;
  }
  if (buffer.mapped != nullptr) {
    vkUnmapMemory(vkDevice, buffer.memory);
    buffer.mapped = nullptr;
  }
  if (buffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(vkDevice, buffer.buffer, nullptr);
  }
  if (buffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(vkDevice, buffer.memory, nullptr);
  }
  buffer = {};
}

bool map_buffer(Buffer& buffer) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE || buffer.memory == VK_NULL_HANDLE) {
    return false;
  }
  if (buffer.mapped != nullptr) {
    return true;
  }
  if (!has_flags(buffer.memoryFlags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
    Log.error("Cannot map non-host-visible Vulkan buffer");
    return false;
  }
  AURORA_VK_RES_CHECK(vkMapMemory(vkDevice, buffer.memory, 0, buffer.size, 0, &buffer.mapped));
  return true;
}

void unmap_buffer(Buffer& buffer) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE || buffer.memory == VK_NULL_HANDLE || buffer.mapped == nullptr) {
    return;
  }
  vkUnmapMemory(vkDevice, buffer.memory);
  buffer.mapped = nullptr;
}

bool flush_mapped_buffer(const Buffer& buffer) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE || buffer.memory == VK_NULL_HANDLE || buffer.mapped == nullptr) {
    return false;
  }
  if (has_flags(buffer.memoryFlags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    return true;
  }
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = buffer.memory;
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;
  AURORA_VK_RES_CHECK(vkFlushMappedMemoryRanges(vkDevice, 1, &range));
  return true;
}

bool write_mapped_buffer(Buffer& buffer, const void* data, std::size_t size, VkDeviceSize offset) noexcept {
  if (data == nullptr || size == 0) {
    return true;
  }
  if (!map_buffer(buffer)) {
    return false;
  }
  if (offset + size > buffer.size) {
    Log.error("Mapped buffer write out of bounds: offset={} size={} capacity={}", offset, size, buffer.size);
    return false;
  }
  std::memcpy(static_cast<uint8_t*>(buffer.mapped) + offset, data, size);
  return flush_mapped_buffer(buffer);
}

VkDescriptorBufferInfo descriptor_buffer_info(const Buffer& buffer, VkDeviceSize range, VkDeviceSize offset) noexcept {
  return {
      .buffer = buffer.buffer,
      .offset = offset,
      .range = range,
  };
}

} // namespace aurora::vk
