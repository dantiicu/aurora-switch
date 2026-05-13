#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.h>

namespace aurora::vk {

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkMemoryPropertyFlags memoryFlags = 0;
  void* mapped = nullptr;
};

uint32_t find_memory_type(uint32_t typeBits, VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred = 0) noexcept;

bool create_buffer(Buffer& out, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags requiredMemory,
                   VkMemoryPropertyFlags preferredMemory = 0, const char* label = nullptr) noexcept;
void destroy_buffer(Buffer& buffer) noexcept;

bool map_buffer(Buffer& buffer) noexcept;
void unmap_buffer(Buffer& buffer) noexcept;
bool flush_mapped_buffer(const Buffer& buffer) noexcept;
bool write_mapped_buffer(Buffer& buffer, const void* data, std::size_t size, VkDeviceSize offset = 0) noexcept;

VkDescriptorBufferInfo descriptor_buffer_info(const Buffer& buffer, VkDeviceSize range = VK_WHOLE_SIZE,
                                              VkDeviceSize offset = 0) noexcept;

} // namespace aurora::vk
