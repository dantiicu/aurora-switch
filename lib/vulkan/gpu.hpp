#pragma once

#include <aurora/math.hpp>

#define VK_USE_PLATFORM_VI_NN 1
#include <vulkan/vulkan.h>

#include <functional>

namespace aurora::vk {
struct Buffer;

struct Frame {
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkImage colorImage = VK_NULL_HANDLE;
  VkImageView colorView = VK_NULL_HANDLE;
  VkImage depthImage = VK_NULL_HANDLE;
  VkImageView depthView = VK_NULL_HANDLE;
  VkExtent2D extent{};
  VkFormat colorFormat = VK_FORMAT_UNDEFINED;
  VkFormat depthFormat = VK_FORMAT_UNDEFINED;
  uint32_t imageIndex = 0;
};

bool initialize(bool vsync);
void shutdown() noexcept;
bool begin_frame(const Vec4<float>& clearColor, bool beginRenderPass = true);
bool begin_render_pass(const Vec4<float>& clearColor, float clearDepth = 1.f, bool loadContents = false,
                       bool clearColorAttachment = true, bool clearDepthAttachment = true);
void end_render_pass() noexcept;
bool end_frame();
bool submit_immediate(const std::function<void(VkCommandBuffer)>& record);
void defer_destroy_buffer(Buffer& buffer) noexcept;
void defer_destroy_framebuffer(VkFramebuffer framebuffer) noexcept;
bool is_initialized() noexcept;
const Frame& current_frame() noexcept;
VkInstance instance() noexcept;
VkPhysicalDevice physical_device() noexcept;
VkDevice device() noexcept;
VkQueue queue() noexcept;
uint32_t queue_family() noexcept;
VkFormat surface_format() noexcept;
VkFormat depth_format() noexcept;
VkExtent2D surface_extent() noexcept;
bool swapchain_sampled() noexcept;

} // namespace aurora::vk
