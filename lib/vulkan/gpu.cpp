#include "gpu.hpp"

#include "resources.hpp"
#include "../internal.hpp"
#include "../window.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include <switch.h>
#include <vulkan/vulkan_vi.h>

namespace aurora::vk {
void shutdown_gx_pipeline_cache() noexcept;
void shutdown_gx_pipeline_layout() noexcept;

namespace {
Module Log("aurora::vk");

VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_physicalDevice = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
uint32_t g_queueFamily = UINT32_MAX;
VkSurfaceKHR g_surface = VK_NULL_HANDLE;
VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
VkFormat g_colorFormat = VK_FORMAT_UNDEFINED;
VkFormat g_depthFormat = VK_FORMAT_D32_SFLOAT;
VkColorSpaceKHR g_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
VkExtent2D g_extent{};
bool g_swapchainSampled = false;
VkRenderPass g_renderPass = VK_NULL_HANDLE;
VkRenderPass g_loadRenderPass = VK_NULL_HANDLE;
VkImage g_depthImage = VK_NULL_HANDLE;
VkDeviceMemory g_depthMemory = VK_NULL_HANDLE;
VkImageView g_depthView = VK_NULL_HANDLE;
VkCommandPool g_commandPool = VK_NULL_HANDLE;
VkSemaphore g_acquireSemaphore = VK_NULL_HANDLE;
VkSemaphore g_renderSemaphore = VK_NULL_HANDLE;
VkFence g_renderFence = VK_NULL_HANDLE;
std::vector<VkImage> g_images;
std::vector<VkImageView> g_imageViews;
std::vector<VkFramebuffer> g_framebuffers;
std::vector<VkFramebuffer> g_loadFramebuffers;
std::vector<VkCommandBuffer> g_commandBuffers;
std::vector<Buffer> g_deferredDestroyBuffers;
std::vector<VkFramebuffer> g_deferredDestroyFramebuffers;
Frame g_frame;
bool g_initialized = false;
bool g_frameActive = false;
bool g_renderPassActive = false;

const char* result_name(VkResult result) noexcept {
  switch (result) {
  case VK_SUCCESS:
    return "VK_SUCCESS";
  case VK_NOT_READY:
    return "VK_NOT_READY";
  case VK_TIMEOUT:
    return "VK_TIMEOUT";
  case VK_ERROR_OUT_OF_HOST_MEMORY:
    return "VK_ERROR_OUT_OF_HOST_MEMORY";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
  case VK_ERROR_INITIALIZATION_FAILED:
    return "VK_ERROR_INITIALIZATION_FAILED";
  case VK_ERROR_DEVICE_LOST:
    return "VK_ERROR_DEVICE_LOST";
  case VK_ERROR_SURFACE_LOST_KHR:
    return "VK_ERROR_SURFACE_LOST_KHR";
  case VK_ERROR_OUT_OF_DATE_KHR:
    return "VK_ERROR_OUT_OF_DATE_KHR";
  case VK_ERROR_EXTENSION_NOT_PRESENT:
    return "VK_ERROR_EXTENSION_NOT_PRESENT";
  case VK_ERROR_FEATURE_NOT_PRESENT:
    return "VK_ERROR_FEATURE_NOT_PRESENT";
  default:
    return "VK_ERROR_UNKNOWN";
  }
}

bool check(VkResult result, const char* call) {
  if (result == VK_SUCCESS) {
    return true;
  }
  Log.error("{} failed: {} ({})", call, result_name(result), static_cast<int>(result));
  return false;
}

#define AURORA_VK_CHECK(call)                                                                                         \
  do {                                                                                                                \
    if (!check((call), #call)) {                                                                                       \
      return false;                                                                                                   \
    }                                                                                                                 \
  } while (false)

bool has_device_extension(VkPhysicalDevice physicalDevice, const char* name) {
  uint32_t count = 0;
  if (vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkExtensionProperties> extensions(count);
  if (vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data()) != VK_SUCCESS) {
    return false;
  }
  return std::ranges::any_of(extensions, [name](const VkExtensionProperties& ext) {
    return std::strcmp(ext.extensionName, name) == 0;
  });
}

bool find_queue_family(VkPhysicalDevice physicalDevice, uint32_t& outQueueFamily) {
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
  for (uint32_t i = 0; i < count; ++i) {
    VkBool32 surfaceSupported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, g_surface, &surfaceSupported);
    if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 && surfaceSupported) {
      outQueueFamily = i;
      return true;
    }
  }
  return false;
}

VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) {
  for (const auto& format : formats) {
    if (format.format == VK_FORMAT_R8G8B8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_UNORM) {
      return format;
    }
  }
  return formats.front();
}

VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& caps) {
  if (caps.currentExtent.width != UINT32_MAX) {
    return caps.currentExtent;
  }
  const auto size = window::get_window_size();
  VkExtent2D extent{
      .width = std::max(1u, size.native_fb_width != 0 ? size.native_fb_width : size.width),
      .height = std::max(1u, size.native_fb_height != 0 ? size.native_fb_height : size.height),
  };
  extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
  extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
  return extent;
}

void destroy_swapchain_resources() noexcept {
  for (auto framebuffer : g_framebuffers) {
    vkDestroyFramebuffer(g_device, framebuffer, nullptr);
  }
  g_framebuffers.clear();
  for (auto framebuffer : g_loadFramebuffers) {
    vkDestroyFramebuffer(g_device, framebuffer, nullptr);
  }
  g_loadFramebuffers.clear();
  if (!g_commandBuffers.empty()) {
    vkFreeCommandBuffers(g_device, g_commandPool, static_cast<uint32_t>(g_commandBuffers.size()), g_commandBuffers.data());
    g_commandBuffers.clear();
  }
  for (auto view : g_imageViews) {
    vkDestroyImageView(g_device, view, nullptr);
  }
  g_imageViews.clear();
  if (g_depthView != VK_NULL_HANDLE) {
    vkDestroyImageView(g_device, g_depthView, nullptr);
    g_depthView = VK_NULL_HANDLE;
  }
  if (g_depthImage != VK_NULL_HANDLE) {
    vkDestroyImage(g_device, g_depthImage, nullptr);
    g_depthImage = VK_NULL_HANDLE;
  }
  if (g_depthMemory != VK_NULL_HANDLE) {
    vkFreeMemory(g_device, g_depthMemory, nullptr);
    g_depthMemory = VK_NULL_HANDLE;
  }
  if (g_renderPass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(g_device, g_renderPass, nullptr);
    g_renderPass = VK_NULL_HANDLE;
  }
  if (g_loadRenderPass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(g_device, g_loadRenderPass, nullptr);
    g_loadRenderPass = VK_NULL_HANDLE;
  }
  if (g_swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(g_device, g_swapchain, nullptr);
    g_swapchain = VK_NULL_HANDLE;
  }
  g_images.clear();
}

void destroy_deferred_buffers() noexcept {
  for (auto& buffer : g_deferredDestroyBuffers) {
    destroy_buffer(buffer);
  }
  g_deferredDestroyBuffers.clear();
  for (auto framebuffer : g_deferredDestroyFramebuffers) {
    if (framebuffer != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(g_device, framebuffer, nullptr);
    }
  }
  g_deferredDestroyFramebuffers.clear();
}

bool create_depth_resources() {
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.format = g_depthFormat;
  imageInfo.extent = {g_extent.width, g_extent.height, 1};
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  AURORA_VK_CHECK(vkCreateImage(g_device, &imageInfo, nullptr, &g_depthImage));

  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(g_device, g_depthImage, &requirements);
  const uint32_t memoryType = find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkMemoryAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.allocationSize = requirements.size;
  allocateInfo.memoryTypeIndex = memoryType;
  AURORA_VK_CHECK(vkAllocateMemory(g_device, &allocateInfo, nullptr, &g_depthMemory));
  AURORA_VK_CHECK(vkBindImageMemory(g_device, g_depthImage, g_depthMemory, 0));

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = g_depthImage;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = g_depthFormat;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.layerCount = 1;
  AURORA_VK_CHECK(vkCreateImageView(g_device, &viewInfo, nullptr, &g_depthView));
  return true;
}

bool create_swapchain_render_pass(bool loadContents, VkRenderPass& out) {
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = g_colorFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = loadContents ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.initialLayout = loadContents ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  VkAttachmentReference colorRef{};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentDescription depthAttachment{};
  depthAttachment.format = g_depthFormat;
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = loadContents ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout =
      loadContents ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  VkAttachmentReference depthRef{};
  depthRef.attachment = 1;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  subpass.pDepthStencilAttachment = &depthRef;

  const std::array attachments{colorAttachment, depthAttachment};
  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  AURORA_VK_CHECK(vkCreateRenderPass(g_device, &renderPassInfo, nullptr, &out));
  return true;
}

bool create_swapchain_framebuffers(VkRenderPass renderPass, std::vector<VkFramebuffer>& out) {
  out.reserve(g_imageViews.size());
  for (auto view : g_imageViews) {
    const std::array framebufferAttachments{view, g_depthView};
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(framebufferAttachments.size());
    framebufferInfo.pAttachments = framebufferAttachments.data();
    framebufferInfo.width = g_extent.width;
    framebufferInfo.height = g_extent.height;
    framebufferInfo.layers = 1;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    AURORA_VK_CHECK(vkCreateFramebuffer(g_device, &framebufferInfo, nullptr, &framebuffer));
    out.push_back(framebuffer);
  }
  return true;
}

bool create_swapchain(bool vsync) {
  VkSurfaceCapabilitiesKHR caps{};
  AURORA_VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_physicalDevice, g_surface, &caps));

  uint32_t formatCount = 0;
  AURORA_VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(g_physicalDevice, g_surface, &formatCount, nullptr));
  if (formatCount == 0) {
    Log.error("No Vulkan surface formats available");
    return false;
  }
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  AURORA_VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(g_physicalDevice, g_surface, &formatCount, formats.data()));

  uint32_t imageCount = std::max(2u, caps.minImageCount);
  if (caps.maxImageCount != 0) {
    imageCount = std::min(imageCount, caps.maxImageCount);
  }
  const auto surfaceFormat = choose_surface_format(formats);
  g_colorFormat = surfaceFormat.format;
  g_colorSpace = surfaceFormat.colorSpace;
  g_extent = choose_extent(caps);

  VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  usage &= caps.supportedUsageFlags;
  if ((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
    usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }
  g_swapchainSampled = (usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;

  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = g_surface;
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = g_colorFormat;
  createInfo.imageColorSpace = g_colorSpace;
  createInfo.imageExtent = g_extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = usage;
  createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createInfo.preTransform = caps.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = vsync ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_IMMEDIATE_KHR;
  createInfo.clipped = VK_TRUE;
  AURORA_VK_CHECK(vkCreateSwapchainKHR(g_device, &createInfo, nullptr, &g_swapchain));

  uint32_t swapImageCount = 0;
  AURORA_VK_CHECK(vkGetSwapchainImagesKHR(g_device, g_swapchain, &swapImageCount, nullptr));
  g_images.resize(swapImageCount);
  AURORA_VK_CHECK(vkGetSwapchainImagesKHR(g_device, g_swapchain, &swapImageCount, g_images.data()));

  g_imageViews.reserve(g_images.size());
  for (auto image : g_images) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = g_colorFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    AURORA_VK_CHECK(vkCreateImageView(g_device, &viewInfo, nullptr, &view));
    g_imageViews.push_back(view);
  }

  if (!create_depth_resources()) {
    return false;
  }

  if (!create_swapchain_render_pass(false, g_renderPass)) {
    return false;
  }
  if (!create_swapchain_render_pass(true, g_loadRenderPass)) {
    return false;
  }
  if (!create_swapchain_framebuffers(g_renderPass, g_framebuffers)) {
    return false;
  }
  if (!create_swapchain_framebuffers(g_loadRenderPass, g_loadFramebuffers)) {
    return false;
  }

  g_commandBuffers.resize(g_images.size());
  VkCommandBufferAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocateInfo.commandPool = g_commandPool;
  allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocateInfo.commandBufferCount = static_cast<uint32_t>(g_commandBuffers.size());
  AURORA_VK_CHECK(vkAllocateCommandBuffers(g_device, &allocateInfo, g_commandBuffers.data()));

  Log.info("Direct Vulkan swapchain ready: {}x{} images={} color={} depth={}", g_extent.width, g_extent.height,
           g_images.size(), static_cast<int>(g_colorFormat), static_cast<int>(g_depthFormat));
  return true;
}

} // namespace

bool initialize(bool vsync) {
  if (g_initialized) {
    return true;
  }

  const std::array instanceExtensions{
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_NN_VI_SURFACE_EXTENSION_NAME,
  };
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Aurora";
  appInfo.apiVersion = VK_API_VERSION_1_0;
  VkInstanceCreateInfo instanceInfo{};
  instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceInfo.pApplicationInfo = &appInfo;
  instanceInfo.enabledExtensionCount = instanceExtensions.size();
  instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
  AURORA_VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &g_instance));

  auto* nwindow = nwindowGetDefault();
  if (nwindow == nullptr) {
    Log.error("nwindowGetDefault returned null");
    return false;
  }
  VkViSurfaceCreateInfoNN surfaceInfo{};
  surfaceInfo.sType = VK_STRUCTURE_TYPE_VI_SURFACE_CREATE_INFO_NN;
  surfaceInfo.window = nwindow;
  AURORA_VK_CHECK(vkCreateViSurfaceNN(g_instance, &surfaceInfo, nullptr, &g_surface));

  uint32_t deviceCount = 0;
  AURORA_VK_CHECK(vkEnumeratePhysicalDevices(g_instance, &deviceCount, nullptr));
  if (deviceCount == 0) {
    Log.error("No Vulkan physical devices available");
    return false;
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  AURORA_VK_CHECK(vkEnumeratePhysicalDevices(g_instance, &deviceCount, devices.data()));
  for (auto device : devices) {
    uint32_t queueFamily = UINT32_MAX;
    if (has_device_extension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME) && find_queue_family(device, queueFamily)) {
      g_physicalDevice = device;
      g_queueFamily = queueFamily;
      break;
    }
  }
  if (g_physicalDevice == VK_NULL_HANDLE) {
    Log.error("No Vulkan device supports graphics+present+swapchain");
    return false;
  }

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(g_physicalDevice, &properties);
  Log.info("Direct Vulkan adapter: {}", properties.deviceName);

  const float priority = 1.f;
  VkDeviceQueueCreateInfo queueInfo{};
  queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfo.queueFamilyIndex = g_queueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;
  const std::array deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkDeviceCreateInfo deviceInfo{};
  deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledExtensionCount = deviceExtensions.size();
  deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
  AURORA_VK_CHECK(vkCreateDevice(g_physicalDevice, &deviceInfo, nullptr, &g_device));
  vkGetDeviceQueue(g_device, g_queueFamily, 0, &g_queue);

  VkCommandPoolCreateInfo commandPoolInfo{};
  commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolInfo.queueFamilyIndex = g_queueFamily;
  commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  AURORA_VK_CHECK(vkCreateCommandPool(g_device, &commandPoolInfo, nullptr, &g_commandPool));

  if (!create_swapchain(vsync)) {
    return false;
  }

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  AURORA_VK_CHECK(vkCreateSemaphore(g_device, &semaphoreInfo, nullptr, &g_acquireSemaphore));
  AURORA_VK_CHECK(vkCreateSemaphore(g_device, &semaphoreInfo, nullptr, &g_renderSemaphore));
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  AURORA_VK_CHECK(vkCreateFence(g_device, &fenceInfo, nullptr, &g_renderFence));

  g_initialized = true;
  return true;
}

void shutdown() noexcept {
  if (g_device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(g_device);
  }
  destroy_deferred_buffers();
  shutdown_gx_pipeline_cache();
  shutdown_gx_pipeline_layout();
  destroy_swapchain_resources();
  if (g_renderFence != VK_NULL_HANDLE) {
    vkDestroyFence(g_device, g_renderFence, nullptr);
    g_renderFence = VK_NULL_HANDLE;
  }
  if (g_renderSemaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(g_device, g_renderSemaphore, nullptr);
    g_renderSemaphore = VK_NULL_HANDLE;
  }
  if (g_acquireSemaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(g_device, g_acquireSemaphore, nullptr);
    g_acquireSemaphore = VK_NULL_HANDLE;
  }
  if (g_commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(g_device, g_commandPool, nullptr);
    g_commandPool = VK_NULL_HANDLE;
  }
  if (g_device != VK_NULL_HANDLE) {
    vkDestroyDevice(g_device, nullptr);
    g_device = VK_NULL_HANDLE;
  }
  if (g_surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(g_instance, g_surface, nullptr);
    g_surface = VK_NULL_HANDLE;
  }
  if (g_instance != VK_NULL_HANDLE) {
    vkDestroyInstance(g_instance, nullptr);
    g_instance = VK_NULL_HANDLE;
  }
  g_physicalDevice = VK_NULL_HANDLE;
  g_queue = VK_NULL_HANDLE;
  g_queueFamily = UINT32_MAX;
  g_frame = {};
  g_initialized = false;
  g_frameActive = false;
}

bool begin_frame(const Vec4<float>& clearColor, bool beginRenderPass) {
  if (!g_initialized || g_frameActive) {
    return false;
  }
  AURORA_VK_CHECK(vkWaitForFences(g_device, 1, &g_renderFence, VK_TRUE, UINT64_MAX));
  destroy_deferred_buffers();
  AURORA_VK_CHECK(vkResetFences(g_device, 1, &g_renderFence));

  uint32_t imageIndex = 0;
  const VkResult acquireResult =
      vkAcquireNextImageKHR(g_device, g_swapchain, UINT64_MAX, g_acquireSemaphore, VK_NULL_HANDLE, &imageIndex);
  if (acquireResult != VK_SUCCESS) {
    Log.error("vkAcquireNextImageKHR failed: {} ({})", result_name(acquireResult), static_cast<int>(acquireResult));
    return false;
  }

  auto commandBuffer = g_commandBuffers[imageIndex];
  AURORA_VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  AURORA_VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

  g_frame = {
      .commandBuffer = commandBuffer,
      .renderPass = g_renderPass,
      .framebuffer = g_framebuffers[imageIndex],
      .colorImage = g_images[imageIndex],
      .colorView = g_imageViews[imageIndex],
      .depthImage = g_depthImage,
      .depthView = g_depthView,
      .extent = g_extent,
      .colorFormat = g_colorFormat,
      .depthFormat = g_depthFormat,
      .imageIndex = imageIndex,
  };
  g_frameActive = true;
  if (beginRenderPass && !begin_render_pass(clearColor)) {
    return false;
  }
  return true;
}

bool begin_render_pass(const Vec4<float>& clearColor, float clearDepth, bool loadContents, bool clearColorAttachment,
                       bool clearDepthAttachment) {
  if (!g_frameActive || g_renderPassActive) {
    return false;
  }
  std::array<VkClearValue, 2> clearValues{};
  clearValues[0].color.float32[0] = clearColor.x();
  clearValues[0].color.float32[1] = clearColor.y();
  clearValues[0].color.float32[2] = clearColor.z();
  clearValues[0].color.float32[3] = clearColor.w();
  clearValues[1].depthStencil.depth = clearDepth;

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = loadContents ? g_loadRenderPass : g_frame.renderPass;
  renderPassInfo.framebuffer = loadContents ? g_loadFramebuffers[g_frame.imageIndex] : g_frame.framebuffer;
  renderPassInfo.renderArea.extent = g_frame.extent;
  renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
  renderPassInfo.pClearValues = clearValues.data();
  vkCmdBeginRenderPass(g_frame.commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
  g_renderPassActive = true;

  if (loadContents && (clearColorAttachment || clearDepthAttachment)) {
    std::array<VkClearAttachment, 2> clearAttachments{};
    uint32_t clearAttachmentCount = 0;
    if (clearColorAttachment) {
      clearAttachments[clearAttachmentCount++] = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .colorAttachment = 0,
          .clearValue = clearValues[0],
      };
    }
    if (clearDepthAttachment) {
      clearAttachments[clearAttachmentCount++] = {
          .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
          .clearValue = clearValues[1],
      };
    }
    const VkClearRect clearRect{
        .rect =
            {
                .offset = {0, 0},
                .extent = g_frame.extent,
            },
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    vkCmdClearAttachments(g_frame.commandBuffer, clearAttachmentCount, clearAttachments.data(), 1, &clearRect);
  }
  return true;
}

void end_render_pass() noexcept {
  if (!g_renderPassActive) {
    return;
  }
  vkCmdEndRenderPass(g_frame.commandBuffer);
  g_renderPassActive = false;
}

bool end_frame() {
  if (!g_frameActive) {
    return false;
  }

  end_render_pass();
  AURORA_VK_CHECK(vkEndCommandBuffer(g_frame.commandBuffer));

  const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &g_acquireSemaphore;
  submitInfo.pWaitDstStageMask = &waitStage;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &g_frame.commandBuffer;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &g_renderSemaphore;
  AURORA_VK_CHECK(vkQueueSubmit(g_queue, 1, &submitInfo, g_renderFence));

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &g_renderSemaphore;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &g_swapchain;
  presentInfo.pImageIndices = &g_frame.imageIndex;
  const VkResult presentResult = vkQueuePresentKHR(g_queue, &presentInfo);
  g_frameActive = false;
  g_renderPassActive = false;
  g_frame = {};
  if (presentResult != VK_SUCCESS) {
    Log.error("vkQueuePresentKHR failed: {} ({})", result_name(presentResult), static_cast<int>(presentResult));
    return false;
  }
  return true;
}

bool submit_immediate(const std::function<void(VkCommandBuffer)>& record) {
  if (!g_initialized || g_device == VK_NULL_HANDLE || g_queue == VK_NULL_HANDLE || g_commandPool == VK_NULL_HANDLE) {
    return false;
  }

  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocateInfo.commandPool = g_commandPool;
  allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocateInfo.commandBufferCount = 1;
  AURORA_VK_CHECK(vkAllocateCommandBuffers(g_device, &allocateInfo, &commandBuffer));

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (!check(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer")) {
    vkFreeCommandBuffers(g_device, g_commandPool, 1, &commandBuffer);
    return false;
  }

  record(commandBuffer);

  if (!check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer")) {
    vkFreeCommandBuffers(g_device, g_commandPool, 1, &commandBuffer);
    return false;
  }

  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (!check(vkCreateFence(g_device, &fenceInfo, nullptr, &fence), "vkCreateFence")) {
    vkFreeCommandBuffers(g_device, g_commandPool, 1, &commandBuffer);
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  bool ok = check(vkQueueSubmit(g_queue, 1, &submitInfo, fence), "vkQueueSubmit");
  if (ok) {
    ok = check(vkWaitForFences(g_device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  }
  vkDestroyFence(g_device, fence, nullptr);
  vkFreeCommandBuffers(g_device, g_commandPool, 1, &commandBuffer);
  return ok;
}

void defer_destroy_buffer(Buffer& buffer) noexcept {
  if (buffer.buffer == VK_NULL_HANDLE && buffer.memory == VK_NULL_HANDLE) {
    buffer = {};
    return;
  }
  g_deferredDestroyBuffers.push_back(buffer);
  buffer = {};
}

void defer_destroy_framebuffer(VkFramebuffer framebuffer) noexcept {
  if (framebuffer != VK_NULL_HANDLE) {
    g_deferredDestroyFramebuffers.push_back(framebuffer);
  }
}

bool is_initialized() noexcept { return g_initialized; }

const Frame& current_frame() noexcept { return g_frame; }

VkInstance instance() noexcept { return g_instance; }

VkPhysicalDevice physical_device() noexcept { return g_physicalDevice; }

VkDevice device() noexcept { return g_device; }

VkQueue queue() noexcept { return g_queue; }

uint32_t queue_family() noexcept { return g_queueFamily; }

VkFormat surface_format() noexcept { return g_colorFormat; }

VkFormat depth_format() noexcept { return g_depthFormat; }

VkExtent2D surface_extent() noexcept { return g_extent; }

bool swapchain_sampled() noexcept { return g_swapchainSampled; }

} // namespace aurora::vk
