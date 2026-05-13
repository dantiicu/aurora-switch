#include "gx_pipeline_cache.hpp"

#include "../internal.hpp"

#include <cstring>
#include <cstdint>
#include <mutex>

#include <absl/container/flat_hash_map.h>

namespace aurora::vk {
namespace {
Module Log("aurora::vk::gx_pipeline_cache");

struct PipelineKey {
  gfx::PipelineRef ref = 0;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  bool hasDepthAttachment = false;

  bool operator==(const PipelineKey& rhs) const noexcept {
    return ref == rhs.ref && renderPass == rhs.renderPass && hasDepthAttachment == rhs.hasDepthAttachment;
  }

  template <typename H>
  friend H AbslHashValue(H h, const PipelineKey& key) {
    return H::combine(std::move(h), key.ref, reinterpret_cast<uintptr_t>(key.renderPass), key.hasDepthAttachment);
  }
};

std::mutex g_pipelineMutex;
absl::flat_hash_map<gfx::PipelineRef, gx::PipelineConfig> g_configs;
absl::flat_hash_map<PipelineKey, GxPipeline> g_pipelines;

uint64_t render_pass_value(VkRenderPass renderPass) {
#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(renderPass));
#else
  return static_cast<uint64_t>(renderPass);
#endif
}
} // namespace

void remember_gx_pipeline_config(gfx::PipelineRef ref, const gx::PipelineConfig& config) noexcept {
  std::lock_guard lock{g_pipelineMutex};
  const auto [it, inserted] = g_configs.try_emplace(ref, config);
  if (!inserted && std::memcmp(&it->second, &config, sizeof(config)) != 0) {
    Log.warn("GX Vulkan pipeline hash collision or config changed for {:x}", ref);
    it->second = config;
  }
}

GxPipeline* get_or_create_gx_pipeline(gfx::PipelineRef ref, VkRenderPass renderPass,
                                      bool hasDepthAttachment) noexcept {
  const PipelineKey key{
      .ref = ref,
      .renderPass = renderPass,
      .hasDepthAttachment = hasDepthAttachment,
  };

  gx::PipelineConfig config{};
  {
    std::lock_guard lock{g_pipelineMutex};
    if (auto it = g_pipelines.find(key); it != g_pipelines.end()) {
      return &it->second;
    }
    const auto configIt = g_configs.find(ref);
    if (configIt == g_configs.end()) {
      Log.error("No remembered GX pipeline config for {:x}", ref);
      return nullptr;
    }
    config = configIt->second;
  }

  Log.debug("[DirectVK] create cached GX pipeline begin ref={:x} renderPass={} depth={}", ref,
            render_pass_value(renderPass), hasDepthAttachment);
  auto pipeline = create_gx_pipeline(config, renderPass, hasDepthAttachment, "GX Vulkan Pipeline");
  if (!pipeline) {
    Log.error("[DirectVK] create cached GX pipeline failed ref={:x}", ref);
    return nullptr;
  }
  Log.debug("[DirectVK] create cached GX pipeline done ref={:x}", ref);

  std::lock_guard lock{g_pipelineMutex};
  auto [it, inserted] = g_pipelines.emplace(key, *pipeline);
  if (!inserted) {
    destroy_gx_pipeline(*pipeline);
  }
  return &it->second;
}

void shutdown_gx_pipeline_cache() noexcept {
  std::lock_guard lock{g_pipelineMutex};
  for (auto& [_, pipeline] : g_pipelines) {
    destroy_gx_pipeline(pipeline);
  }
  g_pipelines.clear();
  g_configs.clear();
}

} // namespace aurora::vk
