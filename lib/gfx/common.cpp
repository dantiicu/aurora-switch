#include "common.hpp"

#include "clear.hpp"
#include "depth_peek.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../gx/pipeline.hpp"
#ifdef AURORA_ENABLE_DIRECT_VULKAN
#include "../vulkan/gpu.hpp"
#include "../vulkan/gx_pipeline_cache.hpp"
#include "../vulkan/gx_resources.hpp"
#include "../vulkan/shader_compiler.hpp"
#endif
#include "pipeline_cache.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "texture_replacement.hpp"
#include "texture.hpp"
#include "../window.hpp"

#include <atomic>
#include <array>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>

#include <absl/container/flat_hash_map.h>
#include <magic_enum.hpp>

#include "tracy/Tracy.hpp"

namespace aurora::gfx {
static Module Log("aurora::gfx");

using webgpu::g_device;
using webgpu::g_instance;
using webgpu::g_queue;

#ifdef AURORA_GFX_DEBUG_GROUPS
std::vector<std::string> g_debugGroupStack;
std::vector<std::string> g_debugMarkers;
#endif

constexpr uint64_t StagingBufferSize = UniformBufferSize + VertexBufferSize + IndexBufferSize + StorageBufferSize +
                                       (UseTextureBuffer ? TextureUploadSize : 0);

struct ShaderDrawCommand {
  ShaderType type;
  union {
    clear::DrawData clear;
    gx::DrawData gx;
  };
};
enum class CommandType {
  SetViewport,
  SetScissor,
  Draw,
  DebugMarker,
};
struct Command {
  CommandType type;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> debugGroupStack;
#endif
  union Data {
    Viewport setViewport;
    ClipRect setScissor;
    ShaderDrawCommand draw;
    size_t debugMarkerIndex;
  } data;
};
} // namespace aurora::gfx

namespace aurora {
// For types that we can't ensure are safe to hash with has_unique_object_representations,
// we create specialized methods to handle them. Note that these are highly dependent on
// the structure definition, which could easily change with Dawn updates.
template <>
inline HashType xxh3_hash(const WGPUBindGroupDescriptor& input, HashType seed) {
  constexpr auto offset = offsetof(WGPUBindGroupDescriptor, layout); // skip nextInChain, label
  const auto hash = xxh3_hash_s(reinterpret_cast<const u8*>(&input) + offset,
                                sizeof(WGPUBindGroupDescriptor) - offset - sizeof(void*) /* skip entries */, seed);
  return xxh3_hash_s(input.entries, sizeof(WGPUBindGroupEntry) * input.entryCount, hash);
}
template <>
inline HashType xxh3_hash(const wgpu::SamplerDescriptor& input, HashType seed) {
  constexpr auto offset = offsetof(wgpu::SamplerDescriptor, addressModeU); // skip nextInChain, label
  return xxh3_hash_s(reinterpret_cast<const u8*>(&input) + offset,
                     sizeof(wgpu::SamplerDescriptor) - offset - 2 /* skip padding */, seed);
}
} // namespace aurora

namespace aurora::gfx {
namespace {
struct CachedBindGroup {
  wgpu::BindGroup bindGroup;
  uint32_t lastUsedFrame = 0;
};

constexpr uint32_t BindGroupCacheRetainFrames = 32;
constexpr uint32_t BindGroupCacheSweepPeriod = 16;
} // namespace

static absl::flat_hash_map<BindGroupRef, CachedBindGroup> g_cachedBindGroups;
static absl::flat_hash_map<SamplerRef, wgpu::Sampler> g_cachedSamplers;

static ByteBuffer g_verts;
static ByteBuffer g_uniforms;
static ByteBuffer g_indices;
static ByteBuffer g_storage;
static ByteBuffer g_textureUpload;
wgpu::Buffer g_vertexBuffer;
wgpu::Buffer g_uniformBuffer;
wgpu::Buffer g_indexBuffer;
wgpu::Buffer g_storageBuffer;
static std::array<wgpu::Buffer, 3> g_stagingBuffers;
static size_t currentStagingBuffer = 0;
enum class BufferMapState {
  Unmapped,
  Mapping,
  Mapped,
};
static std::atomic s_mappingState{BufferMapState::Unmapped};
static wgpu::Limits g_cachedLimits;
static uint32_t g_frameIndex = UINT32_MAX;
static PipelineRef g_currentPipeline;
wgpu::BindGroupLayout g_staticBindGroupLayout;
wgpu::BindGroup g_staticBindGroup;
wgpu::BindGroupLayout g_uniformBindGroupLayout;
wgpu::BindGroup g_uniformBindGroup;
#ifdef AURORA_USE_DIRECT_VULKAN_GX
static vk::GxResources g_vkGxResources;
#endif

// for imgui debug
AuroraStats g_stats{};

using CommandList = std::vector<Command>;
struct RenderPass {
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView; // MSAA resolve target; null if msaaSamples == 1
  wgpu::TextureView depthView;
  wgpu::Texture copySourceTexture;
  wgpu::TextureView copySourceView;
  wgpu::TextureView copySourceDepthView;
  wgpu::Extent3D targetSize;
  uint32_t msaaSamples = 1;

  TextureHandle resolveTarget;
  GXTexFmt resolveFormat = GX_TF_RGBA8;
  ClipRect resolveRect;
  Range resolveUniformRange;
  Vec4<float> clearColorValue{0.f, 0.f, 0.f, 0.f};
  float clearDepthValue = gx::UseReversedZ ? 0.f : 1.f;
  CommandList commands;
  bool clearColor = true;
  bool clearDepth = true;
  std::vector<tex_palette_conv::ConvRequest> paletteConvs;
};
static std::vector<RenderPass> g_renderPasses;
static u32 g_currentRenderPass = UINT32_MAX;
static bool g_inOffscreen = false;
static std::optional<RenderPass> g_suspendedEfbPass;
static Viewport g_suspendedEfbViewport;
static ClipRect g_suspendedEfbScissor;
static webgpu::TextureWithSampler g_offscreenColor;
static webgpu::TextureWithSampler g_offscreenDepth;

static void set_efb_targets(RenderPass& pass) {
  pass.colorView = webgpu::g_frameBuffer.view;
  pass.resolveView = webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.view : nullptr;
  pass.depthView = webgpu::g_depthBuffer.view;
  pass.copySourceTexture =
      webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.texture : webgpu::g_frameBuffer.texture;
  pass.copySourceView =
      webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.view : webgpu::g_frameBuffer.view;
  pass.copySourceDepthView = webgpu::g_depthBuffer.view;
  pass.targetSize = webgpu::g_frameBuffer.size;
  pass.msaaSamples = webgpu::g_graphicsConfig.msaaSamples;
}

struct OffscreenCacheKey {
  uint32_t width;
  uint32_t height;

  bool operator==(const OffscreenCacheKey& rhs) const { return width == rhs.width && height == rhs.height; }
  template <typename H>
  friend H AbslHashValue(H h, const OffscreenCacheKey& key) {
    return H::combine(std::move(h), key.width, key.height);
  }
};
struct OffscreenCacheEntry {
  webgpu::TextureWithSampler color;
  webgpu::TextureWithSampler depth;
};
static absl::flat_hash_map<OffscreenCacheKey, OffscreenCacheEntry> g_offscreenCache;
std::vector<TextureUpload> g_textureUploads;

static inline void push_command(CommandType type, const Command::Data& data) {
  if (g_currentRenderPass == UINT32_MAX)
    UNLIKELY {
      Log.warn("Dropping command {}", magic_enum::enum_name(type));
      return;
    }
  g_renderPasses[g_currentRenderPass].commands.push_back({
      .type = type,
#ifdef AURORA_GFX_DEBUG_GROUPS
      .debugGroupStack = g_debugGroupStack,
#endif
      .data = data,
  });
}

template <>
gx::DrawData* get_last_draw_command() {
  if (g_currentRenderPass >= g_renderPasses.size()) {
    return nullptr;
  }
  auto& last = g_renderPasses[g_currentRenderPass].commands.back();
  if (last.type != CommandType::Draw || last.data.draw.type != ShaderType::GX) {
    return nullptr;
  }
  return &last.data.draw.gx;
}

static void push_draw_command(ShaderDrawCommand data) {
  push_command(CommandType::Draw, Command::Data{.draw = data});
  ++g_stats.drawCallCount;
}

Vec2<uint32_t> get_render_target_size() noexcept {
  if (g_currentRenderPass < g_renderPasses.size()) {
    const auto& size = g_renderPasses[g_currentRenderPass].targetSize;
    return {size.width, size.height};
  }
  const auto windowSize = window::get_window_size();
  return {windowSize.fb_width, windowSize.fb_height};
}

static Viewport g_cachedViewport;
void set_viewport(const Viewport& cmd) noexcept {
  if (cmd != g_cachedViewport) {
    push_command(CommandType::SetViewport, Command::Data{.setViewport = cmd});
    g_cachedViewport = cmd;
  }
}

static ClipRect g_cachedScissor;
void set_scissor(const ClipRect& cmd) noexcept {
  if (cmd != g_cachedScissor) {
    push_command(CommandType::SetScissor, Command::Data{.setScissor = cmd});
    g_cachedScissor = cmd;
  }
}

template <>
void push_draw_command(clear::DrawData data) {
  push_draw_command(ShaderDrawCommand{.type = ShaderType::Clear, .clear = data});
}

template <>
PipelineRef pipeline_ref(const clear::PipelineConfig& config) {
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  return xxh3_hash(config, static_cast<HashType>(ShaderType::Clear));
#else
  return find_pipeline(ShaderType::Clear, config, [=] { return create_pipeline(config); });
#endif
}

void resolve_pass(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                  Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat) {
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  auto& prevPass = g_renderPasses[g_currentRenderPass];
  prevPass.resolveTarget = std::move(texture);
  prevPass.resolveRect = rect;
  prevPass.resolveFormat = resolveFormat;
  const auto srcW = static_cast<float>(prevPass.targetSize.width);
  const auto srcH = static_cast<float>(prevPass.targetSize.height);
  const std::array uvTransform{
      static_cast<float>(rect.x) / srcW,
      static_cast<float>(rect.y) / srcH,
      static_cast<float>(rect.width) / srcW,
      static_cast<float>(rect.height) / srcH,
  };
  prevPass.resolveUniformRange = push_uniform(uvTransform);

  RenderPass newPass{
      .targetSize = prevPass.targetSize,
      .msaaSamples = prevPass.msaaSamples,
      .clearColorValue = clearColorValue,
      .clearDepthValue = clearDepthValue,
      .clearColor = clearColor && clearAlpha,
      .clearDepth = clearDepth,
  };
  g_renderPasses.emplace_back(std::move(newPass));
  ++g_currentRenderPass;

  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
  return;
#else
  // Resolve current render pass
  auto& prevPass = g_renderPasses[g_currentRenderPass];
  prevPass.resolveTarget = std::move(texture);
  prevPass.resolveRect = rect;
  prevPass.resolveFormat = resolveFormat;
  // Push UV transform uniform for tex_copy_conv (crop region in UV space)
  const auto srcW = static_cast<float>(prevPass.targetSize.width);
  const auto srcH = static_cast<float>(prevPass.targetSize.height);
  const std::array uvTransform{
      static_cast<float>(rect.x) / srcW,
      static_cast<float>(rect.y) / srcH,
      static_cast<float>(rect.width) / srcW,
      static_cast<float>(rect.height) / srcH,
  };
  prevPass.resolveUniformRange = push_uniform(uvTransform);

  // Populate new render pass from previous
  const auto msaaSamples = prevPass.msaaSamples;
  RenderPass newPass{
      .colorView = prevPass.colorView,
      .resolveView = prevPass.resolveView,
      .depthView = prevPass.depthView,
      .copySourceTexture = prevPass.copySourceTexture,
      .copySourceView = prevPass.copySourceView,
      .copySourceDepthView = prevPass.copySourceDepthView,
      .targetSize = prevPass.targetSize,
      .msaaSamples = msaaSamples,
      .clearColorValue = clearColorValue,
      .clearDepthValue = clearDepthValue,
      .clearColor = clearColor && clearAlpha,
      .clearDepth = clearDepth,
  };
  g_renderPasses.emplace_back(std::move(newPass));
  ++g_currentRenderPass;

  if (!newPass.clearColor && (clearColor || clearAlpha)) {
    // If we're only clearing color _or_ alpha, perform a clear draw
    push_draw_command(clear::DrawData{
        .pipeline = pipeline_ref(clear::PipelineConfig{
            .msaaSamples = msaaSamples,
            .clearColor = clearColor,
            .clearAlpha = clearAlpha,
            .clearDepth = false, // Depth cleared via render attachment
        }),
        .color =
            wgpu::Color{
                .r = clearColorValue.x(),
                .g = clearColorValue.y(),
                .b = clearColorValue.z(),
                .a = clearColorValue.w(),
            },
    });
  }
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
#endif
}

void queue_palette_conv(tex_palette_conv::ConvRequest req) {
  g_renderPasses[g_currentRenderPass].paletteConvs.push_back(std::move(req));
}

bool is_offscreen() noexcept { return g_inOffscreen; }

uint32_t get_sample_count() noexcept {
  CHECK(g_currentRenderPass != UINT32_MAX, "get_sample_count called outside of a frame");
  return g_renderPasses[g_currentRenderPass].msaaSamples;
}

void clear_caches() noexcept {
  g_offscreenCache.clear();
  g_cachedBindGroups.clear();
}

static OffscreenCacheEntry get_offscreen_textures(uint32_t width, uint32_t height) {
  OffscreenCacheKey key{width, height};
  if (const auto it = g_offscreenCache.find(key); it != g_offscreenCache.end()) {
    return it->second;
  }
  const auto colorFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const wgpu::Extent3D size{width, height, 1};
  const wgpu::TextureDescriptor colorDesc{
      .label = "Offscreen Color",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc |
               wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = colorFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto colorTexture = g_device.CreateTexture(&colorDesc);
  auto colorView = colorTexture.CreateView();
  webgpu::TextureWithSampler color{
      .texture = std::move(colorTexture),
      .view = std::move(colorView),
      .size = size,
      .format = colorFormat,
  };
  const auto depthFormat = webgpu::g_graphicsConfig.depthFormat;
  const wgpu::TextureDescriptor depthDesc{
      .label = "Offscreen Depth",
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = depthFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto depthTexture = g_device.CreateTexture(&depthDesc);
  auto depthView = depthTexture.CreateView();
  webgpu::TextureWithSampler depth{
      .texture = std::move(depthTexture),
      .view = std::move(depthView),
      .size = size,
      .format = depthFormat,
  };
  OffscreenCacheEntry entry{
      .color = std::move(color),
      .depth = std::move(depth),
  };
  auto [insertIt, _] = g_offscreenCache.emplace(key, std::move(entry));
  return insertIt->second;
}

void begin_offscreen(uint32_t width, uint32_t height) {
  ZoneScoped;
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  (void)width;
  (void)height;
  return;
#else
  CHECK(g_currentRenderPass != UINT32_MAX, "begin_offscreen called outside of a frame");

  // If the current EFB pass has no resolve target, its output is unobservable.
  // Suspend it so that we can resume it after the offscreen pass.
  if (!g_inOffscreen) {
    auto& currentPass = g_renderPasses[g_currentRenderPass];
    if (!currentPass.resolveTarget) {
      g_suspendedEfbPass = std::move(currentPass);
      g_renderPasses.pop_back();
      --g_currentRenderPass;
    }
    g_suspendedEfbViewport = g_cachedViewport;
    g_suspendedEfbScissor = g_cachedScissor;
  }

  // Create offscreen textures
  auto offscreenEntry = get_offscreen_textures(width, height);
  g_offscreenColor = std::move(offscreenEntry.color);
  g_offscreenDepth = std::move(offscreenEntry.depth);

  // Start a new pass with offscreen targets
  RenderPass newPass{
      .colorView = g_offscreenColor.view,
      .depthView = g_offscreenDepth.view,
      .copySourceTexture = g_offscreenColor.texture,
      .copySourceView = g_offscreenColor.view,
      .copySourceDepthView = g_offscreenDepth.view,
      .targetSize = {width, height, 1},
      .msaaSamples = 1,
      .clearColorValue = {0.f, 0.f, 0.f, 0.f},
      .clearDepthValue = gx::UseReversedZ ? 0.f : 1.f,
      .clearColor = true,
      .clearDepth = true,
  };
  g_renderPasses.emplace_back(std::move(newPass));
  ++g_currentRenderPass;

  g_inOffscreen = true;

  g_cachedViewport = {0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f};
  g_cachedScissor = {0, 0, static_cast<int32_t>(width), static_cast<int32_t>(height)};
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
#endif
}

void end_offscreen() {
  ZoneScoped;
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  return;
#else
  CHECK(g_inOffscreen, "end_offscreen called without begin_offscreen");

  g_inOffscreen = false;
  g_offscreenColor = {};
  g_offscreenDepth = {};

  // Resume suspended EFB pass, or start a new one (load existing content)
  if (g_suspendedEfbPass) {
    g_renderPasses.emplace_back(std::move(*g_suspendedEfbPass));
    g_suspendedEfbPass.reset();
  } else {
    auto& pass = g_renderPasses.emplace_back();
    pass.clearColor = false;
    pass.clearDepth = false;
  }
  ++g_currentRenderPass;
  set_efb_targets(g_renderPasses[g_currentRenderPass]);

  g_cachedViewport = g_suspendedEfbViewport;
  g_cachedScissor = g_suspendedEfbScissor;
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
#endif
}

template <>
void push_draw_command(gx::DrawData data) {
  push_draw_command(ShaderDrawCommand{.type = ShaderType::GX, .gx = data});
}

template <>
PipelineRef pipeline_ref(const gx::PipelineConfig& config) {
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  const PipelineRef ref = xxh3_hash(config, static_cast<HashType>(ShaderType::GX));
  vk::remember_gx_pipeline_config(ref, config);
  return ref;
#else
  const PipelineRef ref = find_pipeline(ShaderType::GX, config, [=] { return create_pipeline(config); });
#ifdef AURORA_ENABLE_DIRECT_VULKAN
  vk::remember_gx_pipeline_config(ref, config);
#endif
  return ref;
#endif
}

void initialize() {
  Log.info("GFX initialize begin");
  g_frameIndex = 0;
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(vk::physical_device(), &properties);
    g_cachedLimits.minUniformBufferOffsetAlignment =
        static_cast<uint32_t>(properties.limits.minUniformBufferOffsetAlignment);
    g_cachedLimits.minStorageBufferOffsetAlignment =
        static_cast<uint32_t>(properties.limits.minStorageBufferOffsetAlignment);
    g_cachedLimits.maxBufferSize = UINT64_MAX;
    g_cachedLimits.maxUniformBufferBindingSize = gx::MaxUniformSize;
    g_cachedLimits.maxStorageBufferBindingSize = StorageBufferSize;
    Log.info("GFX direct Vulkan limits uniformAlign={} storageAlign={}",
             g_cachedLimits.minUniformBufferOffsetAlignment, g_cachedLimits.minStorageBufferOffsetAlignment);
    ASSERT(vk::create_gx_resources(g_vkGxResources), "Failed to create direct Vulkan GX resources");
    gx::initialize();
    currentStagingBuffer = 0;
    s_mappingState.store(BufferMapState::Unmapped, std::memory_order_release);
    return;
  }
#endif
  Log.info("GFX initialize depth peek");
  depth_peek::initialize();
  Log.info("GFX initialize texture copy conversion");
  tex_copy_conv::initialize();
  Log.info("GFX initialize texture palette conversion");
  tex_palette_conv::initialize();
  Log.info("GFX initialize texture replacement");
  texture_replacement::initialize();

  // For uniform & storage buffer offset alignments
  g_device.GetLimits(&g_cachedLimits);
  Log.info("GFX limits maxBufferSize={} maxUniformBinding={} maxStorageBinding={}", g_cachedLimits.maxBufferSize,
           g_cachedLimits.maxUniformBufferBindingSize, g_cachedLimits.maxStorageBufferBindingSize);

  const auto createBuffer = [](wgpu::Buffer& out, wgpu::BufferUsage usage, uint64_t size, const char* label) {
    if (size <= 0) {
      return;
    }
    Log.info("GFX create buffer {} size={} usage={}", label, size, underlying(usage));
    const wgpu::BufferDescriptor descriptor{
        .label = label,
        .usage = usage,
        .size = size,
    };
    out = g_device.CreateBuffer(&descriptor);
    Log.info("GFX create buffer {} done", label);
  };
  createBuffer(g_uniformBuffer, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, UniformBufferSize,
               "Shared Uniform Buffer");
  createBuffer(g_vertexBuffer, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, VertexBufferSize,
               "Shared Vertex Buffer");
  createBuffer(g_indexBuffer, wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst, IndexBufferSize,
               "Shared Index Buffer");
  createBuffer(g_storageBuffer, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, StorageBufferSize,
               "Shared Storage Buffer");
  for (int i = 0; i < g_stagingBuffers.size(); ++i) {
    const auto label = fmt::format("Staging Buffer {}", i);
    createBuffer(g_stagingBuffers[i], wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, StagingBufferSize,
                 label.c_str());
  }
  currentStagingBuffer = 0;
  s_mappingState.store(BufferMapState::Unmapped, std::memory_order_release);
  Log.info("GFX map staging buffer begin");
  map_staging_buffer();
  Log.info("GFX map staging buffer requested");

  {
    Log.info("GFX create static bind group layout begin");
    constexpr std::array layoutEntries{
        // Vertex data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
        // Storage data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Vertex,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Static bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_staticBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    Log.info("GFX create static bind group layout done");
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = g_vertexBuffer,
        },
        wgpu::BindGroupEntry{
            .binding = 1,
            .buffer = g_storageBuffer,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Static bind group",
        .layout = g_staticBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_staticBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
    Log.info("GFX create static bind group done");
  }

  {
    Log.info("GFX create uniform bind group layout begin");
    constexpr std::array layoutEntries{
        // Uniform buffer (dynamic offset)
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::Uniform,
                    .hasDynamicOffset = true,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Uniform bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_uniformBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    Log.info("GFX create uniform bind group layout done");
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = g_uniformBuffer,
            .size = gx::MaxUniformSize,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Uniform bind group",
        .layout = g_uniformBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_uniformBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
    Log.info("GFX create uniform bind group done");
  }

  Log.info("GFX initialize GX begin");
  gx::initialize();
  Log.info("GFX initialize pipeline cache begin");
  initialize_pipeline_cache();
  Log.info("GFX initialize done");
}

#ifdef AURORA_USE_DIRECT_VULKAN_GX
namespace {
void destroy_depth_resolve_state() noexcept;
void destroy_palette_resolve_state() noexcept;
void destroy_color_resolve_state() noexcept;
void reset_palette_resolve_pool() noexcept;
void reset_color_resolve_pool() noexcept;
}
#endif

void shutdown() {
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  destroy_depth_resolve_state();
  destroy_palette_resolve_state();
  destroy_color_resolve_state();
  gx::shutdown();
  vk::destroy_gx_resources(g_vkGxResources);
  g_renderPasses.clear();
  g_currentRenderPass = UINT32_MAX;
  g_frameIndex = UINT32_MAX;
  currentStagingBuffer = 0;
  s_mappingState.store(BufferMapState::Unmapped, std::memory_order_release);
  return;
#endif
  shutdown_pipeline_cache();
  depth_peek::shutdown();
  tex_copy_conv::shutdown();
  tex_palette_conv::shutdown();
  texture_replacement::shutdown();
  gx::shutdown();

  g_textureUploads.clear();
  g_cachedBindGroups.clear();
  g_cachedSamplers.clear();
  g_vertexBuffer = {};
  g_uniformBuffer = {};
  g_indexBuffer = {};
  g_storageBuffer = {};
  g_stagingBuffers.fill({});
  g_renderPasses.clear();
  g_currentRenderPass = UINT32_MAX;
  g_offscreenCache.clear();
  g_offscreenColor = {};
  g_offscreenDepth = {};
  g_staticBindGroup = {};
  g_staticBindGroupLayout = {};
  g_uniformBindGroup = {};
  g_uniformBindGroupLayout = {};
  g_inOffscreen = false;
  g_frameIndex = UINT32_MAX;
  currentStagingBuffer = 0;
  s_mappingState.store(BufferMapState::Unmapped, std::memory_order_release);
}

void map_staging_buffer() {
  auto expected = BufferMapState::Unmapped;
  if (!s_mappingState.compare_exchange_strong(expected, BufferMapState::Mapping, std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
    return;
  }

  g_stagingBuffers[currentStagingBuffer].MapAsync(
      wgpu::MapMode::Write, 0, StagingBufferSize, wgpu::CallbackMode::AllowSpontaneous,
      [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
        if (status == wgpu::MapAsyncStatus::CallbackCancelled || status == wgpu::MapAsyncStatus::Aborted) {
          Log.warn("Buffer mapping {}: {}", magic_enum::enum_name(status), message);
          s_mappingState.store(BufferMapState::Unmapped, std::memory_order_release);
          return;
        }
        ASSERT(status == wgpu::MapAsyncStatus::Success, "Buffer mapping failed: {} {}", magic_enum::enum_name(status),
               message);
        s_mappingState.store(BufferMapState::Mapped, std::memory_order_release);
      });
}

bool begin_frame() {
  ZoneScoped;
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  {
    auto& staging = g_vkGxResources.stagingBuffers[currentStagingBuffer];
    if (!vk::map_buffer(staging)) {
      return false;
    }
    size_t bufferOffset = 0;
    const auto mapBuffer = [&](ByteBuffer& buf, uint64_t size) {
      if (size <= 0) {
        return;
      }
      buf = ByteBuffer{static_cast<u8*>(staging.mapped) + bufferOffset, static_cast<size_t>(size)};
      bufferOffset += size;
    };
    mapBuffer(g_verts, VertexBufferSize);
    mapBuffer(g_uniforms, UniformBufferSize);
    mapBuffer(g_indices, IndexBufferSize);
    mapBuffer(g_storage, StorageBufferSize);

    g_stats.drawCallCount = 0;
    g_stats.mergedDrawCallCount = 0;
    g_suspendedEfbPass.reset();
    reset_palette_resolve_pool();
    reset_color_resolve_pool();

    g_renderPasses.emplace_back();
    const auto extent = vk::surface_extent();
    g_renderPasses[0].targetSize = {extent.width, extent.height, 1};
    g_renderPasses[0].msaaSamples = 1;
    g_renderPasses[0].clearColorValue = gx::g_gxState.clearColor;
    g_renderPasses[0].clearDepthValue = gx::clear_depth_value();
    g_currentRenderPass = 0;
    g_cachedViewport = gx::map_logical_viewport(gx::g_gxState.logicalViewport);
    g_cachedScissor = gx::map_logical_scissor(gx::g_gxState.logicalScissor);
    push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
    push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
    return true;
  }
#endif
  {
    ZoneScopedN("Wait for buffer map");
    map_staging_buffer();
    while (true) {
      const auto mappingState = s_mappingState.load(std::memory_order_acquire);
      if (mappingState == BufferMapState::Mapped) {
        break;
      }
      if (mappingState == BufferMapState::Unmapped) {
        return false;
      }
      g_instance.ProcessEvents();
    }
  }
  size_t bufferOffset = 0;
  const auto& stagingBuf = g_stagingBuffers[currentStagingBuffer];
  const auto mapBuffer = [&](ByteBuffer& buf, uint64_t size) {
    if (size <= 0) {
      return;
    }
    buf = ByteBuffer{static_cast<u8*>(stagingBuf.GetMappedRange(bufferOffset, size)), static_cast<size_t>(size)};
    bufferOffset += size;
  };
  mapBuffer(g_verts, VertexBufferSize);
  mapBuffer(g_uniforms, UniformBufferSize);
  mapBuffer(g_indices, IndexBufferSize);
  mapBuffer(g_storage, StorageBufferSize);
  if constexpr (UseTextureBuffer) {
    mapBuffer(g_textureUpload, TextureUploadSize);
  }

  g_stats.drawCallCount = 0;
  g_stats.mergedDrawCallCount = 0;
  g_suspendedEfbPass.reset();

  g_renderPasses.emplace_back();
  set_efb_targets(g_renderPasses[0]);
  g_renderPasses[0].clearColorValue = gx::g_gxState.clearColor;
  g_renderPasses[0].clearDepthValue = gx::clear_depth_value();
  g_currentRenderPass = 0;
  // Refresh render viewport/scissor from logical in case FB size changed
  g_cachedViewport = gx::map_logical_viewport(gx::g_gxState.logicalViewport);
  g_cachedScissor = gx::map_logical_scissor(gx::g_gxState.logicalScissor);
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
  begin_pipeline_frame();
  return true;
}

void end_frame(const wgpu::CommandEncoder& cmd) {
  ZoneScoped;
  ASSERT(!g_inOffscreen, "end_frame called while offscreen rendering is active");
  g_uniforms.append_zeroes(gx::MaxUniformSize); // Pad the end of the buffer
  uint64_t bufferOffset = 0;
  const auto writeBuffer = [&](ByteBuffer& buf, wgpu::Buffer& out, uint64_t size, std::string_view label) {
    const auto writeSize = buf.size(); // Only need to copy this many bytes
    if (writeSize > 0) {
      cmd.CopyBufferToBuffer(g_stagingBuffers[currentStagingBuffer], bufferOffset, out, 0, AURORA_ALIGN(writeSize, 4));
      buf.release();
    }
    bufferOffset += size;
    return writeSize;
  };
  g_stagingBuffers[currentStagingBuffer].Unmap();
  s_mappingState.store(BufferMapState::Unmapped, std::memory_order_release);
  g_stats.lastVertSize = writeBuffer(g_verts, g_vertexBuffer, VertexBufferSize, "Vertex");
  g_stats.lastUniformSize = writeBuffer(g_uniforms, g_uniformBuffer, UniformBufferSize, "Uniform");
  g_stats.lastIndexSize = writeBuffer(g_indices, g_indexBuffer, IndexBufferSize, "Index");
  g_stats.lastStorageSize = writeBuffer(g_storage, g_storageBuffer, StorageBufferSize, "Storage");
  if constexpr (UseTextureBuffer) {
    g_stats.lastTextureUploadSize = g_textureUpload.size();
    {
      // Perform texture copies
      for (const auto& item : g_textureUploads) {
        const wgpu::TexelCopyBufferInfo buf{
            .layout =
                wgpu::TexelCopyBufferLayout{
                    .offset = item.layout.offset + bufferOffset,
                    .bytesPerRow = AURORA_ALIGN(item.layout.bytesPerRow, 256),
                    .rowsPerImage = item.layout.rowsPerImage,
                },
            .buffer = g_stagingBuffers[currentStagingBuffer],
        };
        cmd.CopyBufferToTexture(&buf, &item.tex, &item.size);
      }
      g_textureUploads.clear();
      g_textureUpload.release();
    }
  }
  currentStagingBuffer = (currentStagingBuffer + 1) % g_stagingBuffers.size();
  map_staging_buffer();
  g_currentRenderPass = UINT32_MAX;
  for (auto& array : gx::g_gxState.arrays) {
    array.cachedRange = {};
  }
  end_pipeline_frame();
  ++g_frameIndex;
}

#ifdef AURORA_USE_DIRECT_VULKAN_GX
void end_frame(VkCommandBuffer cmd) {
  ZoneScoped;
  ASSERT(!g_inOffscreen, "end_frame called while offscreen rendering is active");
  g_uniforms.append_zeroes(gx::MaxUniformSize);

  auto& staging = g_vkGxResources.stagingBuffers[currentStagingBuffer];
  vk::flush_mapped_buffer(staging);
  vk::unmap_buffer(staging);

  const VkBufferMemoryBarrier stagingReadBarrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = staging.buffer,
      .offset = 0,
      .size = staging.size,
  };
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                       &stagingReadBarrier, 0, nullptr);

  VkDeviceSize bufferOffset = 0;
  std::array<VkBufferMemoryBarrier, 4> uploadedBarriers{};
  uint32_t uploadedBarrierCount = 0;
  const auto addUploadedBarrier = [&](vk::Buffer& out, VkAccessFlags dstAccess) {
    uploadedBarriers[uploadedBarrierCount++] = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = dstAccess,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = out.buffer,
        .offset = 0,
        .size = out.size,
    };
  };
  const auto copyBuffer = [&](ByteBuffer& buf, vk::Buffer& out, uint64_t size) {
    const auto writeSize = buf.size();
    if (writeSize > 0) {
      const VkBufferCopy copy{
          .srcOffset = bufferOffset,
          .dstOffset = 0,
          .size = AURORA_ALIGN(writeSize, 4),
      };
      vkCmdCopyBuffer(cmd, staging.buffer, out.buffer, 1, &copy);
      buf.release();
    }
    bufferOffset += size;
    return writeSize;
  };

  g_stats.lastVertSize = copyBuffer(g_verts, g_vkGxResources.vertexBuffer, VertexBufferSize);
  if (g_stats.lastVertSize > 0) {
    addUploadedBarrier(g_vkGxResources.vertexBuffer, VK_ACCESS_SHADER_READ_BIT);
  }
  g_stats.lastUniformSize = copyBuffer(g_uniforms, g_vkGxResources.uniformBuffer, UniformBufferSize);
  if (g_stats.lastUniformSize > 0) {
    addUploadedBarrier(g_vkGxResources.uniformBuffer, VK_ACCESS_UNIFORM_READ_BIT);
  }
  g_stats.lastIndexSize = copyBuffer(g_indices, g_vkGxResources.indexBuffer, IndexBufferSize);
  if (g_stats.lastIndexSize > 0) {
    addUploadedBarrier(g_vkGxResources.indexBuffer, VK_ACCESS_INDEX_READ_BIT);
  }
  g_stats.lastStorageSize = copyBuffer(g_storage, g_vkGxResources.storageBuffer, StorageBufferSize);
  if (g_stats.lastStorageSize > 0) {
    addUploadedBarrier(g_vkGxResources.storageBuffer, VK_ACCESS_SHADER_READ_BIT);
  }
  if (uploadedBarrierCount > 0) {
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, uploadedBarrierCount, uploadedBarriers.data(), 0, nullptr);
  }
  vk::prepare_gx_resources_for_frame(g_vkGxResources, cmd);

  currentStagingBuffer = (currentStagingBuffer + 1) % g_vkGxResources.stagingBuffers.size();
  g_currentRenderPass = UINT32_MAX;
  for (auto& array : gx::g_gxState.arrays) {
    array.cachedRange = {};
  }
  ++g_frameIndex;
}
#endif

uint32_t current_frame() noexcept { return g_frameIndex; }

static void expire_cached_bind_groups() {
  if (g_cachedBindGroups.empty() || g_frameIndex == UINT32_MAX || g_frameIndex % BindGroupCacheSweepPeriod != 0) {
    return;
  }

  for (auto it = g_cachedBindGroups.begin(); it != g_cachedBindGroups.end();) {
    if (g_frameIndex - it->second.lastUsedFrame > BindGroupCacheRetainFrames) {
      g_cachedBindGroups.erase(it++);
    } else {
      ++it;
    }
  }
}

void render(wgpu::CommandEncoder& cmd) {
  ZoneScoped;
  for (u32 i = 0; i < g_renderPasses.size(); ++i) {
    const auto& passInfo = g_renderPasses[i];
    for (const auto& conv : passInfo.paletteConvs) {
      tex_palette_conv::run(cmd, conv);
    }
    if (i == g_renderPasses.size() - 1) {
      ASSERT(!passInfo.resolveTarget, "Final render pass must not have resolve target");
    } else if (!passInfo.resolveTarget) {
      // Skip intermediate render passes without resolve target
      continue;
    }

    const std::array attachments{
        wgpu::RenderPassColorAttachment{
            .view = passInfo.colorView,
            .resolveTarget = passInfo.resolveView,
            .loadOp = passInfo.clearColor ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load,
            .storeOp = wgpu::StoreOp::Store,
            .clearValue =
                {
                    .r = passInfo.clearColorValue.x(),
                    .g = passInfo.clearColorValue.y(),
                    .b = passInfo.clearColorValue.z(),
                    .a = passInfo.clearColorValue.w(),
                },
        },
    };
    const wgpu::RenderPassDepthStencilAttachment depthStencilAttachment{
        .view = passInfo.depthView,
        .depthLoadOp = passInfo.clearDepth ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = passInfo.clearDepthValue,
    };
    const auto label = fmt::format("Render pass {}", i);
    const wgpu::RenderPassDescriptor renderPassDescriptor{
        .label = label.c_str(),
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
        .depthStencilAttachment = &depthStencilAttachment,
    };

    auto pass = cmd.BeginRenderPass(&renderPassDescriptor);
    render_pass(pass, i);
    pass.End();

    if (i == g_renderPasses.size() - 1) {
      depth_peek::encode_frame_snapshot(cmd, passInfo.copySourceDepthView, passInfo.targetSize, passInfo.msaaSamples);
    }

    if (passInfo.resolveTarget) {
      const auto& dstSize = passInfo.resolveTarget->size;
      const bool needsConversion = tex_copy_conv::needs_conversion(passInfo.resolveFormat);
      const bool needsScaling = dstSize.width != static_cast<uint32_t>(passInfo.resolveRect.width) ||
                                dstSize.height != static_cast<uint32_t>(passInfo.resolveRect.height);
      const bool isDepth = gx::is_depth_format(passInfo.resolveFormat);
      if (isDepth && passInfo.msaaSamples > 1) {
        Log.fatal("Depth tex copies from multisampled EFB targets are not supported");
      }
      const tex_copy_conv::ConvRequest convReq{
          .fmt = passInfo.resolveFormat,
          .srcView = isDepth ? passInfo.copySourceDepthView : passInfo.copySourceView,
          .uniformRange = passInfo.resolveUniformRange,
          .dst = passInfo.resolveTarget,
          .sampleFilter = needsScaling ? tex_copy_conv::SampleFilter::Linear : tex_copy_conv::SampleFilter::Nearest,
      };
      if (needsConversion) {
        tex_copy_conv::run(cmd, convReq);
      } else if (needsScaling) {
        tex_copy_conv::blit(cmd, convReq);
      } else {
        const wgpu::TexelCopyTextureInfo src{
            .texture = passInfo.copySourceTexture,
            .origin =
                wgpu::Origin3D{
                    .x = static_cast<uint32_t>(passInfo.resolveRect.x),
                    .y = static_cast<uint32_t>(passInfo.resolveRect.y),
                },
        };
        const wgpu::TexelCopyTextureInfo dst{
            .texture = passInfo.resolveTarget->texture,
        };
        const wgpu::Extent3D size{
            .width = static_cast<uint32_t>(passInfo.resolveRect.width),
            .height = static_cast<uint32_t>(passInfo.resolveRect.height),
            .depthOrArrayLayers = 1,
        };
        cmd.CopyTextureToTexture(&src, &dst, &size);
      }
    }
  }
  g_renderPasses.clear();
  expire_cached_bind_groups();

#if defined(AURORA_GFX_DEBUG_GROUPS)
  if (!g_debugGroupStack.empty()) {
    for (auto& it : std::ranges::reverse_view(g_debugGroupStack)) {
      Log.warn("Debug group was not popped at end of frame: {}", it);
    }
    g_debugGroupStack.clear();
  }

  if (g_debugMarkers.size() > 0) {
    g_debugMarkers.clear();
  }
#endif
}

#ifdef AURORA_USE_DIRECT_VULKAN_GX
namespace {
uint32_t g_directRenderTraceFrames = 0;
uint32_t g_directDrawTraceCount = 0;
constexpr bool DirectVkDrawTrace = false;

struct DepthResolveState {
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkShaderModule vertexShader = VK_NULL_HANDLE;
  VkShaderModule fragmentShader = VK_NULL_HANDLE;
  VkImageView boundDepthView = VK_NULL_HANDLE;
};

DepthResolveState g_depthResolveState;

struct PaletteResolveState {
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkPipeline fromFloat8Pipeline = VK_NULL_HANDLE;
  VkPipeline fromFloat4Pipeline = VK_NULL_HANDLE;
  VkShaderModule fromFloat8VertexShader = VK_NULL_HANDLE;
  VkShaderModule fromFloat8FragmentShader = VK_NULL_HANDLE;
  VkShaderModule fromFloat4VertexShader = VK_NULL_HANDLE;
  VkShaderModule fromFloat4FragmentShader = VK_NULL_HANDLE;
  uint32_t allocatedSets = 0;
};

PaletteResolveState g_paletteResolveState;
constexpr uint32_t MaxPaletteResolveSets = 1024;

struct ColorResolvePipelineDesc {
  GXTexFmt fmt;
  std::string_view fragShader;
  const char* label;
};

struct ColorResolveState {
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkSampler nearestSampler = VK_NULL_HANDLE;
  VkSampler linearSampler = VK_NULL_HANDLE;
  absl::flat_hash_map<GXTexFmt, VkPipeline> pipelines;
  std::vector<VkShaderModule> shaderModules;
  uint32_t allocatedSets = 0;
  bool initialized = false;
};

ColorResolveState g_colorResolveState;
constexpr uint32_t MaxColorResolveSets = 1024;

bool vk_check_direct(VkResult result, const char* call) {
  if (result == VK_SUCCESS) {
    return true;
  }
  Log.error("{} failed: {}", call, static_cast<int>(result));
  return false;
}

#define AURORA_DIRECT_VK_CHECK(call)                                                                                  \
  do {                                                                                                                \
    if (!vk_check_direct((call), #call)) {                                                                            \
      return false;                                                                                                   \
    }                                                                                                                 \
  } while (false)

void destroy_depth_resolve_state() noexcept {
  const auto vkDevice = vk::device();
  if (vkDevice != VK_NULL_HANDLE) {
    if (g_depthResolveState.pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(vkDevice, g_depthResolveState.pipeline, nullptr);
    }
    if (g_depthResolveState.vertexShader != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vkDevice, g_depthResolveState.vertexShader, nullptr);
    }
    if (g_depthResolveState.fragmentShader != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vkDevice, g_depthResolveState.fragmentShader, nullptr);
    }
    if (g_depthResolveState.pipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(vkDevice, g_depthResolveState.pipelineLayout, nullptr);
    }
    if (g_depthResolveState.descriptorPool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(vkDevice, g_depthResolveState.descriptorPool, nullptr);
    }
    if (g_depthResolveState.descriptorSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(vkDevice, g_depthResolveState.descriptorSetLayout, nullptr);
    }
    if (g_depthResolveState.renderPass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(vkDevice, g_depthResolveState.renderPass, nullptr);
    }
  }
  g_depthResolveState = {};
}

void destroy_palette_resolve_state() noexcept {
  const auto vkDevice = vk::device();
  if (vkDevice != VK_NULL_HANDLE) {
    if (g_paletteResolveState.fromFloat8Pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(vkDevice, g_paletteResolveState.fromFloat8Pipeline, nullptr);
    }
    if (g_paletteResolveState.fromFloat4Pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(vkDevice, g_paletteResolveState.fromFloat4Pipeline, nullptr);
    }
    if (g_paletteResolveState.fromFloat8VertexShader != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vkDevice, g_paletteResolveState.fromFloat8VertexShader, nullptr);
    }
    if (g_paletteResolveState.fromFloat8FragmentShader != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vkDevice, g_paletteResolveState.fromFloat8FragmentShader, nullptr);
    }
    if (g_paletteResolveState.fromFloat4VertexShader != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vkDevice, g_paletteResolveState.fromFloat4VertexShader, nullptr);
    }
    if (g_paletteResolveState.fromFloat4FragmentShader != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vkDevice, g_paletteResolveState.fromFloat4FragmentShader, nullptr);
    }
    if (g_paletteResolveState.pipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(vkDevice, g_paletteResolveState.pipelineLayout, nullptr);
    }
    if (g_paletteResolveState.descriptorPool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(vkDevice, g_paletteResolveState.descriptorPool, nullptr);
    }
    if (g_paletteResolveState.descriptorSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(vkDevice, g_paletteResolveState.descriptorSetLayout, nullptr);
    }
    if (g_paletteResolveState.renderPass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(vkDevice, g_paletteResolveState.renderPass, nullptr);
    }
  }
  g_paletteResolveState = {};
}

void destroy_color_resolve_state() noexcept {
  const auto vkDevice = vk::device();
  if (vkDevice != VK_NULL_HANDLE) {
    for (const auto& [_, pipeline] : g_colorResolveState.pipelines) {
      if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vkDevice, pipeline, nullptr);
      }
    }
    for (auto module : g_colorResolveState.shaderModules) {
      if (module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(vkDevice, module, nullptr);
      }
    }
    if (g_colorResolveState.nearestSampler != VK_NULL_HANDLE) {
      vkDestroySampler(vkDevice, g_colorResolveState.nearestSampler, nullptr);
    }
    if (g_colorResolveState.linearSampler != VK_NULL_HANDLE) {
      vkDestroySampler(vkDevice, g_colorResolveState.linearSampler, nullptr);
    }
    if (g_colorResolveState.pipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(vkDevice, g_colorResolveState.pipelineLayout, nullptr);
    }
    if (g_colorResolveState.descriptorPool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(vkDevice, g_colorResolveState.descriptorPool, nullptr);
    }
    if (g_colorResolveState.descriptorSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(vkDevice, g_colorResolveState.descriptorSetLayout, nullptr);
    }
    if (g_colorResolveState.renderPass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(vkDevice, g_colorResolveState.renderPass, nullptr);
    }
  }
  g_colorResolveState = {};
}

std::string make_depth_z16_shader() {
  static constexpr std::string_view Preamble = R"(
@group(0) @binding(0) var src: texture_depth_2d;

struct UVTransform {
    offset: vec2f,
    scale: vec2f,
};
@group(0) @binding(1) var<uniform> uv_xf: UVTransform;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);
var<private> uvs: array<vec2f, 3> = array(
    vec2f(0.0, 0.0),
    vec2f(0.0, 2.0),
    vec2f(2.0, 0.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi] * uv_xf.scale + uv_xf.offset;
    return out;
}
)";
  static constexpr std::string_view ReversedZ = R"(
fn gx_z24(uv: vec2f) -> u32 {
    let texSize = vec2i(textureDimensions(src));
    let coord = clamp(vec2i(floor(uv * vec2f(texSize))), vec2i(0), texSize - vec2i(1));
    let depth = textureLoad(src, coord, 0);
    return min(u32(clamp(1.0 - depth, 0.0, 1.0) * 16777215.0 + 0.5), 0x00ffffffu);
}
)";
  static constexpr std::string_view ForwardZ = R"(
fn gx_z24(uv: vec2f) -> u32 {
    let texSize = vec2i(textureDimensions(src));
    let coord = clamp(vec2i(floor(uv * vec2f(texSize))), vec2i(0), texSize - vec2i(1));
    let depth = textureLoad(src, coord, 0);
    return min(u32(clamp(depth, 0.0, 1.0) * 16777215.0 + 0.5), 0x00ffffffu);
}
)";
  static constexpr std::string_view Fragment = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let z16 = gx_z24(in.uv) >> 8u;
    let i = f32((z16 >> 8u) & 0xFFu) / 255.0;
    let a = f32(z16 & 0xFFu) / 255.0;
    return vec4f(i, i, i, a);
}
)";

  std::string shader;
  shader.reserve(Preamble.size() + ReversedZ.size() + Fragment.size());
  shader += Preamble;
  shader += gx::UseReversedZ ? ReversedZ : ForwardZ;
  shader += Fragment;
  return shader;
}

std::string make_palette_shader(float indexScale) {
  static constexpr std::string_view Preamble = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var tlut: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);
var<private> uvs: array<vec2f, 3> = array(
    vec2f(0.0, 0.0),
    vec2f(0.0, 2.0),
    vec2f(2.0, 0.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}
)";
  const auto fragment = fmt::format(R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {{
    let srcSize = vec2f(textureDimensions(src));
    let srcCoord = clamp(vec2i(floor(in.uv * srcSize)), vec2i(0), vec2i(textureDimensions(src)) - vec2i(1));
    let tlutWidth = textureDimensions(tlut).x;
    let idx = clamp(i32(textureLoad(src, srcCoord, 0).r * {:.1f}), 0, i32(tlutWidth) - 1);
    return textureLoad(tlut, vec2i(idx, 0), 0);
}}
)",
                                 indexScale);
  std::string shader;
  shader.reserve(Preamble.size() + fragment.size());
  shader += Preamble;
  shader += fragment;
  return shader;
}

static constexpr std::string_view ColorResolvePreamble = R"(
@group(0) @binding(0) var src_samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;

struct UVTransform {
    offset: vec2f,
    scale: vec2f,
};
@group(0) @binding(2) var<uniform> uv_xf: UVTransform;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);
var<private> uvs: array<vec2f, 3> = array(
    vec2f(0.0, 0.0),
    vec2f(0.0, 2.0),
    vec2f(2.0, 0.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi] * uv_xf.scale + uv_xf.offset;
    return out;
}

fn intensity(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.257, 0.504, 0.098)) + 16.0 / 255.0;
}

fn quantize4(v: f32) -> f32 {
    return floor(v * 16.0) / 15.0;
}
)";

static constexpr std::string_view ColorFragI4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let rgb = textureSample(src, src_samp, in.uv).rgb;
    let i = quantize4(intensity(rgb));
    return vec4f(i, i, i, i);
}
)";

static constexpr std::string_view ColorFragI8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let rgb = textureSample(src, src_samp, in.uv).rgb;
    let i = intensity(rgb);
    return vec4f(i, i, i, i);
}
)";

static constexpr std::string_view ColorFragIA4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    let i = quantize4(intensity(c.rgb));
    return vec4f(i, i, i, quantize4(c.a));
}
)";

static constexpr std::string_view ColorFragIA8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    let i = intensity(c.rgb);
    return vec4f(i, i, i, c.a);
}
)";

static constexpr std::string_view ColorFragRGB565 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    return vec4f(c.rgb, 1.0);
}
)";

static constexpr std::string_view ColorFragR4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = quantize4(textureSample(src, src_samp, in.uv).r);
    return vec4f(r, r, r, r);
}
)";

static constexpr std::string_view ColorFragRA4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    let r = quantize4(c.r);
    return vec4f(r, r, r, quantize4(c.a));
}
)";

static constexpr std::string_view ColorFragRA8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    return vec4f(c.r, c.r, c.r, c.a);
}
)";

static constexpr std::string_view ColorFragA8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let a = textureSample(src, src_samp, in.uv).a;
    return vec4f(a, a, a, a);
}
)";

static constexpr std::string_view ColorFragR8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = textureSample(src, src_samp, in.uv).r;
    return vec4f(r, r, r, r);
}
)";

static constexpr std::string_view ColorFragG8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let g = textureSample(src, src_samp, in.uv).g;
    return vec4f(g, g, g, g);
}
)";

static constexpr std::string_view ColorFragB8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let b = textureSample(src, src_samp, in.uv).b;
    return vec4f(b, b, b, b);
}
)";

static constexpr std::string_view ColorFragRG8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    return vec4f(c.r, c.r, c.r, c.g);
}
)";

static constexpr std::string_view ColorFragGB8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    return vec4f(c.g, c.g, c.g, c.b);
}
)";

static constexpr std::array ColorResolvePipelines{
    ColorResolvePipelineDesc{GX_TF_I4, ColorFragI4, "DirectVK TexCopyConv I4"},
    ColorResolvePipelineDesc{GX_TF_I8, ColorFragI8, "DirectVK TexCopyConv I8"},
    ColorResolvePipelineDesc{GX_TF_IA4, ColorFragIA4, "DirectVK TexCopyConv IA4"},
    ColorResolvePipelineDesc{GX_TF_IA8, ColorFragIA8, "DirectVK TexCopyConv IA8"},
    ColorResolvePipelineDesc{GX_TF_RGB565, ColorFragRGB565, "DirectVK TexCopyConv RGB565"},
    ColorResolvePipelineDesc{GX_CTF_R4, ColorFragR4, "DirectVK TexCopyConv R4"},
    ColorResolvePipelineDesc{GX_CTF_RA4, ColorFragRA4, "DirectVK TexCopyConv RA4"},
    ColorResolvePipelineDesc{GX_CTF_RA8, ColorFragRA8, "DirectVK TexCopyConv RA8"},
    ColorResolvePipelineDesc{GX_CTF_A8, ColorFragA8, "DirectVK TexCopyConv A8"},
    ColorResolvePipelineDesc{GX_CTF_R8, ColorFragR8, "DirectVK TexCopyConv R8"},
    ColorResolvePipelineDesc{GX_CTF_G8, ColorFragG8, "DirectVK TexCopyConv G8"},
    ColorResolvePipelineDesc{GX_CTF_B8, ColorFragB8, "DirectVK TexCopyConv B8"},
    ColorResolvePipelineDesc{GX_CTF_RG8, ColorFragRG8, "DirectVK TexCopyConv RG8"},
    ColorResolvePipelineDesc{GX_CTF_GB8, ColorFragGB8, "DirectVK TexCopyConv GB8"},
};

std::string make_color_resolve_shader(std::string_view fragShader) {
  std::string shader;
  shader.reserve(ColorResolvePreamble.size() + fragShader.size());
  shader += ColorResolvePreamble;
  shader += fragShader;
  return shader;
}

const ColorResolvePipelineDesc* color_resolve_pipeline_desc(GXTexFmt fmt) noexcept {
  for (const auto& desc : ColorResolvePipelines) {
    if (desc.fmt == fmt) {
      return &desc;
    }
  }
  return nullptr;
}

bool create_depth_resolve_state(VkFormat dstFormat) {
  const auto vkDevice = vk::device();
  if (vkDevice == VK_NULL_HANDLE || dstFormat == VK_FORMAT_UNDEFINED) {
    return false;
  }

  if (g_depthResolveState.pipeline != VK_NULL_HANDLE && g_depthResolveState.format == dstFormat) {
    return true;
  }
  destroy_depth_resolve_state();
  g_depthResolveState.format = dstFormat;

  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = dstFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkAttachmentReference colorRef{};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  AURORA_DIRECT_VK_CHECK(vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, &g_depthResolveState.renderPass));

  const std::array bindings{
      VkDescriptorSetLayoutBinding{
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      VkDescriptorSetLayoutBinding{
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      },
  };
  VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
  setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  setLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  setLayoutInfo.pBindings = bindings.data();
  AURORA_DIRECT_VK_CHECK(
      vkCreateDescriptorSetLayout(vkDevice, &setLayoutInfo, nullptr, &g_depthResolveState.descriptorSetLayout));

  const std::array poolSizes{
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = 1,
      },
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 1,
      },
  };
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  AURORA_DIRECT_VK_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &g_depthResolveState.descriptorPool));

  VkDescriptorSetAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.descriptorPool = g_depthResolveState.descriptorPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = &g_depthResolveState.descriptorSetLayout;
  AURORA_DIRECT_VK_CHECK(vkAllocateDescriptorSets(vkDevice, &allocateInfo, &g_depthResolveState.descriptorSet));

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &g_depthResolveState.descriptorSetLayout;
  AURORA_DIRECT_VK_CHECK(vkCreatePipelineLayout(vkDevice, &layoutInfo, nullptr, &g_depthResolveState.pipelineLayout));

  const auto shaderSource = make_depth_z16_shader();
  g_depthResolveState.vertexShader = vk::create_shader_module_from_wgsl(shaderSource, "vs_main", "DirectVK Z16 VS");
  g_depthResolveState.fragmentShader = vk::create_shader_module_from_wgsl(shaderSource, "fs_main", "DirectVK Z16 FS");
  if (g_depthResolveState.vertexShader == VK_NULL_HANDLE || g_depthResolveState.fragmentShader == VK_NULL_HANDLE) {
    return false;
  }

  const std::array stages{
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = g_depthResolveState.vertexShader,
          .pName = "vs_main",
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = g_depthResolveState.fragmentShader,
          .pName = "fs_main",
      },
  };
  VkPipelineVertexInputStateCreateInfo vertexInput{};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.lineWidth = 1.f;
  VkPipelineMultisampleStateCreateInfo multisample{};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo colorBlend{};
  colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &colorBlendAttachment;
  const std::array dynamicStates{
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();
  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages = stages.data();
  pipelineInfo.pVertexInputState = &vertexInput;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisample;
  pipelineInfo.pColorBlendState = &colorBlend;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = g_depthResolveState.pipelineLayout;
  pipelineInfo.renderPass = g_depthResolveState.renderPass;
  AURORA_DIRECT_VK_CHECK(
      vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &g_depthResolveState.pipeline));
  return true;
}

bool create_palette_pipeline(std::string_view shaderSource, VkPipeline& pipeline, VkShaderModule& vertexShader,
                             VkShaderModule& fragmentShader, const char* label) {
  const auto vkDevice = vk::device();
  vertexShader = vk::create_shader_module_from_wgsl(shaderSource, "vs_main", fmt::format("{} VS", label));
  fragmentShader = vk::create_shader_module_from_wgsl(shaderSource, "fs_main", fmt::format("{} FS", label));
  if (vertexShader == VK_NULL_HANDLE || fragmentShader == VK_NULL_HANDLE) {
    return false;
  }

  const std::array stages{
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vertexShader,
          .pName = "vs_main",
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fragmentShader,
          .pName = "fs_main",
      },
  };
  VkPipelineVertexInputStateCreateInfo vertexInput{};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.lineWidth = 1.f;
  VkPipelineMultisampleStateCreateInfo multisample{};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo colorBlend{};
  colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &colorBlendAttachment;
  const std::array dynamicStates{
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();
  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages = stages.data();
  pipelineInfo.pVertexInputState = &vertexInput;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisample;
  pipelineInfo.pColorBlendState = &colorBlend;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = g_paletteResolveState.pipelineLayout;
  pipelineInfo.renderPass = g_paletteResolveState.renderPass;
  return vk_check_direct(vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
                         "vkCreateGraphicsPipelines");
}

bool create_palette_resolve_state(VkFormat dstFormat) {
  const auto vkDevice = vk::device();
  if (vkDevice == VK_NULL_HANDLE || dstFormat == VK_FORMAT_UNDEFINED) {
    return false;
  }
  if (g_paletteResolveState.fromFloat8Pipeline != VK_NULL_HANDLE && g_paletteResolveState.format == dstFormat) {
    return true;
  }
  destroy_palette_resolve_state();
  g_paletteResolveState.format = dstFormat;

  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = dstFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkAttachmentReference colorRef{};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  AURORA_DIRECT_VK_CHECK(vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, &g_paletteResolveState.renderPass));

  const std::array bindings{
      VkDescriptorSetLayoutBinding{
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      VkDescriptorSetLayoutBinding{
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
  };
  VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
  setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  setLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  setLayoutInfo.pBindings = bindings.data();
  AURORA_DIRECT_VK_CHECK(
      vkCreateDescriptorSetLayout(vkDevice, &setLayoutInfo, nullptr, &g_paletteResolveState.descriptorSetLayout));

  const VkDescriptorPoolSize poolSize{
      .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .descriptorCount = MaxPaletteResolveSets * 2,
  };
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = MaxPaletteResolveSets;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  AURORA_DIRECT_VK_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &g_paletteResolveState.descriptorPool));

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &g_paletteResolveState.descriptorSetLayout;
  AURORA_DIRECT_VK_CHECK(
      vkCreatePipelineLayout(vkDevice, &layoutInfo, nullptr, &g_paletteResolveState.pipelineLayout));

  const auto fromFloat8Shader = make_palette_shader(255.f);
  const auto fromFloat4Shader = make_palette_shader(15.f);
  return create_palette_pipeline(fromFloat8Shader, g_paletteResolveState.fromFloat8Pipeline,
                                 g_paletteResolveState.fromFloat8VertexShader,
                                 g_paletteResolveState.fromFloat8FragmentShader, "DirectVK Palette FromFloat8") &&
         create_palette_pipeline(fromFloat4Shader, g_paletteResolveState.fromFloat4Pipeline,
                                 g_paletteResolveState.fromFloat4VertexShader,
                                 g_paletteResolveState.fromFloat4FragmentShader, "DirectVK Palette FromFloat4");
}

bool create_color_resolve_pipeline(const ColorResolvePipelineDesc& desc) {
  const auto vkDevice = vk::device();
  const auto shaderSource = make_color_resolve_shader(desc.fragShader);
  auto vertexShader = vk::create_shader_module_from_wgsl(shaderSource, "vs_main", fmt::format("{} VS", desc.label));
  auto fragmentShader =
      vk::create_shader_module_from_wgsl(shaderSource, "fs_main", fmt::format("{} FS", desc.label));
  if (vertexShader == VK_NULL_HANDLE || fragmentShader == VK_NULL_HANDLE) {
    if (vertexShader != VK_NULL_HANDLE) {
      g_colorResolveState.shaderModules.push_back(vertexShader);
    }
    if (fragmentShader != VK_NULL_HANDLE) {
      g_colorResolveState.shaderModules.push_back(fragmentShader);
    }
    return false;
  }
  g_colorResolveState.shaderModules.push_back(vertexShader);
  g_colorResolveState.shaderModules.push_back(fragmentShader);

  const std::array stages{
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vertexShader,
          .pName = "vs_main",
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fragmentShader,
          .pName = "fs_main",
      },
  };
  VkPipelineVertexInputStateCreateInfo vertexInput{};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.lineWidth = 1.f;
  VkPipelineMultisampleStateCreateInfo multisample{};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo colorBlend{};
  colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &colorBlendAttachment;
  const std::array dynamicStates{
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();
  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages = stages.data();
  pipelineInfo.pVertexInputState = &vertexInput;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisample;
  pipelineInfo.pColorBlendState = &colorBlend;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = g_colorResolveState.pipelineLayout;
  pipelineInfo.renderPass = g_colorResolveState.renderPass;

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!vk_check_direct(vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
                       "vkCreateGraphicsPipelines")) {
    return false;
  }
  g_colorResolveState.pipelines[desc.fmt] = pipeline;
  return true;
}

bool create_color_resolve_state(VkFormat dstFormat) {
  const auto vkDevice = vk::device();
  if (vkDevice == VK_NULL_HANDLE || dstFormat == VK_FORMAT_UNDEFINED) {
    return false;
  }
  if (!vk::swapchain_sampled()) {
    static bool s_warnedSwapchainSampling = false;
    if (!s_warnedSwapchainSampling) {
      Log.warn("[DirectVK] swapchain images are not sampleable; GXCopyTex conversions will fall back to raw copies");
      s_warnedSwapchainSampling = true;
    }
    return false;
  }
  if (g_colorResolveState.initialized && g_colorResolveState.format == dstFormat) {
    return true;
  }
  destroy_color_resolve_state();
  g_colorResolveState.format = dstFormat;

  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = dstFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkAttachmentReference colorRef{};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  AURORA_DIRECT_VK_CHECK(vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, &g_colorResolveState.renderPass));

  const std::array bindings{
      VkDescriptorSetLayoutBinding{
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      VkDescriptorSetLayoutBinding{
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      VkDescriptorSetLayoutBinding{
          .binding = 2,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      },
  };
  VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
  setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  setLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  setLayoutInfo.pBindings = bindings.data();
  AURORA_DIRECT_VK_CHECK(
      vkCreateDescriptorSetLayout(vkDevice, &setLayoutInfo, nullptr, &g_colorResolveState.descriptorSetLayout));

  const std::array poolSizes{
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = MaxColorResolveSets,
      },
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = MaxColorResolveSets,
      },
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = MaxColorResolveSets,
      },
  };
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = MaxColorResolveSets;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  AURORA_DIRECT_VK_CHECK(vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &g_colorResolveState.descriptorPool));

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &g_colorResolveState.descriptorSetLayout;
  AURORA_DIRECT_VK_CHECK(vkCreatePipelineLayout(vkDevice, &layoutInfo, nullptr, &g_colorResolveState.pipelineLayout));

  const auto createSampler = [&](VkFilter filter, VkSampler& out) -> bool {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = filter;
    samplerInfo.minFilter = filter;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 1.f;
    return vk_check_direct(vkCreateSampler(vkDevice, &samplerInfo, nullptr, &out), "vkCreateSampler");
  };
  if (!createSampler(VK_FILTER_NEAREST, g_colorResolveState.nearestSampler) ||
      !createSampler(VK_FILTER_LINEAR, g_colorResolveState.linearSampler)) {
    return false;
  }

  g_colorResolveState.initialized = true;
  return true;
}

void reset_palette_resolve_pool() noexcept {
  if (g_paletteResolveState.descriptorPool != VK_NULL_HANDLE) {
    vkResetDescriptorPool(vk::device(), g_paletteResolveState.descriptorPool, 0);
    g_paletteResolveState.allocatedSets = 0;
  }
}

void reset_color_resolve_pool() noexcept {
  if (g_colorResolveState.descriptorPool != VK_NULL_HANDLE) {
    vkResetDescriptorPool(vk::device(), g_colorResolveState.descriptorPool, 0);
    g_colorResolveState.allocatedSets = 0;
  }
}

VkPipeline palette_pipeline_for_variant(tex_palette_conv::Variant variant) noexcept {
  switch (variant) {
  case tex_palette_conv::Variant::FromFloat8:
    return g_paletteResolveState.fromFloat8Pipeline;
  case tex_palette_conv::Variant::FromFloat4:
    return g_paletteResolveState.fromFloat4Pipeline;
  default:
    return VK_NULL_HANDLE;
  }
}

bool update_depth_resolve_descriptor(VkImageView depthView) {
  if (depthView == VK_NULL_HANDLE || depthView == g_depthResolveState.boundDepthView) {
    return depthView != VK_NULL_HANDLE;
  }
  const auto vkDevice = vk::device();
  const VkDescriptorImageInfo imageInfo{
      .imageView = depthView,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
  };
  const VkDescriptorBufferInfo uniformInfo{
      .buffer = g_vkGxResources.uniformBuffer.buffer,
      .offset = 0,
      .range = gx::MaxUniformSize,
  };
  const std::array writes{
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = g_depthResolveState.descriptorSet,
          .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .pImageInfo = &imageInfo,
      },
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = g_depthResolveState.descriptorSet,
          .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .pBufferInfo = &uniformInfo,
      },
  };
  vkUpdateDescriptorSets(vkDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  g_depthResolveState.boundDepthView = depthView;
  return true;
}

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspectMask, VkImageLayout oldLayout,
                      VkImageLayout newLayout, VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                      VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
  if (image == VK_NULL_HANDLE || oldLayout == newLayout) {
    return;
  }
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = srcAccess;
  barrier.dstAccessMask = dstAccess;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = aspectMask;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

VkAccessFlags access_for_layout(VkImageLayout layout) noexcept {
  switch (layout) {
  case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
    return VK_ACCESS_TRANSFER_WRITE_BIT;
  case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
    return VK_ACCESS_TRANSFER_READ_BIT;
  case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
    return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
    return VK_ACCESS_SHADER_READ_BIT;
  default:
    return 0;
  }
}

VkPipelineStageFlags stage_for_layout(VkImageLayout layout) noexcept {
  switch (layout) {
  case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
  case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
    return VK_PIPELINE_STAGE_TRANSFER_BIT;
  case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
    return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
    return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  default:
    return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  }
}

void transition_texture_to_shader_read(TextureRef& texture, VkCommandBuffer cmd) {
  transition_image(cmd, texture.vkImage, VK_IMAGE_ASPECT_COLOR_BIT, texture.vkLayout,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, access_for_layout(texture.vkLayout),
                   VK_ACCESS_SHADER_READ_BIT, stage_for_layout(texture.vkLayout),
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  texture.vkLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

bool run_palette_conv_vk(const tex_palette_conv::ConvRequest& req, VkCommandBuffer cmd) {
  if (!req.src || !req.dst || !req.tlut || req.src->vkImage == VK_NULL_HANDLE || req.dst->vkImage == VK_NULL_HANDLE ||
      req.tlut->vkImage == VK_NULL_HANDLE || req.src->vkImageView == VK_NULL_HANDLE ||
      req.dst->vkImageView == VK_NULL_HANDLE || req.tlut->vkImageView == VK_NULL_HANDLE) {
    return false;
  }
  if (!create_palette_resolve_state(req.dst->vkFormat)) {
    return false;
  }
  const auto pipeline = palette_pipeline_for_variant(req.variant);
  if (pipeline == VK_NULL_HANDLE || g_paletteResolveState.allocatedSets >= MaxPaletteResolveSets) {
    static bool s_warnedPaletteVariant = false;
    if (!s_warnedPaletteVariant) {
      Log.warn("[DirectVK] dynamic palette conversion variant is not available");
      s_warnedPaletteVariant = true;
    }
    return false;
  }

  transition_texture_to_shader_read(*req.src, cmd);
  transition_texture_to_shader_read(*req.tlut, cmd);

  const VkImageLayout dstOldLayout = req.dst->vkLayout;
  transition_image(cmd, req.dst->vkImage, VK_IMAGE_ASPECT_COLOR_BIT, dstOldLayout,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, access_for_layout(dstOldLayout),
                   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, stage_for_layout(dstOldLayout),
                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.descriptorPool = g_paletteResolveState.descriptorPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = &g_paletteResolveState.descriptorSetLayout;
  if (!vk_check_direct(vkAllocateDescriptorSets(vk::device(), &allocateInfo, &descriptorSet),
                       "vkAllocateDescriptorSets")) {
    return false;
  }
  ++g_paletteResolveState.allocatedSets;

  const std::array imageInfos{
      VkDescriptorImageInfo{
          .imageView = req.src->vkImageView,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
      VkDescriptorImageInfo{
          .imageView = req.tlut->vkImageView,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
  };
  const std::array writes{
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = descriptorSet,
          .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .pImageInfo = &imageInfos[0],
      },
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = descriptorSet,
          .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .pImageInfo = &imageInfos[1],
      },
  };
  vkUpdateDescriptorSets(vk::device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = g_paletteResolveState.renderPass;
  framebufferInfo.attachmentCount = 1;
  framebufferInfo.pAttachments = &req.dst->vkImageView;
  framebufferInfo.width = req.dst->size.width;
  framebufferInfo.height = req.dst->size.height;
  framebufferInfo.layers = 1;
  if (!vk_check_direct(vkCreateFramebuffer(vk::device(), &framebufferInfo, nullptr, &framebuffer),
                       "vkCreateFramebuffer")) {
    return false;
  }

  VkClearValue clearValue{};
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = g_paletteResolveState.renderPass;
  renderPassInfo.framebuffer = framebuffer;
  renderPassInfo.renderArea.extent = {req.dst->size.width, req.dst->size.height};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearValue;
  vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
  const VkViewport viewport{
      .x = 0.f,
      .y = 0.f,
      .width = static_cast<float>(req.dst->size.width),
      .height = static_cast<float>(req.dst->size.height),
      .minDepth = 0.f,
      .maxDepth = 1.f,
  };
  const VkRect2D scissor{
      .extent = {req.dst->size.width, req.dst->size.height},
  };
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_paletteResolveState.pipelineLayout, 0, 1,
                          &descriptorSet, 0, nullptr);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmd);
  vk::defer_destroy_framebuffer(framebuffer);
  req.dst->vkLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  return true;
}

bool resolve_depth_z16_vk(const RenderPass& passInfo, VkCommandBuffer cmd) {
  const auto& frame = vk::current_frame();
  auto& dst = *passInfo.resolveTarget;
  if (frame.depthImage == VK_NULL_HANDLE || frame.depthView == VK_NULL_HANDLE || dst.vkImage == VK_NULL_HANDLE ||
      dst.vkImageView == VK_NULL_HANDLE || passInfo.resolveFormat != GX_TF_Z16) {
    return false;
  }
  if (!create_depth_resolve_state(dst.vkFormat) || !update_depth_resolve_descriptor(frame.depthView)) {
    return false;
  }

  transition_image(cmd, frame.depthImage, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

  const VkImageLayout dstOldLayout = dst.vkLayout;
  const VkAccessFlags dstSrcAccess =
      dstOldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_ACCESS_SHADER_READ_BIT : 0;
  const VkPipelineStageFlags dstSrcStage =
      dstOldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                                               : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  transition_image(cmd, dst.vkImage, VK_IMAGE_ASPECT_COLOR_BIT, dstOldLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   dstSrcAccess, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, dstSrcStage,
                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = g_depthResolveState.renderPass;
  framebufferInfo.attachmentCount = 1;
  framebufferInfo.pAttachments = &dst.vkImageView;
  framebufferInfo.width = dst.size.width;
  framebufferInfo.height = dst.size.height;
  framebufferInfo.layers = 1;
  if (!vk_check_direct(vkCreateFramebuffer(vk::device(), &framebufferInfo, nullptr, &framebuffer),
                       "vkCreateFramebuffer")) {
    transition_image(cmd, frame.depthImage, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
    if (dstOldLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
      transition_image(cmd, dst.vkImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       dstOldLayout, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, dstSrcAccess,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, dstSrcStage);
    }
    return false;
  }

  VkClearValue clearValue{};
  clearValue.color.float32[0] = 0.f;
  clearValue.color.float32[1] = 0.f;
  clearValue.color.float32[2] = 0.f;
  clearValue.color.float32[3] = 0.f;
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = g_depthResolveState.renderPass;
  renderPassInfo.framebuffer = framebuffer;
  renderPassInfo.renderArea.extent = {dst.size.width, dst.size.height};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearValue;
  vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  const VkViewport viewport{
      .x = 0.f,
      .y = 0.f,
      .width = static_cast<float>(dst.size.width),
      .height = static_cast<float>(dst.size.height),
      .minDepth = 0.f,
      .maxDepth = 1.f,
  };
  const VkRect2D scissor{
      .extent = {dst.size.width, dst.size.height},
  };
  const uint32_t uniformOffset = passInfo.resolveUniformRange.offset;
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_depthResolveState.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_depthResolveState.pipelineLayout, 0, 1,
                          &g_depthResolveState.descriptorSet, 1, &uniformOffset);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmd);
  vk::defer_destroy_framebuffer(framebuffer);

  dst.vkLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  transition_image(cmd, frame.depthImage, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
  return true;
}

bool resolve_color_conv_vk(const RenderPass& passInfo, VkCommandBuffer cmd) {
  const auto& frame = vk::current_frame();
  auto& dst = *passInfo.resolveTarget;
  if (frame.colorImage == VK_NULL_HANDLE || frame.colorView == VK_NULL_HANDLE || dst.vkImage == VK_NULL_HANDLE ||
      dst.vkImageView == VK_NULL_HANDLE) {
    return false;
  }
  if (!create_color_resolve_state(dst.vkFormat)) {
    return false;
  }
  auto pipelineIt = g_colorResolveState.pipelines.find(passInfo.resolveFormat);
  if (pipelineIt == g_colorResolveState.pipelines.end()) {
    const auto* desc = color_resolve_pipeline_desc(passInfo.resolveFormat);
    if (desc == nullptr || !create_color_resolve_pipeline(*desc)) {
      return false;
    }
    pipelineIt = g_colorResolveState.pipelines.find(passInfo.resolveFormat);
  }
  if (pipelineIt == g_colorResolveState.pipelines.end() || pipelineIt->second == VK_NULL_HANDLE ||
      g_colorResolveState.allocatedSets >= MaxColorResolveSets) {
    static bool s_warnedColorPipeline = false;
    if (!s_warnedColorPipeline) {
      Log.warn("[DirectVK] color GXCopyTex conversion pipeline is not available");
      s_warnedColorPipeline = true;
    }
    return false;
  }

  const int32_t srcX = std::max<int32_t>(0, passInfo.resolveRect.x);
  const int32_t srcY = std::max<int32_t>(0, passInfo.resolveRect.y);
  const uint32_t srcW = std::min<uint32_t>(static_cast<uint32_t>(std::max<int32_t>(0, passInfo.resolveRect.width)),
                                           frame.extent.width - std::min<uint32_t>(srcX, frame.extent.width));
  const uint32_t srcH = std::min<uint32_t>(static_cast<uint32_t>(std::max<int32_t>(0, passInfo.resolveRect.height)),
                                           frame.extent.height - std::min<uint32_t>(srcY, frame.extent.height));
  if (srcW == 0 || srcH == 0 || dst.size.width == 0 || dst.size.height == 0) {
    return false;
  }

  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.descriptorPool = g_colorResolveState.descriptorPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts = &g_colorResolveState.descriptorSetLayout;
  if (!vk_check_direct(vkAllocateDescriptorSets(vk::device(), &allocateInfo, &descriptorSet),
                       "vkAllocateDescriptorSets")) {
    return false;
  }
  ++g_colorResolveState.allocatedSets;

  const bool needsScaling = dst.size.width != srcW || dst.size.height != srcH;
  const VkDescriptorImageInfo samplerInfo{
      .sampler = needsScaling ? g_colorResolveState.linearSampler : g_colorResolveState.nearestSampler,
  };
  const VkDescriptorImageInfo imageInfo{
      .imageView = frame.colorView,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  const VkDescriptorBufferInfo uniformInfo{
      .buffer = g_vkGxResources.uniformBuffer.buffer,
      .offset = 0,
      .range = gx::MaxUniformSize,
  };
  const std::array writes{
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = descriptorSet,
          .dstBinding = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
          .pImageInfo = &samplerInfo,
      },
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = descriptorSet,
          .dstBinding = 1,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .pImageInfo = &imageInfo,
      },
      VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = descriptorSet,
          .dstBinding = 2,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .pBufferInfo = &uniformInfo,
      },
  };
  vkUpdateDescriptorSets(vk::device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  transition_image(cmd, frame.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

  const VkImageLayout dstOldLayout = dst.vkLayout;
  const VkAccessFlags dstSrcAccess =
      dstOldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_ACCESS_SHADER_READ_BIT : 0;
  const VkPipelineStageFlags dstSrcStage =
      dstOldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                                               : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  transition_image(cmd, dst.vkImage, VK_IMAGE_ASPECT_COLOR_BIT, dstOldLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   dstSrcAccess, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, dstSrcStage,
                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = g_colorResolveState.renderPass;
  framebufferInfo.attachmentCount = 1;
  framebufferInfo.pAttachments = &dst.vkImageView;
  framebufferInfo.width = dst.size.width;
  framebufferInfo.height = dst.size.height;
  framebufferInfo.layers = 1;
  if (!vk_check_direct(vkCreateFramebuffer(vk::device(), &framebufferInfo, nullptr, &framebuffer),
                       "vkCreateFramebuffer")) {
    transition_image(cmd, frame.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_SHADER_READ_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    if (dstOldLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
      transition_image(cmd, dst.vkImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       dstOldLayout, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, dstSrcAccess,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, dstSrcStage);
    }
    return false;
  }

  VkClearValue clearValue{};
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = g_colorResolveState.renderPass;
  renderPassInfo.framebuffer = framebuffer;
  renderPassInfo.renderArea.extent = {dst.size.width, dst.size.height};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearValue;
  vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  const VkViewport viewport{
      .x = 0.f,
      .y = 0.f,
      .width = static_cast<float>(dst.size.width),
      .height = static_cast<float>(dst.size.height),
      .minDepth = 0.f,
      .maxDepth = 1.f,
  };
  const VkRect2D scissor{
      .extent = {dst.size.width, dst.size.height},
  };
  const uint32_t uniformOffset = passInfo.resolveUniformRange.offset;
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineIt->second);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_colorResolveState.pipelineLayout, 0, 1,
                          &descriptorSet, 1, &uniformOffset);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmd);
  vk::defer_destroy_framebuffer(framebuffer);

  dst.vkLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  transition_image(cmd, frame.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_SHADER_READ_BIT, 0,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  return true;
}

void resolve_pass_vk(const RenderPass& passInfo, VkCommandBuffer cmd) {
  if (!passInfo.resolveTarget || passInfo.resolveTarget->vkImage == VK_NULL_HANDLE) {
    return;
  }
  if (gx::is_depth_format(passInfo.resolveFormat)) {
    if (resolve_depth_z16_vk(passInfo, cmd)) {
      return;
    }
    static bool s_warnedDepthResolve = false;
    if (!s_warnedDepthResolve) {
      Log.warn("[DirectVK] depth GXCopyTex resolve is not implemented yet");
      s_warnedDepthResolve = true;
    }
    return;
  }
  if (tex_copy_conv::needs_conversion(passInfo.resolveFormat)) {
    if (resolve_color_conv_vk(passInfo, cmd)) {
      return;
    }
    static bool s_warnedConversionResolve = false;
    if (!s_warnedConversionResolve) {
      Log.warn("[DirectVK] GXCopyTex format conversion failed; using raw color copy");
      s_warnedConversionResolve = true;
    }
  }

  const auto& frame = vk::current_frame();
  auto& dst = *passInfo.resolveTarget;
  const int32_t srcX = std::max<int32_t>(0, passInfo.resolveRect.x);
  const int32_t srcY = std::max<int32_t>(0, passInfo.resolveRect.y);
  const uint32_t srcW = std::min<uint32_t>(static_cast<uint32_t>(std::max<int32_t>(0, passInfo.resolveRect.width)),
                                           frame.extent.width - std::min<uint32_t>(srcX, frame.extent.width));
  const uint32_t srcH = std::min<uint32_t>(static_cast<uint32_t>(std::max<int32_t>(0, passInfo.resolveRect.height)),
                                           frame.extent.height - std::min<uint32_t>(srcY, frame.extent.height));
  if (srcW == 0 || srcH == 0 || dst.size.width == 0 || dst.size.height == 0) {
    return;
  }

  transition_image(cmd, frame.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);

  const VkImageLayout dstOldLayout = dst.vkLayout;
  const VkAccessFlags dstSrcAccess =
      dstOldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_ACCESS_SHADER_READ_BIT : 0;
  const VkPipelineStageFlags dstSrcStage =
      dstOldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                                               : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  transition_image(cmd, dst.vkImage, VK_IMAGE_ASPECT_COLOR_BIT, dstOldLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   dstSrcAccess, VK_ACCESS_TRANSFER_WRITE_BIT, dstSrcStage, VK_PIPELINE_STAGE_TRANSFER_BIT);

  const bool needsScaling = dst.size.width != srcW || dst.size.height != srcH;
  if (needsScaling) {
    const VkImageBlit blit{
        .srcSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        .srcOffsets = {{srcX, srcY, 0}, {srcX + static_cast<int32_t>(srcW), srcY + static_cast<int32_t>(srcH), 1}},
        .dstSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        .dstOffsets = {{0, 0, 0}, {static_cast<int32_t>(dst.size.width), static_cast<int32_t>(dst.size.height), 1}},
    };
    vkCmdBlitImage(cmd, frame.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.vkImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
  } else {
    const VkImageCopy copy{
        .srcSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        .srcOffset = {srcX, srcY, 0},
        .dstSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1,
            },
        .extent = {srcW, srcH, 1},
    };
    vkCmdCopyImage(cmd, frame.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.vkImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  }

  transition_image(cmd, dst.vkImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  dst.vkLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  transition_image(cmd, frame.colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

void render_gx_vk(const gx::DrawData& data, VkCommandBuffer cmd) {
  const uint32_t drawId = g_directDrawTraceCount++;
  const bool trace = DirectVkDrawTrace && drawId < 96;
  if (trace) {
    Log.info("[DirectVK] draw {} pipeline={:x} indices={} instances={} uniformOff={} vertexOff={} indexOff={}",
             drawId, data.pipeline, data.indexCount, data.instanceCount, data.uniformRange.offset,
             data.vertRange.offset, data.idxRange.offset);
  }
  auto* pipeline = vk::get_or_create_gx_pipeline(data.pipeline, vk::current_frame().renderPass, true);
  if (pipeline == nullptr) {
    if (trace) {
      Log.warn("[DirectVK] draw {} pipeline lookup failed for {:x}", drawId, data.pipeline);
    }
    return;
  }

  if (trace) {
    Log.info("[DirectVK] draw {} bind pipeline begin", drawId);
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
  if (trace) {
    Log.info("[DirectVK] draw {} bind static descriptors begin", drawId);
  }
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk::gx_pipeline_layout(), 0, 1,
                          &g_vkGxResources.staticDescriptorSet, 0, nullptr);
  const uint32_t uniformOffset = data.uniformRange.offset;
  if (trace) {
    Log.info("[DirectVK] draw {} bind uniform descriptors begin", drawId);
  }
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk::gx_pipeline_layout(), 1, 1,
                          &g_vkGxResources.uniformDescriptorSet, 1, &uniformOffset);
  if (trace) {
    Log.info("[DirectVK] draw {} texture descriptor lookup begin", drawId);
  }
  const auto textureSet = vk::get_gx_texture_descriptor_set(g_vkGxResources, data.bindGroups);
  if (trace) {
    Log.info("[DirectVK] draw {} bind texture descriptors begin", drawId);
  }
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk::gx_pipeline_layout(), 2, 1,
                          &textureSet, 0, nullptr);
  if (trace) {
    Log.info("[DirectVK] draw {} bind index buffer begin", drawId);
  }
  vkCmdBindIndexBuffer(cmd, g_vkGxResources.indexBuffer.buffer, data.idxRange.offset, VK_INDEX_TYPE_UINT16);
  if (data.dstAlpha != UINT32_MAX) {
    const float blendConstants[4]{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    if (trace) {
      Log.info("[DirectVK] draw {} set blend constants begin", drawId);
    }
    vkCmdSetBlendConstants(cmd, blendConstants);
  }
  if (trace) {
    Log.info("[DirectVK] draw {} indexed draw begin", drawId);
  }
  vkCmdDrawIndexed(cmd, data.indexCount, data.instanceCount, 0, 0, 0);
  if (trace) {
    Log.info("[DirectVK] draw {} done", drawId);
  }
}
} // namespace

void render(VkCommandBuffer cmd) {
  ZoneScoped;
  const bool traceFrame = DirectVkDrawTrace && g_directRenderTraceFrames < 4;
  if (traceFrame) {
    Log.info("[DirectVK] gfx render pass count={}", g_renderPasses.size());
  }
  for (u32 i = 0; i < g_renderPasses.size(); ++i) {
    const auto& passInfo = g_renderPasses[i];
    for (const auto& conv : passInfo.paletteConvs) {
      run_palette_conv_vk(conv, cmd);
    }
    if (i != g_renderPasses.size() - 1 && !passInfo.resolveTarget) {
      continue;
    }

    if (traceFrame) {
      Log.info("[DirectVK] render pass {} begin commands={}", i, passInfo.commands.size());
    }
    const bool loadContents = i != 0;
    if (!vk::begin_render_pass(passInfo.clearColorValue, passInfo.clearDepthValue, loadContents, passInfo.clearColor,
                               passInfo.clearDepth)) {
      if (traceFrame) {
        Log.warn("[DirectVK] render pass {} begin failed", i);
      }
      continue;
    }

    for (const auto& command : passInfo.commands) {
      switch (command.type) {
      case CommandType::SetViewport: {
        const auto& vp = command.data.setViewport;
        const float minDepth = gx::UseReversedZ ? 1.f - vp.zfar : vp.znear;
        const float maxDepth = gx::UseReversedZ ? 1.f - vp.znear : vp.zfar;
        const VkViewport viewport{
            .x = vp.left,
            .y = vp.top + vp.height,
            .width = vp.width,
            .height = -vp.height,
            .minDepth = minDepth,
            .maxDepth = maxDepth,
        };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        break;
      }
      case CommandType::SetScissor: {
        const auto& sc = command.data.setScissor;
        const VkRect2D rect{
            .offset = {std::max<int32_t>(0, sc.x), std::max<int32_t>(0, sc.y)},
            .extent = {static_cast<uint32_t>(std::max<int32_t>(0, sc.width)),
                       static_cast<uint32_t>(std::max<int32_t>(0, sc.height))},
        };
        vkCmdSetScissor(cmd, 0, 1, &rect);
        break;
      }
      case CommandType::Draw:
        if (command.data.draw.type == ShaderType::GX) {
          render_gx_vk(command.data.draw.gx, cmd);
        }
        break;
      case CommandType::DebugMarker:
        break;
      }
    }

    vk::end_render_pass();
    resolve_pass_vk(passInfo, cmd);
    if (traceFrame) {
      Log.info("[DirectVK] render pass {} done", i);
    }
  }
  g_renderPasses.clear();
  ++g_directRenderTraceFrames;
}
#endif

void after_submit() noexcept {
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  return;
#else
  depth_peek::after_submit();
#endif
}

void render_pass(const wgpu::RenderPassEncoder& pass, u32 idx) {
  g_currentPipeline = UINTPTR_MAX;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> lastDebugGroupStack;
#endif

  // Bind static bind group for the whole pass
  pass.SetBindGroup(0, g_staticBindGroup);
  pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);

  for (const auto& cmd : g_renderPasses[idx].commands) {
#ifdef AURORA_GFX_DEBUG_GROUPS
    {
      size_t firstDiff = lastDebugGroupStack.size();
      for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
        if (i >= cmd.debugGroupStack.size() || cmd.debugGroupStack[i] != lastDebugGroupStack[i]) {
          firstDiff = i;
          break;
        }
      }
      for (size_t i = firstDiff; i < lastDebugGroupStack.size(); ++i) {
        pass.PopDebugGroup();
      }
      for (size_t i = firstDiff; i < cmd.debugGroupStack.size(); ++i) {
        pass.PushDebugGroup(cmd.debugGroupStack[i].c_str());
      }
      lastDebugGroupStack = cmd.debugGroupStack;
    }
#endif
    switch (cmd.type) {
    case CommandType::SetViewport: {
      const auto& vp = cmd.data.setViewport;
      const float minDepth = gx::UseReversedZ ? 1.f - vp.zfar : vp.znear;
      const float maxDepth = gx::UseReversedZ ? 1.f - vp.znear : vp.zfar;
      pass.SetViewport(vp.left, vp.top, vp.width, vp.height, minDepth, maxDepth);
    } break;
    case CommandType::SetScissor: {
      const auto& sc = cmd.data.setScissor;
      const auto& size = g_renderPasses[idx].targetSize;
      const auto x = std::clamp(static_cast<uint32_t>(sc.x), 0u, size.width);
      const auto y = std::clamp(static_cast<uint32_t>(sc.y), 0u, size.height);
      const auto w = std::clamp(static_cast<uint32_t>(sc.width), 0u, size.width - x);
      const auto h = std::clamp(static_cast<uint32_t>(sc.height), 0u, size.height - y);
      pass.SetScissorRect(x, y, w, h);
    } break;
    case CommandType::Draw: {
      const auto& draw = cmd.data.draw;
      switch (draw.type) {
      case ShaderType::Clear:
        clear::render(draw.clear, pass, g_renderPasses[idx].targetSize);
        break;
      case ShaderType::GX:
        gx::render(draw.gx, pass);
        break;
      }
    } break;
    case CommandType::DebugMarker: {
#if defined(AURORA_GFX_DEBUG_GROUPS)
      pass.InsertDebugMarker(wgpu::StringView(g_debugMarkers[cmd.data.debugMarkerIndex]));
#endif
    } break;
    }
  }

#ifdef AURORA_GFX_DEBUG_GROUPS
  for (size_t i = 0; i < lastDebugGroupStack.size(); ++i) {
    pass.PopDebugGroup();
  }
#endif
}

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass) {
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  (void)ref;
  (void)pass;
  return false;
#else
  if (ref == g_currentPipeline) {
    return true;
  }
  wgpu::RenderPipeline pipeline;
  if (!get_pipeline(ref, pipeline)) {
    return false;
  }
  pass.SetPipeline(pipeline);
  g_currentPipeline = ref;
  return true;
#endif
}

static inline Range push(ByteBuffer& target, const uint8_t* data, size_t length, size_t alignment) {
  size_t padding = 0;
  if (alignment != 0) {
    const size_t remainder = length % alignment;
    if (remainder != 0) {
      padding = alignment - remainder;
    }
  }
  auto begin = target.size();
  if (length == 0) {
    length = alignment;
    target.append_zeroes(alignment);
  } else {
    target.append(data, length);
    if (padding > 0) {
      target.append_zeroes(padding);
    }
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length + padding)};
}
static inline Range map(ByteBuffer& target, size_t length, size_t alignment) {
  size_t padding = 0;
  if (alignment != 0) {
    const size_t remainder = length % alignment;
    if (remainder != 0) {
      padding = alignment - remainder;
    }
  }
  if (length == 0) {
    length = alignment;
  }
  auto begin = target.size();
  target.append_zeroes(length + padding);
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length + padding)};
}
Range push_verts(const uint8_t* data, size_t length) { return push(g_verts, data, length, 0); }
Range push_indices(const uint8_t* data, size_t length) { return push(g_indices, data, length, 0); }
Range push_uniform(const uint8_t* data, size_t length) {
  return push(g_uniforms, data, length, g_cachedLimits.minUniformBufferOffsetAlignment);
}
Range push_storage(const uint8_t* data, size_t length) {
  return push(g_storage, data, length, g_cachedLimits.minStorageBufferOffsetAlignment);
}
Range push_texture_data(const uint8_t* data, size_t length, u32 bytesPerRow, u32 rowsPerImage) {
  // For CopyBufferToTexture, we need an alignment of 256 per row (see Dawn kTextureBytesPerRowAlignment)
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, 256);
  const auto range = map(g_textureUpload, copyBytesPerRow * rowsPerImage, 0);
  u8* dst = g_textureUpload.data() + range.offset;
  for (u32 i = 0; i < rowsPerImage; ++i) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  return range;
}
std::pair<ByteBuffer, Range> map_verts(size_t length) {
  const auto range = map(g_verts, length, 4);
  return {ByteBuffer{g_verts.data() + range.offset, range.size}, range};
}
std::pair<ByteBuffer, Range> map_indices(size_t length) {
  const auto range = map(g_indices, length, 4);
  return {ByteBuffer{g_indices.data() + range.offset, range.size}, range};
}
std::pair<ByteBuffer, Range> map_uniform(size_t length) {
  const auto range = map(g_uniforms, length, g_cachedLimits.minUniformBufferOffsetAlignment);
  return {ByteBuffer{g_uniforms.data() + range.offset, range.size}, range};
}
std::pair<ByteBuffer, Range> map_storage(size_t length) {
  const auto range = map(g_storage, length, g_cachedLimits.minStorageBufferOffsetAlignment);
  return {ByteBuffer{g_storage.data() + range.offset, range.size}, range};
}

BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor) {
  const auto id = xxh3_hash(descriptor);
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  (void)descriptor;
  static bool s_warned = false;
  if (!s_warned) {
    Log.warn("Ignoring WebGPU bind group request on direct Vulkan GX path");
    s_warned = true;
  }
  return id;
#else
  const auto it = g_cachedBindGroups.find(id);
  if (it == g_cachedBindGroups.end()) {
    auto bg = wgpu::BindGroup::Acquire(wgpuDeviceCreateBindGroup(g_device.Get(), &descriptor));
    g_cachedBindGroups.emplace(id, CachedBindGroup{
                                       .bindGroup = std::move(bg),
                                       .lastUsedFrame = g_frameIndex,
                                   });
  } else {
    it->second.lastUsedFrame = g_frameIndex;
  }
  return id;
#endif
}

wgpu::BindGroup& find_bind_group(BindGroupRef id) {
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  (void)id;
  static wgpu::BindGroup s_directVulkanNullBindGroup;
  return s_directVulkanNullBindGroup;
#else
  const auto it = g_cachedBindGroups.find(id);
  CHECK(it != g_cachedBindGroups.end(), "get_bind_group: failed to locate {:x}", id);
  return it->second.bindGroup;
#endif
}

wgpu::Sampler& sampler_ref(const wgpu::SamplerDescriptor& descriptor) {
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  (void)descriptor;
  static wgpu::Sampler s_directVulkanNullSampler;
  static bool s_warned = false;
  if (!s_warned) {
    Log.warn("Ignoring WebGPU sampler request on direct Vulkan GX path");
    s_warned = true;
  }
  return s_directVulkanNullSampler;
#else
  const auto id = xxh3_hash(descriptor);
  auto it = g_cachedSamplers.find(id);
  if (it == g_cachedSamplers.end()) {
    it = g_cachedSamplers.try_emplace(id, g_device.CreateSampler(&descriptor)).first;
  }
  return it->second;
#endif
}

uint32_t align_uniform(uint32_t value) { return AURORA_ALIGN(value, g_cachedLimits.minUniformBufferOffsetAlignment); }

void insert_debug_marker(std::string label) {
#if defined(AURORA_GFX_DEBUG_GROUPS)
  auto idx = g_debugMarkers.size();
  g_debugMarkers.emplace_back(std::move(label));
  push_command(CommandType::DebugMarker, {.debugMarkerIndex = idx});
#endif
}

} // namespace aurora::gfx

void aurora::gfx::push_debug_group(std::string label) {
#if defined(AURORA_GFX_DEBUG_GROUPS)
  g_debugGroupStack.push_back(std::move(label));
#endif
}
void push_debug_group(const char* label) {
#ifdef AURORA_GFX_DEBUG_GROUPS
  aurora::gfx::g_debugGroupStack.emplace_back(label);
#endif
}
void pop_debug_group() {
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (aurora::gfx::g_debugGroupStack.empty()) {
    aurora::gfx::Log.error("Debug group stack underflowed!");
    return;
  }

  aurora::gfx::g_debugGroupStack.pop_back();
#endif
}

const AuroraStats* aurora_get_stats() { return &aurora::gfx::g_stats; }
