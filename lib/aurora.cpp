#include <aurora/aurora.h>

#ifdef AURORA_ENABLE_GX
#include "gfx/common.hpp"
#include "gx/fifo.hpp"
#ifdef AURORA_USE_DIRECT_VULKAN_GX
#include "vulkan/gpu.hpp"
#endif
#include "webgpu/gpu.hpp"
#include <webgpu/webgpu_cpp.h>
#endif
#ifdef AURORA_ENABLE_IMGUI
#include "imgui.hpp"
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "input.hpp"
#include "internal.hpp"
#include "window.hpp"

#if !defined(__SWITCH__)
#include <SDL3/SDL_filesystem.h>
#endif
#include <chrono>
#include <magic_enum.hpp>

#include "tracy/Tracy.hpp"

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
Module Log("aurora");
#if defined(AURORA_USE_DIRECT_VULKAN_GX)
uint32_t g_directTraceFrame = 0;
constexpr bool DirectVkFrameTrace = false;
#endif

#if defined(__SWITCH__)
const char* SDL_GetError() {
  return "Switch standalone backend";
}
#endif

#if defined(__SWITCH__) && defined(AURORA_ENABLE_GX)
using SwitchProfileClock = std::chrono::steady_clock;

static double elapsed_ms(SwitchProfileClock::time_point start, SwitchProfileClock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct SwitchAuroraEndFrameProfile {
  uint32_t frames = 0;
  double total = 0.0;
  double fifoDrain = 0.0;
  double gfxEnd = 0.0;
  double render = 0.0;
  double rml = 0.0;
  double copyPass = 0.0;
  double finish = 0.0;
  double submit = 0.0;
  double afterSubmit = 0.0;
  double present = 0.0;
};

static SwitchAuroraEndFrameProfile g_switchAuroraEndFrameProfile;

static void record_switch_aurora_end_frame_profile(double total, double fifoDrain, double gfxEnd, double render,
                                                   double rml, double copyPass, double finish, double submit,
                                                   double afterSubmit, double present) {
  constexpr uint32_t LogEveryFrames = 300;
  auto& profile = g_switchAuroraEndFrameProfile;
  profile.frames++;
  profile.total += total;
  profile.fifoDrain += fifoDrain;
  profile.gfxEnd += gfxEnd;
  profile.render += render;
  profile.rml += rml;
  profile.copyPass += copyPass;
  profile.finish += finish;
  profile.submit += submit;
  profile.afterSubmit += afterSubmit;
  profile.present += present;

  if (profile.frames < LogEveryFrames) {
    return;
  }

  const double inv = 1.0 / static_cast<double>(profile.frames);
  Log.info("[SwitchProfile][aurora_end] avg_ms total={:.3f} fifo={:.3f} gfxEnd={:.3f} render={:.3f} rml={:.3f} copy={:.3f} finish={:.3f} submit={:.3f} afterSubmit={:.3f} present={:.3f}",
           profile.total * inv, profile.fifoDrain * inv, profile.gfxEnd * inv, profile.render * inv,
           profile.rml * inv, profile.copyPass * inv, profile.finish * inv, profile.submit * inv,
           profile.afterSubmit * inv, profile.present * inv);
  profile = {};
}
#endif

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
// #ifdef DAWN_ENABLE_BACKEND_OPENGLES
//     BACKEND_OPENGLES,
// #endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.configPath == nullptr) {
#if defined(__SWITCH__)
    g_config.configPath = strdup("sdmc:/aurora");
#else
    g_config.configPath = SDL_GetPrefPath(nullptr, g_config.appName);
#endif
  } else {
    g_config.configPath = strdup(g_config.configPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  ASSERT(window::initialize(), "Error initializing window");

#if defined(__SWITCH__)
  g_sdlCustomEventsStart = 0x8000;
#else
  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
#endif
  ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  selectedBackend = BACKEND_VULKAN;
  if (window::create_window(selectedBackend)) {
    if (vk::initialize(g_config.vsync)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }
#else
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }
#endif

  ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

#ifndef AURORA_USE_DIRECT_VULKAN_GX
  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#endif
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();

#ifdef AURORA_ENABLE_GX
  gfx::initialize();

#ifdef AURORA_ENABLE_IMGUI
  imgui::create_context();
#endif
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_IMGUI
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
#ifndef AURORA_USE_DIRECT_VULKAN_GX
  rmlui::initialize(size);
#endif
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .configPath = g_config.configPath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

#ifdef AURORA_ENABLE_GX
wgpu::TextureView g_currentView;
#endif

void shutdown() noexcept {
#ifdef AURORA_ENABLE_RMLUI
#ifndef AURORA_USE_DIRECT_VULKAN_GX
  rmlui::shutdown();
#endif
#endif
#ifdef AURORA_ENABLE_GX
  g_currentView = {};
#ifdef AURORA_ENABLE_IMGUI
  imgui::shutdown();
#endif
  gfx::shutdown();
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  vk::shutdown();
#else
  webgpu::shutdown();
#endif
#endif
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
  }
  return window::poll_events();
}

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  {
    window::SurfaceLock surfaceLock;
    if (!window::is_presentable() || window::is_paused()) {
      return false;
    }
    if (DirectVkFrameTrace && g_directTraceFrame < 4) {
      Log.info("[DirectVK] frame {} vk begin_frame begin", g_directTraceFrame);
    }
    if (!vk::begin_frame({0.f, 0.f, 0.f, 1.f}, false)) {
      return false;
    }
    if (DirectVkFrameTrace && g_directTraceFrame < 4) {
      Log.info("[DirectVK] frame {} vk begin_frame done", g_directTraceFrame);
    }
  }
  if (DirectVkFrameTrace && g_directTraceFrame < 4) {
    Log.info("[DirectVK] frame {} gfx begin_frame begin", g_directTraceFrame);
  }
  if (!gfx::begin_frame()) {
    vk::end_frame();
    return false;
  }
  if (DirectVkFrameTrace && g_directTraceFrame < 4) {
    Log.info("[DirectVK] frame {} gfx begin_frame done", g_directTraceFrame);
  }
  return true;
#else
  {
    window::SurfaceLock surfaceLock;
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (!g_surface) {
      webgpu::refresh_surface(true);
      if (!g_surface) {
        return false;
      }
    }
    wgpu::SurfaceTexture surfaceTexture;
    g_surface.GetCurrentTexture(&surfaceTexture);
    switch (surfaceTexture.status) {
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal:
      g_currentView = surfaceTexture.texture.CreateView();
      break;
    case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
      Log.warn("Surface texture acquisition timed out");
      return false;
    case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
    case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
      Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceTexture.status));
      webgpu::refresh_surface(false);
      return false;
    case wgpu::SurfaceGetCurrentTextureStatus::Lost:
      Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceTexture.status));
      webgpu::release_surface();
    case wgpu::SurfaceGetCurrentTextureStatus::Error:
      Log.warn("Surface texture is {}, dropping surface", magic_enum::enum_name(surfaceTexture.status));
      g_surface = {};
      return false;
    default:
      Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceTexture.status));
      return false;
    }
  }

#ifdef AURORA_ENABLE_IMGUI
  imgui::new_frame(window::get_window_size());
#endif
  if (!gfx::begin_frame()) {
    g_currentView = {};
    return false;
  }
#endif
#endif
  return true;
}

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
#ifdef AURORA_USE_DIRECT_VULKAN_GX
  if (DirectVkFrameTrace && g_directTraceFrame < 4) {
    Log.info("[DirectVK] frame {} fifo drain begin", g_directTraceFrame);
  }
  gx::fifo::drain();
  if (DirectVkFrameTrace && g_directTraceFrame < 4) {
    Log.info("[DirectVK] frame {} fifo drain done", g_directTraceFrame);
  }
  const auto& frame = vk::current_frame();
  if (frame.commandBuffer != VK_NULL_HANDLE) {
    if (DirectVkFrameTrace && g_directTraceFrame < 4) {
      Log.info("[DirectVK] frame {} gfx end_frame begin", g_directTraceFrame);
    }
    gfx::end_frame(frame.commandBuffer);
    if (DirectVkFrameTrace && g_directTraceFrame < 4) {
      Log.info("[DirectVK] frame {} gfx end_frame done", g_directTraceFrame);
    }
    if (DirectVkFrameTrace && g_directTraceFrame < 4) {
      Log.info("[DirectVK] frame {} gfx render begin", g_directTraceFrame);
    }
    gfx::render(frame.commandBuffer);
    if (DirectVkFrameTrace && g_directTraceFrame < 4) {
      Log.info("[DirectVK] frame {} gfx render done", g_directTraceFrame);
    }
  }
  if (DirectVkFrameTrace && g_directTraceFrame < 4) {
    Log.info("[DirectVK] frame {} vk end_frame begin", g_directTraceFrame);
  }
  vk::end_frame();
  if (DirectVkFrameTrace && g_directTraceFrame < 4) {
    Log.info("[DirectVK] frame {} vk end_frame done", g_directTraceFrame);
  }
  ++g_directTraceFrame;
  TracyPlot("aurora: drawCallCount", static_cast<int64_t>(gfx::g_stats.drawCallCount));
  TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(gfx::g_stats.mergedDrawCallCount));
  TracyPlot("aurora: lastVertSize", static_cast<int64_t>(gfx::g_stats.lastVertSize));
  TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(gfx::g_stats.lastUniformSize));
  TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(gfx::g_stats.lastIndexSize));
  TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(gfx::g_stats.lastStorageSize));
  return;
#else
#if defined(__SWITCH__)
  const auto profileStart = SwitchProfileClock::now();
  auto profileStepStart = profileStart;
  double profileFifoDrain = 0.0;
  double profileGfxEnd = 0.0;
  double profileRender = 0.0;
  double profileRml = 0.0;
  double profileCopyPass = 0.0;
  double profileFinish = 0.0;
  double profileSubmit = 0.0;
  double profileAfterSubmit = 0.0;
  double profilePresent = 0.0;
#endif
  gx::fifo::drain();
#if defined(__SWITCH__)
  auto profileNow = SwitchProfileClock::now();
  profileFifoDrain = elapsed_ms(profileStepStart, profileNow);
  profileStepStart = profileNow;
#endif
  const auto encoderDescriptor = wgpu::CommandEncoderDescriptor{
      .label = "Redraw encoder",
  };
  auto encoder = g_device.CreateCommandEncoder(&encoderDescriptor);
  gfx::end_frame(encoder);
#if defined(__SWITCH__)
  profileNow = SwitchProfileClock::now();
  profileGfxEnd = elapsed_ms(profileStepStart, profileNow);
  profileStepStart = profileNow;
#endif
  gfx::render(encoder);
#if defined(__SWITCH__)
  profileNow = SwitchProfileClock::now();
  profileRender = elapsed_ms(profileStepStart, profileNow);
  profileStepStart = profileNow;
#endif
  {
    window::SurfaceLock surfaceLock;
    if (window::is_presentable() && g_surface && g_currentView) {
      const auto& presentSource = webgpu::present_source();
      auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                         webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                         presentSource.size.width, presentSource.size.height);
      wgpu::BindGroup presentBindGroup = webgpu::g_CopyBindGroup;
    #if AURORA_ENABLE_RMLUI
      if (rmlui::is_initialized()) {
      #if defined(__SWITCH__)
        const auto profileRmlStart = SwitchProfileClock::now();
      #endif
        const auto rmlOutput = rmlui::render(encoder, viewport);
      #if defined(__SWITCH__)
        profileRml += elapsed_ms(profileRmlStart, SwitchProfileClock::now());
      #endif
        if (rmlOutput.texture != nullptr) {
          presentBindGroup = rmlOutput.copyBindGroup;
        }
      }
    #endif
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = g_currentView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        pass.SetPipeline(webgpu::g_CopyPipeline);
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
        pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);

        pass.Draw(3);
        pass.End();
      }
#if defined(__SWITCH__)
      profileNow = SwitchProfileClock::now();
      profileCopyPass = elapsed_ms(profileStepStart, profileNow) - profileRml;
      profileStepStart = profileNow;
#endif
#if defined(AURORA_ENABLE_IMGUI) && !defined(AURORA_PLATFORM_SWITCH)
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = g_currentView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass);
        pass.End();
      }
#endif
    } else {
      Log.info("Skipping present; window not presentable");
      webgpu::release_surface();
    }
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    const auto buffer = encoder.Finish(&cmdBufDescriptor);
#if defined(__SWITCH__)
    profileNow = SwitchProfileClock::now();
    profileFinish = elapsed_ms(profileStepStart, profileNow);
    profileStepStart = profileNow;
#endif
    g_queue.Submit(1, &buffer);
#if defined(__SWITCH__)
    profileNow = SwitchProfileClock::now();
    profileSubmit = elapsed_ms(profileStepStart, profileNow);
    profileStepStart = profileNow;
#endif
    gfx::after_submit();
#if defined(__SWITCH__)
    profileNow = SwitchProfileClock::now();
    profileAfterSubmit = elapsed_ms(profileStepStart, profileNow);
    profileStepStart = profileNow;
#endif
    if (window::is_presentable() && g_surface) {
      auto presentStatus = g_surface.Present();
#if defined(__SWITCH__)
      profileNow = SwitchProfileClock::now();
      profilePresent = elapsed_ms(profileStepStart, profileNow);
      profileStepStart = profileNow;
#endif
      if (presentStatus != wgpu::Status::Success) {
        Log.warn("Surface present failed: {}", static_cast<int>(presentStatus));
        webgpu::release_surface();
      }
    } else if (g_surface) {
      webgpu::release_surface();
    }
    g_currentView = {};
  }
#if defined(__SWITCH__)
  record_switch_aurora_end_frame_profile(elapsed_ms(profileStart, SwitchProfileClock::now()), profileFifoDrain,
                                         profileGfxEnd, profileRender, profileRml, profileCopyPass, profileFinish,
                                         profileSubmit, profileAfterSubmit, profilePresent);
#endif

  TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

  TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(gfx::g_stats.queuedPipelines));
  TracyPlot("aurora: createdPipelines", static_cast<int64_t>(gfx::g_stats.createdPipelines));
  TracyPlot("aurora: drawCallCount", static_cast<int64_t>(gfx::g_stats.drawCallCount));
  TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(gfx::g_stats.mergedDrawCallCount));
  TracyPlot("aurora: lastVertSize", static_cast<int64_t>(gfx::g_stats.lastVertSize));
  TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(gfx::g_stats.lastUniformSize));
  TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(gfx::g_stats.lastIndexSize));
  TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(gfx::g_stats.lastStorageSize));
  TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(gfx::g_stats.lastTextureUploadSize));

#endif
#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
