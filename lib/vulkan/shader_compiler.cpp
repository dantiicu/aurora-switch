#include "shader_compiler.hpp"

#include "gpu.hpp"
#include "../internal.hpp"

#include "src/tint/lang/spirv/writer/writer.h"
#include "src/tint/lang/wgsl/reader/reader.h"
#include "src/tint/utils/diagnostic/source.h"

#include <exception>
#include <atomic>
#include <cstdint>
#include <utility>

namespace aurora::vk {
namespace {
Module Log("aurora::vk::shader");
std::atomic<uint32_t> g_compileTraceCount{0};

uint64_t handle_value(VkShaderModule handle) {
#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
#else
  return static_cast<uint64_t>(handle);
#endif
}

std::optional<SpirvCompileResult> fail(std::string error) {
  Log.error("WGSL -> SPIR-V failed: {}", error);
  return SpirvCompileResult{
      .words = {},
      .error = std::move(error),
  };
}
} // namespace

std::optional<SpirvCompileResult> compile_wgsl_to_spirv(std::string_view source, std::string_view entryPoint,
                                                        std::string_view label) noexcept {
  try {
    const uint32_t traceId = g_compileTraceCount.fetch_add(1);
    const bool trace = traceId < 16;
    if (trace) {
      Log.debug("[DirectVK] shader {} compile begin label={} entry={} bytes={}", traceId, label, entryPoint,
                source.size());
    }
    tint::Source::File file{std::string(label), source};
    if (trace) {
      Log.debug("[DirectVK] shader {} WGSL parse begin", traceId);
    }
    tint::Program program = tint::wgsl::reader::Parse(&file);
    if (!program.IsValid()) {
      return fail(program.Diagnostics().Str());
    }
    if (trace) {
      Log.debug("[DirectVK] shader {} WGSL parse done", traceId);
    }

    tint::wgsl::reader::IROptions irOptions{};
    if (trace) {
      Log.debug("[DirectVK] shader {} IR lower begin", traceId);
    }
    auto ir = tint::wgsl::reader::ProgramToLoweredIR(program, irOptions);
    if (ir != tint::Success) {
      return fail(ir.Failure().reason);
    }
    if (trace) {
      Log.debug("[DirectVK] shader {} IR lower done", traceId);
    }

    tint::spirv::writer::Options options{};
    options.entry_point_name = std::string(entryPoint);
    options.spirv_version = tint::spirv::writer::SpvVersion::kSpv13;
    options.disable_robustness = true;
    options.disable_workgroup_init = true;
    options.strip_all_names = true;

    if (trace) {
      Log.debug("[DirectVK] shader {} SPIR-V generate begin", traceId);
    }
    auto spirv = tint::spirv::writer::Generate(ir.Get(), options);
    if (spirv != tint::Success) {
      return fail(spirv.Failure().reason);
    }
    if (trace) {
      Log.debug("[DirectVK] shader {} SPIR-V generate done words={}", traceId, spirv->spirv.size());
    }

    return SpirvCompileResult{
        .words = std::move(spirv->spirv),
        .error = {},
    };
  } catch (const std::exception& ex) {
    return fail(ex.what());
  } catch (...) {
    return fail("unknown exception");
  }
}

VkShaderModule create_shader_module_from_wgsl(std::string_view source, std::string_view entryPoint,
                                              std::string_view label) noexcept {
  const auto vkDevice = device();
  if (vkDevice == VK_NULL_HANDLE) {
    Log.error("Cannot create shader module before Vulkan device initialization");
    return VK_NULL_HANDLE;
  }

  const auto spirv = compile_wgsl_to_spirv(source, entryPoint, label);
  if (!spirv || spirv->words.empty()) {
    return VK_NULL_HANDLE;
  }
  Log.debug("[DirectVK] vkCreateShaderModule begin label={} entry={} words={}", label, entryPoint,
            spirv->words.size());

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = spirv->words.size() * sizeof(uint32_t);
  createInfo.pCode = spirv->words.data();

  VkShaderModule module = VK_NULL_HANDLE;
  const VkResult result = vkCreateShaderModule(vkDevice, &createInfo, nullptr, &module);
  if (result != VK_SUCCESS) {
    Log.error("vkCreateShaderModule failed for {}: {}", label, static_cast<int>(result));
    return VK_NULL_HANDLE;
  }
  Log.debug("[DirectVK] vkCreateShaderModule done label={} handle={}", label, handle_value(module));
  return module;
}

} // namespace aurora::vk
