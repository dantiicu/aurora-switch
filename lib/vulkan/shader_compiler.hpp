#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

namespace aurora::vk {

struct SpirvCompileResult {
  std::vector<uint32_t> words;
  std::string error;
};

std::optional<SpirvCompileResult> compile_wgsl_to_spirv(std::string_view source, std::string_view entryPoint,
                                                        std::string_view label) noexcept;
VkShaderModule create_shader_module_from_wgsl(std::string_view source, std::string_view entryPoint,
                                              std::string_view label) noexcept;

} // namespace aurora::vk
