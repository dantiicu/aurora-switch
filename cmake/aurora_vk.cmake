add_library(aurora_vk STATIC
        lib/vulkan/gpu.cpp
        lib/vulkan/gx_pipeline.cpp
        lib/vulkan/gx_pipeline_cache.cpp
        lib/vulkan/gx_resources.cpp
        lib/vulkan/resources.cpp
        lib/vulkan/shader_compiler.cpp
)
add_library(aurora::vk ALIAS aurora_vk)
set_target_properties(aurora_vk PROPERTIES FOLDER "aurora")

target_link_libraries(aurora_vk PUBLIC aurora::core)
target_link_libraries(aurora_vk PRIVATE aurora::gx dawn::webgpu_dawn absl::flat_hash_map)
if (TARGET tint_api)
    target_link_libraries(aurora_vk PRIVATE tint_api)
else ()
    message(FATAL_ERROR "aurora_vk requires Tint's tint_api target for WGSL -> SPIR-V compilation")
endif ()
target_compile_definitions(aurora_vk PUBLIC AURORA_ENABLE_DIRECT_VULKAN VK_USE_PLATFORM_VI_NN)

if (AURORA_PLATFORM_SWITCH)
    set(DAWN_SWITCH_NVK_ROOT "/opt/nvk-switch" CACHE PATH
        "Mesa/NVK Switch install root")
    set(DAWN_SWITCH_NVK_LIBRARY "${DAWN_SWITCH_NVK_ROOT}/lib/libvulkan.a"
        CACHE FILEPATH "Mesa/NVK Switch Vulkan static archive")
    if (NOT EXISTS "${DAWN_SWITCH_NVK_LIBRARY}")
        message(FATAL_ERROR
            "DAWN_SWITCH_NVK_LIBRARY does not exist: ${DAWN_SWITCH_NVK_LIBRARY}")
    endif ()
    target_compile_definitions(aurora_vk PUBLIC __SWITCH__ NX)
    target_include_directories(aurora_vk PRIVATE
        "${DAWN_SWITCH_NVK_ROOT}/include")
    target_link_libraries(aurora_vk PRIVATE
        "${DAWN_SWITCH_NVK_LIBRARY}"
        drm_nouveau
        zstd
        z
        nx
        m)
endif ()
