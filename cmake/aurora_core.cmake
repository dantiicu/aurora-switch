set(_aurora_core_platform_sources
        lib/input.cpp
        lib/window.cpp
)
if (AURORA_PLATFORM_SWITCH)
    set(_aurora_core_platform_sources
            lib/input_switch.cpp
            lib/window_switch.cpp
    )
endif ()

add_library(aurora_core STATIC
        lib/aurora.cpp
        ${_aurora_core_platform_sources}
        lib/logging.cpp
)
add_library(aurora::core ALIAS aurora_core)
set_target_properties(aurora_core PROPERTIES FOLDER "aurora")

target_compile_definitions(aurora_core PUBLIC AURORA)
if (AURORA_PLATFORM_SWITCH)
    target_compile_definitions(aurora_core PUBLIC AURORA_PLATFORM_SWITCH TARGET_PC)
else ()
    target_compile_definitions(aurora_core PUBLIC TARGET_PC)
endif ()
if (AURORA_PLATFORM_SWITCH AND AURORA_ENABLE_DIRECT_VULKAN)
    target_compile_definitions(aurora_core PUBLIC AURORA_USE_DIRECT_VULKAN_GX)
endif ()
target_include_directories(aurora_core PUBLIC include)
target_link_libraries(aurora_core PUBLIC fmt::fmt ${AURORA_SDL3_TARGET} xxhash)
target_link_libraries(aurora_core PRIVATE absl::btree absl::flat_hash_map TracyClient)
if (AURORA_ENABLE_GX AND AURORA_ENABLE_GPU_CACHE)
    target_link_libraries(aurora_core PRIVATE sqlite3)
endif ()
if (AURORA_ENABLE_GX AND AURORA_ENABLE_GPU_CACHE AND AURORA_CACHE_USE_ZSTD)
    target_compile_definitions(aurora_core PRIVATE AURORA_CACHE_USE_ZSTD)
    target_link_libraries(aurora_core PRIVATE libzstd_static)
endif ()

if (AURORA_ENABLE_GX AND AURORA_ENABLE_IMGUI)
    target_compile_definitions(aurora_core PUBLIC AURORA_ENABLE_IMGUI)
    target_sources(aurora_core PRIVATE lib/imgui.cpp)
    target_link_libraries(aurora_core PUBLIC imgui)
endif ()

if(AURORA_ENABLE_RMLUI)
    target_compile_definitions(aurora_core PUBLIC AURORA_ENABLE_RMLUI)

    target_sources(aurora_core PRIVATE
            lib/rmlui.cpp
            lib/rmlui/RmlUi_Backend_Aurora.cpp
            lib/rmlui/WebGPURenderInterface.cpp
            lib/rmlui/SystemInterface_Aurora.cpp
            lib/rmlui/FileInterface_SDL.cpp
    )
    target_link_libraries(aurora_core PUBLIC rmlui)

    target_link_libraries(aurora_core PUBLIC rmlui_backends)

    if (AURORA_PLATFORM_SWITCH)
        find_library(AURORA_SWITCH_PNG_LIBRARY NAMES png png16
                PATHS "$ENV{DEVKITPRO}/portlibs/switch/lib"
                NO_DEFAULT_PATH)
        if (NOT AURORA_SWITCH_PNG_LIBRARY)
            message(FATAL_ERROR "aurora: Missing Switch libpng dependency")
        endif ()
        target_link_libraries(aurora_core PRIVATE ${AURORA_SWITCH_PNG_LIBRARY})
    endif ()
endif ()

if (AURORA_ENABLE_GX)
    target_compile_definitions(aurora_core PUBLIC AURORA_ENABLE_GX WEBGPU_DAWN)
    if (AURORA_ENABLE_GPU_CACHE)
        set(_aurora_gpu_cache_source lib/webgpu/gpu_cache.cpp)
    else ()
        set(_aurora_gpu_cache_source lib/webgpu/gpu_cache_null.cpp)
    endif ()
    target_sources(aurora_core PRIVATE lib/webgpu/gpu.cpp ${_aurora_gpu_cache_source} lib/dawn/BackendBinding.cpp)
    target_link_libraries(aurora_core PRIVATE dawn::webgpu_dawn)
    if (AURORA_PLATFORM_SWITCH)
        set(DAWN_SWITCH_NVK_ROOT "/opt/nvk-switch" CACHE PATH
            "Mesa/NVK Switch install root")
        set(DAWN_SWITCH_NVK_LIBRARY "${DAWN_SWITCH_NVK_ROOT}/lib/libvulkan.a"
            CACHE FILEPATH "Mesa/NVK Switch Vulkan static archive")
        if (NOT EXISTS "${DAWN_SWITCH_NVK_LIBRARY}")
            message(FATAL_ERROR
                "DAWN_SWITCH_NVK_LIBRARY does not exist: ${DAWN_SWITCH_NVK_LIBRARY}")
        endif ()
        target_compile_definitions(aurora_core PUBLIC
            __SWITCH__
            NX
            VK_USE_PLATFORM_VI_NN)
        target_include_directories(aurora_core PRIVATE
            "${DAWN_SWITCH_NVK_ROOT}/include")
        target_link_libraries(aurora_core PRIVATE
            "${DAWN_SWITCH_NVK_LIBRARY}"
            drm_nouveau
            zstd
            z
            nx
            m)
        if (TARGET webgpu_dawn)
            target_include_directories(webgpu_dawn INTERFACE
                "${DAWN_SWITCH_NVK_ROOT}/include")
            target_link_libraries(webgpu_dawn INTERFACE
                "${DAWN_SWITCH_NVK_LIBRARY}"
                drm_nouveau
                zstd
                z
                nx
                m)
        endif ()
    endif ()
    if (DAWN_ENABLE_VULKAN)
        target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_VULKAN)
    endif ()
    if (DAWN_ENABLE_METAL)
        target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_METAL)
        target_sources(aurora_core PRIVATE lib/dawn/MetalBinding.mm)
        set_source_files_properties(lib/dawn/MetalBinding.mm PROPERTIES COMPILE_FLAGS -fobjc-arc)
    endif ()
    if (DAWN_ENABLE_D3D11)
        target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_D3D11)
    endif ()
    if (DAWN_ENABLE_D3D12)
        target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_D3D12)
    endif ()
    if (DAWN_ENABLE_DESKTOP_GL OR DAWN_ENABLE_OPENGLES)
        target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_OPENGL)
        if (DAWN_ENABLE_DESKTOP_GL)
            target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_DESKTOP_GL)
        endif ()
        if (DAWN_ENABLE_OPENGLES)
            target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_OPENGLES)
        endif ()
    endif ()
    if (DAWN_ENABLE_NULL)
        target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_NULL)
    endif ()
endif ()
