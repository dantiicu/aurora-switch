if (AURORA_ENABLE_GX)
  add_executable(simple simple.c)
  target_link_libraries(simple PRIVATE aurora::core aurora::gx aurora::main aurora::vi)

  if (AURORA_PLATFORM_SWITCH)
    # The Switch SDL3 shim (include/SDL3/SDL.h), pulled in transitively via
    # <aurora/event.h>, is C++-only. Build the example as C++ so it can consume it;
    # the aurora/dolphin headers are extern "C", so linkage is unaffected.
    set_source_files_properties(simple.c PROPERTIES LANGUAGE CXX)

    set(DAWN_SWITCH_NVK_ROOT "/opt/nvk-switch" CACHE PATH
      "Mesa/NVK Switch install root")
    set(DAWN_SWITCH_NVK_LIBRARY "${DAWN_SWITCH_NVK_ROOT}/lib/libvulkan.a"
      CACHE FILEPATH "Mesa/NVK Switch Vulkan static archive")

    if (NOT EXISTS "${DAWN_SWITCH_NVK_LIBRARY}")
      message(FATAL_ERROR
        "DAWN_SWITCH_NVK_LIBRARY does not exist: ${DAWN_SWITCH_NVK_LIBRARY}")
    endif ()

    target_compile_definitions(simple PRIVATE
      __SWITCH__
      NX
      VK_USE_PLATFORM_VI_NN)
    target_include_directories(simple PRIVATE
      "${DAWN_SWITCH_NVK_ROOT}/include")
    target_link_libraries(simple PRIVATE
      "${DAWN_SWITCH_NVK_LIBRARY}"
      drm_nouveau
      zstd
      z
      nx
      m)
    target_link_options(simple PRIVATE
      -Wl,--gc-sections
      -Wl,--allow-multiple-definition)

    find_program(AURORA_SWITCH_ELF2NRO elf2nro REQUIRED)
    find_program(AURORA_SWITCH_NACPTOOL nacptool REQUIRED)

    set(_simple_nacp "${CMAKE_CURRENT_BINARY_DIR}/simple.nacp")
    set(_simple_nro "${CMAKE_CURRENT_BINARY_DIR}/simple.nro")
    add_custom_command(
      OUTPUT "${_simple_nacp}"
      COMMAND "${AURORA_SWITCH_NACPTOOL}" --create
              "Aurora Simple" "Aurora" "0.1.0"
              "${_simple_nacp}"
      VERBATIM)
    add_custom_command(
      OUTPUT "${_simple_nro}"
      COMMAND "${AURORA_SWITCH_ELF2NRO}"
              "$<TARGET_FILE:simple>"
              "${_simple_nro}"
              --nacp=${_simple_nacp}
      DEPENDS simple "${_simple_nacp}"
      VERBATIM)
    add_custom_target(simple_nro DEPENDS "${_simple_nro}")
  endif ()
endif ()
