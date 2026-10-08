cmake_minimum_required(VERSION 3.20)

project(avbd2d VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release builds carry a PDB so a crash can be traced to a function and line. /Z7 puts the
# debug info in each object (no shared compile-time PDB under parallel Ninja); /OPT:REF,ICF
# restore what /DEBUG turns off. Code generation is unchanged.
if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/Z7>)
    add_link_options($<$<CONFIG:Release>:/DEBUG> $<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
endif()

if(NOT WIN32)
    message(FATAL_ERROR "avbd2d v1 is only built and tested on Windows.")
endif()

# ---------------------------------------------------------------------------
# Toolchain: the Vulkan SDK (loader + slangc) and Python (SPIR-V embedding).
# ---------------------------------------------------------------------------
find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND AND DEFINED ENV{VULKAN_SDK})
    find_package(Vulkan QUIET HINTS "$ENV{VULKAN_SDK}")
endif()
if(NOT Vulkan_FOUND)
    find_package(Vulkan QUIET HINTS "C:/VulkanSDK/1.4.357.0")
endif()
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "The Vulkan SDK was not found. "
        "Set VULKAN_SDK or pass -DVulkan_ROOT=<path to SDK>.")
endif()

# slangc.exe ships in the SDK's Bin directory next to vulkan-1.lib; Vulkan::Vulkan does not
# expose it, so it is found by hand.
get_filename_component(AVBD_VULKAN_SDK_BIN "${Vulkan_INCLUDE_DIR}/../Bin" ABSOLUTE)
find_program(AVBD_SLANGC slangc
    HINTS "${AVBD_VULKAN_SDK_BIN}" "$ENV{VULKAN_SDK}/Bin" "C:/VulkanSDK/1.4.357.0/Bin")
if(NOT AVBD_SLANGC)
    message(FATAL_ERROR "slangc was not found next to the Vulkan SDK.")
endif()
message(STATUS "avbd2d Vulkan: SDK ${Vulkan_INCLUDE_DIR}, slangc ${AVBD_SLANGC}")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(AVBD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# Compute shaders: <dir>/<name>.slang -> SPIR-V -> an embedded C array
# (<gen>/<name>_spirv.h exposing `<name>_spirv` / `<name>_spirv_words`).
#
# Modules import each other and slangc has no depfile output, so every SPIR-V depends on
# every .slang in its directory: an edit to an imported module must recompile its importers.
# ---------------------------------------------------------------------------
function(avbd_compile_slang dir gen name out_header)
    file(GLOB all_slang CONFIGURE_DEPENDS ${dir}/*.slang)
    set(src ${dir}/${name}.slang)
    set(spv ${gen}/${name}.spv)
    set(hdr ${gen}/${name}_spirv.h)

    add_custom_command(
        OUTPUT ${spv}
        COMMAND ${AVBD_SLANGC} ${src}
                -target spirv -profile spirv_1_5
                -fvk-use-scalar-layout
                -fvk-use-entrypoint-name
                -emit-spirv-directly
                -O2
                -I ${dir}
                -o ${spv}
        DEPENDS ${all_slang}
        COMMENT "slangc: ${name}.slang -> SPIR-V"
        VERBATIM)

    add_custom_command(
        OUTPUT ${hdr}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
                ${spv} ${hdr} ${name}_spirv
        DEPENDS ${spv} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
        COMMENT "embed: ${name}_spirv.h"
        VERBATIM)

    set(${out_header} ${hdr} PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# avbd2d_gpu -- the shared Vulkan layer: device, buffers, command lists, the radix sort and
# the CSR constraint graph with Jones-Plassmann colouring.
# ---------------------------------------------------------------------------
set(AVBD_GPU_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders_gpu_gen)
file(MAKE_DIRECTORY ${AVBD_GPU_GEN_DIR})

set(AVBD_GPU_SPIRV_HEADERS)
foreach(module scan compact radix constraints)
    avbd_compile_slang(${AVBD_SRC}/gpu/shaders ${AVBD_GPU_GEN_DIR} ${module} hdr)
    list(APPEND AVBD_GPU_SPIRV_HEADERS ${hdr})
endforeach()

add_library(avbd2d_gpu STATIC
    src/gpu/vk_device.cpp
    src/gpu/vk_buffer.cpp
    src/gpu/vk_pipeline.cpp
    src/gpu/vk_cmd.cpp
    src/gpu/vk_primitives.cpp
    src/gpu/vk_constraints.cpp
    src/gpu/vk_gate.cpp
    ${AVBD_GPU_SPIRV_HEADERS})
target_include_directories(avbd2d_gpu PUBLIC
    ${AVBD_SRC}/gpu
    ${AVBD_GPU_GEN_DIR})
target_link_libraries(avbd2d_gpu PUBLIC Vulkan::Vulkan)
set_target_properties(avbd2d_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# avbd2d_vk -- VkWorld2D + the embedded 2D kernels.
# ---------------------------------------------------------------------------
set(AVBD_2D_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders2d_gen)
file(MAKE_DIRECTORY ${AVBD_2D_GEN_DIR})

set(AVBD_2D_SPIRV_HEADERS)
foreach(module integrate2d broadphase2d narrowphase2d solver2d sleep2d events2d ccd2d constraints2d query2d)
    avbd_compile_slang(${AVBD_SRC}/shaders ${AVBD_2D_GEN_DIR} ${module} hdr)
    list(APPEND AVBD_2D_SPIRV_HEADERS ${hdr})
endforeach()

add_library(avbd2d_vk STATIC
    src/world/vk_world2d.cpp
    src/query/query2d.cpp
    ${AVBD_2D_SPIRV_HEADERS})
target_include_directories(avbd2d_vk PUBLIC
    ${AVBD_SRC}/core
    ${AVBD_SRC}/scene
    ${AVBD_SRC}/query
    ${AVBD_SRC}/world
    ${AVBD_2D_GEN_DIR})
target_link_libraries(avbd2d_vk PUBLIC avbd2d_gpu)
set_target_properties(avbd2d_vk PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# avbd2d_solver -- the C ABI (include/avbd2d/avbd2d.h), a shared library.
# ---------------------------------------------------------------------------
add_library(avbd2d_api OBJECT src/api/avbd2d_c_api.cpp)
target_include_directories(avbd2d_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/avbd2d)
target_link_libraries(avbd2d_api PUBLIC avbd2d_vk)
target_compile_definitions(avbd2d_api PRIVATE AVBD2D_BUILD_SHARED)
set_target_properties(avbd2d_api PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(avbd2d_solver SHARED $<TARGET_OBJECTS:avbd2d_api>)
target_include_directories(avbd2d_solver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/avbd2d>)
target_link_libraries(avbd2d_solver PRIVATE avbd2d_vk)
target_compile_definitions(avbd2d_solver INTERFACE AVBD2D_USE_SHARED)

# ---------------------------------------------------------------------------
# Third-party (demo only): SDL2 and ImGui, vendored as submodules in external/.
# ---------------------------------------------------------------------------
add_subdirectory(external/SDL)

file(GLOB IMGUI_SRC
    external/imgui/*.cpp
    external/imgui/backends/imgui_impl_sdl2.cpp)

add_library(imgui STATIC ${IMGUI_SRC})
target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends
    external/SDL/include)
target_link_libraries(imgui PUBLIC SDL2::SDL2 SDL2::SDL2main)

# ---------------------------------------------------------------------------
# avbd2d_demo -- SDL2 + ImGui demo.
# ---------------------------------------------------------------------------
set(AVBD_DEMO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/demo_shaders_gen)
file(MAKE_DIRECTORY ${AVBD_DEMO_GEN_DIR})
avbd_compile_slang(${CMAKE_CURRENT_SOURCE_DIR}/demo/shaders ${AVBD_DEMO_GEN_DIR} scene2d scene2d_hdr)

add_executable(avbd2d_demo
    demo/main2d.cpp
    demo/renderer2d.cpp
    demo/renderer2d.h
    external/imgui/backends/imgui_impl_vulkan.cpp
    ${scene2d_hdr})
target_include_directories(avbd2d_demo PRIVATE
    ${AVBD_SRC}/scene
    ${CMAKE_CURRENT_SOURCE_DIR}/demo
    ${AVBD_DEMO_GEN_DIR}
    external/SDL/include)
target_link_libraries(avbd2d_demo PRIVATE avbd2d_vk imgui SDL2::SDL2 SDL2::SDL2main)
add_custom_command(TARGET avbd2d_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL2::SDL2> $<TARGET_FILE_DIR:avbd2d_demo>
    COMMENT "Copying SDL2.dll next to avbd2d_demo")

# ---------------------------------------------------------------------------
# Install: the shared library, its import library and the one public header.
# ---------------------------------------------------------------------------
install(TARGETS avbd2d_solver
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
install(FILES include/avbd2d/avbd2d.h DESTINATION include/avbd2d)
