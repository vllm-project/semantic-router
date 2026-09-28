cmake_minimum_required(VERSION 3.21)
project(ort_ck_flash_attn LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── ROCm / HIP ──────────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
find_package(hip REQUIRED)
find_package(rocblas QUIET)
find_package(Threads REQUIRED)

set(GPU_TARGETS "gfx942" CACHE STRING "GPU architectures to compile for")

# ── ORT C API headers ───────────────────────────────────────────
# Only the raw C API is used; pin its bytes to the runtime's release.
set(ORT_VERSION "1.22.1" CACHE STRING "ORT release version for C API headers")
set(ORT_C_API_SHA256 "d683537d0fdc29e977b5520f7f15d87a0ac212ae6d94fa9be8893a52655621ae"
    CACHE STRING "SHA256 of the selected ORT C API header")
set(ORT_HEADER_DIR "${CMAKE_BINARY_DIR}/ort_headers" CACHE PATH "ORT C API header directory")

if(NOT EXISTS "${ORT_HEADER_DIR}/onnxruntime_c_api.h")
    message(STATUS "Downloading ORT C API header v${ORT_VERSION} ...")
    set(_base "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ORT_VERSION}/include/onnxruntime/core/session")
    file(DOWNLOAD "${_base}/onnxruntime_c_api.h"
         "${ORT_HEADER_DIR}/onnxruntime_c_api.h"
         EXPECTED_HASH "SHA256=${ORT_C_API_SHA256}" TLS_VERIFY ON SHOW_PROGRESS)
endif()
file(SHA256 "${ORT_HEADER_DIR}/onnxruntime_c_api.h" _actual_ort_header_sha)
if(NOT _actual_ort_header_sha STREQUAL ORT_C_API_SHA256)
    message(FATAL_ERROR "ORT C API header differs from its configured SHA256")
endif()

# ── CK-tile headers ─────────────────────────────────────────────
set(CK_INCLUDE_DIR "/opt/rocm/include" CACHE PATH "Path to CK-tile headers")

# ── FMHA dispatch kernel (compiled with hipcc) ──────────────────
set_source_files_properties(src/ck_fmha_dispatch.hip PROPERTIES LANGUAGE HIP)
add_library(ck_fmha_dispatch OBJECT src/ck_fmha_dispatch.hip)
target_include_directories(ck_fmha_dispatch PUBLIC
    ${CK_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(ck_fmha_dispatch PRIVATE
    --offload-arch=${GPU_TARGETS}
    -O3
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1
)
target_link_libraries(ck_fmha_dispatch PRIVATE hip::device)

# ── ORT custom op library (.so) ─────────────────────────────────
add_library(ort_ck_flash_attn SHARED
    src/ort_custom_op.cpp
    $<TARGET_OBJECTS:ck_fmha_dispatch>
)
target_include_directories(ort_ck_flash_attn PRIVATE
    ${ORT_HEADER_DIR}
    ${CK_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(ort_ck_flash_attn PRIVATE hip::host Threads::Threads)
set_target_properties(ort_ck_flash_attn PROPERTIES
    OUTPUT_NAME "ort_ck_flash_attn"
    SOVERSION 1
)

install(TARGETS ort_ck_flash_attn LIBRARY DESTINATION lib)

# ── Validation test ──────────────────────────────────────────────
enable_testing()

set_source_files_properties(tests/test_fa_vs_sdpa.hip PROPERTIES LANGUAGE HIP)
add_executable(test_fa_vs_sdpa tests/test_fa_vs_sdpa.hip)
target_include_directories(test_fa_vs_sdpa PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_fa_vs_sdpa PRIVATE --offload-arch=${GPU_TARGETS} -O3)
target_link_libraries(test_fa_vs_sdpa PRIVATE ort_ck_flash_attn hip::host)

add_test(NAME fa_vs_sdpa COMMAND test_fa_vs_sdpa)

# Runs without a GPU: invokes the real registration/create/compute callbacks
# against a checked API stub and verifies that failures never become success.
add_executable(test_custom_op_contract tests/test_custom_op_contract.cpp)
target_include_directories(test_custom_op_contract PRIVATE ${ORT_HEADER_DIR} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_custom_op_contract PRIVATE hip::host Threads::Threads)
add_test(NAME custom_op_contract COMMAND test_custom_op_contract)
