cmake_minimum_required(VERSION 3.13)
project(openvino_semantic_router VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Suppress undefined variable warnings in generated Makefiles
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Find OpenVINO - try multiple approaches
find_package(OpenVINO QUIET COMPONENTS Runtime)

if(NOT OpenVINO_FOUND)
    message(STATUS "OpenVINO not found via find_package, trying Python site-packages...")
    
    # Try to find OpenVINO in Python site-packages
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import openvino; print(openvino.__path__[0])"
            OUTPUT_VARIABLE OPENVINO_PYTHON_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PYTHON_IMPORT_RESULT
        )
        
        if(PYTHON_IMPORT_RESULT EQUAL 0 AND EXISTS "${OPENVINO_PYTHON_PATH}")
            message(STATUS "Found OpenVINO Python installation at: ${OPENVINO_PYTHON_PATH}")
            
            # Set paths for CMake
            set(OpenVINO_DIR "${OPENVINO_PYTHON_PATH}/cmake")
            set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${OPENVINO_PYTHON_PATH}/cmake")
            
            # Try to find OpenVINO again with the Python path
            find_package(OpenVINO QUIET COMPONENTS Runtime PATHS "${OPENVINO_PYTHON_PATH}/cmake" NO_DEFAULT_PATH)
            
            if(OpenVINO_FOUND)
                message(STATUS "Successfully configured OpenVINO from Python site-packages")
            else()
                # Manual configuration fallback
                message(STATUS "Manual OpenVINO configuration from Python site-packages")
                set(OpenVINO_FOUND TRUE)
                set(OPENVINO_INCLUDE_DIRS "${OPENVINO_PYTHON_PATH}/runtime/include")
                set(OPENVINO_LIBRARY_DIRS "${OPENVINO_PYTHON_PATH}/libs")
                
                # Create imported target manually
                add_library(openvino::runtime SHARED IMPORTED)
                set_target_properties(openvino::runtime PROPERTIES
                    IMPORTED_LOCATION "${OPENVINO_LIBRARY_DIRS}/libopenvino.so"
                    INTERFACE_INCLUDE_DIRECTORIES "${OPENVINO_INCLUDE_DIRS}"
                )
            endif()
        endif()
    endif()
endif()

if(NOT OpenVINO_FOUND)
    message(FATAL_ERROR "OpenVINO not found. Please install OpenVINO or set OpenVINO_DIR environment variable.")
endif()

message(STATUS "OpenVINO found and configured successfully")

# Find OpenVINO Tokenizers library
set(OPENVINO_TOKENIZERS_LIB_DIR "")
if(Python3_FOUND AND OPENVINO_PYTHON_PATH)
    # Check if openvino_tokenizers exists in same Python installation
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import openvino_tokenizers; print(openvino_tokenizers.__path__[0])"
        OUTPUT_VARIABLE OPENVINO_TOKENIZERS_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TOKENIZERS_IMPORT_RESULT
    )
    
    if(TOKENIZERS_IMPORT_RESULT EQUAL 0 AND EXISTS "${OPENVINO_TOKENIZERS_PATH}")
        set(OPENVINO_TOKENIZERS_LIB_DIR "${OPENVINO_TOKENIZERS_PATH}/lib")
        message(STATUS "Found OpenVINO Tokenizers: ${OPENVINO_TOKENIZERS_LIB_DIR}")
        
        # Verify library files exist
        if(EXISTS "${OPENVINO_TOKENIZERS_LIB_DIR}/libopenvino_tokenizers.so")
            message(STATUS "  ✓ libopenvino_tokenizers.so found")
        endif()
        if(EXISTS "${OPENVINO_TOKENIZERS_LIB_DIR}/libcore_tokenizers.so")
            message(STATUS "  ✓ libcore_tokenizers.so found")
        endif()
    endif()
endif()

if(NOT OPENVINO_TOKENIZERS_LIB_DIR OR NOT EXISTS "${OPENVINO_TOKENIZERS_LIB_DIR}")
    message(WARNING "OpenVINO Tokenizers library not found. Install with: pip install openvino-tokenizers")
endif()

# Library sources (modular architecture)
set(SOURCES
    # Utils module
    cpp/src/utils/math_utils.cpp
    cpp/src/utils/preprocessing.cpp
    
    # Core module
    cpp/src/core/model_manager.cpp
    cpp/src/core/tokenizer.cpp
    
    # Classifiers module
    cpp/src/classifiers/text_classifier.cpp
    cpp/src/classifiers/token_classifier.cpp
    
    # Embeddings module
    cpp/src/embeddings/embedding_generator.cpp
    
    # FFI layer (C API for Go CGO)
    cpp/src/ffi/openvino_semantic_router_ffi.cpp
)

set(HEADERS
    # C API header (public interface)
    cpp/include/openvino_semantic_router.h
    
    # Core headers
    cpp/include/core/types.h
    cpp/include/core/model_manager.h
    cpp/include/core/tokenizer.h
    
    # Classifier headers
    cpp/include/classifiers/text_classifier.h
    cpp/include/classifiers/token_classifier.h
    
    # Embedding headers
    cpp/include/embeddings/embedding_generator.h
    
    # Utility headers
    cpp/include/utils/math_utils.h
    cpp/include/utils/preprocessing.h
)

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
)

# Link OpenVINO and OpenVINO Tokenizers
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        openvino::runtime
)

# Link OpenVINO Tokenizers if available
if(OPENVINO_TOKENIZERS_LIB_DIR AND EXISTS "${OPENVINO_TOKENIZERS_LIB_DIR}/libopenvino_tokenizers.so")
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            ${OPENVINO_TOKENIZERS_LIB_DIR}/libopenvino_tokenizers.so
    )
    
    # Add rpath so the library can be found at runtime
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "${OPENVINO_TOKENIZERS_LIB_DIR}"
        INSTALL_RPATH "${OPENVINO_TOKENIZERS_LIB_DIR}"
    )
    
    message(STATUS "Linked OpenVINO Tokenizers library")
endif()

# Compiler options
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Set library output properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER "${HEADERS}"
)

# Installation rules
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Create package configuration files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "OpenVINO Semantic Router Configuration")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")

