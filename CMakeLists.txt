cmake_minimum_required(VERSION 3.25)
project(cppBpe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug, RElease, RelWithDebInfo, MinSizeRel)
endif()

option(CPPBPE_USE_BUNDLED_PCRE2 "Build PCRE2 from source with JIT enabled" ON)

if(CPPBPE_USE_BUNDLED_PCRE2)
    include(FetchContent)
    set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
    set(PCRE2_SUPPORT_JIT ON CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2POSIX OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        pcre2
        URL https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.47/pcre2-10.47.tar.gz
    )
    FetchContent_MakeAvailable(pcre2)

    set(PCRE2_LIBRARIES pcre2-8)
    set(PCRE2_INCLUDE_DIRS "")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PCRE2 REQUIRED libpcre2-8)
endif()

find_package(TBB REQUIRED)
cmake_policy(SET CMP0148 NEW)
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

add_library(cppBpe_core STATIC
        src/tokenizer.cpp
        include/cppBpe/tokenizer.h
        include/cppBpe/python_bindings.h)

set_target_properties(cppBpe_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(cppBpe_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PCRE2_INCLUDE_DIRS}
)

target_link_libraries(cppBpe_core
    PUBLIC
        TBB::tbb
    PRIVATE
        ${PCRE2_LIBRARIES}
)

target_compile_options(cppBpe_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g -fsanitize=address,undefined>
    >
)

target_link_options(cppBpe_core PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Debug>>:
        -fsanitize=address,undefined
    >
)

pybind11_add_module(cppBpe MODULE src/python_bindings.cpp)
target_link_libraries(cppBpe PRIVATE cppBpe_core)
target_include_directories(cppBpe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(cppBpe PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

option(CPPBPE_BUILD_TESTS "Build c++ unit tests" ON)

if(CPPBPE_BUILD_TESTS)
    enable_testing()

    add_executable(cppBpe_tests tests/cpp/test_tokenizer.cpp)
    target_link_libraries(cppBpe_tests PRIVATE cppBpe_core)
    target_include_directories(cppBpe_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(cppBpe_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:
            -Wall -Wextra
            $<$<CONFIG:Release>:-O2>
            $<$<CONFIG:Debug>:-O0 -g>
        >
    )

    add_test(NAME cppBpe_tests COMMAND cppBpe_tests)
endif()

install(TARGETS cppBpe_core
    EXPORT cppBpeTargets
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/cppBpe DESTINATION include)
install(TARGETS cppBpe LIBRARY DESTINATION ${Python3_SITEARCH})