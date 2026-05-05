cmake_minimum_required(VERSION 3.16)

project(marsh_muddpile LANGUAGES CXX)

# ------------------------------------------------------------------------------
# build options
# ------------------------------------------------------------------------------

option(marsh_muddpile_use_openmp "Enable OpenMP support if available" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------------------
# check for C++17 support
# ------------------------------------------------------------------------------

include(CheckCXXSourceCompiles)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "Checking for C++17 compiler support")

set(_marsh_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")

if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} /std:c++17")
else()
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++17")
endif()

check_cxx_source_compiles("
    #include <tuple>
    #include <vector>
    int main()
    {
        std::vector<int> v = {1, 2, 3};
        auto [a, b, c] = std::tuple<int, int, int>{1, 2, 3};
        return a + b + c + static_cast<int>(v.size());
    }
" marsh_muddpile_has_cxx17)

set(CMAKE_REQUIRED_FLAGS "${_marsh_saved_required_flags}")

if(NOT marsh_muddpile_has_cxx17)
    message(FATAL_ERROR
        "A compiler with C++17 support is required to build marsh_muddpile.")
else()
    message(STATUS "C++17 compiler support: found")
    message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# ------------------------------------------------------------------------------
# dependencies
# ------------------------------------------------------------------------------

message(STATUS "Checking for Eigen3")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(TARGET Eigen3::Eigen)
    if(DEFINED EIGEN3_VERSION_STRING)
        message(STATUS "Eigen3 found: ${EIGEN3_VERSION_STRING}")
    else()
        message(STATUS "Eigen3 found")
    endif()
else()
    message(FATAL_ERROR "Eigen3 was found by CMake, but target Eigen3::Eigen is missing.")
endif()

message(STATUS "Checking for yaml-cpp")
find_package(yaml-cpp REQUIRED)

if(TARGET yaml-cpp)
    if(DEFINED yaml-cpp_VERSION)
        message(STATUS "yaml-cpp found: ${yaml-cpp_VERSION}")
    else()
        message(STATUS "yaml-cpp found")
    endif()
else()
    message(FATAL_ERROR "yaml-cpp was found by CMake, but target yaml-cpp is missing.")
endif()

message(STATUS "Checking for pkg-config")
find_package(PkgConfig REQUIRED)

message(STATUS "Checking for NetCDF C")
pkg_check_modules(NETCDF_C REQUIRED IMPORTED_TARGET netcdf)

if(DEFINED NETCDF_C_VERSION)
    message(STATUS "NetCDF C found: ${NETCDF_C_VERSION}")
else()
    message(STATUS "NetCDF C found")
endif()

message(STATUS "Checking for NetCDF C++4")
pkg_check_modules(NETCDF_CXX REQUIRED IMPORTED_TARGET netcdf-cxx4)

if(DEFINED NETCDF_CXX_VERSION)
    message(STATUS "NetCDF C++4 found: ${NETCDF_CXX_VERSION}")
else()
    message(STATUS "NetCDF C++4 found")
endif()

# ------------------------------------------------------------------------------
# optional OpenMP
# ------------------------------------------------------------------------------

if(marsh_muddpile_use_openmp)
    message(STATUS "Checking for OpenMP")
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
        message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
    else()
        message(WARNING "OpenMP requested but not found. Building without OpenMP.")
    endif()
else()
    message(STATUS "OpenMP support disabled")
endif()

# ------------------------------------------------------------------------------
# library target
# ------------------------------------------------------------------------------

add_library(marsh_model_lib
    src/core/material_catalog.cpp
    src/core/column_state.cpp
    src/core/forcing_series.cpp
    src/core/parameter_set.cpp

    src/engine/simulator.cpp
    src/engine/process_factory.cpp
    src/engine/layer_merger.cpp
    src/engine/surface_property_summarizer.cpp

    src/io/config_io.cpp
    src/io/result_io.cpp

    src/processes/zero_deposition_model.cpp
    src/processes/null_biomass_model.cpp
    src/processes/null_root_allocation_model.cpp
    src/processes/identity_decay_model.cpp
    src/processes/identity_compaction_model.cpp

    src/processes/simple_canopy_et_model.cpp
    src/processes/marsh_gpp_biomass_model.cpp
    src/processes/hydro_salinity_decomposition_modifier_model.cpp

    src/processes/two_stage_compaction_model.cpp
    src/processes/first_order_decay_model.cpp
    src/processes/marsh_decay_model.cpp
    src/processes/exponential_root_allocation_model.cpp
    src/processes/seasonal_biomass_model.cpp
    src/processes/tke_deposition_model.cpp
    src/processes/edge_distance_deposition_model.cpp

    src/processes/distance_flushing_salinity_model.cpp
    src/processes/composite_water_level_model.cpp

)

target_include_directories(marsh_model_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(marsh_model_lib
    PUBLIC
        Eigen3::Eigen
        yaml-cpp
        PkgConfig::NETCDF_C
        PkgConfig::NETCDF_CXX
)

target_compile_features(marsh_model_lib PUBLIC cxx_std_17)

if(marsh_muddpile_use_openmp AND OpenMP_CXX_FOUND)
    target_link_libraries(marsh_model_lib
        PUBLIC
            OpenMP::OpenMP_CXX
    )

    target_compile_definitions(marsh_model_lib
        PUBLIC
            marsh_muddpile_use_openmp=1
    )
endif()

# ------------------------------------------------------------------------------
# warnings
# ------------------------------------------------------------------------------

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(marsh_model_lib
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
elseif(MSVC)
    target_compile_options(marsh_model_lib
        PRIVATE
            /W4
    )
endif()

# ------------------------------------------------------------------------------
# executable
# ------------------------------------------------------------------------------

add_executable(marsh_cli
    apps/marsh_cli.cpp
)

target_link_libraries(marsh_cli
    PRIVATE
        marsh_model_lib
)

target_compile_features(marsh_cli PRIVATE cxx_std_17)

# ------------------------------------------------------------------------------
# install rules
# ------------------------------------------------------------------------------

install(TARGETS marsh_model_lib marsh_cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)
