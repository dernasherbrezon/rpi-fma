cmake_minimum_required(VERSION 3.16.3)
project(rpi_fma C)

set(CMAKE_C_FLAGS "-mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard")
set(CMAKE_C_STANDARD 99)

add_executable(rpi_fma main.c)

find_package(PkgConfig REQUIRED)

pkg_check_modules(PC_OpenCL REQUIRED OpenCL)
include_directories(${PC_OpenCL_INCLUDE_DIRS})
target_link_libraries(rpi_fma ${PC_OpenCL_LINK_LIBRARIES})

