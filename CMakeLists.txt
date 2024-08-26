cmake_minimum_required(VERSION 3.10)
project(tpuv7_test)

# set(CMAKE_CXX_FLAGS "-O0")
message("cur dir: ${PROJECT_SOURCE_DIR}")

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if (${TARGET_ARCH} STREQUAL "pcie")
    message( "${TARGET_ARCH} mode, starting......")
    # set(lib_DIR /usr/lib/x84_64-linux-gnu)
    # link_directories(${lib_DIR})
    # use ffmpeg
    # set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    # find_package(FFMPEG REQUIRED)
    # include_directories(${FFMPEG_INCLUDE_DIRS})
    # link_directories(${FFMPEG_LIB_DIRS})
    # use opencv
    # set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    # find_package(OpenCV REQUIRED)
    # include_directories(${OpenCV_INCLUDE_DIRS})
    # link_directories(${OpenCV_LIB_DIRS})
    # use libbmrt libbmlib
    set(tpuv7_DIR /opt/tpuv7/tpuv7-runtime-emulator-onednn_0.1.0/data/)
    find_package(tpuv7 REQUIRED)
    include_directories(${TPUV7_INCLUDE_DIRS})
    link_directories(${TPUV7_LIB_DIRS})

    add_executable(tpuv7_test main.cc tpu_utils.h)
    target_link_libraries(tpuv7_test tpuv7_rt tpuv7_modelrt)

elseif (${TARGET_ARCH} STREQUAL "soc")
    
endif ()
