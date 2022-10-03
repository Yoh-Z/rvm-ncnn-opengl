cmake_minimum_required(VERSION 3.15)

project(rvm_ncnn_gl)

set(OpenCV_DIR "./3rdparty/opencv" CACHE PATH "Directory that contains OpenCVConfig.cmake")

message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

find_package(OpenCV QUIET COMPONENTS opencv_world)

if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
endif()
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
endif()

if(OpenCV_FOUND)
    message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
    if(${OpenCV_VERSION_MAJOR} GREATER 3)
        set(CMAKE_CXX_STANDARD 11)
    endif()
else()
    message(FATAL_ERROR "Error! OpenCV not found!")
endif()

add_executable(rvm_ncnn_gl
    src/main.cpp
    src/rvm.h
    src/rvm.cpp
)

target_link_libraries(rvm_ncnn_gl ${OpenCV_LIBS})