cmake_minimum_required(VERSION 3.10)
project("android_ncnn_yolo11")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-2.4.13.7-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240410-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)



add_library(${CMAKE_PROJECT_NAME} SHARED
        src/yolo11.cpp
        src/yolo11.h
        native-lib.cpp)



target_link_libraries(${CMAKE_PROJECT_NAME}
        ncnn
        ${OpenCV_LIBS}
        # List libraries link to the target library
        android
        log)