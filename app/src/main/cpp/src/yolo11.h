//
// Created by PIPIKAI on 2024/10/12.
//

#ifndef ANDROID_NCNN_YOLO11_YOLO11_H
#define ANDROID_NCNN_YOLO11_YOLO11_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"
#include <net.h>
#include <iostream>

#if defined(WIN32) || defined(_WIN32) || defined(_WIN32_) || defined(WIN64) || defined(_WIN64) || defined(_WIN64_)
#define PLATFORM_WINDOWS 1
#elif defined(ANDROID) || defined(_ANDROID_)
#define PLATFORM_ANDROID 1
#elif defined(__linux__)
#define PLATFORM_LINUX	 1
#elif defined(__APPLE__) || defined(TARGET_OS_IPHONE) || defined(TARGET_IPHONE_SIMULATOR) || defined(TARGET_OS_MAC)
#define PLATFORM_IOS	 1
#else
#define PLATFORM_UNKNOWN 1
#endif


const int MAX_STRIDE = 32;
const int REG_MAX_1 = 16;

struct DetectRes {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class Yolo {
public:
    Yolo();

    int load(AAssetManager *mgr, const char *model_path, bool use_gpu);

    std::vector<DetectRes> detect(const cv::Mat &img, float prob_threshold, float nms_threshold);

private:
    ncnn::Net yolo;
    //RGB
    const float mean_vals[3] = {0.01712475383f, 0.0175070028f, 0.01742919389f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //ANDROID_NCNN_YOLO11_YOLO11_H
