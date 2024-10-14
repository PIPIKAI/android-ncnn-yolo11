//
// Created by PIPIKAI on 2024/10/12.
//

#include "yolo11.h"


static float softmax(
        const float *src,
        float *dst,
        int length
) {
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++) {
        float score = src[c];
        if (score > alpha) {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i) {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}

static float clamp(
        float val,
        float min = 0.f,
        float max = 1280.f
) {
    return val > min ? (val < max ? val : max) : min;
}

static void qsort_descent_inplace(std::vector<DetectRes> &detect_res, int left, int right) {
    int i = left;
    int j = right;
    float p = detect_res[(left + right) / 2].prob;

    while (i <= j) {
        while (detect_res[i].prob > p)
            i++;

        while (detect_res[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(detect_res[i], detect_res[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(detect_res, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(detect_res, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<DetectRes> &detect_res) {
    if (detect_res.empty())
        return;

    qsort_descent_inplace(detect_res, 0, detect_res.size() - 1);
}

static void generate_proposals(
        int stride,
        const ncnn::Mat &feat_blob,
        const float prob_threshold,
        std::vector<DetectRes> &objects
) {
    const int reg_max = REG_MAX_1;
    float dst[16];
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    const int num_class = num_w - 4 * reg_max;
    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {

            const float *matat = feat_blob.channel(i).row(j);

            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int c = 0; c < num_class; c++) {
                float score = matat[4 * reg_max + c];
                if (score > class_score) {
                    class_index = c;
                    class_score = score;
                }
            }

            if (class_score >= prob_threshold) {

                float x0 = j + 0.5f - softmax(matat, dst, 16);
                float y0 = i + 0.5f - softmax(matat + 16, dst, 16);
                float x1 = j + 0.5f + softmax(matat + 2 * 16, dst, 16);
                float y1 = i + 0.5f + softmax(matat + 3 * 16, dst, 16);

                x0 *= stride;
                y0 *= stride;
                x1 *= stride;
                y1 *= stride;

                DetectRes obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = class_score;
                objects.push_back(obj);
            }
        }
    }

}

static float intersection_area(const DetectRes &a, const DetectRes &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void non_max_suppression(
        std::vector<DetectRes> &proposals,
        std::vector<DetectRes> &results,
        int orin_h,
        int orin_w,
        float dh = 0,
        float dw = 0,
        float ratio_h = 1.0f,
        float ratio_w = 1.0f,
        float conf_thres = 0.25f,
        float iou_thres = 0.65f
) {
    results.clear();
    std::vector<int> picked;
    const int n = proposals.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = proposals[i].rect.width * proposals[i].rect.height;
    }
    for (int i = 0; i < n; i++) {
        const DetectRes &a = proposals[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const DetectRes &b = proposals[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float IoU = inter_area / union_area;
            if (IoU > iou_thres) {
                keep = 0;
                break;
            }
        }
        if (keep)
            picked.push_back(i);
    }
    for (auto i: picked) {
        DetectRes obj = proposals[i];
        float x0 = obj.rect.x;
        float y0 = obj.rect.y;
        float x1 = obj.rect.x + obj.rect.width;
        float y1 = obj.rect.y + obj.rect.height;
        float &score = obj.prob;
        int &label = obj.label;

        x0 = (x0 - dw) / ratio_w;
        y0 = (y0 - dh) / ratio_h;
        x1 = (x1 - dw) / ratio_w;
        y1 = (y1 - dh) / ratio_h;

        x0 = clamp(x0, 0.f, orin_w);
        y0 = clamp(y0, 0.f, orin_h);
        x1 = clamp(x1, 0.f, orin_w);
        y1 = clamp(y1, 0.f, orin_h);

        DetectRes res;
        res.rect.x = x0;
        res.rect.y = y0;
        res.rect.width = x1 - x0;
        res.rect.height = y1 - y0;
        res.prob = score;
        res.label = label;
        results.push_back(res);
    }
}

Yolo::Yolo() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolo::load(AAssetManager *mgr, const char *model_path, bool use_gpu) {
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(0);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", model_path);
    sprintf(modelpath, "%s.bin", model_path);
#if defined(PLATFORM_ANDROID)
    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);
#elif defined(PLATFORM_WINDOWS)
    int res = yolo.load_param(parampath);
    res += yolo.load_model(modelpath);
#endif

    return 1;
}

std::vector<DetectRes> Yolo::detect(const cv::Mat &img, float prob_threshold, float nms_threshold) {
    cv::Mat bgrMat;
    if (img.channels() == 1) {
        cv::cvtColor(img, bgrMat, cv::COLOR_GRAY2BGR);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, bgrMat, cv::COLOR_RGBA2BGR);
    } else {
        bgrMat = img;
    }
    int width = bgrMat.cols;
    int height = bgrMat.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;

    const int target_w = 640;
    const int target_h = 640;
    if (w > h) {
        scale = (float) target_w / w;
        w = target_w;
        h = h * scale;
    } else {
        scale = (float) target_h / h;
        h = target_h;
        w = w * scale;
    }

    ncnn::Extractor ex = yolo.create_extractor();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgrMat.data, ncnn::Mat::PIXEL_BGR2RGB, width,
                                                 height, w, h);
    // pad to target_size rectangle
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ex.input("in0", in_pad);

    std::vector<DetectRes> proposals;

    ncnn::Mat out;

    //stride 8
    {
        ncnn::Mat out;
        ex.extract("out0", out);

        std::vector<DetectRes> objects8;
        generate_proposals(8, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;

        ex.extract("out1", out);

        std::vector<DetectRes> objects16;
        generate_proposals(16, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;

        ex.extract("out2", out);

        std::vector<DetectRes> objects32;
        generate_proposals(32, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    qsort_descent_inplace(proposals);
    std::vector<DetectRes> objects;
    non_max_suppression(proposals, objects,
                        height, width, hpad / 2, wpad / 2,
                        scale, scale, prob_threshold, nms_threshold);


    return objects;
}