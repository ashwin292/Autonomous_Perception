#pragma once
#include <opencv2/opencv.hpp>

class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    virtual void process_frame(cv::Mat& image) = 0;
};