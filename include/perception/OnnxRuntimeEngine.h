#pragma once

#include "IInferenceEngine.h"
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

class OnnxRuntimeEngine : public IInferenceEngine {
public:
    OnnxRuntimeEngine(const std::string& model_path);

    void process_frame(cv::Mat& image) override;

private:
    const float INPUT_WIDTH = 1280.0;
    const float INPUT_HEIGHT = 1280.0;
    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.45;
    const std::vector<std::string> CLASS_NAMES = {
        "person", "rider", "car", "truck", "bus", "train",
        "motor", "bike", "traffic light", "traffic sign"
    };

    cv::dnn::Net net;
};