#include "perception/OnnxRuntimeEngine.h"
#include <iostream>

OnnxRuntimeEngine::OnnxRuntimeEngine(const std::string& model_path) {
    try {
        this->net = cv::dnn::readNet(model_path);
        std::cout << "ONNX model loaded successfully from: "<<model_path << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        exit(-1);
    }
}

void OnnxRuntimeEngine::process_frame(cv::Mat& image) {
    cv::Mat blob;
    std::vector<cv::Mat> outputs;

    cv::dnn::blobFromImage(image, blob, 1./255.,
        cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    this->net.setInput(blob);
    this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());

    // Calculate scaling factors
    float x_factor = image.cols / INPUT_WIDTH;
    float y_factor = image.rows / INPUT_HEIGHT;

    // Storage for detections
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Reshape and transpose the output for easier parsing
    const int dimensions = CLASS_NAMES.size() + 4;
    cv::Mat mat(dimensions, outputs[0].size[2], CV_32F, (float*)outputs[0].data);
    cv::Mat mat_t = mat.t();

    // Iterate over each potential detection
    float* data = (float*)mat_t.data;
    for (int i = 0; i < mat_t.rows; ++i) {
        float* classes_scores = data + 4;
        cv::Mat scores(1, CLASS_NAMES.size(), CV_32FC1, classes_scores);
        cv::Point class_id_point;
        double max_score;
        cv::minMaxLoc(scores, 0, &max_score, 0, &class_id_point);

        if (max_score > SCORE_THRESHOLD) {
            confidences.push_back(max_score);
            class_ids.push_back(class_id_point.x);
            float cx = data[0], cy = data[1], w = data[2], h = data[3];
            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += dimensions;
    }

    // Apply Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    // Draw the final, filtered bounding boxes
    for (int idx : indices) {
        const cv::Rect& box = boxes[idx];
        int class_id = class_ids[idx];
        const std::string& class_name = CLASS_NAMES[class_id];
        cv::Scalar color = (class_name == "person") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        cv::rectangle(image, box, color, 2);
        std::string label = class_name + ": " + cv::format("%.2f", confidences[idx]);
        cv::putText(image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }

}