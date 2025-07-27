#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// --- Constants ---
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// --- FINAL: Class names from our bdd100k.yaml file ---
const std::vector<std::string> CLASS_NAMES = {
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign"};

int main() {
    cv::VideoCapture cap("driving.mov");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNet("best.onnx");

    cv::Mat frame, blob;
    std::vector<cv::Mat> outputs;

    while (cap.read(frame)) {
        cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        net.setInput(blob);
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        float x_factor = frame.cols / INPUT_WIDTH;
        float y_factor = frame.rows / INPUT_HEIGHT;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // --- START OF FIX, RANDOM GRID-LIKE BOXES ---
        // The output shape from YOLOv8 is [1, 14, 8400], but sometimes the ONNX model
        // has a transposed shape of [1, 8400, 14]. We need to handle this.
        const int dimensions = CLASS_NAMES.size() + 4;
        cv::Mat mat(dimensions, outputs[0].size[2], CV_32F, (float*)outputs[0].data);
        cv::Mat mat_t = mat.t(); // Transpose the matrix to get shape [8400, 14]

        float* data = (float*)mat_t.data;
        const int rows = mat_t.rows;
        // --- END OF FIX ---


        for (int i = 0; i < rows; ++i) {
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

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        for (int idx : indices) {
            const cv::Rect& box = boxes[idx];
            int class_id = class_ids[idx];
            const std::string& class_name = CLASS_NAMES[class_id];

            cv::Scalar color;
            
            if (class_name == "pedestrian" || class_name == "bicycle" || class_name == "motorcycle" || class_name == "rider") {
                color = cv::Scalar(0, 0, 255);
                std::cout << "CRITICAL: " << class_name << " detected! TRIGGER BRAKE." << std::endl;
            } else if (class_name == "traffic sign" || class_name == "traffic light") {
                color = cv::Scalar(255, 255, 0);
                std::cout << "INFO: " << class_name << " detected." << std::endl;
            } else {
                color = cv::Scalar(0, 255, 0);
            }

            cv::rectangle(frame, box, color, 2);
            std::string label = class_name + ": " + cv::format("%.2f", confidences[idx]);
            cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }

        cv::imshow("Autonomous Perception Module", frame);
        if (cv::waitKey(1) == 27) { // Press 'ESC' key to exit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
