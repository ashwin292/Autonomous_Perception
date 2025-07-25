#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// --- Constants for the YOLOv8 Model ---
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// For the placeholder yolov8s.onnx model, the classes are from the COCO dataset.
// We'll update this list when we swap in our custom model later.
const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

int main() {
    // 1. Load the video and the ONNX model
    cv::VideoCapture cap("driving.mov");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    // Use the placeholder model for now
    cv::dnn::Net net = cv::dnn::readNet("yolov8s.onnx");

    cv::Mat frame, blob;
    std::vector<cv::Mat> outputs;

    while (cap.read(frame)) {
        // 2. Pre-process the frame for the model
        cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        net.setInput(blob);

        // 3. Run forward pass to get model output
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // 4. Post-process the raw output
        float x_factor = frame.cols / INPUT_WIDTH;
        float y_factor = frame.rows / INPUT_HEIGHT;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float* data = (float*)outputs[0].data;
        const int dimensions = CLASS_NAMES.size() + 4;
        const int rows = outputs[0].size[2];

        for (int i = 0; i < rows; ++i) {
            float* classes_scores = data + 4;
            cv::Mat scores(1, CLASS_NAMES.size(), CV_32FC1, classes_scores);
            cv::Point class_id_point;
            double max_score;
            cv::minMaxLoc(scores, 0, &max_score, 0, &class_id_point);

            if (max_score > SCORE_THRESHOLD) {
                confidences.push_back(max_score);
                class_ids.push_back(class_id_point.x);

                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
            data += dimensions;
        }

        // Apply Non-Maximum Suppression to remove redundant boxes
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        // 5. Draw the final bounding boxes
        for (int idx : indices) {
            const cv::Rect& box = boxes[idx];
            int class_id = class_ids[idx];

            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            std::string label = CLASS_NAMES[class_id] + ": " + cv::format("%.2f", confidences[idx]);
            cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // 6. Display the result
        cv::imshow("Object Detection", frame);
        if (cv::waitKey(1) == 27) { // Press 'ESC' key to exit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
