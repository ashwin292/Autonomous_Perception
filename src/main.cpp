/**
 * @file main.cpp
 * @brief Object detection application using OpenCV's DNN module.
 *
 * This program reads a video file, runs a pre-trained ONNX model (like YOLO)
 * on each frame to detect objects, and saves the resulting video with
 * bounding boxes drawn on it.
 *
 */
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// --- Model & Detection Configuration ---

// The input dimensions for the neural network. Must match the model's trained size.
const float INPUT_WIDTH = 1280.0;
const float INPUT_HEIGHT = 1280.0;

// Thresholds for filtering detections after inference.
const float SCORE_THRESHOLD = 0.5;      // Minimum confidence score to consider a detection.
const float NMS_THRESHOLD = 0.45;       // Threshold for non-maximum suppression.

// The class names that correspond to the model's output IDs, in the correct order.
// IMPORTANT: This must exactly match the classes the ONNX model was trained on.
const std::vector<std::string> CLASS_NAMES = {
    "person", "rider", "car", "truck", "bus", "train",
    "motor", "bike", "traffic light", "traffic sign"};

int main() {
    // Attempt to open the input video file.
    cv::VideoCapture cap("data/driving.mov");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file in data/ folder." << std::endl;
        return -1;
    }

    // Set up the VideoWriter to save the processed video.
    // Same properties (width, height, fps) used as the input video.
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter video_writer("demo/output_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    // Load the pre-trained object detection model from the ONNX file.
    cv::dnn::Net net = cv::dnn::readNet("models/best.onnx");

    cv::Mat frame, blob;
    std::vector<cv::Mat> outputs;
    int frame_count = 0;

    // Process the video frame by frame.
    while (cap.read(frame)) {
        frame_count++;
        std::cout << "Processing frame " << frame_count << "..." << std::endl;

        // Convert the frame to a "blob" which is the input format for the network.
        // This includes resizing and normalizing pixel values to the [0, 1] range.
        cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        net.setInput(blob);

        // Perform a forward pass through the network to get the raw detections.
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Calculate scaling factors to map detection coordinates from the model's input size
        // back to the original frame's dimensions.
        float x_factor = frame.cols / INPUT_WIDTH;
        float y_factor = frame.rows / INPUT_HEIGHT;

        // Storage for all valid detections found in this frame.
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // The raw output from YOLO-style models is often in a [channels, num_detections] format.
        // We reshape and transpose it to [num_detections, channels] to make it easier to parse.
        // Each row will now represent one potential detection.
        const int dimensions = CLASS_NAMES.size() + 4; // 4 for bbox (cx, cy, w, h) + num_classes
        cv::Mat mat(dimensions, outputs[0].size[2], CV_32F, (float*)outputs[0].data);
        cv::Mat mat_t = mat.t(); // Transpose to get detections as rows

        // Iterate over each potential detection to filter and process it.
        float* data = (float*)mat_t.data;
        for (int i = 0; i < mat_t.rows; ++i) {
            // The first 4 values are bounding box, the rest are class scores.
            float* classes_scores = data + 4;
            cv::Mat scores(1, CLASS_NAMES.size(), CV_32FC1, classes_scores);

            // Find the class with the highest confidence score for this detection.
            cv::Point class_id_point;
            double max_score;
            cv::minMaxLoc(scores, 0, &max_score, 0, &class_id_point);

            // Keep the detection only if its confidence is above the threshold.
            if (max_score > SCORE_THRESHOLD) {
                confidences.push_back(max_score);
                class_ids.push_back(class_id_point.x);

                // Extract and scale the bounding box coordinates.
                float cx = data[0], cy = data[1], w = data[2], h = data[3];
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
            // Move the pointer to the next detection in the buffer.
            data += dimensions;
        }

        // Apply Non-Maximum Suppression (NMS) to remove redundant, overlapping boxes.
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        // Draw the final, filtered bounding boxes and labels on the frame.
        for (int idx : indices) {
            const cv::Rect& box = boxes[idx];
            int class_id = class_ids[idx];
            const std::string& class_name = CLASS_NAMES[class_id];
            
            // Use red for "person" and green for all other objects.
            cv::Scalar color = (class_name == "person") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

            cv::rectangle(frame, box, color, 2);
            std::string label = class_name + ": " + cv::format("%.2f", confidences[idx]);
            cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }

        // Write the processed frame to the output video file.
        video_writer.write(frame);
    }

    // Clean up and release all resources.
    std::cout << "Processing complete. Saving video file to demo/ folder." << std::endl;
    cap.release();
    video_writer.release();

    return 0;
}
