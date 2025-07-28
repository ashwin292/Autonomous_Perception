#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <zmq.hpp> // Include the ZMQ C++ binding

// Constants remain the same
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.3; // Lowered for better detection of weaker classes
const float NMS_THRESHOLD = 0.45;
const std::vector<std::string> CLASS_NAMES = {
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign"};

int main() {
    // --- ZMQ Setup ---
    // Create a ZMQ context and a REP (Reply) socket
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:5555");
    std::cout << "Perception Module listening on tcp://*:5555" << std::endl;

    // Load your custom-trained model
    cv::dnn::Net net = cv::dnn::readNet("best.onnx");

    while (true) { // The server runs forever
        // 1. Receive Image Data from Client (Python)
        zmq::message_t request;
        socket.recv(request, zmq::recv_flags::none);
        
        // Decode the received bytes into an OpenCV image
        std::vector<uchar> buffer(static_cast<uchar*>(request.data()), static_cast<uchar*>(request.data()) + request.size());
        cv::Mat frame = cv::imdecode(buffer, cv::IMREAD_COLOR);
        
        if (frame.empty()) {
            std::cerr << "Error: Received empty frame." << std::endl;
            // Send a default reply to prevent client from hanging
            socket.send(zmq::buffer("CONTINUE"), zmq::send_flags::none);
            continue;
        }

        // 2. Run the Perception Pipeline (same as before)
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        
        // Post-processing logic (same as Week 3)
        // ... (This includes transposing, looping, and NMS)
        float x_factor = frame.cols / INPUT_WIDTH;
        float y_factor = frame.rows / INPUT_HEIGHT;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        const int dimensions = CLASS_NAMES.size() + 4;
        cv::Mat mat(dimensions, outputs[0].size[2], CV_32F, (float*)outputs[0].data);
        cv::Mat mat_t = mat.t();
        float* data = (float*)mat_t.data;
        const int rows = mat_t.rows;

        for (int i = 0; i < rows; ++i) {
            // ... (rest of the processing loop is identical to Week 3)
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
        
        // 3. Make a Decision and Send Reply
        std::string command = "CONTINUE"; // Default command
        for (int idx : indices) {
            int class_id = class_ids[idx];
            if (CLASS_NAMES[class_id] == "pedestrian" || CLASS_NAMES[class_id] == "rider") {
                command = "BRAKE";
                break; // One critical object is enough to brake
            }
        }
        
        // Send the command string back to the Python client
        socket.send(zmq::buffer(command), zmq::send_flags::none);

        // (Optional) Visualize the output
        // ... (drawing loop from Week 3 can be added here) ...
        // cv::imshow("Perception Module", frame);
        // if (cv::waitKey(1) == 27) break;
    }
    return 0;
}
