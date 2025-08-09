// src/main.cpp

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>
#include "json.hpp"

#include "perception/OnnxRuntimeEngine.h"

using json = nlohmann::json;

int main() {
    OnnxRuntimeEngine engine("models/best.onnx");

    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind("tcp://*:5555");
    std::cout << "C++ ZMQ Server listening on tcp://*:5555" << std::endl;

    while (true) {
        zmq::message_t metadata_msg;
        socket.recv(metadata_msg, zmq::recv_flags::none);
        json metadata = json::parse(static_cast<char*>(metadata_msg.data()), static_cast<char*>(metadata_msg.data()) + metadata_msg.size());
        
        int height = metadata["height"];
        int width = metadata["width"];

        zmq::message_t image_data_msg;
        socket.recv(image_data_msg, zmq::recv_flags::none);

        cv::Mat bgra_image(height, width, CV_8UC4, image_data_msg.data());

        cv::Mat bgr_image;
        cv::cvtColor(bgra_image, bgr_image, cv::COLOR_BGRA2BGR);
        
        std::cout << "Received frame " << metadata["frame"] << ". Processing..." << std::endl;

        engine.process_frame(bgr_image);

        std::string reply_str = "OK";
        socket.send(zmq::buffer(reply_str), zmq::send_flags::none);
    }

    return 0;
}
