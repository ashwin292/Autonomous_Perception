cmake_minimum_required(VERSION 3.10)
project(AutonomousPerception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Find Dependencies ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
find_package(ZeroMQ REQUIRED)

# --- NEW: Tell the linker where to find Homebrew libraries ---
link_directories("/opt/homebrew/lib")

# --- 2. Add Your Executable ---
add_executable(perception_app
    src/main.cpp
    src/OnnxRuntimeEngine.cpp
)

# --- 3. Add Include Directories ---
target_include_directories(perception_app PUBLIC
    "/opt/homebrew/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json" 
    ${OpenCV_INCLUDE_DIRS}
)

# --- 4. Link Libraries ---
target_link_libraries(perception_app PRIVATE
    ${OpenCV_LIBS}
    zmq
)
