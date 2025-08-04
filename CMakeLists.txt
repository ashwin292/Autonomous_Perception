cmake_minimum_required(VERSION 3.10)
project(AutonomousPerception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
# Find ZMQ installed by Homebrew
find_package(ZeroMQ REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${ZeroMQ_INCLUDE_DIRS})

add_executable(perception_app src/main.cpp)

# Link against both OpenCV and ZMQ libraries
target_link_libraries(perception_app ${OpenCV_LIBS} ${ZeroMQ_LIBRARIES})
