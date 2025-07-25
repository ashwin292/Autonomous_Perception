# Required CMake version and project name
cmake_minimum_required(VERSION 3.10)
project(AutonomousPerception)

# Use the C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the OpenCV package installed by Homebrew
find_package(OpenCV REQUIRED)

# Add OpenCV's header files to our project's include paths
include_directories(${OpenCV_INCLUDE_DIRS})

# Define our executable, named 'perception_app', built from main.cpp
add_executable(perception_app src/main.cpp)

# Link our app against the OpenCV libraries
target_link_libraries(perception_app ${OpenCV_LIBS})
