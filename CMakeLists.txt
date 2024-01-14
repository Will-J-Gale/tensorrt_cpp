cmake_minimum_required(VERSION 3.25)
project(tensorrt_test)

# Set C++ version and optimization level
file(COPY yolov8n.onnx DESTINATION .)
set(CMAKE_CXX_STANDARD 17)

# CMake Options
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Build and link the executables
add_executable(tensorrt_build main_build.cpp )
target_link_libraries(tensorrt_build nvinfer nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(tensorrt_inference_image main_inference_image.cpp)
target_link_libraries(tensorrt_inference_image nvinfer nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(tensorrt_inference_video main_inference_video.cpp)
target_link_libraries(tensorrt_inference_video nvinfer nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})
