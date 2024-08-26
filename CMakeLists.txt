cmake_minimum_required(VERSION 3.25)
project(tensorrt_test)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

file(
    GLOB_RECURSE 
    SRC_FILES    
    src/*.cpp
)

add_executable(tensorrt_build main_build.cpp ${SRC_FILES})
target_link_libraries(tensorrt_build nvinfer nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(tensorrt_inference_image main_inference_image.cpp ${SRC_FILES})
target_link_libraries(tensorrt_inference_image nvinfer nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(tensorrt_inference_video main_inference_video.cpp ${SRC_FILES})
target_link_libraries(tensorrt_inference_video nvinfer nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})
