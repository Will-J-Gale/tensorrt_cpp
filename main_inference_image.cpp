//Extracted from Yolov8-TensorRT-CPP
#include <iostream>
#include <fstream>
#include <filesystem>

#include <tensorrt_inference.h>

// Runs object detection on video stream then displays annotated results.
int main(int argc, char* argv[]) 
{
    if(argc != 3)
    {
        std::cout << "Required args: engine_path image_path" << std::endl;
        return 1;
    }

    std::filesystem::path engine_path = argv[1];
    std::filesystem::path image_path = argv[2];

    if(!std::filesystem::exists(engine_path))
    {
        std::cout << "engine files does does note exist";
    }

    if(!std::filesystem::exists(image_path))
    {
        std::cout << "Image does note exist";
    }

    cv::Mat image = cv::imread(image_path);
    TensorrtInference inference = TensorrtInference(engine_path);

    for(int i = 0; i < 10; i++)
    {
        auto inference_start = Time();
        auto objects = inference.detect(image);
        auto inference_dt = Time() - inference_start;
        std::cout << "Inference time: " << inference_dt << " FPS: " << 1/inference_dt << std::endl;
    }

    auto objects = inference.detect(image);
    std::cout << "Detected: " << objects.size() << std::endl;

    for(auto& object : objects)
    {
        cv::rectangle(image, object.rect, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Image", image);
    cv::waitKey(0);
    
	return 0;
}