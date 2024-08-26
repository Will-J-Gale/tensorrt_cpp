//Extracted from Yolov8-TensorRT-CPP
#include <iostream>
#include <string>
#include <filesystem>

#include <tensorrt_inference.h>

inline bool isWebcam(std::string videoSource)
{
    try
    {
        std::stoi(videoSource);
        return true;
    }
    catch(...)
    {
        return false;
    }

    return false;
}

int main(int argc, char* argv[]) 
{
    std::string modelPath = "yolov8n.trt";
    std::string videoSource = "0";

    if(argc != 3)
    {
        std::cout << "Required args: engine_path video_source" << std::endl;
    }

    modelPath = argv[1];
    videoSource = argv[2];

    std::cout << "Loading " << modelPath << std::endl;
    std::cout.precision(4);

    TensorrtInference inference = TensorrtInference(modelPath);

	cv::VideoCapture cap;

    if(isWebcam(videoSource))
    {
        cap.open(std::stoi(videoSource), cv::CAP_V4L2);
    
        //Manually set webcam width/height/fps
        auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        auto fps = cap.get(cv::CAP_PROP_FPS);
        auto fourcc = cap.get(cv::CAP_PROP_FOURCC);
        std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
        std::cout << "Original FPS:" << fps << std::endl;
        std::cout << "Original fourcc:" << fourcc << std::endl;
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FPS, 30);
        resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        fps = cap.get(cv::CAP_PROP_FPS);
        fourcc = cap.get(cv::CAP_PROP_FOURCC);
        std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;
        std::cout << "New fps:" << fps << std::endl;
        std::cout << "New fourcc:" << fourcc << std::endl;
    }
    else
    {
        cap.open(videoSource);
    }

	if (!cap.isOpened())
    {
		throw std::runtime_error("Unable to open video capture with input " + videoSource);
    }

	while (true) 
    {
        auto start = Time();
		cv::Mat img;
        cap.read(img);
        auto capture_dt = Time() - start;

		if (img.empty())
        {
			throw std::runtime_error("Unable to decode image from video stream.");
        }

        auto detect_start = Time();
        auto objects = inference.detect(img);
        auto detect_dt = Time() - detect_start;

        for(auto& object : objects)
        {
            cv::rectangle(img, object.rect, cv::Scalar(0, 255, 0), 2);
        }

		cv::imshow("Object Detection", img);
		if (cv::waitKey(1) >= 0)
        {
			break;
        }
        
        auto frame_dt = Time() - start;

        std::cout << "Frame time: " << frame_dt << " Cap time: " << capture_dt << " Detect time: " << detect_dt << std::endl;
	}

	return 0;
}