//Extracted from Yolov8-TensorRT-CPP
#include <chrono>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <filesystem>

#include "NvOnnxParser.h"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <utils.h>

class TensorrtInference
{
public:
    TensorrtInference(std::string enginePath)
    {
        load_tensorrt_model(enginePath);
    }
    std::vector<Object> detect(const cv::Mat& imageBGR);

private:
    void load_tensorrt_model(std::string enginePath);
    cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batchInput, bool normalize);
    cv::cuda::GpuMat preprocessImage(const cv::Mat& imageBGR);
    std::vector<float> runInference(std::vector<cv::cuda::GpuMat>& input);
    std::vector<Object> parseModelOutput(std::vector<float>& modelOutput, float imageWidth, float imageHeight);

private:
    std::unique_ptr<nvinfer1::IRuntime> _runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> _engine;
    std::unique_ptr<nvinfer1::IExecutionContext> _context;
    Logger _logger;
    IOBinding inputBinding;
    IOBinding outputBinding;
    float _imageWidth = -1;
    float _imageHeight = -1;
    float _ratio = -1;
    float _widthRatio = -1;
    float _heightRatio = -1;
};