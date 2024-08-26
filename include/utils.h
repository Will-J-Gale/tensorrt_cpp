#pragma once
#include <chrono>
#include <string>

#include "NvOnnxParser.h"
#include <opencv4/opencv2/opencv.hpp>

inline double Time()
{
    auto currentTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration<double>(currentTime.time_since_epoch());

    return duration.count();
}

struct Object
{
    int label = -1;
    float probabability{};
    cv::Rect_<float> rect;
};



inline void checkCudaErrorCode(cudaError_t code) 
{
    if (code != 0) 
    {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suepress info-level messages
        if (severity <= Severity::kWARNING)
        {
            std::cout << "WARNING: " << msg << std::endl;
        }
    }
};

struct IOBinding
{
    const char* name = nullptr;
    void* deviceBuffer = nullptr;
    nvinfer1::Dims shape;

    inline size_t getSize()
    {
        if(_size == 0)
        {
            _size = 1;
            for(int i = 0; i < shape.nbDims; i++)
            {
                _size *= shape.d[i];
            }

            _size *= sizeof(float);

        }

        return _size;
    }

private:
    size_t _size = 0;
};