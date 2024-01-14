//Extracted from Yolov8-TensorRT-CPP
#include <chrono>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <filesystem>

#include "NvOnnxParser.h"
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn/dnn.hpp>

inline double Time()
{
    auto currentTime = std::chrono::system_clock::now();
    auto duration = std::chrono::duration<double>(currentTime.time_since_epoch());

    return duration.count();
}

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
        if (severity >= Severity::kWARNING)
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

struct Object
{
    int label = -1;
    float probabability{};
    cv::Rect_<float> rect;
};

class TensorrtInference
{
public:
    TensorrtInference(std::string enginePath)
    {
        load_tensorrt_model(enginePath);
    }

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

    inline void load_tensorrt_model(std::string enginePath)
    {
        std::cout << enginePath << std::endl;
        std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> modelBytes(size);

        if(!file.read(modelBytes.data(), size))
        {
            throw std::runtime_error("Unable to read engine file");
        }


        //Select which GPU to use
        int deviceIndex = 0;
        if(cudaSetDevice(deviceIndex) != 0)
        {
            int numGPUs;
            cudaGetDeviceCount(&numGPUs);
            std::string msg = "Unable to set GPU 0\n";
            msg += "Has " + std::to_string(numGPUs) + "GPU(s)";
            throw std::runtime_error(msg);
        }

        _runtime.reset(nvinfer1::createInferRuntime(_logger));
        _engine.reset(_runtime->deserializeCudaEngine(modelBytes.data(), modelBytes.size()));

        if(! _engine)
            throw std::runtime_error("Failed to create engine");

        _context.reset(_engine->createExecutionContext());

        if(! _context)
            throw std::runtime_error("Failed to create context");
        
        cudaStream_t stream;
        checkCudaErrorCode(cudaStreamCreate(&stream));

        inputBinding.name = _engine->getIOTensorName(0);
        outputBinding.name = _engine->getIOTensorName(1);

        inputBinding.shape = _engine->getTensorShape(inputBinding.name);
        outputBinding.shape = _engine->getTensorShape(outputBinding.name);

        //Allocate device memory
        checkCudaErrorCode(cudaMallocAsync(&inputBinding.deviceBuffer, inputBinding.getSize(), stream));
        checkCudaErrorCode(cudaMallocAsync(&outputBinding.deviceBuffer, outputBinding.getSize(), stream));

        checkCudaErrorCode(cudaStreamSynchronize(stream));
        checkCudaErrorCode(cudaStreamDestroy(stream));
    }

    inline cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batchInput, bool normalize) 
    {
        cv::cuda::GpuMat gpuDst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

        size_t width = batchInput[0].cols * batchInput[0].rows;
        for (size_t imageIndex = 0; imageIndex < batchInput.size(); imageIndex++) 
        {
            std::vector<cv::cuda::GpuMat> inputChannels
            {
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpuDst.ptr()[0 + width * 3 * imageIndex])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpuDst.ptr()[width + width * 3 * imageIndex])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpuDst.ptr()[width * 2 + width * 3 * imageIndex]))
            };
            cv::cuda::split(batchInput[imageIndex], inputChannels);  // HWC -> CHW
        }

        cv::cuda::GpuMat mfloat;
        if (normalize) {
            // [0.f, 1.f]
            gpuDst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
        } else {
            // [0.f, 255.f]
            gpuDst.convertTo(mfloat, CV_32FC3);
        }

        // cv::Mat image;
        // mfloat.download(image);

        // std::cout << "###########################" << std::endl;
        // for(int i = 0; i < 100; i++)
        // {
        //     std::cout << (char)image.data[i] << std::endl;
        // }
        // std::cout << "###########################" << std::endl;
        return mfloat;
    }

    inline cv::cuda::GpuMat preprocessImage(const cv::Mat& imageBGR)
    {
        cv::cuda::GpuMat gpuImage;
        gpuImage.upload(imageBGR);

        //Preprocess image
        cv::cuda::GpuMat imageRGB;
        cv::cuda::cvtColor(gpuImage, imageRGB, cv::COLOR_BGR2RGB);

        cv::Size inputSize(inputBinding.shape.d[2], inputBinding.shape.d[3]);
        cv::cuda::GpuMat resized;
        cv::cuda::resize(imageRGB, resized, inputSize);

        _imageHeight = (float)imageBGR.rows;
        _imageWidth = (float)imageBGR.cols;
        auto inputDims = inputBinding.shape.d;
        _widthRatio =  _imageWidth / inputDims[2];
        _heightRatio =  _imageHeight / inputDims[3];

        return resized;
    }

    inline std::vector<float> runInference(std::vector<cv::cuda::GpuMat>& input)
    {
        //Run inference
        cudaStream_t inferenceCudaStream;
        checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

        //This is used instead of cv:dnn::blobFromImage because that function does not accept GpuMat is an input
        auto modelInputBlob = blobFromGpuMats(input, true);
        auto modelInputPtr = modelInputBlob.ptr<void>();
        auto blobSize = modelInputBlob.cols * modelInputBlob.rows * modelInputBlob.channels() * sizeof(float);

        checkCudaErrorCode(
            cudaMemcpyAsync(
                inputBinding.deviceBuffer,
                modelInputPtr,
                blobSize,
                cudaMemcpyDeviceToDevice, 
                inferenceCudaStream
            )
        );

        if (!_context->allInputDimensionsSpecified()) 
            throw std::runtime_error("Error, not all required dimensions specified.");
        
        bool status = false;
        status = _context->setTensorAddress(inputBinding.name, inputBinding.deviceBuffer);
        status = _context->setTensorAddress(outputBinding.name, outputBinding.deviceBuffer);

        if(!status)
            throw std::runtime_error("Error settings tensor address");
        
        status = _context->enqueueV3(inferenceCudaStream);

        if(!status)
            throw std::runtime_error("Error running inference");

        std::vector<float> output;
        output.clear();
        output.resize(outputBinding.getSize());

        checkCudaErrorCode(
            cudaMemcpyAsync(
                output.data(), 
                static_cast<char*>(outputBinding.deviceBuffer),
                outputBinding.getSize(),
                cudaMemcpyDeviceToHost,
                inferenceCudaStream
            )
        );

        checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
        checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

        return output;
    }

    inline std::vector<Object> parseModelOutput(std::vector<float>& modelOutput, float imageWidth, float imageHeight)
    {
        auto numChannels = outputBinding.shape.d[1];
        auto numAnchors = outputBinding.shape.d[2];
        auto numClasses = 80;

        cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, modelOutput.data());
        output = output.t();

        std::vector<int> labels;
        std::vector<float> scores;
        std::vector<cv::Rect> bboxes;
        std::vector<int> indices;

        for(int i = 0; i < numAnchors; i++)
        {
            auto rowPtr = output.row(i).ptr<float>();
            auto bboxPtr = rowPtr;
            auto scoresPtr = rowPtr + 4;
            auto maxSPtr = std::max_element(scoresPtr, bboxPtr + numClasses);
            float score = *maxSPtr;

            if(score > 0.25f)
            {
                float x = *bboxPtr++;
                float y = *bboxPtr++;
                float w = *bboxPtr++;
                float h = *bboxPtr;

                float x0 = std::clamp((x - 0.5f * w) * _widthRatio, 0.f, imageWidth);
                float y0 = std::clamp((y - 0.5f * h) * _heightRatio, 0.f, imageHeight);
                float x1 = std::clamp((x + 0.5f * w) * _widthRatio, 0.f, imageWidth);
                float y1 = std::clamp((y + 0.5f * h) * _heightRatio, 0.f, imageHeight);

                int label = maxSPtr - scoresPtr;
                cv::Rect_<float> bbox;
                bbox.x = x0;
                bbox.y = y0;
                bbox.width = x1-x0;
                bbox.height = y1-y0;

                bboxes.push_back(bbox);
                labels.push_back(label);
                scores.push_back(score);
            }
        }

        cv::dnn::NMSBoxesBatched(
            bboxes,
            scores,
            labels,
            0.25,
            0.65,
            indices
        );

        std::vector<Object> objects;

        int count = 0;

        for(auto& i : indices)
        {
            if(count >= 100)//TOP_K
                break;

            Object object;
            object.label = labels[i];
            object.rect = bboxes[i];
            object.probabability = scores[i];
            objects.push_back(object);
            count++;
        }

        return objects;
    }

public:
    inline std::vector<Object> detect(const cv::Mat& imageBGR)
    {
        cv::cuda::GpuMat preprocessedImage = preprocessImage(imageBGR);
        //Convert to batched input (HWC -> BHWC)
        std::vector<cv::cuda::GpuMat> input{std::move(preprocessedImage)};
        std::vector<float> modelOutput = runInference(input);
        return parseModelOutput(modelOutput, (float)imageBGR.cols, (float)imageBGR.rows);
    }
};


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