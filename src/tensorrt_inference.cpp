#include <tensorrt_inference.h>

void TensorrtInference::load_tensorrt_model(std::string enginePath)
{
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

cv::cuda::GpuMat TensorrtInference::blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batchInput, bool normalize) 
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

    if (normalize) 
    {
        gpuDst.convertTo(gpuDst, CV_32FC3, 1.f / 255.f);
    } 
    else 
    {
        gpuDst.convertTo(gpuDst, CV_32FC3);
    }

    return gpuDst;
}

cv::cuda::GpuMat TensorrtInference::preprocessImage(const cv::Mat& imageBGR)
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

std::vector<float> TensorrtInference::runInference(std::vector<cv::cuda::GpuMat>& input)
{
    //Run inference
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

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

std::vector<Object> TensorrtInference::parseModelOutput(std::vector<float>& modelOutput, float imageWidth, float imageHeight)
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

std::vector<Object> TensorrtInference::detect(const cv::Mat& imageBGR)
{
    cv::cuda::GpuMat preprocessedImage = preprocessImage(imageBGR);
    //Convert to batched input (HWC -> BHWC)
    std::vector<cv::cuda::GpuMat> input{std::move(preprocessedImage)};
    std::vector<float> modelOutput = runInference(input);
    return parseModelOutput(modelOutput, (float)imageBGR.cols, (float)imageBGR.rows);
}
