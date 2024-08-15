#include <fstream>
#include <iostream>
#include <memory>
#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace nvonnxparser;
using namespace nvinfer1;

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << "WARNING: " << msg << std::endl;
    }
} logger;

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        std::cout << "Expected <model>.onnx <output_filename> as arguments" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string output_path = argv[2];

    std::cout << "Loading " << modelPath << std::endl;

    std::unique_ptr<IBuilder> builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
    std::unique_ptr<INetworkDefinition> network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(1U));
    std::unique_ptr<IParser> parser = std::unique_ptr<IParser>(createParser(*network, logger));


    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();

    if(size == -1)
        throw std::runtime_error("Error reading ONNX file");

    file.seekg(0, std::ios::beg);

    std::cout << "Creating onnx buffer vector of size: " << size << std::endl;

    std::vector<char> buffer(size);
    if(!file.read(buffer.data(), size))
        throw std::runtime_error("Unable to read ONNX file");

    std::cout <<"Created buffer" << std::endl;

    bool parseSuccess = parser->parse(buffer.data(), buffer.size());
    if(!parseSuccess)
    {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }

        throw std::runtime_error("Parsed error");
    }

    std::unique_ptr<IBuilderConfig> config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());

    IOptimizationProfile* optimizationProfile = builder->createOptimizationProfile();

    for(size_t i = 0; i < network->getNbInputs(); i++)
    {
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();

        int32_t batch_size = inputDims.d[0];
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        optimizationProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(batch_size, inputC, inputH, inputW));
        optimizationProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(batch_size, inputC, inputH, inputW));
        optimizationProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(batch_size, inputC, inputH, inputW));
    }

    config->addOptimizationProfile(optimizationProfile);
    config->setFlag(BuilderFlag::kFP16);

    cudaStream_t profileStream;
    cudaStreamCreate(&profileStream);
    config->setProfileStream(profileStream);

    std::unique_ptr<IHostMemory> serializedModel = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if(!serializedModel)
        throw std::runtime_error("Error serializing model");
    
    std::ofstream ofs(output_path, std::ios::out | std::ios::binary);
    ofs.write((char*)(serializedModel->data()), serializedModel->size());
    ofs.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());

    std::cout << "Finished. Model size: " << serializedModel->size() << std::endl;

    return 0;
}