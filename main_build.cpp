#include <fstream>
#include <filesystem>
#include <iostream>

#include <utils.h>

using namespace nvonnxparser;
using namespace nvinfer1;


int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cout << "Expected <model>.onnx as argument" << std::endl;
        return 1;
    }

    Logger logger;
    std::filesystem::path modelPath = argv[1];
    std::string outputPath = modelPath.stem().string() + ".engine";

    std::cout << "Building. This may take a few minutes. " << modelPath << std::endl;

    std::unique_ptr<IBuilder> builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
    std::unique_ptr<INetworkDefinition> network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(1U));
    std::unique_ptr<IParser> parser = std::unique_ptr<IParser>(createParser(*network, logger));

    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();

    if(size == -1)
    {
        throw std::runtime_error("Error reading ONNX file");
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);

    if(!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Unable to read ONNX file");
    }

    bool parseSuccess = parser->parse(buffer.data(), buffer.size());
    if(!parseSuccess)
    {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }

        throw std::runtime_error("Error parsing onnx file.");
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
    {
        throw std::runtime_error("Error serializing model");
    }
    
    std::ofstream ofs(outputPath, std::ios::out | std::ios::binary);
    ofs.write((char*)(serializedModel->data()), serializedModel->size());
    ofs.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());

    std::cout << "Finished. Model saved to: " << outputPath << std::endl;

    return 0;
}