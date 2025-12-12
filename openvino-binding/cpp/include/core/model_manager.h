#pragma once

#include "types.h"
#include <openvino/openvino.hpp>
#include <memory>
#include <string>
#include <mutex>

namespace openvino_sr {
namespace core {

/**
 * @brief ModelManager handles OpenVINO Core initialization and model management
 */
class ModelManager {
public:
    static ModelManager& getInstance();
    
    // Initialize OpenVINO Core if not already initialized
    void ensureCoreInitialized();
    
    // Get the OpenVINO Core instance
    ov::Core& getCore();
    
    // Load a model from file
    std::shared_ptr<ov::CompiledModel> loadModel(
        const std::string& model_path,
        const std::string& device = "CPU",
        const ov::AnyMap& config = {}
    );
    
    // Create InferRequest pool for concurrent execution
    void createInferPool(
        ModelInstance& model,
        size_t pool_size = 16
    );
    
    // Get an InferRequest from the pool
    InferRequestSlot* getInferRequest(ModelInstance& model);
    
private:
    ModelManager() = default;
    ~ModelManager() = default;
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    
    std::unique_ptr<ov::Core> core_;
    std::mutex mutex_;
};

} // namespace core
} // namespace openvino_sr

