#include "../../include/core/model_manager.h"
#include <iostream>
#include <fstream>
#include <thread>

namespace openvino_sr {
namespace core {

// Helper to get OpenVINO tokenizers extension library path
static std::string getTokenizersExtension() {
    const char* env_path = std::getenv("OPENVINO_TOKENIZERS_LIB");
    if (!env_path) {
        throw std::runtime_error(
            "OPENVINO_TOKENIZERS_LIB environment variable not set.\n"
            "Please set it to the path of libopenvino_tokenizers.so"
        );
    }
    
    std::ifstream test_file(env_path);
    if (!test_file.good()) {
        throw std::runtime_error(
            std::string("OpenVINO tokenizers library not found at: ") + env_path + "\n"
            "Please verify the path specified in OPENVINO_TOKENIZERS_LIB"
        );
    }
    
    return env_path;
}

ModelManager& ModelManager::getInstance() {
    static ModelManager instance;
    return instance;
}

void ModelManager::ensureCoreInitialized() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!core_) {
        core_ = std::make_unique<ov::Core>();
        
        // Load OpenVINO tokenizers extension (required)
        std::string tokenizers_lib = getTokenizersExtension();
        core_->add_extension(tokenizers_lib);
        std::cout << "✓ Loaded OpenVINO tokenizers extension from: " << tokenizers_lib << std::endl;
    }
}

ov::Core& ModelManager::getCore() {
    ensureCoreInitialized();
    return *core_;
}

std::shared_ptr<ov::CompiledModel> ModelManager::loadModel(
    const std::string& model_path,
    const std::string& device,
    const ov::AnyMap& config
) {
    ensureCoreInitialized();
    
    try {
        // Read model
        auto model = core_->read_model(model_path);
        
        // Compile model
        auto compiled_model = std::make_shared<ov::CompiledModel>(
            core_->compile_model(model, device, config)
        );
        
        return compiled_model;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return nullptr;
    }
}

void ModelManager::createInferPool(ModelInstance& model, size_t pool_size) {
    if (!model.compiled_model) {
        std::cerr << "Cannot create InferRequest pool: model not compiled" << std::endl;
        return;
    }
    
    try {
        model.infer_pool.clear();
        model.infer_pool.reserve(pool_size);
        
        for (size_t i = 0; i < pool_size; ++i) {
            auto slot = std::make_unique<InferRequestSlot>();
            slot->request = model.compiled_model->create_infer_request();
            model.infer_pool.push_back(std::move(slot));
        }
        
        model.pool_index.store(0);
        std::cout << "✓ Created InferRequest pool with " << pool_size << " requests" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to create InferRequest pool: " << e.what() << std::endl;
    }
}

InferRequestSlot* ModelManager::getInferRequest(ModelInstance& model) {
    if (model.infer_pool.empty()) {
        std::cerr << "InferRequest pool is empty" << std::endl;
        return nullptr;
    }
    
    // Round-robin selection (lock-free)
    size_t pool_idx = model.pool_index.fetch_add(1, std::memory_order_relaxed) % model.infer_pool.size();
    return model.infer_pool[pool_idx].get();
}

} // namespace core
} // namespace openvino_sr

