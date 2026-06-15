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

        // Enable model compilation cache to avoid ~22s recompilation on restart.
        // Default: /tmp/ov_model_cache, overridable via OPENVINO_CACHE_DIR env var.
        const char* cache_dir = std::getenv("OPENVINO_CACHE_DIR");
        std::string cache_path = cache_dir ? cache_dir : "/tmp/ov_model_cache";
        core_->set_property(ov::cache_dir(cache_path));
        std::cout << "✓ Model cache enabled: " << cache_path << std::endl;

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
    
    const size_t pool_size = model.infer_pool.size();
    
    // Phase 1: Try to find an idle slot via try_lock (non-blocking scan).
    // Start from round-robin index to maintain fair distribution.
    size_t start = model.pool_index.fetch_add(1, std::memory_order_relaxed) % pool_size;
    for (size_t i = 0; i < pool_size; ++i) {
        size_t idx = (start + i) % pool_size;
        if (model.infer_pool[idx]->mutex.try_lock()) {
            model.infer_pool[idx]->mutex.unlock();  // Caller will lock via lock_guard
            return model.infer_pool[idx].get();
        }
    }
    
    // Phase 2: All slots busy — fall back to blocking on the round-robin slot.
    return model.infer_pool[start].get();
}

InferRequestSlot* ModelManager::acquireInferRequest(ModelInstance& model) {
    if (model.infer_pool.empty()) {
        std::cerr << "InferRequest pool is empty" << std::endl;
        return nullptr;
    }

    const size_t pool_size = model.infer_pool.size();
    const size_t start = model.pool_index.fetch_add(1, std::memory_order_relaxed) % pool_size;

    for (size_t i = 0; i < pool_size; ++i) {
        size_t idx = (start + i) % pool_size;
        auto* slot = model.infer_pool[idx].get();
        if (slot->mutex.try_lock()) {
            return slot;
        }
    }

    auto* slot = model.infer_pool[start].get();
    slot->mutex.lock();
    return slot;
}

} // namespace core
} // namespace openvino_sr

