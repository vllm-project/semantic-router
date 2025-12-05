#include "../../include/classifiers/lora_adapter.h"
#include "../../include/core/model_manager.h"
#include <iostream>
#include <cstring>

namespace openvino_sr {
namespace classifiers {

bool LoRAAdapter::load(
    const std::string& adapter_model_path,
    const LoRAConfig& config,
    const std::string& device
) {
    try {
        config_ = config;
        
        auto& manager = core::ModelManager::getInstance();
        manager.ensureCoreInitialized();
        
        // Configure for inference
        ov::AnyMap ov_config;
        ov_config[ov::inference_num_threads.name()] = 2;
        ov_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
        
        // Load and compile LoRA adapter model
        compiled_model_ = manager.loadModel(adapter_model_path, device, ov_config);
        if (!compiled_model_) {
            std::cerr << "Failed to load LoRA adapter model: " << adapter_model_path << std::endl;
            return false;
        }
        
        // Create infer request
        infer_request_ = compiled_model_->create_infer_request();
        
        std::cout << "✓ LoRA adapter loaded: " << adapter_model_path 
                  << " (rank=" << config_.rank << ", alpha=" << config_.alpha << ")" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load LoRA adapter: " << e.what() << std::endl;
        return false;
    }
}

ov::Tensor LoRAAdapter::forward(const ov::Tensor& input) {
    if (!isLoaded()) {
        throw std::runtime_error("LoRA adapter not loaded");
    }
    
    try {
        // Set input tensor
        infer_request_.set_input_tensor(input);
        
        // Run inference (LoRA forward pass: B(A(x)))
        infer_request_.infer();
        
        // Get output tensor
        auto output = infer_request_.get_output_tensor();
        
        // Apply scaling factor: alpha / rank
        // Note: In a real implementation, scaling should be applied within the model
        // or as a post-processing step. For now, we assume the model includes scaling.
        
        return output;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("LoRA forward pass failed: ") + e.what());
    }
}

} // namespace classifiers
} // namespace openvino_sr

