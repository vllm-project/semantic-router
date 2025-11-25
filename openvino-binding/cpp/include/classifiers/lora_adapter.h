#pragma once

#include <openvino/openvino.hpp>
#include <vector>
#include <memory>
#include <string>

namespace openvino_sr {
namespace classifiers {

/**
 * @brief LoRA configuration
 */
struct LoRAConfig {
    size_t rank = 16;                    // LoRA rank
    double alpha = 32.0;                 // LoRA alpha for scaling
    double dropout = 0.1;                // Dropout rate (used during training)
    bool use_bias = false;               // Whether to use bias in LoRA layers
    
    double get_scaling() const {
        return alpha / static_cast<double>(rank);
    }
};

/**
 * @brief LoRA adapter for parameter-efficient fine-tuning
 * 
 * Implements Low-Rank Adaptation by applying:
 * output = input + LoRA_B(LoRA_A(input)) * scaling
 */
class LoRAAdapter {
public:
    LoRAAdapter() = default;
    
    /**
     * @brief Load LoRA adapter from OpenVINO IR model
     * @param adapter_model_path Path to LoRA adapter model (.xml file)
     * @param config LoRA configuration
     * @param device Device name ("CPU", "GPU", etc.)
     * @return true if successful
     */
    bool load(
        const std::string& adapter_model_path,
        const LoRAConfig& config,
        const std::string& device
    );
    
    /**
     * @brief Apply LoRA adapter to input tensor
     * @param input Input tensor (pooled output from BERT/ModernBERT)
     * @return Output tensor after LoRA transformation
     */
    ov::Tensor forward(const ov::Tensor& input);
    
    /**
     * @brief Check if adapter is loaded
     */
    bool isLoaded() const { return compiled_model_ != nullptr; }
    
    /**
     * @brief Get LoRA configuration
     */
    const LoRAConfig& getConfig() const { return config_; }
    
private:
    std::shared_ptr<ov::CompiledModel> compiled_model_;
    LoRAConfig config_;
    ov::InferRequest infer_request_;
};

} // namespace classifiers
} // namespace openvino_sr

