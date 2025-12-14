#include "../../include/core/tokenizer.h"
#include "../../include/core/model_manager.h"
#include <iostream>
#include <fstream>

namespace openvino_sr {
namespace core {

bool OVNativeTokenizer::loadVocab(const std::string& model_dir) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    
    // Look for tokenizer.xml in the specified model directory
    tokenizer_path_ = model_dir + "/tokenizer.xml";
    
    std::ifstream test_file(tokenizer_path_);
    if (!test_file.good()) {
        throw std::runtime_error(
            "Native tokenizer not found at: " + tokenizer_path_ + "\n"
            "Please ensure tokenizer.xml exists in the specified model directory"
        );
    }
    
    try {
        auto& manager = ModelManager::getInstance();
        manager.ensureCoreInitialized();
        
        auto& core = manager.getCore();
        auto model = core.read_model(tokenizer_path_);
        compiled_tokenizer_ = std::make_shared<ov::CompiledModel>(
            core.compile_model(model, "CPU")
        );
        initialized_.store(true, std::memory_order_release);
        std::cout << "✓ Loaded native OpenVINO tokenizer: " << tokenizer_path_ << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load tokenizer: " << e.what() << std::endl;
        return false;
    }
}

bool OVNativeTokenizer::ensureInitialized() {
    // Fast path: already initialized (no lock needed)
    if (initialized_.load(std::memory_order_acquire)) {
        return true;
    }
    
    // Tokenizer must be explicitly initialized via loadVocab()
    std::cerr << "Tokenizer not initialized. Call loadVocab() with a valid model directory first." << std::endl;
    return false;
}

std::vector<int> OVNativeTokenizer::tokenize(const std::string& text, int max_length) {
    if (!initialized_.load(std::memory_order_acquire)) {
        if (!ensureInitialized()) {
            std::cerr << "Tokenizer not initialized" << std::endl;
            return {};
        }
    }
    
    try {
        // Create input tensor (string)
        ov::Tensor input_tensor(ov::element::string, ov::Shape{1});
        input_tensor.data<std::string>()[0] = text;
        
        // Create per-thread InferRequest (thread-safe, no locking needed)
        auto infer_request = compiled_tokenizer_->create_infer_request();
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        
        // Get input_ids output
        auto input_ids_tensor = infer_request.get_tensor("input_ids");
        const int64_t* input_ids_data = input_ids_tensor.data<const int64_t>();
        auto shape = input_ids_tensor.get_shape();
        
        if (shape.size() < 2) {
            std::cerr << "Unexpected tokenizer output shape" << std::endl;
            return {};
        }
        
        size_t sequence_length = shape[1];
        
        // Truncate to max_length if needed
        size_t actual_length = std::min(sequence_length, static_cast<size_t>(max_length));
        
        std::vector<int> tokens;
        tokens.reserve(actual_length);
        for (size_t i = 0; i < actual_length; ++i) {
            tokens.push_back(static_cast<int>(input_ids_data[i]));
        }
        
        return tokens;
        
    } catch (const std::exception& e) {
        std::cerr << "Tokenization error: " << e.what() << std::endl;
        return {};
    }
}

TokenizationResult OVNativeTokenizer::tokenizeFull(const std::string& text, int max_length) {
    TokenizationResult result;
    
    if (!initialized_.load(std::memory_order_acquire)) {
        if (!ensureInitialized()) {
            std::cerr << "Tokenizer not initialized" << std::endl;
            return result;
        }
    }
    
    try {
        // Create input tensor (string)
        ov::Tensor input_tensor(ov::element::string, ov::Shape{1});
        input_tensor.data<std::string>()[0] = text;
        
        // Create per-thread InferRequest (thread-safe, no locking needed)
        auto infer_request = compiled_tokenizer_->create_infer_request();
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        
        // Get outputs
        auto input_ids_tensor = infer_request.get_tensor("input_ids");
        auto attention_mask_tensor = infer_request.get_tensor("attention_mask");
        
        const int64_t* input_ids_data = input_ids_tensor.data<const int64_t>();
        const int64_t* attention_mask_data = attention_mask_tensor.data<const int64_t>();
        
        auto shape = input_ids_tensor.get_shape();
        size_t sequence_length = shape[1];
        size_t actual_length = std::min(sequence_length, static_cast<size_t>(max_length));
        
        // Copy input_ids
        result.input_ids.assign(input_ids_data, input_ids_data + actual_length);
        result.attention_mask.assign(attention_mask_data, attention_mask_data + actual_length);
        
        // Try to get token_type_ids (might not exist for all models)
        try {
            auto token_type_ids_tensor = infer_request.get_tensor("token_type_ids");
            const int64_t* token_type_ids_data = token_type_ids_tensor.data<const int64_t>();
            result.token_type_ids.assign(token_type_ids_data, token_type_ids_data + actual_length);
        } catch (...) {
            // If not present, fill with zeros
            result.token_type_ids.resize(actual_length, 0);
        }
        
        result.success = true;
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Tokenization error: " << e.what() << std::endl;
        return result;
    }
}

} // namespace core
} // namespace openvino_sr

