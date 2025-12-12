#include "../../include/utils/preprocessing.h"
#include <cstring>
#include <iostream>

namespace openvino_sr {
namespace utils {

std::map<std::string, ov::Tensor> prepareBertInputs(
    const std::string& text,
    int max_length,
    core::OVNativeTokenizer& tokenizer,
    const ov::CompiledModel& model
) {
    std::map<std::string, ov::Tensor> tensors;
    
    try {
        // Get full tokenization result
        auto token_result = tokenizer.tokenizeFull(text, max_length);
        if (!token_result.success || token_result.input_ids.empty()) {
            std::cerr << "Tokenization failed" << std::endl;
            return tensors;
        }
        
        size_t seq_len = token_result.input_ids.size();
        ov::Shape input_shape = {1, seq_len};
        
        // Create input_ids tensor
        ov::Tensor input_ids_tensor(ov::element::i64, input_shape);
        std::memcpy(input_ids_tensor.data<int64_t>(), 
                    token_result.input_ids.data(), 
                    seq_len * sizeof(int64_t));
        tensors["input_ids"] = input_ids_tensor;
        
        // Create attention_mask tensor
        if (!token_result.attention_mask.empty()) {
            ov::Tensor attention_mask_tensor(ov::element::i64, input_shape);
            std::memcpy(attention_mask_tensor.data<int64_t>(), 
                        token_result.attention_mask.data(), 
                        seq_len * sizeof(int64_t));
            tensors["attention_mask"] = attention_mask_tensor;
            // Some models use different names
            tensors["101"] = attention_mask_tensor;  // Fallback name
        }
        
        // Create token_type_ids tensor
        if (!token_result.token_type_ids.empty()) {
            ov::Tensor token_type_tensor(ov::element::i64, input_shape);
            std::memcpy(token_type_tensor.data<int64_t>(), 
                        token_result.token_type_ids.data(), 
                        seq_len * sizeof(int64_t));
            tensors["token_type_ids"] = token_type_tensor;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error preparing BERT inputs: " << e.what() << std::endl;
    }
    
    return tensors;
}

char* strDup(const char* str) {
    if (!str) return nullptr;
    size_t len = std::strlen(str);
    char* dup = new char[len + 1];
    std::strcpy(dup, str);
    return dup;
}

} // namespace utils
} // namespace openvino_sr

