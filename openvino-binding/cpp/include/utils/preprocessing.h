#pragma once

#include "../core/types.h"
#include "../core/tokenizer.h"
#include <openvino/openvino.hpp>
#include <string>
#include <map>

namespace openvino_sr {
namespace utils {

/**
 * @brief Prepare BERT input tensors from text
 * 
 * @param text Input text to tokenize
 * @param max_length Maximum sequence length
 * @param tokenizer Tokenizer instance
 * @param model Compiled model (to get input tensor specs)
 * @return Map of input tensor names to tensors
 */
std::map<std::string, ov::Tensor> prepareBertInputs(
    const std::string& text,
    int max_length,
    core::OVNativeTokenizer& tokenizer,
    const ov::CompiledModel& model
);

/**
 * @brief Helper to duplicate a C string (for FFI)
 */
char* strDup(const char* str);

} // namespace utils
} // namespace openvino_sr

