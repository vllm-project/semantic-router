#include "../../include/classifiers/text_classifier.h"
#include "../../include/core/model_manager.h"
#include "../../include/utils/math_utils.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace openvino_sr {
namespace classifiers {

bool TextClassifier::initialize(
    const std::string& model_path,
    int num_classes,
    const std::string& device
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        auto& manager = core::ModelManager::getInstance();
        manager.ensureCoreInitialized();
        
        // Create model instance
        model_ = std::make_shared<core::ModelInstance>();
        model_->num_classes = num_classes;
        model_->model_path = model_path;
          // OPENVINO_CLASSIFIER_TOTAL_THREADS: total threads OV can use (0 = auto from CPU affinity).
          // OPENVINO_CLASSIFIER_NUM_STREAMS: number of parallel inference streams.
          // OPENVINO_CLASSIFIER_NUM_REQUESTS: hint for expected concurrent requests.
          const int total_threads = []() {
              if (const char* v = std::getenv("OPENVINO_CLASSIFIER_TOTAL_THREADS")) {
                  int parsed = std::atoi(v);
                  return parsed > 0 ? parsed : 0;
              }
              return 0;
          }();
          const int num_streams = []() {
              if (const char* v = std::getenv("OPENVINO_CLASSIFIER_NUM_STREAMS")) {
                  int parsed = std::atoi(v);
                  return parsed > 0 ? parsed : 0;
              }
              return 0;
          }();
          const int num_requests = []() {
              if (const char* v = std::getenv("OPENVINO_CLASSIFIER_NUM_REQUESTS")) {
                  int parsed = std::atoi(v);
                  return parsed > 0 ? parsed : 16;
              }
              return 16;
          }();
          int infer_pool_size = num_requests;
          if (const char* v = std::getenv("OPENVINO_CLASSIFIER_INFER_POOL_SIZE")) {
              int parsed = std::atoi(v);
              if (parsed > 0) {
                  infer_pool_size = parsed;
              }
          }

          ov::AnyMap config;
          config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
          config[ov::hint::num_requests.name()] = num_requests;
          if (total_threads > 0) {
              config[ov::inference_num_threads.name()] = total_threads;
          }
          if (num_streams > 0) {
              config[ov::num_streams.name()] = num_streams;
          }

          // Load and compile model
          model_->compiled_model = manager.loadModel(model_path, device, config);
          if (!model_->compiled_model) {
              return false;
          }

          // Query OV for optimal number of infer requests, use as pool default if env var not set.
          int optimal_requests = static_cast<int>(
              model_->compiled_model->get_property(ov::optimal_number_of_infer_requests));
          if (const char* v = std::getenv("OPENVINO_CLASSIFIER_INFER_POOL_SIZE")) {
              int parsed = std::atoi(v);
              if (parsed > 0) infer_pool_size = parsed;
          } else {
              infer_pool_size = std::min(optimal_requests, num_requests);
          }

          std::cout << "â Configured classifier concurrency: total_threads=" << total_threads
                    << ", num_streams=" << num_streams
                    << ", num_requests=" << num_requests
                    << ", optimal_infer_requests=" << optimal_requests
                    << ", actual_pool_size=" << infer_pool_size << std::endl;

          // Create InferRequest pool for concurrent inference
          manager.createInferPool(*model_, static_cast<size_t>(infer_pool_size));
        
        // Load tokenizer vocabulary
        std::string model_dir = model_path;
        auto last_slash = model_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            model_dir = model_dir.substr(0, last_slash);
        }
        tokenizer_.loadVocab(model_dir);

          // Detect attention_mask input name at init time to avoid per-inference try-catch.
          attention_mask_name_ = "attention_mask";
          {
              auto inputs = model_->compiled_model->inputs();
              bool found = false;
              for (const auto& input : inputs) {
                  for (const auto& name : input.get_names()) {
                      if (name == "attention_mask") { found = true; break; }
                  }
                  if (found) break;
              }
              if (!found) {
                  // Fallback: some models use numeric index names
                  for (const auto& input : inputs) {
                      for (const auto& name : input.get_names()) {
                          if (name != "input_ids" && name != "token_type_ids") {
                              attention_mask_name_ = name;
                              break;
                          }
                      }
                  }
              }
              std::cout << "✓ attention_mask input name: " << attention_mask_name_ << std::endl;
          }

        
        std::cout << "OpenVINO classifier initialized: " << model_path 
                  << " on " << device << " with " << num_classes << " classes" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize classifier: " << e.what() << std::endl;
        return false;
    }
}

core::ClassificationResult TextClassifier::classify(const std::string& text) {
    core::ClassificationResult result;
    result.predicted_class = -1;
    result.confidence = 0.0f;
    
    if (!model_ || !model_->compiled_model) {
        std::cerr << "Classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        // Tokenize input
        std::vector<int> token_ids = tokenizer_.tokenize(text, 8192);
        
        if (token_ids.empty()) {
            std::cerr << "Tokenization failed or returned empty" << std::endl;
            return result;
        }
        
        // Create attention mask (ModernBERT uses 50283 as PAD token)
        const int MODERNBERT_PAD = 50283;
        std::vector<int64_t> attention_mask(token_ids.size());
        for (size_t i = 0; i < token_ids.size(); ++i) {
            attention_mask[i] = (token_ids[i] != MODERNBERT_PAD) ? 1 : 0;
        }
        
        // Convert to i64 for ModernBERT
        std::vector<int64_t> token_ids_i64(token_ids.begin(), token_ids.end());
        
        // Create input tensors
        ov::Tensor input_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
        std::memcpy(input_ids_tensor.data<int64_t>(), token_ids_i64.data(), 
                    token_ids_i64.size() * sizeof(int64_t));
        
        ov::Tensor attention_mask_tensor(ov::element::i64, {1, attention_mask.size()});
        std::memcpy(attention_mask_tensor.data<int64_t>(), attention_mask.data(), 
                    attention_mask.size() * sizeof(int64_t));
        
        // Acquire an InferRequest slot and keep it locked for the full inference.
        auto& manager = core::ModelManager::getInstance();
        auto* slot = manager.acquireInferRequest(*model_);
        
        // Keep the slot locked until tensors and inference are complete.
        std::unique_lock<std::mutex> request_lock(slot->mutex, std::adopt_lock);
        
        // Set tensors and run inference
        slot->request.set_tensor("input_ids", input_ids_tensor);
        slot->request.set_tensor(attention_mask_name_, attention_mask_tensor);
        slot->request.start_async();
        slot->request.wait();
        
        // Get output tensor by name (logits: [batch_size, num_classes])
        auto output_tensor = slot->request.get_tensor("logits");
        const float* logits = output_tensor.data<const float>();
        
        auto shape = output_tensor.get_shape();
        size_t num_classes = shape[1];
        
        // Apply softmax to logits
        std::vector<float> logits_vec(logits, logits + num_classes);
        auto probs = utils::softmax(logits_vec);
        
        // Find max probability and corresponding class
        auto max_it = std::max_element(probs.begin(), probs.end());
        result.predicted_class = static_cast<int>(std::distance(probs.begin(), max_it));
        result.confidence = *max_it;
        
    } catch (const std::exception& e) {
        std::cerr << "Classification error: " << e.what() << std::endl;
    }
    
    return result;
}

core::ClassificationResultWithProbs TextClassifier::classifyWithProbabilities(const std::string& text) {
    core::ClassificationResultWithProbs result;
    result.predicted_class = -1;
    result.confidence = 0.0f;
    
    if (!model_ || !model_->compiled_model) {
        std::cerr << "Classifier not initialized" << std::endl;
        return result;
    }
    
    try {
        // Tokenize input
        std::vector<int> token_ids = tokenizer_.tokenize(text, 8192);
        
        if (token_ids.empty()) {
            std::cerr << "Tokenization failed or returned empty" << std::endl;
            return result;
        }
        
        // Create attention mask (ModernBERT uses 50283 as PAD token)
        const int MODERNBERT_PAD = 50283;
        std::vector<int64_t> attention_mask(token_ids.size());
        for (size_t i = 0; i < token_ids.size(); ++i) {
            attention_mask[i] = (token_ids[i] != MODERNBERT_PAD) ? 1 : 0;
        }
        
        // Convert to i64
        std::vector<int64_t> token_ids_i64(token_ids.begin(), token_ids.end());
        
        // Create input tensors
        ov::Tensor input_ids_tensor(ov::element::i64, {1, token_ids_i64.size()});
        std::memcpy(input_ids_tensor.data<int64_t>(), token_ids_i64.data(), 
                    token_ids_i64.size() * sizeof(int64_t));
        
        ov::Tensor attention_mask_tensor(ov::element::i64, {1, attention_mask.size()});
        std::memcpy(attention_mask_tensor.data<int64_t>(), attention_mask.data(), 
                    attention_mask.size() * sizeof(int64_t));
        
        // Acquire an InferRequest slot and keep it locked for the full inference.
        auto& manager = core::ModelManager::getInstance();
        auto* slot = manager.acquireInferRequest(*model_);
        
        // Keep the slot locked until tensors and inference are complete.
        std::unique_lock<std::mutex> request_lock(slot->mutex, std::adopt_lock);
        
        // Set tensors and run inference
        slot->request.set_tensor("input_ids", input_ids_tensor);
        slot->request.set_tensor(attention_mask_name_, attention_mask_tensor);
        slot->request.start_async();
        slot->request.wait();
        
        // Get output tensor
        auto output_tensor = slot->request.get_tensor("logits");
        const float* logits = output_tensor.data<const float>();
        
        auto shape = output_tensor.get_shape();
        size_t num_classes = shape[1];
        
        // Apply softmax to logits
        std::vector<float> logits_vec(logits, logits + num_classes);
        auto probs = utils::softmax(logits_vec);
        
        // Find max probability and corresponding class
        auto max_it = std::max_element(probs.begin(), probs.end());
        result.predicted_class = static_cast<int>(std::distance(probs.begin(), max_it));
        result.confidence = *max_it;
        result.probabilities = probs;
        
    } catch (const std::exception& e) {
        std::cerr << "Classification with probabilities error: " << e.what() << std::endl;
    }
    
    return result;
}

} // namespace classifiers
} // namespace openvino_sr

