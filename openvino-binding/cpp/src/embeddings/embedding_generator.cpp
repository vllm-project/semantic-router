#include "../../include/embeddings/embedding_generator.h"
#include "../../include/core/model_manager.h"
#include "../../include/utils/math_utils.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>

namespace openvino_sr {
namespace embeddings {

// Constants for special tokens (ModernBERT)
static const int MODERNBERT_PAD = 50283;

static int getPositiveEnvOrDefault(const char* key, int def) {
    if (const char* v = std::getenv(key)) {
        int parsed = std::atoi(v);
        return parsed > 0 ? parsed : def;
    }
    return def;
}

bool EmbeddingGenerator::initialize(
    const std::string& model_path,
    const std::string& device
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        auto& manager = core::ModelManager::getInstance();
        manager.ensureCoreInitialized();
        
        // Create model instance
        model_ = std::make_shared<core::ModelInstance>();
        model_->model_path = model_path;
        
          // Configure for better concurrency.
          // OPENVINO_EMBEDDING_TOTAL_THREADS: total threads OV can use (0 = auto from CPU affinity).
          // OPENVINO_EMBEDDING_NUM_STREAMS: number of parallel inference streams.
          // OPENVINO_EMBEDDING_NUM_REQUESTS: hint for expected concurrent requests.
          const int total_threads = getPositiveEnvOrDefault("OPENVINO_EMBEDDING_TOTAL_THREADS", 0);
          const int num_streams = getPositiveEnvOrDefault("OPENVINO_EMBEDDING_NUM_STREAMS", 0);
          const int num_requests = getPositiveEnvOrDefault("OPENVINO_EMBEDDING_NUM_REQUESTS", 16);

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
          int actual_pool_size = getPositiveEnvOrDefault("OPENVINO_EMBEDDING_INFER_POOL_SIZE", std::min(optimal_requests, num_requests));

          // Create InferRequest pool for concurrent inference
          manager.createInferPool(*model_, static_cast<size_t>(actual_pool_size));

          std::cout << "✓ Configured embedding concurrency: total_threads=" << total_threads
                    << ", num_streams=" << num_streams
                    << ", num_requests=" << num_requests
                    << ", optimal_infer_requests=" << optimal_requests
                    << ", actual_pool_size=" << actual_pool_size << std::endl;
        
        // Load tokenizer vocabulary
        std::string model_dir = model_path;
        auto last_slash = model_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            model_dir = model_dir.substr(0, last_slash);
        }
        tokenizer_.loadVocab(model_dir);
        
        std::cout << "OpenVINO embedding model initialized: " << model_path 
                  << " on " << device << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize embedding model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> EmbeddingGenerator::generateEmbedding(
    const std::string& text,
    int max_length
) {
    if (!model_ || !model_->compiled_model) {
        std::cerr << "Embedding model not initialized" << std::endl;
        return {};
    }
    
    try {
        // Tokenize text
        auto token_ids = tokenizer_.tokenize(text, max_length);
        if (token_ids.empty()) {
            std::cerr << "Tokenization failed or returned empty" << std::endl;
            return {};
        }
        
        size_t seq_len = token_ids.size();
        
          // Acquire an InferRequest slot and keep it locked for the full inference.
          auto& manager = core::ModelManager::getInstance();
          auto* slot = manager.acquireInferRequest(*model_);
          if (!slot) {
              std::cerr << "Failed to acquire InferRequest slot" << std::endl;
              return {};
          }

          // Keep the slot locked until tensors and inference are complete.
          std::unique_lock<std::mutex> request_lock(slot->mutex, std::adopt_lock);
        
        // Get model inputs
        auto inputs = model_->compiled_model->inputs();
        
        // Prepare input tensors for BERT (input_ids, attention_mask, token_type_ids)
        ov::Shape input_shape = {1, seq_len};
        
        // Set input_ids
        auto input_ids_tensor = ov::Tensor(ov::element::i64, input_shape);
        auto input_ids_data = input_ids_tensor.data<int64_t>();
        for (size_t i = 0; i < seq_len; ++i) {
            input_ids_data[i] = static_cast<int64_t>(token_ids[i]);
        }
        slot->request.set_input_tensor(0, input_ids_tensor);
        
        // Set attention_mask (1 for non-padding tokens, 0 for padding)
        if (inputs.size() > 1) {
            auto attention_mask_tensor = ov::Tensor(ov::element::i64, input_shape);
            auto mask_data = attention_mask_tensor.data<int64_t>();
            for (size_t i = 0; i < seq_len; ++i) {
                mask_data[i] = (token_ids[i] != MODERNBERT_PAD) ? 1 : 0;
            }
            slot->request.set_input_tensor(1, attention_mask_tensor);
        }
        
        // Set token_type_ids (all zeros for single sentence)
        if (inputs.size() > 2) {
            auto token_type_tensor = ov::Tensor(ov::element::i64, input_shape);
            auto type_data = token_type_tensor.data<int64_t>();
            std::fill(type_data, type_data + seq_len, 0);
            slot->request.set_input_tensor(2, token_type_tensor);
        }
        
        // Run inference
        slot->request.start_async();
        slot->request.wait();
        
        // Get output tensor
        auto output_tensor = slot->request.get_output_tensor(0);
        auto output_shape = output_tensor.get_shape();
        auto output_data = output_tensor.data<float>();
        
        // Extract embedding vector
        std::vector<float> embedding;
        
        if (output_shape.size() == 3) {
            // Output shape: [batch_size, seq_len, hidden_size]
            // For sentence-transformers models, use mean pooling
            size_t batch_size = output_shape[0];
            size_t sequence_length = output_shape[1];
            size_t hidden_size = output_shape[2];
            
            if (batch_size != 1) {
                std::cerr << "Unexpected batch size: " << batch_size << std::endl;
                return {};
            }
            
            // Mean pooling: average over all non-padding tokens
            embedding.resize(hidden_size, 0.0f);
            int valid_token_count = 0;
            
            for (size_t seq_idx = 0; seq_idx < sequence_length && seq_idx < seq_len; ++seq_idx) {
                if (token_ids[seq_idx] != MODERNBERT_PAD) {
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = seq_idx * hidden_size + h;
                        embedding[h] += output_data[idx];
                    }
                    valid_token_count++;
                }
            }
            
            // Average
            if (valid_token_count > 0) {
                for (size_t h = 0; h < hidden_size; ++h) {
                    embedding[h] /= valid_token_count;
                }
            }
            
        } else if (output_shape.size() == 2) {
            // Pooled output: [batch_size, hidden_size]
            size_t hidden_size = output_shape[1];
            embedding.assign(output_data, output_data + hidden_size);
        }
        
        return embedding;
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating embedding: " << e.what() << std::endl;
        return {};
    }
}

float EmbeddingGenerator::computeSimilarity(
    const std::string& text1,
    const std::string& text2,
    int max_length
) {
    try {
        auto emb1 = generateEmbedding(text1, max_length);
        auto emb2 = generateEmbedding(text2, max_length);
        
        if (emb1.empty() || emb2.empty()) {
            return -1.0f;
        }
        
        return utils::cosineSimilarity(emb1, emb2);
        
    } catch (const std::exception& e) {
        std::cerr << "Similarity calculation error: " << e.what() << std::endl;
        return -1.0f;
    }
}

core::SimilarityResult EmbeddingGenerator::findMostSimilar(
    const std::string& query,
    const std::vector<std::string>& candidates,
    int max_length
) {
    core::SimilarityResult result;
    result.index = -1;
    result.score = -1.0f;
    
    if (candidates.empty()) {
        return result;
    }
    
    try {
        auto query_emb = generateEmbedding(query, max_length);
        
        if (query_emb.empty()) {
            return result;
        }
        
        float best_score = -1.0f;
        int best_idx = -1;
        
        for (size_t i = 0; i < candidates.size(); ++i) {
            auto candidate_emb = generateEmbedding(candidates[i], max_length);
            if (candidate_emb.empty()) {
                continue;
            }
            
            float score = utils::cosineSimilarity(query_emb, candidate_emb);
            if (score > best_score) {
                best_score = score;
                best_idx = static_cast<int>(i);
            }
        }
        
        result.index = best_idx;
        result.score = best_score;
        
    } catch (const std::exception& e) {
        std::cerr << "Find most similar error: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<core::SimilarityMatch> EmbeddingGenerator::findTopKSimilar(
    const std::string& query,
    const std::vector<std::string>& candidates,
    int top_k,
    int max_length
) {
    std::vector<core::SimilarityMatch> matches;
    
    if (candidates.empty()) {
        return matches;
    }
    
    try {
        auto query_emb = generateEmbedding(query, max_length);
        
        if (query_emb.empty()) {
            return matches;
        }
        
        // Calculate similarities for all candidates
        for (size_t i = 0; i < candidates.size(); ++i) {
            auto candidate_emb = generateEmbedding(candidates[i], max_length);
            if (candidate_emb.empty()) {
                continue;
            }
            
            float score = utils::cosineSimilarity(query_emb, candidate_emb);
            matches.push_back({static_cast<int>(i), score});
        }
        
        // Sort by similarity (descending)
        std::sort(matches.begin(), matches.end(),
                  [](const core::SimilarityMatch& a, const core::SimilarityMatch& b) {
                      return a.similarity > b.similarity;
                  });
        
        // Take top-k (or all if top_k == 0)
        int k = (top_k == 0 || top_k > static_cast<int>(matches.size())) 
                ? static_cast<int>(matches.size()) : top_k;
        
        matches.resize(k);
        
    } catch (const std::exception& e) {
        std::cerr << "Find top-K similar error: " << e.what() << std::endl;
    }
    
    return matches;
}

} // namespace embeddings
} // namespace openvino_sr

