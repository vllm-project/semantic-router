//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"runtime"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (s *ClassificationAPIServer) handleModelsInfo(w http.ResponseWriter, _ *http.Request) {
	response := s.buildModelsInfoResponse()
	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleEmbeddingModelsInfo handles GET /api/v1/embeddings/models
// Returns ONLY embedding models information
func (s *ClassificationAPIServer) handleEmbeddingModelsInfo(w http.ResponseWriter, r *http.Request) {
	embeddingModels := s.getEmbeddingModelsInfo()

	response := map[string]interface{}{
		"models": embeddingModels,
		"count":  len(embeddingModels),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleClassifierInfo(w http.ResponseWriter, _ *http.Request) {
	if s.config == nil {
		s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
			"status": "no_config",
			"config": nil,
		})
		return
	}

	// Return the config directly
	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"status": "config_loaded",
		"config": s.config,
	})
}

// buildModelsInfoResponse builds the models info response
func (s *ClassificationAPIServer) buildModelsInfoResponse() ModelsInfoResponse {
	var models []ModelInfo

	// Check if we have a real classification service with classifier
	if s.classificationSvc != nil && s.classificationSvc.HasClassifier() {
		// Get model information from the classifier
		models = s.getLoadedModelsInfo()
	} else {
		// Return placeholder model info
		models = s.getPlaceholderModelsInfo()
	}

	// Add embedding models information
	embeddingModels := s.getEmbeddingModelsInfo()
	models = append(models, embeddingModels...)

	// Get system information
	systemInfo := s.getSystemInfo()

	return ModelsInfoResponse{
		Models: models,
		System: systemInfo,
	}
}

// getLoadedModelsInfo returns information about actually loaded models
func (s *ClassificationAPIServer) getLoadedModelsInfo() []ModelInfo {
	var models []ModelInfo

	if s.config == nil {
		return models
	}

	// Category classifier model
	if s.config.CategoryMappingPath != "" {
		categories := []string{}
		// Extract category names from config.Categories
		for _, cat := range s.config.Categories {
			categories = append(categories, cat.Name)
		}

		models = append(models, ModelInfo{
			Name:       "category_classifier",
			Type:       "intent_classification",
			Loaded:     true,
			ModelPath:  s.config.CategoryModel.ModelID,
			Categories: categories,
			Metadata: map[string]string{
				"mapping_path": s.config.CategoryMappingPath,
				"model_type":   "modernbert",
				"threshold":    fmt.Sprintf("%.2f", s.config.CategoryModel.Threshold),
			},
		})
	}

	// PII classifier model
	if s.config.PIIMappingPath != "" {
		models = append(models, ModelInfo{
			Name:      "pii_classifier",
			Type:      "pii_detection",
			Loaded:    true,
			ModelPath: s.config.PIIModel.ModelID,
			Metadata: map[string]string{
				"mapping_path": s.config.PIIMappingPath,
				"model_type":   "modernbert_token",
				"threshold":    fmt.Sprintf("%.2f", s.config.PIIModel.Threshold),
			},
		})
	}

	// Jailbreak classifier model
	if s.config.PromptGuard.Enabled {
		models = append(models, ModelInfo{
			Name:      "jailbreak_classifier",
			Type:      "security_detection",
			Loaded:    true,
			ModelPath: s.config.PromptGuard.JailbreakMappingPath,
			Metadata: map[string]string{
				"enabled": "true",
			},
		})
	}

	// BERT similarity model
	if s.config.BertModel.ModelID != "" {
		models = append(models, ModelInfo{
			Name:      "bert_similarity_model",
			Type:      "similarity",
			Loaded:    true,
			ModelPath: s.config.BertModel.ModelID,
			Metadata: map[string]string{
				"model_type": "sentence_transformer",
				"threshold":  fmt.Sprintf("%.2f", s.config.BertModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", s.config.BertModel.UseCPU),
			},
		})
	}

	return models
}

// getPlaceholderModelsInfo returns placeholder model information
func (s *ClassificationAPIServer) getPlaceholderModelsInfo() []ModelInfo {
	return []ModelInfo{
		{
			Name:   "category_classifier",
			Type:   "intent_classification",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "pii_classifier",
			Type:   "pii_detection",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "jailbreak_classifier",
			Type:   "security_detection",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
	}
}

// getSystemInfo returns system information
func (s *ClassificationAPIServer) getSystemInfo() SystemInfo {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return SystemInfo{
		GoVersion:    runtime.Version(),
		Architecture: runtime.GOARCH,
		OS:           runtime.GOOS,
		MemoryUsage:  fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
		GPUAvailable: false, // TODO: Implement GPU detection
	}
}

// getEmbeddingModelsInfo returns information about loaded embedding models
func (s *ClassificationAPIServer) getEmbeddingModelsInfo() []ModelInfo {
	var models []ModelInfo

	// Query embedding models info from Rust FFI
	embeddingInfo, err := candle_binding.GetEmbeddingModelsInfo()
	if err != nil {
		logging.Warnf("Failed to get embedding models info: %v", err)
		return models
	}

	// Convert to ModelInfo format
	for _, model := range embeddingInfo.Models {
		models = append(models, ModelInfo{
			Name:      fmt.Sprintf("%s_embedding_model", model.ModelName),
			Type:      "embedding",
			Loaded:    model.IsLoaded,
			ModelPath: model.ModelPath,
			Metadata: map[string]string{
				"model_type":           model.ModelName,
				"max_sequence_length":  fmt.Sprintf("%d", model.MaxSequenceLength),
				"default_dimension":    fmt.Sprintf("%d", model.DefaultDimension),
				"matryoshka_supported": "true",
			},
		})
	}

	return models
}
