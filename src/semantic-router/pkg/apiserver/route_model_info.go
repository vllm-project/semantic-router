//go:build !windows && cgo

package apiserver

import (
	"net/http"
)

func (s *ClassificationAPIServer) handleModelsInfo(w http.ResponseWriter, _ *http.Request) {
	response := s.buildModelsInfoResponse()
	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleEmbeddingModelsInfo handles GET /api/v1/embeddings/models
// Returns ONLY embedding models information
func (s *ClassificationAPIServer) handleEmbeddingModelsInfo(w http.ResponseWriter, r *http.Request) {
	embeddingModels := s.getEmbeddingModelsInfo(s.loadModelsRuntimeState())

	response := map[string]interface{}{
		"models": embeddingModels,
		"count":  len(embeddingModels),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleClassifierInfo(w http.ResponseWriter, _ *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
			"status": "no_config",
			"config": nil,
		})
		return
	}

	// Return the config directly
	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"status": "config_loaded",
		"config": jsonCompatibleValue(cfg),
	})
}

type classifierModelAvailability struct {
	core                   bool
	factCheck              bool
	hallucination          bool
	hallucinationExplainer bool
	feedback               bool
}

// buildModelsInfoResponse builds the models info response
func (s *ClassificationAPIServer) buildModelsInfoResponse() ModelsInfoResponse {
	runtimeState := s.loadModelsRuntimeState()
	models := s.getClassifierModelsInfo(s.classifierModelAvailability(), runtimeState)

	// Add embedding models information
	embeddingModels := s.getEmbeddingModelsInfo(runtimeState)
	models = append(models, embeddingModels...)

	// Get system information
	systemInfo := s.getSystemInfo()

	return ModelsInfoResponse{
		Models:  models,
		Summary: buildModelsInfoSummary(runtimeState, models),
		System:  systemInfo,
	}
}
