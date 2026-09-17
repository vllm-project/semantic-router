//go:build !windows && cgo

package apiserver

import (
	"net/http"
)

type embeddingModelsResponse struct {
	Models []ModelInfo `json:"models"`
	Count  int         `json:"count"`
}

type classifierConfigResponse struct {
	Status string `json:"status"`
	Config any    `json:"config"`
}

func (s *ClassificationAPIServer) handleModelsInfo(w http.ResponseWriter, _ *http.Request) {
	response := s.buildModelsInfoResponse()
	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleEmbeddingModelsInfo handles GET /api/v1/inventory/embedding-models
// Returns ONLY embedding models information
func (s *ClassificationAPIServer) handleEmbeddingModelsInfo(w http.ResponseWriter, r *http.Request) {
	embeddingModels := s.getEmbeddingModelsInfo(s.loadModelsRuntimeState())

	response := embeddingModelsResponse{
		Models: embeddingModels,
		Count:  len(embeddingModels),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleClassifierInfo returns live classifier/runtime config.
// Access requires config.read; plaintext secrets require secret_view (otherwise redacted).
func (s *ClassificationAPIServer) handleClassifierInfo(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeJSONResponse(w, http.StatusOK, classifierConfigResponse{
			Status: "no_config",
			Config: nil,
		})
		return
	}

	s.writeJSONResponse(w, http.StatusOK, classifierConfigResponse{
		Status: "config_loaded",
		Config: s.maybeRedactConfigView(r, jsonCompatibleValue(cfg)),
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
	cfg, service, release := s.acquireClassificationRuntime()
	defer release()
	models, prepared := preparedModelsInfo(service)
	if !prepared {
		models = s.getClassifierModelsInfo(cfg, classificationAvailabilityForService(service), runtimeState)
		models = append(models, s.getEmbeddingModelsInfo(runtimeState)...)
	}
	systemInfo := s.getSystemInfo()
	systemInfo.GPUAvailable = modelsUseGPU(models)
	summary := buildModelsInfoSummary(runtimeState, models)
	if prepared {
		// Startup status counts downloaded artifacts; the live inventory counts
		// task bindings. Do not inflate a published snapshot with unused models.
		summary.LoadedModels = len(models)
		if runtimeState == nil || runtimeState.Ready {
			summary.TotalModels = len(models)
		}
	}

	return ModelsInfoResponse{
		Models:  models,
		Summary: summary,
		System:  systemInfo,
	}
}
