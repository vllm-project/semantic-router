//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// handleOpenAIModels handles OpenAI-compatible model listing at /v1/models
// It returns the configured auto model name and optionally the underlying models from config.
// Whether to include configured models is controlled by the config's IncludeConfigModelsInList setting (default: false)
func (s *ClassificationAPIServer) handleOpenAIModels(w http.ResponseWriter, _ *http.Request) {
	now := time.Now().Unix()
	cfg := s.currentConfig()

	// Start with all configured auto model aliases.
	models := []OpenAIModel{}
	seenModels := map[string]bool{}

	autoModelNames := config.DefaultAutoModelNames()
	if cfg != nil {
		autoModelNames = cfg.EffectiveAutoModelNames()
	}
	for _, autoModelName := range autoModelNames {
		if autoModelName == "" || seenModels[autoModelName] {
			continue
		}
		seenModels[autoModelName] = true
		models = append(models, OpenAIModel{
			ID:          autoModelName,
			Object:      "model",
			Created:     now,
			OwnedBy:     "vllm-semantic-router",
			Description: "Intelligent Router for Mixture-of-Models",
		})
	}

	// Append underlying models from config (if available and configured to include them)
	if cfg != nil && cfg.IncludeConfigModelsInList {
		for _, m := range cfg.GetAllModels() {
			if seenModels[m] {
				continue
			}
			seenModels[m] = true
			models = append(models, OpenAIModel{
				ID:      m,
				Object:  "model",
				Created: now,
				OwnedBy: "upstream-endpoint",
			})
		}
	}

	resp := OpenAIModelList{
		Object: "list",
		Data:   models,
	}

	s.writeJSONResponse(w, http.StatusOK, resp)
}
