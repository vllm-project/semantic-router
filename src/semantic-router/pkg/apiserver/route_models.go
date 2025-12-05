//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"time"
)

// handleOpenAIModels handles OpenAI-compatible model listing at /v1/models
// It returns the configured auto model name and optionally the underlying models from config.
// Whether to include configured models is controlled by the config's IncludeConfigModelsInList setting (default: false)
func (s *ClassificationAPIServer) handleOpenAIModels(w http.ResponseWriter, _ *http.Request) {
	now := time.Now().Unix()

	// Start with the configured auto model name (or default "MoM")
	// The model list uses the actual configured name, not "auto"
	// However, "auto" is still accepted as an alias in request handling for backward compatibility
	models := []OpenAIModel{}

	// Add the effective auto model name (configured or default "MoM")
	if s.config != nil {
		effectiveAutoModelName := s.config.GetEffectiveAutoModelName()
		models = append(models, OpenAIModel{
			ID:          effectiveAutoModelName,
			Object:      "model",
			Created:     now,
			OwnedBy:     "vllm-semantic-router",
			Description: "Intelligent Router for Mixture-of-Models",
			LogoURL:     "https://github.com/vllm-project/semantic-router/blob/main/website/static/img/vllm.png", // You can customize this URL
		})
	} else {
		// Fallback if no config
		models = append(models, OpenAIModel{
			ID:          "MoM",
			Object:      "model",
			Created:     now,
			OwnedBy:     "vllm-semantic-router",
			Description: "Intelligent Router for Mixture-of-Models",
			LogoURL:     "https://github.com/vllm-project/semantic-router/blob/main/website/static/img/vllm.png", // You can customize this URL
		})
	}

	// Append underlying models from config (if available and configured to include them)
	if s.config != nil && s.config.IncludeConfigModelsInList {
		for _, m := range s.config.GetAllModels() {
			// Skip if already added as the configured auto model name (avoid duplicates)
			if m == s.config.GetEffectiveAutoModelName() {
				continue
			}
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
