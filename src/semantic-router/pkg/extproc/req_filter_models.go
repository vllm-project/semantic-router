package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// OpenAIModel represents a single model in the OpenAI /v1/models response
type OpenAIModel struct {
	ID          string `json:"id"`
	Object      string `json:"object"`
	Created     int64  `json:"created"`
	OwnedBy     string `json:"owned_by"`
	Description string `json:"description,omitempty"` // Optional description for Chat UI
	LogoURL     string `json:"logo_url,omitempty"`    // Optional logo URL for Chat UI
}

// OpenAIModelList is the container for the models list response
type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// handleModelsRequest handles GET /v1/models requests and returns a direct response
// Whether to include configured models is controlled by the config's IncludeConfigModelsInList setting (default: false)
func (r *OpenAIRouter) handleModelsRequest(_ string) (*ext_proc.ProcessingResponse, error) {
	now := time.Now().Unix()

	// Start with all configured auto model aliases.
	models := []OpenAIModel{}
	seenModels := map[string]bool{}

	autoModelNames := config.DefaultAutoModelNames()
	if r.Config != nil {
		autoModelNames = r.Config.EffectiveAutoModelNames()
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
	if r.Config != nil && r.Config.IncludeConfigModelsInList {
		for _, m := range r.Config.GetAllModels() {
			if seenModels[m] {
				continue
			}
			seenModels[m] = true
			models = append(models, OpenAIModel{
				ID:      m,
				Object:  "model",
				Created: now,
				OwnedBy: "vllm-semantic-router",
			})
		}
	}

	resp := OpenAIModelList{
		Object: "list",
		Data:   models,
	}

	return r.createJSONResponse(200, resp), nil
}
