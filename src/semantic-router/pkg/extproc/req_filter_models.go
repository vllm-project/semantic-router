package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
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

	// Start with the configured auto model name (or default "MoM")
	// The model list uses the actual configured name, not "auto"
	// However, "auto" is still accepted as an alias in request handling for backward compatibility
	models := []OpenAIModel{}

	// Add the effective auto model name (configured or default "MoM")
	if r.Config != nil {
		effectiveAutoModelName := r.Config.GetEffectiveAutoModelName()
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
	if r.Config != nil && r.Config.IncludeConfigModelsInList {
		for _, m := range r.Config.GetAllModels() {
			// Skip if already added as the configured auto model name (avoid duplicates)
			if m == r.Config.GetEffectiveAutoModelName() {
				continue
			}
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
