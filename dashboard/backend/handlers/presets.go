package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
)

type PresetModel struct {
	Name string `json:"name"`
	Role string `json:"role"`
}

type PresetInfo struct {
	ID        string        `json:"id"`
	Label     string        `json:"label"`
	Summary   string        `json:"summary"`
	Models    []PresetModel `json:"required_models"`
	RecipeURL string        `json:"recipe_url"`
}

type PresetDeltaRequest struct {
	PresetID string   `json:"preset_id"`
	Models   []string `json:"models"`
}

type PresetDeltaResponse struct {
	PresetID         string        `json:"preset_id"`
	ConfiguredModels []string      `json:"configured_models"`
	MissingModels    []PresetModel `json:"missing_models"`
	Ready            bool          `json:"ready"`
	RecipeURL        string        `json:"recipe_url"`
}

var builtinPresets = []PresetInfo{
	{
		ID:      "balance",
		Label:   "Balance",
		Summary: "Default trade-off between cost, quality, and safety across five model tiers.",
		Models: []PresetModel{
			{Name: "qwen/qwen3.5-rocm", Role: "simple"},
			{Name: "google/gemini-2.5-flash-lite", Role: "medium"},
			{Name: "google/gemini-3.1-pro", Role: "complex"},
			{Name: "openai/gpt5.4", Role: "reasoning"},
			{Name: "anthropic/claude-opus-4.6", Role: "premium"},
		},
		RecipeURL: "https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/recipes/balance.yaml",
	},
	{
		ID:      "security",
		Label:   "Security",
		Summary: "Privacy-first routing that keeps sensitive requests on local models and restricts cloud egress.",
		Models: []PresetModel{
			{Name: "local/private-qwen", Role: "local-private"},
			{Name: "cloud/frontier-reasoning", Role: "cloud-reasoning"},
		},
		RecipeURL: "https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/recipes/privacy/privacy-router.yaml",
	},
}

func PresetsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		data, err := json.Marshal(builtinPresets)
		if err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(data); err != nil {
			log.Printf("presets: write error: %v", err)
		}
	}
}

func PresetDeltaHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req PresetDeltaRequest
		if status, err := decodeBoundedJSON(w, r, smallJSONRequestBodyLimit, &req); err != nil {
			http.Error(w, "Invalid request body", status)
			return
		}

		preset := findPreset(req.PresetID)
		if preset == nil {
			http.Error(w, "Unknown preset ID", http.StatusNotFound)
			return
		}

		configured, missing := computeModelDelta(preset.Models, req.Models)

		data, err := json.Marshal(PresetDeltaResponse{
			PresetID:         preset.ID,
			ConfiguredModels: configured,
			MissingModels:    missing,
			Ready:            len(missing) == 0,
			RecipeURL:        preset.RecipeURL,
		})
		if err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(data); err != nil {
			log.Printf("presets/delta: write error: %v", err)
		}
	}
}

func findPreset(id string) *PresetInfo {
	for i := range builtinPresets {
		if builtinPresets[i].ID == id {
			return &builtinPresets[i]
		}
	}
	return nil
}

func computeModelDelta(required []PresetModel, userModels []string) (configured []string, missing []PresetModel) {
	have := make(map[string]bool, len(userModels))
	for _, m := range userModels {
		have[strings.ToLower(strings.TrimSpace(m))] = true
	}

	for _, rm := range required {
		if have[strings.ToLower(rm.Name)] {
			configured = append(configured, rm.Name)
		} else {
			missing = append(missing, rm)
		}
	}

	if configured == nil {
		configured = []string{}
	}
	if missing == nil {
		missing = []PresetModel{}
	}

	return configured, missing
}
