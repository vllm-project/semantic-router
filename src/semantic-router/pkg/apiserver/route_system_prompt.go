//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type SystemPromptInfo struct {
	Category string `json:"category"`
	Prompt   string `json:"prompt"`
	Enabled  bool   `json:"enabled"`
	Mode     string `json:"mode"` // "replace" or "insert"
}

// SystemPromptsResponse represents the response for GET /config/system-prompts
type SystemPromptsResponse struct {
	SystemPrompts []SystemPromptInfo `json:"system_prompts"`
}

// SystemPromptUpdateRequest represents a request to update system prompt settings
type SystemPromptUpdateRequest struct {
	Category string `json:"category,omitempty"` // If empty, applies to all categories
	Enabled  *bool  `json:"enabled,omitempty"`  // true to enable, false to disable
	Mode     string `json:"mode,omitempty"`     // "replace" or "insert"
}

// handleGetSystemPrompts handles GET /config/system-prompts
func (s *ClassificationAPIServer) handleGetSystemPrompts(w http.ResponseWriter, _ *http.Request) {
	cfg := s.config
	if cfg == nil {
		http.Error(w, "Configuration not available", http.StatusInternalServerError)
		return
	}

	var systemPrompts []SystemPromptInfo
	for _, category := range cfg.Categories {
		systemPrompts = append(systemPrompts, SystemPromptInfo{
			Category: category.Name,
			Prompt:   category.SystemPrompt,
			Enabled:  category.IsSystemPromptEnabled(),
			Mode:     category.GetSystemPromptMode(),
		})
	}

	response := SystemPromptsResponse{
		SystemPrompts: systemPrompts,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

// handleUpdateSystemPrompts handles PUT /config/system-prompts
func (s *ClassificationAPIServer) handleUpdateSystemPrompts(w http.ResponseWriter, r *http.Request) {
	var req SystemPromptUpdateRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Enabled == nil && req.Mode == "" {
		http.Error(w, "either enabled or mode field is required", http.StatusBadRequest)
		return
	}

	// Validate mode if provided
	if req.Mode != "" && req.Mode != "replace" && req.Mode != "insert" {
		http.Error(w, "mode must be either 'replace' or 'insert'", http.StatusBadRequest)
		return
	}

	cfg := s.config
	if cfg == nil {
		http.Error(w, "Configuration not available", http.StatusInternalServerError)
		return
	}

	// Create a copy of the config to modify
	newCfg := *cfg
	newCategories := make([]config.Category, len(cfg.Categories))
	copy(newCategories, cfg.Categories)
	newCfg.Categories = newCategories

	updated := false
	if req.Category == "" {
		// Update all categories
		for i := range newCfg.Categories {
			if newCfg.Categories[i].SystemPrompt != "" {
				if req.Enabled != nil {
					newCfg.Categories[i].SystemPromptEnabled = req.Enabled
				}
				if req.Mode != "" {
					newCfg.Categories[i].SystemPromptMode = req.Mode
				}
				updated = true
			}
		}
	} else {
		// Update specific category
		for i := range newCfg.Categories {
			if newCfg.Categories[i].Name == req.Category {
				if newCfg.Categories[i].SystemPrompt == "" {
					http.Error(w, fmt.Sprintf("Category '%s' has no system prompt configured", req.Category), http.StatusBadRequest)
					return
				}
				if req.Enabled != nil {
					newCfg.Categories[i].SystemPromptEnabled = req.Enabled
				}
				if req.Mode != "" {
					newCfg.Categories[i].SystemPromptMode = req.Mode
				}
				updated = true
				break
			}
		}
		if !updated {
			http.Error(w, fmt.Sprintf("Category '%s' not found", req.Category), http.StatusNotFound)
			return
		}
	}

	if !updated {
		http.Error(w, "No categories with system prompts found to update", http.StatusBadRequest)
		return
	}

	// Update the configuration
	s.config = &newCfg
	s.classificationSvc.UpdateConfig(&newCfg)

	// Return the updated system prompts
	var systemPrompts []SystemPromptInfo
	for _, category := range newCfg.Categories {
		systemPrompts = append(systemPrompts, SystemPromptInfo{
			Category: category.Name,
			Prompt:   category.SystemPrompt,
			Enabled:  category.IsSystemPromptEnabled(),
			Mode:     category.GetSystemPromptMode(),
		})
	}

	response := SystemPromptsResponse{
		SystemPrompts: systemPrompts,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}
