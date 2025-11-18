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
	for _, decision := range cfg.Decisions {
		systemPromptConfig := decision.GetSystemPromptConfig()
		prompt := ""
		if systemPromptConfig != nil {
			prompt = systemPromptConfig.SystemPrompt
		}
		systemPrompts = append(systemPrompts, SystemPromptInfo{
			Category: decision.Name,
			Prompt:   prompt,
			Enabled:  decision.IsSystemPromptEnabled(),
			Mode:     decision.GetSystemPromptMode(),
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
	newDecisions := make([]config.Decision, len(cfg.Decisions))
	copy(newDecisions, cfg.Decisions)
	newCfg.Decisions = newDecisions

	updated := false
	if req.Category == "" {
		// Update all decisions
		for i := range newCfg.Decisions {
			systemPromptConfig := newCfg.Decisions[i].GetSystemPromptConfig()
			if systemPromptConfig != nil && systemPromptConfig.SystemPrompt != "" {
				// Update the plugin configuration
				for j := range newCfg.Decisions[i].Plugins {
					if newCfg.Decisions[i].Plugins[j].Type == "system_prompt" {
						// Convert Configuration to map[string]interface{}
						configMap, ok := newCfg.Decisions[i].Plugins[j].Configuration.(map[string]interface{})
						if !ok {
							// If not a map, create a new one
							configMap = make(map[string]interface{})
						}
						if req.Enabled != nil {
							configMap["enabled"] = *req.Enabled
						}
						if req.Mode != "" {
							configMap["mode"] = req.Mode
						}
						newCfg.Decisions[i].Plugins[j].Configuration = configMap
						updated = true
						break
					}
				}
			}
		}
	} else {
		// Update specific decision
		for i := range newCfg.Decisions {
			if newCfg.Decisions[i].Name == req.Category {
				systemPromptConfig := newCfg.Decisions[i].GetSystemPromptConfig()
				if systemPromptConfig == nil || systemPromptConfig.SystemPrompt == "" {
					http.Error(w, fmt.Sprintf("Decision '%s' has no system prompt configured", req.Category), http.StatusBadRequest)
					return
				}
				// Update the plugin configuration
				for j := range newCfg.Decisions[i].Plugins {
					if newCfg.Decisions[i].Plugins[j].Type == "system_prompt" {
						// Convert Configuration to map[string]interface{}
						configMap, ok := newCfg.Decisions[i].Plugins[j].Configuration.(map[string]interface{})
						if !ok {
							// If not a map, create a new one
							configMap = make(map[string]interface{})
						}
						if req.Enabled != nil {
							configMap["enabled"] = *req.Enabled
						}
						if req.Mode != "" {
							configMap["mode"] = req.Mode
						}
						newCfg.Decisions[i].Plugins[j].Configuration = configMap
						updated = true
						break
					}
				}
				break
			}
		}
		if !updated {
			http.Error(w, fmt.Sprintf("Decision '%s' not found", req.Category), http.StatusNotFound)
			return
		}
	}

	if !updated {
		http.Error(w, "No decisions with system prompts found to update", http.StatusBadRequest)
		return
	}

	// Update the configuration
	s.config = &newCfg
	s.classificationSvc.UpdateConfig(&newCfg)

	// Return the updated system prompts from decisions
	var systemPrompts []SystemPromptInfo
	for _, decision := range newCfg.Decisions {
		systemPromptConfig := decision.GetSystemPromptConfig()
		prompt := ""
		if systemPromptConfig != nil {
			prompt = systemPromptConfig.SystemPrompt
		}
		systemPrompts = append(systemPrompts, SystemPromptInfo{
			Category: decision.Name,
			Prompt:   prompt,
			Enabled:  decision.IsSystemPromptEnabled(),
			Mode:     decision.GetSystemPromptMode(),
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
